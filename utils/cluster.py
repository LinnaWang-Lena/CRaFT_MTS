# utils/cluster.py
import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch_geometric.nn import MessagePassing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import hashlib
from datetime import datetime
import random

# --------------------------- 可复现：统一设种 ---------------------------
def set_seed_all(seed: int = 42):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------- 工具 ---------------------------
def _dataset_signature(input_folder, feature_cols):
    key = os.path.abspath(input_folder) + "||" + "|".join(sorted([str(c) for c in feature_cols]))
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _maybe_load_cached_model(model_dir, sig, device):
    meta_path = os.path.join(model_dir, "metadata.json")
    model_path = os.path.join(model_dir, "model.pth")
    if not (os.path.isfile(meta_path) and os.path.isfile(model_path)):
        return None, None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None, None
    if meta.get("signature") == sig:
        try:
            _ = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            _ = torch.load(model_path, map_location=device)
        return model_path, meta
    return None, None


def normalize_cols(cols):
    """只做去空格，不删除列，避免 Length mismatch。"""
    return [str(c).strip() for c in cols]


def df_to_batch_tensors(df, feature_cols, device):
    batch = df.shape[0]
    num_modalities = len(feature_cols)
    vals = df[feature_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    x_flag = ~np.isnan(vals)
    x_np = np.nan_to_num(vals, nan=0.0).astype(np.float32).reshape(batch, num_modalities, 1)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    x_flag = torch.from_numpy(x_flag.astype(np.uint8)).to(device=device)
    return x, x_flag


# --------------------------- 模型（省略，和你原来一样） ---------------------------
class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, normalize_emb, aggr="mean", **kwargs):
        super(EdgeSAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.normalize_emb = normalize_emb
        self.message_lin = nn.Linear(in_channels + edge_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)
        self.message_activation = nn.ReLU()
        self.update_activation = nn.ReLU()

    def forward(self, x, edge_attr, edge_index):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index):
        m_j = torch.cat((x_j, edge_attr), dim=-1)
        m_j = self.message_activation(self.message_lin(m_j))
        return m_j

    def update(self, aggr_out, x):
        aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x), dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class GNNStack(nn.Module):
    def __init__(self, node_channels, edge_channels, normalize_embs, num_layers, dropout):
        super(GNNStack, self).__init__()
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.normalize_embs = normalize_embs if normalize_embs is not None else [False] * num_layers
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList([
            EdgeSAGEConv(node_channels, node_channels, edge_channels, self.normalize_embs[l])
            for l in range(num_layers)
        ])
        self.edge_update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_channels + node_channels + edge_channels, edge_channels),
                nn.ReLU()
            ) for _ in range(num_layers - 1)
        ])

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]
        edge_attr = mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return edge_attr

    def forward(self, x, edge_attr, edge_index):
        for l, conv in enumerate(self.convs):
            x = conv(x, edge_attr, edge_index)
            if l < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
        return x


class SiameseGNN(nn.Module):
    def __init__(self, num_modalities, hidden_channels=128, normalize_embs=None, num_layers=2, dropout=0.25):
        super(SiameseGNN, self).__init__()
        self.num_modalities = num_modalities
        self.hidden_channels = hidden_channels
        self.modality_nodes = nn.Parameter(torch.randn(self.num_modalities, self.hidden_channels))
        self.gnn = GNNStack(self.hidden_channels, self.hidden_channels, normalize_embs, num_layers, dropout)
        self.relu = nn.ReLU()
        self.tau = nn.Parameter(torch.tensor(1 / 0.07), requires_grad=False)
        self.cl_projection = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.Tanh(),
        )
        self.simiProj = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            self.relu,
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            self.relu,
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
        )
        self.bn = nn.BatchNorm1d(self.hidden_channels)
        self.sigmoid = nn.Sigmoid()
        self.eps0 = nn.Parameter(torch.ones(1) + 1)

    @staticmethod
    def euclidean_dist(x, y):
        b = x.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(b, b)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(b, b).t()
        dist = xx + yy - 2 * torch.mm(x, y.t())
        return dist

    def gaussian_kernel(self, source, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
        n = source.size(0)
        L2_distance = self.euclidean_dist(source, source)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n**2 - n)
        if bandwidth < 1e-3:
            bandwidth = 1.0
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / b) for b in bandwidth_list]
        return sum(kernel_val) / len(kernel_val)

    @staticmethod
    def unsup_ce_loss(zaz_s):
        target = torch.arange(zaz_s.size(0), device=zaz_s.device)
        loss = F.cross_entropy(zaz_s, target)
        loss_t = F.cross_entropy(zaz_s.t(), target)
        return (loss + loss_t) / 2

    def edgedrop(self, flag):
        n, m = flag.size()
        flag = flag.clone()
        for i in range(n):
            count_ones = flag[i].sum().item()
            if torch.rand(1) < 0.5 or count_ones <= 1:
                continue
            keep_count = torch.randint(1, count_ones, (1,)).item()
            one_indices = (flag[i] == 1).nonzero(as_tuple=True)[0]
            one_indices = one_indices[torch.randperm(one_indices.size(0))]
            mask_count = count_ones - keep_count
            flag[i][one_indices[:mask_count]] = 0
        return flag

    def forward(self, x, x_flag):
        batch_size = x.size(0)
        device = x.device

        g_patient_nodes = torch.ones(batch_size, self.hidden_channels, device=device)
        g_nodes = torch.cat([g_patient_nodes, self.modality_nodes], dim=0)
        g_edge_index = x_flag.nonzero().t()
        g_edge_index[1] += batch_size
        g_edge_index = torch.cat([g_edge_index, g_edge_index.flip([0])], dim=1)
        edge_vals = x[x_flag]
        g_edge_attr = edge_vals.expand(-1, self.hidden_channels).repeat(2, 1)

        z = self.gnn(g_nodes, g_edge_attr, g_edge_index)[:batch_size]

        ag_flag = self.edgedrop(x_flag)
        ag_nodes = torch.cat([g_patient_nodes, self.modality_nodes], dim=0)
        ag_edge_index = ag_flag.nonzero().t()
        ag_edge_index[1] += batch_size
        ag_edge_index = torch.cat([ag_edge_index, ag_edge_index.flip([0])], dim=1)
        ag_edge_vals = x[ag_flag]
        ag_edge_attr = ag_edge_vals.expand(-1, self.hidden_channels).repeat(2, 1)
        az = self.gnn(ag_nodes, ag_edge_attr, ag_edge_index)[:batch_size]

        u = F.normalize(self.cl_projection(z), dim=-1)
        au = F.normalize(self.cl_projection(az), dim=-1)
        logits = torch.matmul(u, au.t()) * self.tau
        loss = self.unsup_ce_loss(logits)

        z_k1 = self.gaussian_kernel(self.bn(self.simiProj(z)), kernel_mul=2.0, kernel_num=3)
        z_k2 = self.gaussian_kernel(self.bn(z), kernel_mul=2.0, kernel_num=3)
        sim = ((1 - self.sigmoid(self.eps0)) * z_k1 + self.sigmoid(self.eps0)) * z_k2
        sim = torch.clamp(sim, 0.0, 1.0)
        return loss, sim


# --------------------------- 数据集 ---------------------------
class PatientDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, device):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_df = self.df.iloc[[idx]]
        x, x_flag = df_to_batch_tensors(sample_df, self.feature_cols, device=self.device)
        return x.squeeze(0), x_flag.squeeze(0)


# --------------------------- 训练/验证 ---------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total = []
    for x, x_flag in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)
        x_flag = x_flag.to(device).bool()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(x, x_flag)
        loss.backward()
        optimizer.step()
        total.append(loss.item())
    return float(np.mean(total)) if total else float("inf")


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total = []
    for x, x_flag in tqdm(dataloader, desc="Validation", leave=False):
        x = x.to(device)
        x_flag = x_flag.to(device).bool()
        loss, _ = model(x, x_flag)
        total.append(loss.item())
    return float(np.mean(total)) if total else float("inf")


def cosine_warmup_lambda(epoch, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    T = epoch - warmup_epochs
    T_max = max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * T / T_max))


# --------------------------- 聚类 ---------------------------
def classify(score, n, k):
    num = 0
    colors = [0 for _ in range(n)]
    ans = []
    for i in range(n):
        if colors[i] != 0:
            continue
        res = [i]
        num += 1
        colors[i] = num
        de = [colors[j] for j in range(n)]
        for j in range(n):
            if de[j] != 0:
                continue
            if score[i][j] < k:
                de[j] = 1
        for j in range(n):
            if de[j] != 0:
                continue
            res.append(j)
            colors[j] = num
            for t in range(n):
                if de[t] != 0:
                    continue
                if score[j][t] < k:
                    de[t] = 1
        ans.append(res)
    return ans, colors


def get_center(score, n, sets):
    now = sets[0]
    for s in sets:
        mn1 = 1e9
        mn2 = 1e9
        for s2 in sets:
            mn1 = min(mn1, score[now][s2])
            mn2 = min(mn2, score[s][s2])
        if mn1 < mn2:
            now = s
    return now


# --------------------------- 主流程 ---------------------------
def run(input_folder, output_folder, seed: int = 42):
    set_seed_all(seed)

    os.makedirs(output_folder, exist_ok=True)
    model_dir = os.path.join(output_folder, "PretrainedModels")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    meta_path = os.path.join(model_dir, "metadata.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 读取并合并 CSV（以第一个文件为基准对齐）
    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {input_folder}")

    df0 = pd.read_csv(csv_files[0])
    df0.columns = normalize_cols(df0.columns)
    # 仅此处筛掉 Unnamed 列，避免纳入特征集合；不在 rename 阶段删列
    base_cols = [c for c in df0.columns if c != "ICUSTAY" and not str(c).startswith("Unnamed:")]
    print(f"[Schema] base feature count = {len(base_cols)}")

    patients_summary = []
    for file in csv_files:
        ICUSTAY = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        df.columns = normalize_cols(df.columns)
        # 只关心 base_cols，对齐到相同列；一次性 reindex，避免碎片化
        df = df.reindex(columns=base_cols, fill_value=np.nan)
        mv = df.apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)
        mv["ICUSTAY"] = ICUSTAY
        patients_summary.append(mv)

    merged_df = pd.DataFrame(patients_summary)
    merged_df = merged_df[["ICUSTAY"] + base_cols]
    print(f"All the patient data have been merged. Total patients = {len(merged_df)}")
    print(f"Found {len(base_cols)} feature columns for training/inference.")

    # 2) 数据集签名 & 特征列持久化
    feature_cols = base_cols
    with open(os.path.join(model_dir, "feature_cols.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_cols))
    sig = _dataset_signature(input_folder, feature_cols)

    # DataLoader 生成器（确保 shuffle 可复现）
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # 3) 缓存模型检查
    cached_model_path, cached_meta = _maybe_load_cached_model(model_dir, sig, device)
    if cached_model_path is not None:
        print(f"[Cache] Found pretrained model for signature={sig}. Skip training.")
        model = SiameseGNN(num_modalities=len(feature_cols), hidden_channels=128,
                           normalize_embs=None, num_layers=2, dropout=0.25).to(device)
        try:
            state = torch.load(cached_model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(cached_model_path, map_location=device)
        model.load_state_dict(state)
    else:
        # 4) 训练
        df_train, df_val = train_test_split(merged_df, test_size=0.2, random_state=seed)
        train_ds = PatientDataset(df_train, feature_cols, device=device)
        val_ds = PatientDataset(df_val, feature_cols, device=device)

        train_bs = min(64, max(8, len(train_ds)))
        val_bs = min(128, max(8, len(val_ds)))

        train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True,
                                  num_workers=0, generator=g, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False,
                                num_workers=0, generator=g, pin_memory=False)

        model = SiameseGNN(num_modalities=len(feature_cols), hidden_channels=128,
                           normalize_embs=None, num_layers=2, dropout=0.25).to(device)
        optimizer = Adam(model.parameters(), lr=2e-3, weight_decay=5e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda ep: cosine_warmup_lambda(ep, total_epochs=20, warmup_epochs=10)
        )

        print("==== Start Training ====")
        best_val = float("inf")
        for epoch in range(1, 20 + 1):
            set_seed_all(seed + epoch)
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = validate(model, val_loader, device)
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")
            scheduler.step()
        print("==== Training Finished ====")

        meta = {
            "signature": sig,
            "feature_count": len(feature_cols),
            "train_patients": int(len(df_train)),
            "val_patients": int(len(df_val)),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "notes": "SiameseGNN clustering model for this dataset signature",
            "seed": int(seed)
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

    # 5) 推理（相似度矩阵）
    model.eval()
    full_ds = PatientDataset(merged_df, feature_cols, device=device)
    infer_bs = len(full_ds)
    full_loader = DataLoader(full_ds, batch_size=infer_bs, shuffle=False,
                             num_workers=0, generator=g, pin_memory=False)

    with torch.no_grad():
        for x, x_flag in full_loader:
            x = x.to(device)
            x_flag = x_flag.to(device).bool()
            _, simi_mat = model(x, x_flag)
            break

    mat = simi_mat.detach().cpu().numpy()
    np.fill_diagonal(mat, 1.0)
    print("Similarity matrix has been corrected.")

    # 6) 阈值聚类 & 输出
    threshold = 0.6
    n = mat.shape[0]
    ans, colors = classify(mat, n, threshold)
    max_len = max((len(g) for g in ans), default=0)

    data = {}
    for i, group in enumerate(ans, start=1):
        padded = group + [''] * (max_len - len(group))
        data[f'Group_{i}'] = padded

    groups_csv = os.path.join(output_folder, f'groups_{n}_{threshold}.csv')
    centers_csv = os.path.join(output_folder, f'centers_{n}_{threshold}.csv')
    labels_csv = os.path.join(output_folder, f'labels_{n}_{threshold}.csv')

    pd.DataFrame(data).to_csv(groups_csv, index=False, encoding='utf-8-sig')

    centers = [get_center(mat, n, group) for group in ans]
    pd.DataFrame({'Group': [f'Group_{i+1}' for i in range(len(ans))],
                  'Center_Index': centers}).to_csv(centers_csv, index=False, encoding='utf-8-sig')

    pd.DataFrame({'Sample_Index': list(range(n)), 'Group_Label': colors}).to_csv(labels_csv, index=False, encoding='utf-8-sig')

    print("These files were generated:")
    print(f" - {groups_csv}")
    print(f" - {centers_csv}")
    print(f" - {labels_csv}")

    return {
        "groups_csv": groups_csv,
        "centers_csv": centers_csv,
        "labels_csv": labels_csv,
        "signature": sig,
        "seed": int(seed)
    }


# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    run(
        input_folder="data_raw/lorenz",
        output_folder="result",
        seed=42,
    )
