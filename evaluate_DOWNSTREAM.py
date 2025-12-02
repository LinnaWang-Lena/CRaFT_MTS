# -*- coding: utf-8 -*-
"""
下游任务评估（多GPU + 可复现 + 多填补方法对比；my_model 走预填补目录）
- 其它基线：从 data_dir 读取带缺失原始数据 -> baseline 现场填补 -> 训练 RNN（LSTM/GRU）
- my_model：不现场填补，直接从 preimputed_dir 读取已填补矩阵 -> 训练 RNN
- k 折交叉验证（在 TrainVal 上）+ 固定一次性的 80/20 Hold-out Test（汇总到同一张表）
- 通过函数参数直接控制，无 argparse/命令行依赖
"""

import os
import random
import warnings
from typing import List, Tuple, Callable, Union, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# sklearn
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ====================== 复现 & 多进程 ======================
def _set_spawn():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

def set_seed(seed: int = 42):
    import numpy as _np
    import torch as _torch
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed(seed)
        if hasattr(_torch.cuda, 'manual_seed_all'):
            _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ====================== 数据集 ======================
class MatrixDataset(Dataset):
    def __init__(self, matrices: List[np.ndarray], labels: List[int]):
        self.matrices = matrices
        self.labels = labels
    def __len__(self):
        return len(self.matrices)
    def __getitem__(self, idx):
        x = torch.tensor(self.matrices[idx], dtype=torch.float32)  # (T,D_eff)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ====================== 读数据 ======================
def _read_matrices_and_labels(
    data_dir: str, label_csv: str, id_col: str, label_col: str
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    file_list = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".csv")])
    label_df = pd.read_csv(label_csv)
    label_df[id_col] = label_df[id_col].astype(str)

    matrices, labels, ids = [], [], []
    for fn in tqdm(file_list, desc=f"Reading & matching: {os.path.basename(data_dir)}"):
        arr = pd.read_csv(os.path.join(data_dir, fn)).to_numpy()
        case_id = os.path.splitext(fn)[0]
        row = label_df[label_df[id_col] == case_id]
        if len(row) == 0:
            continue
        y = int(row.iloc[0][label_col])
        matrices.append(arr)
        labels.append(y)
        ids.append(case_id)

    return matrices, labels, ids


# ====================== 特征预处理 ======================
def _compute_fold_scaler(train_mats: List[np.ndarray], eps: float = 1e-8) -> Dict[str, Any]:
    D = train_mats[0].shape[1]
    concat = np.concatenate(train_mats, axis=0)
    mean = np.nanmean(concat, axis=0)
    std  = np.nanstd(concat, axis=0)
    const_mask = std < eps
    keep_idx = np.where(~const_mask)[0]
    info = {
        "mean": mean.astype(np.float32),
        "std":  np.maximum(std, eps).astype(np.float32),
        "keep_idx": keep_idx.astype(np.int32),
        "D_raw": int(D),
        "D_eff": int(len(keep_idx)),
        "const_cnt": int(np.sum(const_mask)),
    }
    return info

def _apply_scaler(mats: List[np.ndarray], scaler: Dict[str, Any]) -> List[np.ndarray]:
    keep = scaler["keep_idx"]
    mean = scaler["mean"][keep]
    std  = scaler["std"][keep]
    out = []
    for arr in mats:
        x = arr[:, keep]
        x = (x - mean) / std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        out.append(x.astype(np.float32))
    return out

def _prepare_fold_data(
    matrices: List[np.ndarray], labels: List[int], tr_idx: np.ndarray, va_idx: np.ndarray,
    std_eps: float = 1e-8
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int], Dict[str, Any]]:
    mats_tr = [matrices[i] for i in tr_idx]
    mats_va = [matrices[i] for i in va_idx]
    scaler = _compute_fold_scaler(mats_tr, eps=std_eps)
    mats_tr_z = _apply_scaler(mats_tr, scaler)
    mats_va_z = _apply_scaler(mats_va, scaler)
    y_tr = [labels[i] for i in tr_idx]
    y_va = [labels[i] for i in va_idx]
    return mats_tr_z, mats_va_z, y_tr, y_va, scaler


# ====================== 模型定义（LSTM/GRU 可选） ======================
USE_LAYERNORM = True

def _weight_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GRU, nn.LSTM)):
        for name, param in m.named_parameters():
            if "weight_ih" in name: nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name: nn.init.orthogonal_(param.data)
            elif "bias" in name: nn.init.zeros_(param.data)

class LSTMClassifierLN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.ln_in = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True,
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=8, dropout=dropout, batch_first=True)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.apply(_weight_init)
    def forward(self, x):  # x:(B,T,D)
        x = self.ln_in(x)
        h, _ = self.lstm(x)
        h, _ = self.attn(h, h, h)
        pooled = h.mean(1)
        return self.cls(pooled)

class GRUClassifierLN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.ln_in = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True,
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=8, dropout=dropout, batch_first=True)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.apply(_weight_init)
    def forward(self, x):
        x = self.ln_in(x)
        h, _ = self.gru(x)
        h, _ = self.attn(h, h, h)
        pooled = h.mean(1)
        return self.cls(pooled)

def _build_model(input_dim: int, rnn_type: str, device: torch.device):
    r = (rnn_type or "lstm").lower()
    if USE_LAYERNORM:
        if r == "gru":
            model = GRUClassifierLN(input_dim).to(device)
        else:
            model = LSTMClassifierLN(input_dim).to(device)
    else:
        model = LSTMClassifierLN(input_dim).to(device)
    return model


# ====================== 评估 ======================
@torch.no_grad()
def _eval_loader(model, loader, device):
    was_training = model.training
    model.eval()
    all_y, all_prob = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        prob = torch.sigmoid(model(xb)).squeeze(1).cpu().numpy()
        all_prob.extend(prob.tolist())
        all_y.extend(yb.numpy().tolist())
    if was_training: model.train()
    all_y = np.array(all_y, dtype=int); all_prob = np.array(all_prob, dtype=float)
    pred = (all_prob >= 0.5).astype(int)
    acc = accuracy_score(all_y, pred)
    pre = precision_score(all_y, pred, zero_division=0)
    rec = recall_score(all_y, pred, zero_division=0)
    if len(np.unique(all_y)) > 1:
        auc = roc_auc_score(all_y, all_prob)
        auprc = average_precision_score(all_y, all_prob)
    else:
        auc, auprc = float("nan"), float("nan")
    f1 = f1_score(all_y, pred, zero_division=0)
    return acc, pre, rec, auc, auprc, f1, all_prob


# ====================== 早停器 ======================
class EarlyStopper:
    def __init__(self, patience=12, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.bad = 0
    def step(self, value: float) -> bool:
        if value > self.best + self.min_delta:
            self.best = value; self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience


def _pos_weight_from_labels(y: List[int]) -> torch.Tensor:
    y_np = np.array(y, dtype=int)
    pos = (y_np == 1).sum(); neg = (y_np == 0).sum()
    if pos == 0: w = 1.0
    else: w = max(1.0, neg / max(1, pos))
    return torch.tensor([w], dtype=torch.float32)


# ====================== 单折训练 ======================
def _train_one_fold(args):
    (
        fold_id, tr_idx, va_idx, matrices, labels, rnn_type,
        epochs, batch_size, lr, seed, gpu_id, debug, weight_decay
    ) = args

    _set_spawn(); set_seed(seed + fold_id)
    device = torch.device(f"cuda:{gpu_id}") if (torch.cuda.is_available() and gpu_id is not None) else torch.device("cpu")

    mats_tr_z, mats_va_z, y_tr, y_va, scaler = _prepare_fold_data(matrices, labels, tr_idx, va_idx, std_eps=1e-8)

    dataset_tr = MatrixDataset(mats_tr_z, y_tr)
    dataset_va = MatrixDataset(mats_va_z, y_va)
    train_loader = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(dataset_va, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = mats_tr_z[0].shape[1]
    model = _build_model(input_dim, rnn_type, device)
    pos_w = _pos_weight_from_labels(y_tr).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, epochs), eta_min=1e-5)

    stopper = EarlyStopper(patience=12, min_delta=1e-4)
    last_loss = None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            y  = yb.to(device).unsqueeze(1)
            logits = model(xb)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            last_loss = loss.item()
        scheduler.step()

        # 监控验证 AUC（无正负样本则回退 loss）
        _, _, _, va_auc, _, _, _ = _eval_loader(model, val_loader, device)
        score = (va_auc if not np.isnan(va_auc) else -(last_loss if last_loss is not None else 0.0))
        if stopper.step(score):
            break

    va_acc, va_pre, va_rec, va_auc, va_auprc, va_f1, _ = _eval_loader(model, val_loader, device)
    return va_acc, va_pre, va_rec, va_auc, va_auprc, va_f1


# ====================== 多 GPU 调度（CV 用） ======================
def _fold_worker(worker_id, gpu_id, task_q, res_dict, matrices, labels, rnn_type, epochs, batch_size, lr, seed, debug, weight_decay):
    _set_spawn()
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
    while True:
        task = task_q.get()
        if task is None:
            break
        fold_id, tr_idx, va_idx = task
        try:
            metrics = _train_one_fold(
                (fold_id, tr_idx, va_idx, matrices, labels, rnn_type, epochs, batch_size, lr, seed, gpu_id, debug, weight_decay)
            )
            res_dict[fold_id] = metrics
        except Exception:
            res_dict[fold_id] = (float("nan"),)*6

def _kfold_indices(matrices, labels, k, seed):
    y = np.array(labels, dtype=int)
    if len(np.unique(y)) >= 2:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [(i, tr, va) for i, (tr, va) in enumerate(kf.split(np.arange(len(matrices)), y))]
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        return [(i, tr, va) for i, (tr, va) in enumerate(kf.split(np.arange(len(matrices))))]

def _kfold_train_multi_gpu(matrices, labels, rnn_type, k, epochs, batch_size, lr, seed, debug, weight_decay):
    import multiprocessing as mp
    _set_spawn(); set_seed(seed)
    num_gpus = torch.cuda.device_count()
    folds = _kfold_indices(matrices, labels, k, seed)

    if num_gpus == 0 or len(folds) == 1:
        results = []
        for fold_id, tr, va in folds:
            metrics = _train_one_fold(
                (fold_id, tr, va, matrices, labels, rnn_type, epochs, batch_size, lr, seed, None, debug, weight_decay)
            )
            results.append(metrics)
    else:
        import multiprocessing as mp
        manager = mp.Manager()
        task_q = manager.Queue()
        res_dict = manager.dict()
        for item in folds: task_q.put(item)
        for _ in range(min(num_gpus, len(folds))): task_q.put(None)

        workers = []
        for wid, gpu_id in enumerate(range(min(num_gpus, len(folds)))):  # 每 GPU 并行处理 1 折
            p = mp.Process(
                target=_fold_worker,
                args=(wid, gpu_id, task_q, res_dict, matrices, labels, rnn_type, epochs, batch_size, lr, seed, debug, weight_decay),
            )
            p.start(); workers.append(p)
        for p in workers: p.join()

        results = [res_dict.get(i, (float("nan"),)*6) for i in range(len(folds))]

    accs, pres, recs, aurocs, auprcs, f1s = list(zip(*results)) if len(results)>0 else ([],[],[],[],[],[])
    mean_std = lambda x: (float(np.nanmean(x)), float(np.nanstd(x))) if len(x)>0 else (float("nan"), float("nan"))
    return {
        "accuracy": mean_std(accs),
        "precision": mean_std(pres),
        "recall": mean_std(recs),
        "auroc": mean_std(aurocs),
        "auprc": mean_std(auprcs),
        "f1": mean_std(f1s),
    }


# ====================== 训练/测试 划分与测试评估 ======================
def _stratified_train_test_split(n_samples: int, labels: List[int], seed: int, test_size: float = 0.2):
    y = np.array(labels, dtype=int)
    idx = np.arange(n_samples)
    if len(np.unique(y)) >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        (tr, te), = sss.split(idx, y)
    else:
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        (tr, te), = ss.split(idx)
    return tr, te

def _fit_on_trainval_and_eval_test(
    matrices: List[np.ndarray],
    labels: List[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rnn_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    weight_decay: float,
) -> Dict[str, float]:
    set_seed(seed + 999)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 二次划分 val
    y = np.array(labels, dtype=int)
    if len(np.unique(y[train_idx])) >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=max(1, int(0.1 * len(train_idx))), random_state=seed+7)
        (train_sub, val_sub), = sss.split(train_idx, y[train_idx])
    else:
        ss = ShuffleSplit(n_splits=1, test_size=max(1, int(0.1 * len(train_idx))), random_state=seed+7)
        (train_sub, val_sub), = ss.split(train_idx)
    tr_idx = train_idx[train_sub]
    va_idx = train_idx[val_sub]

    mats_tr_z, mats_va_z, y_tr, y_va, scaler = _prepare_fold_data(matrices, labels, tr_idx, va_idx, std_eps=1e-8)
    mats_te = [matrices[i] for i in test_idx]
    mats_te_z = _apply_scaler(mats_te, scaler)
    y_te = [labels[i] for i in test_idx]

    train_loader = DataLoader(MatrixDataset(mats_tr_z, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(MatrixDataset(mats_va_z, y_va), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(MatrixDataset(mats_te_z, y_te), batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = mats_tr_z[0].shape[1]
    model = _build_model(input_dim, rnn_type, device)
    pos_w = _pos_weight_from_labels(y_tr).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, epochs), eta_min=1e-5)
    stopper = EarlyStopper(patience=12, min_delta=1e-4)

    last_loss = None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device).unsqueeze(1)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_loss = loss.item()
        scheduler.step()

        _, _, _, va_auc, _, _, _ = _eval_loader(model, val_loader, device)
        score = (va_auc if not np.isnan(va_auc) else -(last_loss if last_loss is not None else 0.0))
        if stopper.step(score):
            break

    te_acc, te_pre, te_rec, te_auc, te_auprc, te_f1, _ = _eval_loader(model, test_loader, device)
    return {
        "test_accuracy": float(te_acc),
        "test_precision": float(te_pre),
        "test_recall": float(te_rec),
        "test_auroc": float(te_auc),
        "test_auprc": float(te_auprc),
        "test_f1": float(te_f1),
    }


# ====================== 填补（解析 & 并行） ======================
def _resolve_imputer(name_or_fn):
    if isinstance(name_or_fn, (tuple, list)) and len(name_or_fn) == 2 and callable(name_or_fn[1]):
        name, fn = name_or_fn
        if not isinstance(name, str): name = getattr(fn, "__name__", "custom_imputer")
        return name, fn
    if callable(name_or_fn):
        return getattr(name_or_fn, "__name__", "custom_imputer"), name_or_fn
    if isinstance(name_or_fn, str):
        if name_or_fn == "my_model":
            return "my_model", None
        import importlib
        bl = importlib.import_module("baseline"); fn = getattr(bl, name_or_fn, None)
        if not callable(fn):
            try:
                pi = importlib.import_module("pipeline_imputer"); fn = getattr(pi, name_or_fn, None)
            except Exception:
                fn = None
        if callable(fn): return name_or_fn, fn
        raise ValueError(f"Cannot resolve imputer '{name_or_fn}'.")
    raise TypeError(f"imputer must be str, callable, or (str, callable); got {type(name_or_fn)}")


def _impute_one_gpu(args):
    idx, arr, fn, gpu_id, seed = args
    try:
        _set_spawn(); set_seed(seed + idx)
        if torch.cuda.is_available() and gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        out = fn(arr)
        return idx, out
    except Exception:
        return idx, None


def _impute_dataset_gpu_parallel(matrices: List[np.ndarray], imputer_fn: Callable, seed: int) -> List[np.ndarray]:
    from multiprocessing import get_context
    set_seed(seed)
    num_gpus = torch.cuda.device_count()
    gpu_ids = [i for i in range(num_gpus)] if num_gpus > 0 else [None]
    tasks = [(i, matrices[i], imputer_fn, gpu_ids[i % len(gpu_ids)], seed) for i in range(len(matrices))]
    ctx = get_context("spawn")
    with ctx.Pool(processes=max(1, len(gpu_ids))) as pool:
        results = list(tqdm(pool.imap(_impute_one_gpu, tasks), total=len(tasks), desc=f"Imputing ({imputer_fn.__name__})"))

    imputed = [None] * len(matrices)
    for i, out in results:
        if out is None:
            imputed[i] = matrices[i].astype(np.float32, copy=False)
        else:
            imputed[i] = out.astype(np.float32, copy=False)
    return imputed


# ====================== 主函数（关键：my_model 读预填补，其余现场填补） ======================
def evaluate_imputation_downstream(
    data_dir: str,
    label_csv: str,
    id_col: str,
    label_col: str,
    imputer_list: List[Union[str, Callable]],
    rnn_type: str = 'lstm',
    seed: int = 42,
    k: int = 5,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    out_csv: str = './results/imputation_comparison_results.csv',
    debug: bool = True,
    preimputed_dir: str = './results/III_res/diffusion_imputed',
) -> pd.DataFrame:
    """
    - 其它基线：从 data_dir 读取原始矩阵 -> baseline 现场填补
    - my_model：从 preimputed_dir 读取已填补矩阵
    - 同一固定 Train/Test 划分 & k 折 CV（基于各自的数据源）
    - 输出单张结果表
    """
    _set_spawn(); set_seed(seed)

    # 为 baseline 读取原始矩阵（带缺失）
    raw_matrices, raw_labels, _ = _read_matrices_and_labels(data_dir, label_csv, id_col, label_col)
    raw_labels = list(map(int, raw_labels))

    rows = []
    for name_or_fn in imputer_list:
        imputer_name, imputer_fn = _resolve_imputer(name_or_fn)

        if imputer_name == 'my_model':
            # 读取预填补矩阵
            mats_all, labels_all, _ = _read_matrices_and_labels(preimputed_dir, label_csv, id_col, label_col)
            labels_all = list(map(int, labels_all))
        else:
            # 现场填补：先对 raw 进行全量填补
            if imputer_fn is None:
                raise ValueError(f"Imputer function for '{imputer_name}' is None.")
            mats_all = _impute_dataset_gpu_parallel(raw_matrices, imputer_fn, seed)
            labels_all = raw_labels[:len(mats_all)]

        if len(mats_all) == 0:
            if debug:
                print(f"[SKIP] {imputer_name}: no samples.")
            continue

        # 固定一次 Train/Test 划分
        tr_idx_all, te_idx_all = _stratified_train_test_split(len(mats_all), labels_all, seed, test_size=0.2)

        # —— CV：只在 TrainVal 上 —— #
        mats_trainval = [mats_all[i] for i in tr_idx_all]
        labels_trainval = [labels_all[i] for i in tr_idx_all]
        metrics_cv = _kfold_train_multi_gpu(
            mats_trainval, labels_trainval, rnn_type, k, epochs, batch_size, lr, seed, debug, weight_decay
        )

        # —— Test 评估 —— #
        test_metrics = _fit_on_trainval_and_eval_test(
            matrices=mats_all,
            labels=labels_all,
            train_idx=tr_idx_all,
            test_idx=te_idx_all,
            rnn_type=rnn_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            weight_decay=weight_decay,
        )

        row = {
            "Method": imputer_name,
            "Seed": seed,
            "RNN": rnn_type,
            # CV（TrainVal）均值±方差
            "Accuracy (mean ± std)": f"{metrics_cv['accuracy'][0]:.4f} ± {metrics_cv['accuracy'][1]:.4f}",
            "Precision (mean ± std)": f"{metrics_cv['precision'][0]:.4f} ± {metrics_cv['precision'][1]:.4f}",
            "Recall (mean ± std)": f"{metrics_cv['recall'][0]:.4f} ± {metrics_cv['recall'][1]:.4f}",
            "AUROC (mean ± std)": f"{metrics_cv['auroc'][0]:.4f} ± {metrics_cv['auroc'][1]:.4f}",
            "AUPRC (mean ± std)": f"{metrics_cv['auprc'][0]:.4f} ± {metrics_cv['auprc'][1]:.4f}",
            "F1 (mean ± std)": f"{metrics_cv['f1'][0]:.4f} ± {metrics_cv['f1'][1]:.4f}",
            # Test
            "Test Accuracy": f"{test_metrics['test_accuracy']:.4f}",
            "Test Precision": f"{test_metrics['test_precision']:.4f}",
            "Test Recall": f"{test_metrics['test_recall']:.4f}",
            "Test AUROC": f"{test_metrics['test_auroc']:.4f}" if not np.isnan(test_metrics['test_auroc']) else "nan",
            "Test AUPRC": f"{test_metrics['test_auprc']:.4f}" if not np.isnan(test_metrics['test_auprc']) else "nan",
            "Test F1": f"{test_metrics['test_f1']:.4f}",
        }
        rows.append(row)

        print(f"\n[DONE] {imputer_name} (RNN={rnn_type}):")
        print("  " + "; ".join([
            f"Accuracy={row['Accuracy (mean ± std)']}",
            f"Precision={row['Precision (mean ± std)']}",
            f"Recall={row['Recall (mean ± std)']}",
            f"AUROC={row['AUROC (mean ± std)']}",
            f"AUPRC={row['AUPRC (mean ± std)']}",
            f"F1={row['F1 (mean ± std)']}",
        ]))

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No results produced. Please check imputers, data paths, and labels mapping.")

    cols = [
        "Method", "Seed", "RNN",
        "Accuracy (mean ± std)", "Precision (mean ± std)", "Recall (mean ± std)",
        "AUROC (mean ± std)", "AUPRC (mean ± std)", "F1 (mean ± std)",
        "Test Accuracy", "Test Precision", "Test Recall", "Test AUROC", "Test AUPRC", "Test F1",
    ]
    df = df[cols]

    print("\n=== Final Results (CV on TrainVal; Single Hold-out Test) ===")
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results -> {out_csv}")
    return df


# ====================== 直接函数调用示例（按需改动） ======================
if __name__ == '__main__':
    # 在这里列出所有要评估的填补方法名称（与 ./baseline.py / pipeline_imputer.py 中的函数同名；my_model 走预填补目录）
    imputers = ['my_model', 'zero_impu', 'mean_impu', 'bfill_impu', 'knn_impu', 'mice_impu', 'tefn_impu', 'saits_impu', 'timemixerpp_impu', 'timesnet_impu', 'grin_impu', 'miracle_impu', 'tsde_impu', 'csdi_impu', 'diffputer_impu']  # 示例，可按需增删

    evaluate_imputation_downstream(
        data_dir='./data_raw/downstreamIV',                    # 原始带缺失矩阵目录
        label_csv='./AAAI_3_4_labels.csv',                 # 标签文件
        id_col='ICUSTAY_ID',                         # 样本 ID 列
        label_col='DIEINHOSPITAL',                       # 标签列
        imputer_list=imputers,
        rnn_type='gru',                          # 或 'gru'
        seed=42,
        k=5,
        epochs=100,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-2,
        out_csv='./results/imputation_comparison_results.csv',
        debug=True,
        preimputed_dir='./IV/diffusion_imputed/diffusion_imputed'
    )