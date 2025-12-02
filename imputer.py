# imputer_pipeline.py
import os
import glob
import math
import random
import json
import traceback
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from utils.initial_imputer import initial_process
from utils.cluster import run as cluster_run
from utils.causal_discovery import compute_causal_strength_and_lag
from utils.diffusion_backbone import DiffusionBackbone


# ---------------- 通用：可复现设置 ----------------
def set_seed_all(seed: int = 42):
    import os as _os
    _os.environ.setdefault("PYTHONHASHSEED", str(seed))
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ---------------- 基本工具 ----------------
def list_csvs(input_folder):
    return [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".csv")
    ]


def read_csv_to_array(file):
    df = pd.read_csv(file)
    return df.values.astype(np.float32)


def save_matrix(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def log(s):
    print(s, flush=True)

beta = 10
def get_feature_names_from_file(csv_path, Fdim):
    try:
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        cols = [c for c in cols if not str(c).startswith("Unnamed:")]
        if len(cols) == Fdim:
            return cols
    except Exception:
        pass
    return [f"X{i}" for i in range(Fdim)]


# ---------------- Step 1: 聚类（仅用 utils.cluster.run 的结果） ----------------
def run_clustering(input_folder, output_folder, seed: int = 42):
    """
    严格使用 utils.cluster.run(input_folder, output_folder, seed) 进行聚类。
    不做任何兜底聚类。根据其返回的 groups_csv / centers_csv 重建 clusters 与 centers。
    """
    log("[Step 1] 运行聚类（使用 utils.cluster.run）...")
    try:
        result = cluster_run(input_folder, output_folder, seed=seed)
    except TypeError:
        # 兼容旧签名
        result = cluster_run(input_folder, output_folder)
    if not isinstance(result, dict):
        raise RuntimeError("utils.cluster.run 未返回 dict。请检查 utils/cluster.py 的 run() 实现。")

    groups_csv = result.get("groups_csv")
    centers_csv = result.get("centers_csv")
    if not (groups_csv and centers_csv and os.path.isfile(groups_csv) and os.path.isfile(centers_csv)):
        raise RuntimeError(
            f"utils.cluster.run 未提供有效的 groups/centers CSV：\n"
            f"  groups_csv={groups_csv}\n  centers_csv={centers_csv}"
        )

    # 保持与 cluster.run 一致的文件顺序（run 内部使用 sorted(glob(...))）
    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {input_folder}")
    n_files = len(csv_files)

    # 读取 centers 索引并映射为文件路径
    cdf = pd.read_csv(centers_csv)
    if "Center_Index" in cdf.columns:
        center_indices = cdf["Center_Index"].tolist()
    elif cdf.shape[1] == 1:
        center_indices = cdf.iloc[:, 0].tolist()
    else:
        raise ValueError("centers CSV 无 'Center_Index' 列且不为单列，无法解析中心索引。")

    centers = []
    for v in center_indices:
        if pd.isna(v) or str(v).strip() == "":
            continue
        try:
            idx = int(float(v))
        except Exception:
            continue
        if not (0 <= idx < n_files):
            raise IndexError(f"Center_Index {idx} 超出文件数范围 [0, {n_files-1}]")
        centers.append(os.path.abspath(csv_files[idx]))

    # 读取各簇样本索引并映射为文件路径
    gdf = pd.read_csv(groups_csv)
    group_cols = [c for c in gdf.columns if str(c).startswith("Group_")]
    if not group_cols:
        raise ValueError("groups CSV 未找到形如 'Group_1' 的列。")

    clusters = []
    for gc in group_cols:
        idx_list = []
        for v in gdf[gc].tolist():
            if pd.isna(v) or str(v).strip() == "":
                continue
            try:
                idx = int(float(v))
            except Exception:
                continue
            if 0 <= idx < n_files:
                idx_list.append(idx)
        clusters.append([os.path.abspath(csv_files[i]) for i in idx_list])

    # 对齐中心与簇数量
    if len(centers) != len(clusters):
        log(f"[WARN] centers 数量({len(centers)})与 clusters 数量({len(clusters)})不一致，将按较小者对齐。")
    m = min(len(centers), len(clusters))
    centers = centers[:m]
    clusters = clusters[:m]

    log(f"[Step 1] 簇数量: {len(clusters)}，中心数量: {len(centers)}（来自 utils.cluster.run 的 CSV）")
    return clusters, centers


# ---------------- Step 2: 中心初始填补 ----------------
def initial_impute_centers(centers, threshold=0.8, perturbation_prob=0.1, perturbation_scale=0.1):
    log("[Step 2] 对每个簇中心进行初始插补...")
    imputed_center_mats = []
    for i, file in enumerate(centers):
        mat = read_csv_to_array(file)
        mat_imp = initial_process(mat, threshold, perturbation_prob, perturbation_scale)
        imputed_center_mats.append(mat_imp)
        log(f"  - 中心 {i}: {os.path.basename(file)} 完成初始插补，shape={mat_imp.shape}")
    return imputed_center_mats


# ---------------- Step 3: 因果发现（多 GPU 子进程） ----------------
def _causal_worker_center_shard(shard_indices, npy_paths, params, gpu_id, out_dir, res_q, base_seed: int):
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        set_seed_all(base_seed + int(gpu_id))

        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        centers_dir = os.path.join(out_dir, "centers")
        os.makedirs(centers_dir, exist_ok=True)

        for idx in shard_indices:
            try:
                set_seed_all(base_seed + int(idx))

                mat = np.load(npy_paths[idx])
                S, L = compute_causal_strength_and_lag(
                    mat, params, gpu_id=gpu_id, seed=int(base_seed) + int(idx)
                )
                if np.all(S == 0):
                    res_q.put(("error", idx, "ALL_ZERO_CAUSAL: strength matrix is all zeros"))
                    continue
                np.save(os.path.join(centers_dir, f"center_{idx}_strength.npy"), S)
                np.save(os.path.join(centers_dir, f"center_{idx}_lag.npy"), L)
                res_q.put(("ok", idx))
            except Exception:
                res_q.put(("error", idx, traceback.format_exc()))
    except Exception:
        res_q.put(("error", -1, traceback.format_exc()))


def causal_discovery_on_centers_multi_gpu(imputed_center_mats, params, out_dir, base_seed: int = 42):
    log("[Step 3] 多 GPU 并行对簇中心进行因果发现...")
    os.makedirs(os.path.join(out_dir, "centers"), exist_ok=True)

    n_centers = len(imputed_center_mats)
    if n_centers == 0:
        return [], []

    tmp_dir = os.path.join(out_dir, "tmp_center_npy")
    os.makedirs(tmp_dir, exist_ok=True)
    npy_paths = []
    for i, mat in enumerate(imputed_center_mats):
        pth = os.path.join(tmp_dir, f"center_{i}.npy")
        np.save(pth, mat)
        npy_paths.append(pth)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        log("  * 未检测到 CUDA，使用 CPU 串行计算。")
        centers_dir = os.path.join(out_dir, "centers")
        S_list, L_list = [], []
        for i in range(n_centers):
            set_seed_all(base_seed + int(i))
            S, L = compute_causal_strength_and_lag(np.load(npy_paths[i]), params, gpu_id=0, seed=int(base_seed) + int(i))

            if np.all(S == 0):
                raise RuntimeError(f"[FATAL] 中心 {i} 产生全 0 因果图，已停止。")
            np.save(os.path.join(centers_dir, f"center_{i}_strength.npy"), S)
            np.save(os.path.join(centers_dir, f"center_{i}_lag.npy"), L)
            S_list.append(S)
            L_list.append(L)
        try:
            for p in npy_paths:
                os.remove(p)
            os.rmdir(tmp_dir)
        except Exception:
            pass
        return S_list, L_list

    ctx = mp.get_context("spawn")
    res_q = ctx.SimpleQueue()

    n_procs = min(num_gpus, n_centers)
    shards = [[] for _ in range(n_procs)]
    for i in range(n_centers):
        shards[i % n_procs].append(i)

    procs = []
    for gid in range(n_procs):
        if not shards[gid]:
            continue
        p = ctx.Process(
            target=_causal_worker_center_shard,
            args=(shards[gid], npy_paths, params, gid, out_dir, res_q, base_seed),
        )
        p.start()
        procs.append(p)

    finished = 0
    target_total = n_centers
    had_error = False
    error_msgs = []

    while finished < target_total:
        if not any(p.is_alive() for p in procs) and res_q.empty():
            break
        try:
            tag, idx, *rest = res_q.get()
            if tag != "ok":
                msgtxt = rest[0] if rest else "Unknown error"
                if "ALL_ZERO_CAUSAL" in msgtxt:
                    had_error = True
                    error_msgs.append(f"[FATAL] 中心 {idx} 产生全 0 因果图。")
                else:
                    error_msgs.append(f"[ERROR] 中心 {idx} 失败：\n{msgtxt}")
            finished += 1
        except Exception:
            pass

    for p in procs:
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()

    if had_error:
        raise RuntimeError("\n".join(error_msgs))

    centers_dir = os.path.join(out_dir, "centers")
    S_list, L_list, remaining = [], [], []
    for i in range(n_centers):
        S_path = os.path.join(centers_dir, f"center_{i}_strength.npy")
        L_path = os.path.join(centers_dir, f"center_{i}_lag.npy")
        if os.path.exists(S_path) and os.path.exists(L_path):
            S_list.append(np.load(S_path, allow_pickle=False))
            L_list.append(np.load(L_path, allow_pickle=False))
        else:
            remaining.append(i)

    if remaining:
        log(f"[WARN] 并行有 {len(remaining)} 个中心未生成结果，CPU 回退：{remaining}")
        for i in remaining:
            set_seed_all(base_seed + int(i))
            S, L = compute_causal_strength_and_lag(np.load(npy_paths[i]), params, gpu_id=0, seed=int(base_seed) + int(i))

            if np.all(S == 0):
                raise RuntimeError(f"[FATAL] 中心 {i} 产生全 0 因果图（CPU 回退仍为 0），已停止。")
            np.save(os.path.join(centers_dir, f"center_{i}_strength.npy"), S)
            np.save(os.path.join(centers_dir, f"center_{i}_lag.npy"), L)
            S_list.append(S)
            L_list.append(L)

    shapes = [S.shape for S in S_list]
    if any(S is None for S in S_list) or len(set(shapes)) != 1 or len(S_list) != n_centers:
        raise RuntimeError("中心因果结果形状不一致或存在缺失，请检查日志。")

    try:
        for p in npy_paths:
            os.remove(p)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    return S_list, L_list


# ---------------- Step 4: 全局因果图汇总 ----------------
def aggregate_global_graph(strength_list, lag_list, out_dir, method="mean_vote", threshold=0.5, feature_names=None):
    log("[Step 4] 汇总总因果强度/时滞矩阵...")
    if len(strength_list) == 0:
        raise ValueError("没有中心强度矩阵可供汇总。")

    S_stack = np.stack(strength_list, axis=0)
    L_stack = np.stack(lag_list, axis=0)

    S_cont = S_stack.mean(axis=0)
    S_bin = (S_cont >= threshold).astype(int)

    Fdim = S_cont.shape[0]
    L_global = np.zeros_like(S_cont, dtype=int)
    for i in range(Fdim):
        for j in range(Fdim):
            if S_cont[i, j] > 0:
                vals = L_stack[:, i, j]
                counts = {}
                for v in vals:
                    counts[int(v)] = counts.get(int(v), 0) + 1
                mode_v = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
                L_global[i, j] = int(mode_v)
            else:
                L_global[i, j] = 0

    os.makedirs(out_dir, exist_ok=True)
    save_matrix(os.path.join(out_dir, "global_strength_cont.npy"), S_cont)
    if feature_names is not None and len(feature_names) == S_cont.shape[0]:
        pd.DataFrame(S_cont, index=feature_names, columns=feature_names)\
          .to_csv(os.path.join(out_dir, "global_strength_cont.csv"), float_format="%.6f")
    else:
        np.savetxt(os.path.join(out_dir, "global_strength_cont.csv"), S_cont, fmt="%.6f", delimiter=",")

    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(S_cont, aspect='equal')
    plt.colorbar()
    plt.title("Global Causal Strength (continuous 0-1)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_strength_cont_heatmap.png"))
    plt.close()

    save_matrix(os.path.join(out_dir, "global_strength_bin.npy"), S_bin)
    if feature_names is not None and len(feature_names) == S_bin.shape[0]:
        pd.DataFrame(S_bin, index=feature_names, columns=feature_names)\
          .to_csv(os.path.join(out_dir, "global_strength_bin.csv"), float_format="%.0f")
    else:
        np.savetxt(os.path.join(out_dir, "global_strength_bin.csv"), S_bin, fmt="%d", delimiter=",")
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(S_bin, aspect='equal')
    plt.colorbar()
    plt.title(f"Global Causal Strength (binary, thr={threshold})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_strength_bin_heatmap.png"))
    plt.close()

    save_matrix(os.path.join(out_dir, "global_lag.npy"), L_global)
    log(f"[Step 4] 全局因果图完成：形状 {S_cont.shape}")
    return S_cont, L_global, S_bin


# ---------------- Step 5: 初始填补（保留原观测） ----------------
def initial_impute_all_files(file_list, threshold=0.8, perturbation_prob=0.1, perturbation_scale=0.1):
    log("[Step 5] 对所有文件进行初始插补...")
    results = []
    for f in file_list:
        X_orig_nan = read_csv_to_array(f)                 # 原始矩阵（含 NaN）
        obs_mask = (~np.isnan(X_orig_nan)).astype(np.int32)
        miss_cnt = int((obs_mask == 0).sum())
        log(f"    · {os.path.basename(f)} 原始缺失个数: {miss_cnt}")

        X_init = initial_process(X_orig_nan, threshold, perturbation_prob, perturbation_scale)  # 完整
        # 强制观测位保持原值
        X_init[obs_mask == 1] = X_orig_nan[obs_mask == 1]

        results.append((f, X_init.astype(np.float32), obs_mask.astype(np.int32), X_orig_nan.astype(np.float32)))
        log(f"  - {os.path.basename(f)} 完成初始插补，shape={X_init.shape}")
    return results


# ---------------- Step 6: Diffusion（训练 + 留出评估 + 推理） ----------------
def _masked_mse(pred, target, mask, eps=1e-8):
    sel = (mask > 0)
    if torch.is_tensor(pred):
        sel = sel & torch.isfinite(pred)
    if torch.is_tensor(target):
        sel = sel & torch.isfinite(target)
    n = sel.sum()
    if n.item() == 0:
        return pred.new_tensor(0.0)
    diff = pred[sel] - target[sel]
    return (diff * diff).mean()


def _masked_mae(pred, target, mask, eps=1e-8):
    sel = (mask > 0)
    if torch.is_tensor(pred):
        sel = sel & torch.isfinite(pred)
    if torch.is_tensor(target):
        sel = sel & torch.isfinite(target)
    n = sel.sum()
    if n.item() == 0:
        return pred.new_tensor(0.0)
    return (pred[sel] - target[sel]).abs().mean()


def _pad_stack(arrs, pad_value=0.0, nan_pad=False):
    """把不同长度 (T,F) pad 成 (B,T_max,F)。"""
    F = arrs[0].shape[1]
    T_max = max(a.shape[0] for a in arrs)
    out = []
    for a in arrs:
        T = a.shape[0]
        if T < T_max:
            pad = np.full((T_max - T, F), np.nan if nan_pad else pad_value, dtype=np.float32)
            a = np.concatenate([a, pad], axis=0)
        out.append(torch.tensor(a, dtype=torch.float32))
    return torch.stack(out, dim=0)  # (B,T_max,F)


def _make_epoch_train_masks(obs_b, drop_prob: float, device):
    """
    在观测位内随机抽样一部分作为当前 epoch 的“伪缺失”，用于监督。
    返回：train_mask (B,T,F) ∈ {0,1}；每个样本若存在观测位，则至少保证 1 个被遮盖。
    """
    obs_b = obs_b.to(device).float()
    if drop_prob <= 0:
        return torch.zeros_like(obs_b, device=device)

    rand = torch.rand_like(obs_b, device=device)
    train_mask = ((rand < drop_prob) & (obs_b > 0)).float()

    with torch.no_grad():
        B = obs_b.size(0)
        flat_obs = obs_b.view(B, -1)
        flat_mask = train_mask.view(B, -1)
        for b in range(B):
            if flat_obs[b].sum() > 0 and flat_mask[b].sum() == 0:
                idxs = (flat_obs[b] > 0).nonzero(as_tuple=False).view(-1)
                j = idxs[torch.randint(0, idxs.numel(), (1,), device=device)]
                flat_mask[b, j] = 1.0
        train_mask = flat_mask.view_as(obs_b)
    return train_mask


# ====== 多 GPU 独立训练：worker（加入微批） ======
def _diffusion_train_worker(args):
    """
    子进程在指定 GPU 上独立训练一份模型，返回 CPU 版本的 state_dict。
    采用微批训练：把 shard 的 batch 切成若干 micro-batches，梯度累积后再 step。
    """
    (rank, device_id, shard_np, shard_nan_np, shard_gt_np, shard_obs_np,
     S_np, L_np, D, T, width, diff_epochs, diff_lr, diff_weight_decay,
     seed, drop_prob, noise_scale, diff_micro_bsz) = args

    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        set_seed_all(seed + rank)

        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        # to tensors（整 shard 常驻显存，但按微批前向）
        X_init_all = torch.tensor(shard_np, dtype=torch.float32, device=device)
        X_nan_all  = torch.tensor(shard_nan_np, dtype=torch.float32, device=device)
        X_gt_all   = torch.tensor(shard_gt_np, dtype=torch.float32, device=device)
        obs_all    = torch.tensor(shard_obs_np, dtype=torch.float32, device=device)
        S = torch.tensor(S_np, dtype=torch.float32, device=device)
        L = torch.tensor(L_np, dtype=torch.float32, device=device)

        B = X_init_all.size(0)
        mb = max(1, int(diff_micro_bsz))
        n_steps = (B + mb - 1) // mb

        model = DiffusionBackbone(D=D, T=T, device=device, width=width).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=diff_lr, weight_decay=diff_weight_decay)
        torch.autograd.set_detect_anomaly(False)

        model.train()
        for epoch in range(1, diff_epochs + 1):
            set_seed_all(seed + rank*1000 + epoch)
            optimizer.zero_grad()

            # 整批生成 train_mask（与原逻辑一致）
            train_mask_all = _make_epoch_train_masks(obs_all, drop_prob=drop_prob, device=device)
            if train_mask_all.sum() == 0:
                # 极端兜底：不改变统计意义
                flat_obs = obs_all.view(B, -1)
                flat_mask = train_mask_all.view(B, -1)
                for b in range(B):
                    if flat_obs[b].sum() > 0 and flat_mask[b].sum() == 0:
                        idxs = (flat_obs[b] > 0).nonzero(as_tuple=False).view(-1)
                        j = idxs[0]
                        flat_mask[b, j] = 1.0
                train_mask_all = flat_mask.view_as(obs_all)

            # 微批循环：均分 loss 进行梯度累积，等价于整批
            for step in range(n_steps):
                s = step * mb
                e = min(B, s + mb)
                if s >= e:
                    break

                X_init_b = X_init_all[s:e]
                X_nan_b  = X_nan_all[s:e]
                X_gt_b   = X_gt_all[s:e]
                train_mask = train_mask_all[s:e]

                X_nan_train = X_nan_b.clone()
                X_nan_train[train_mask.bool()] = float('nan')
                X_nan_train_safe = torch.nan_to_num(X_nan_train, nan=0.0, posinf=0.0, neginf=0.0)

                X_in_noisy = X_init_b + (noise_scale * torch.randn_like(X_init_b) if (noise_scale and noise_scale > 0) else 0.0)

                pred = model(X_in_noisy, X_nan_train_safe, S, L, lock_observed=False)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

                loss = _masked_mse(pred, X_gt_b, train_mask)
                # 均分累积，保持与整批一致的总梯度幅度
                (loss / n_steps).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 返回 CPU 版本 state_dict
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return ("ok", rank, sd)
    except Exception as e:
        return ("error", rank, traceback.format_exc())


def _average_state_dicts(state_dict_list):
    """对多个 worker 返回的 state_dict 做逐参数算术平均。"""
    if not state_dict_list:
        raise RuntimeError("Empty state_dict_list in _average_state_dicts")
    avg_sd = {}
    keys = state_dict_list[0].keys()
    for k in keys:
        tensors = [sd[k].float() for sd in state_dict_list]
        stacked = torch.stack(tensors, dim=0)
        avg_sd[k] = stacked.mean(dim=0)
    return avg_sd


def diffusion_train_eval_and_infer(
    imputed_files, global_S, global_L, out_dir,
    D=5, T=50, width=256,
    diff_epochs=30, diff_lr=5e-4, diff_weight_decay=0.0,
    phaseB_iter=3, val_ratio=0.1, seed=42,
    drop_prob=0.2, noise_scale=0.01,
    diff_micro_bsz=2  # <<< 新增：微批大小，默认 2（保守防 OOM）
):
    """
    训练目标：随机遮盖观测位的自监督补全（避免恒等）。
      - 训练：在“当前 epoch 被遮盖的观测位”上计算损失（train_mask）；允许模型预测所有位置（lock_observed=False）。
      - 推理：对所有位置输出预测（lock_observed=False），仅在缺失位写回（obs_mask==0）。
    """
    set_seed_all(seed)
    print("[Step 6] 用总因果图指导 Diffusion（训练：随机遮盖观测位；推理：填补缺失）...")

    os.makedirs(out_dir, exist_ok=True)

    # 1) 准备批数据
    X_init_list, X_nan_list, X_gt_list, obs_mask_list = [], [], [], []
    for (path, X_init, obs_mask, X_orig_nan) in imputed_files:
        X_in = X_init.copy()
        X_in[obs_mask == 1] = X_orig_nan[obs_mask == 1]

        X_nan = X_in.copy()
        X_nan[obs_mask == 0] = np.nan

        X_gt = np.zeros_like(X_in, dtype=np.float32)
        X_gt[obs_mask == 1] = X_orig_nan[obs_mask == 1]

        X_init_list.append(X_in.astype(np.float32))
        X_nan_list.append(X_nan.astype(np.float32))
        X_gt_list.append(X_gt.astype(np.float32))
        obs_mask_list.append(obs_mask.astype(np.float32))

    X_init_b = _pad_stack(X_init_list, pad_value=0.0, nan_pad=False).cpu().numpy()
    X_nan_b  = _pad_stack(X_nan_list,  pad_value=np.nan, nan_pad=True).cpu().numpy()
    X_gt_b   = _pad_stack(X_gt_list,   pad_value=0.0, nan_pad=False).cpu().numpy()
    obs_b    = _pad_stack(obs_mask_list, pad_value=0.0, nan_pad=False).cpu().numpy()

    obs_count = int(np.sum(obs_b))
    total = int(np.prod(obs_b.shape))
    print(f"    [DEBUG] 观测位总数 obs_count = {obs_count} / {total}")
    if obs_count == 0:
        print("    [FATAL] 数据集中没有任何观测位，无法训练。")
        return

    S_np = np.asarray(global_S, dtype=np.float32)
    L_np = np.asarray(global_L, dtype=np.float32)
    print(f"    [DEBUG] 全局因果图 S shape: {S_np.shape}, L shape: {L_np.shape}")

    # 2) 多 GPU 独立训练（按样本维 shard 到各卡；每个子进程互不依赖；每卡使用微批）
    num_gpus = torch.cuda.device_count()
    B = X_init_b.shape[0]
    if num_gpus >= 2 and B >= num_gpus:
        print(f"  * 检测到 {num_gpus} 张 GPU，启动独立子进程并行训练（按样本划分，互不依赖）。")
        indices = [list(range(i, B, num_gpus)) for i in range(num_gpus)]
        args_list = []
        for rank, (dev, idxs) in enumerate(zip(range(num_gpus), indices)):
            if not idxs:
                continue
            shard_np = X_init_b[idxs]
            shard_nan_np = X_nan_b[idxs]
            shard_gt_np = X_gt_b[idxs]
            shard_obs_np = obs_b[idxs]
            args_list.append((rank, dev, shard_np, shard_nan_np, shard_gt_np, shard_obs_np,
                              S_np, L_np, D, T, width, diff_epochs, diff_lr, diff_weight_decay,
                              seed, drop_prob, noise_scale, diff_micro_bsz))

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(args_list)) as pool:
            results = pool.map(_diffusion_train_worker, args_list)

        ok_sd = []
        for tag, rk, payload in results:
            if tag != "ok":
                raise RuntimeError(f"[Worker-{rk}] 训练失败：\n{payload}")
            ok_sd.append(payload)

        avg_sd = _average_state_dicts(ok_sd)

        device_main = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DiffusionBackbone(D=D, T=T, device=device_main, width=width).to(device_main)
        model.load_state_dict(avg_sd, strict=True)
    else:
        # 单卡训练（同样采用微批）
        device_main = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DiffusionBackbone(D=D, T=T, device=device_main, width=width).to(device_main)
        optimizer = torch.optim.Adam(model.parameters(), lr=diff_lr, weight_decay=diff_weight_decay)
        torch.autograd.set_detect_anomaly(False)

        X_init_all = torch.tensor(X_init_b, dtype=torch.float32, device=device_main)
        X_nan_all  = torch.tensor(X_nan_b, dtype=torch.float32, device=device_main)
        X_gt_all   = torch.tensor(X_gt_b, dtype=torch.float32, device=device_main)
        obs_all    = torch.tensor(obs_b, dtype=torch.float32, device=device_main)
        S_t = torch.tensor(S_np, dtype=torch.float32, device=device_main)
        L_t = torch.tensor(L_np, dtype=torch.float32, device=device_main)

        B = X_init_all.size(0)
        mb = max(1, int(diff_micro_bsz))
        n_steps = (B + mb - 1) // mb

        model.train()
        for epoch in range(1, diff_epochs + 1):
            set_seed_all(seed + epoch)
            optimizer.zero_grad()

            train_mask_all = _make_epoch_train_masks(obs_all, drop_prob=drop_prob, device=device_main)
            if train_mask_all.sum() == 0:
                flat_obs = obs_all.view(B, -1)
                flat_mask = train_mask_all.view(B, -1)
                for b in range(B):
                    if flat_obs[b].sum() > 0 and flat_mask[b].sum() == 0:
                        idxs = (flat_obs[b] > 0).nonzero(as_tuple=False).view(-1)
                        j = idxs[0]
                        flat_mask[b, j] = 1.0
                train_mask_all = flat_mask.view_as(obs_all)

            for step in range(n_steps):
                s = step * mb
                e = min(B, s + mb)
                if s >= e:
                    break

                X_init_b_t = X_init_all[s:e]
                X_nan_b_t  = X_nan_all[s:e]
                X_gt_b_t   = X_gt_all[s:e]
                train_mask = train_mask_all[s:e]

                X_nan_train = X_nan_b_t.clone()
                X_nan_train[train_mask.bool()] = float('nan')
                X_nan_train_safe = torch.nan_to_num(X_nan_train, nan=0.0, posinf=0.0, neginf=0.0)

                X_in_noisy = X_init_b_t + (noise_scale * torch.randn_like(X_init_b_t) if (noise_scale and noise_scale > 0) else 0.0)

                pred = model(X_in_noisy, X_nan_train_safe, S_t, L_t, lock_observed=False)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

                loss = _masked_mse(pred, X_gt_b_t, train_mask)
                (loss / n_steps).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % max(1, diff_epochs // 5) == 0 or epoch in (1, diff_epochs):
                # 复用最后一微批的指标，仅作监控
                with torch.no_grad():
                    mae_masked = _masked_mae(pred, X_gt_b_t, train_mask).item()
                    mae_obs_full = _masked_mae(
                        model(torch.nan_to_num(X_init_all[s:e], nan=0.0),
                              torch.nan_to_num(X_nan_all[s:e], nan=0.0),
                              S_t, L_t, lock_observed=False),
                        X_gt_all[s:e],
                        obs_all[s:e]
                    ).item()
                print(f"    [Train] epoch={epoch:03d}/{diff_epochs} "
                      f"loss(masked MSE)={loss.item():.6f}  "
                      f"mae(masked)={mae_masked:.6f}  mae(observed_ref)={mae_obs_full:.6f}  "
                      f"mask_ratio≈{train_mask.float().mean().item():.3f}")

    # 3) 训练后评估（微批前向）
    model.eval()
    with torch.no_grad():
        device_eval = next(model.parameters()).device
        X_init_all = torch.tensor(X_init_b, dtype=torch.float32, device=device_eval)
        X_nan_all  = torch.tensor(X_nan_b, dtype=torch.float32, device=device_eval)
        X_gt_all   = torch.tensor(X_gt_b, dtype=torch.float32, device=device_eval)
        obs_all    = torch.tensor(obs_b, dtype=torch.float32, device=device_eval)
        S_t = torch.tensor(S_np, dtype=torch.float32, device=device_eval)
        L_t = torch.tensor(L_np, dtype=torch.float32, device=device_eval)

        B = X_init_all.size(0)
        mb = max(1, int(diff_micro_bsz))
        n_steps = (B + mb - 1) // mb

        # 聚合指标
        num_sum = 0.0
        se_sum = 0.0
        ae_sum = 0.0

        for step in range(n_steps):
            s = step * mb
            e = min(B, s + mb)
            Xt  = X_init_all[s:e]
            Xnt = X_nan_all[s:e]
            Xnt_safe = torch.nan_to_num(Xnt, nan=0.0, posinf=0.0, neginf=0.0)

            pred = model(Xt, Xnt_safe, S_t, L_t, lock_observed=False)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

            gt = X_gt_all[s:e]
            ob = obs_all[s:e]

            sel = (ob > 0) & torch.isfinite(pred) & torch.isfinite(gt)
            n = sel.sum().item()
            if n > 0:
                diff = pred[sel] - gt[sel]
                se_sum += float((diff * diff).sum().item())
                ae_sum += float(diff.abs().sum().item())
                num_sum += float(n)

        rmse_obs = math.sqrt(se_sum / num_sum) if num_sum > 0 else 0.0
        mae_obs  = (ae_sum / num_sum) if num_sum > 0 else 0.0

    df = pd.DataFrame([{"method": "my_model", "RMSE_on_observed": rmse_obs, "MAE_on_observed": mae_obs}])
    print(df.to_string(index=False), flush=True)

    # 4) 推理：预测全域（lock_observed=False），仅在缺失位写回，确保覆盖初始插补
    device_inf = next(model.parameters()).device
    S_t = torch.tensor(S_np, dtype=torch.float32, device=device_inf)
    L_t = torch.tensor(L_np, dtype=torch.float32, device=device_inf)

    for (path, X_init, obs_mask, X_orig_nan) in imputed_files:
        X_in = X_init.copy()
        X_in[obs_mask == 1] = X_orig_nan[obs_mask == 1]
        X_nan = X_in.copy()
        X_nan[obs_mask == 0] = np.nan

        Xt  = torch.tensor(X_in,  dtype=torch.float32, device=device_inf).unsqueeze(0)
        Xnt = torch.tensor(X_nan, dtype=torch.float32, device=device_inf).unsqueeze(0)
        with torch.no_grad():
            Xnt_safe = torch.nan_to_num(Xnt, nan=0.0, posinf=0.0, neginf=0.0)
            pred_one = model(Xt, Xnt_safe, S_t, L_t, lock_observed=False).squeeze(0)
            pred_one = torch.nan_to_num(pred_one, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()

        out = X_in.copy()
        out[obs_mask == 0] = pred_one[obs_mask == 0]
        out_path = os.path.join(out_dir, os.path.basename(path))
        np.savetxt(out_path, out, delimiter=",")


# ---------------- 主流程 ----------------
def main(input_folder, output_folder, params,
         gpu_id=0,
         init_threshold=0.8, perturbation_prob=0.1, perturbation_scale=0.1,
         agg_threshold=0.5,
         D=5, T=50, width=256,
         output_folder_causal='./result',
         # diffusion 训练与评估超参（默认更稳）
         diff_epochs=20, diff_lr=5e-4, diff_weight_decay=0.0,
         phaseB_iter=3, val_ratio=0.1, seed=42,
         # 新增防退化训练超参
         drop_prob=0.2, noise_scale=0.01,
         # 新增：显存友好微批
         diff_micro_bsz=2):

    set_seed_all(seed)

    # Step 1
    clusters, centers = run_clustering(input_folder, output_folder=output_folder_causal, seed=seed)

    # 特征名
    sample_csv = centers[0] if centers else list_csvs(input_folder)[0]
    Fdim = pd.read_csv(sample_csv, nrows=0).shape[1]
    feature_names = get_feature_names_from_file(sample_csv, Fdim)

    # Step 2
    center_mats_imputed = initial_impute_centers(
        centers,
        threshold=init_threshold,
        perturbation_prob=perturbation_prob,
        perturbation_scale=perturbation_scale
    )

    # Step 3
    causal_dir = os.path.join(output_folder, "causal")
    S_list, L_list = causal_discovery_on_centers_multi_gpu(center_mats_imputed, params, causal_dir, base_seed=seed)

    # Step 4
    S_cont, L_global, S_bin = aggregate_global_graph(
        S_list, L_list, out_dir=causal_dir, method="mean_vote",
        threshold=agg_threshold, feature_names=feature_names
    )

    # Step 5
    files_all = list_csvs(input_folder)
    imputed_all = initial_impute_all_files(
        files_all,
        threshold=init_threshold,
        perturbation_prob=perturbation_prob,
        perturbation_scale=perturbation_scale
    )

    # Step 6 训练+评估+推理
    diff_out_dir = os.path.join(output_folder, "diffusion_imputed")
    # print("hahhahhaha_shape of S:", S_cont.shape)
    diffusion_train_eval_and_infer(
        imputed_all, S_cont, L_global,
        out_dir=diff_out_dir, D=D, T=T, width=width,
        diff_epochs=diff_epochs, diff_lr=diff_lr, diff_weight_decay=diff_weight_decay,
        phaseB_iter=phaseB_iter, val_ratio=val_ratio, seed=seed,
        drop_prob=drop_prob, noise_scale=noise_scale,
        diff_micro_bsz=diff_micro_bsz
    )

    log("✅ 全流程完成！")

# ==========================================================
# ========   消融实验：基于主流程 main 的统一封装   ========
# ==========================================================
import torch.optim as optim
import utils.causal_discovery as cd


def _simple_mean_impute(X_nan: np.ndarray) -> np.ndarray:
    """
    最基本的列均值插补。用于“无初始插补器”消融。
    """
    X = X_nan.astype(np.float32).copy()
    T, F = X.shape
    for j in range(F):
        col = X[:, j]
        m = ~np.isnan(col)
        if m.any():
            v = float(col[m].mean())
        else:
            v = 0.0
        col[~m] = v
        X[:, j] = col
    return X


def _diffusion_initial_only(
    imputed_files,
    global_S,
    global_L,
    out_dir,
    D=5, T=50, width=256,
    diff_epochs=30, diff_lr=5e-4, diff_weight_decay=0.0,
    phaseB_iter=3, val_ratio=0.1, seed=42,
    drop_prob=0.2, noise_scale=0.01,
    diff_micro_bsz=2,
):
    """
    no_diffusion 消融版本：
    - 不做任何 Diffusion 训练
    - 不再在这里挖缺失 / 抽样 / 计算 RMSE
    - 只对上游传来的带 NaN 矩阵做一次 initial_process，
      并把插补结果保存到 out_dir，供 evaluate_MSE 统一评估
    """
    set_seed_all(seed)
    os.makedirs(out_dir, exist_ok=True)

    for (path, X_init, obs_mask, X_orig_nan) in imputed_files:
        # X_orig_nan：上游传进来的带 NaN 的矩阵（缺失模式已经在外面确定好）
        X_eval = initial_process(X_orig_nan.copy(), 0.8, 0.5, 0.3)   # 使用 utils/initial_imputer.py
        X_eval = np.asarray(X_eval, dtype=np.float32)

        # 写到 diffusion_imputed 目录（main 里一般会把 out_dir 设成 diffusion_imputed）
        out_path = os.path.join(out_dir, os.path.basename(path))
        np.savetxt(out_path, X_eval, delimiter=",")

def main_ablation(
    input_folder,
    output_folder,
    params,
    ablation: str = "none",
    **kwargs
):
    """
    统一入口：在完整 main() 流程上做 6 种消融实验。

    ablation 可选值：
      - "none"                  : 原始完整模型（等价直接调用 main）
      - "no_causal_graph"       : 因果图 S=I, L=0
      - "no_diffusion"          : 只用初始插补，不训练 Diffusion
      - "no_initial_imputer"    : initial_process -> 简单列均值
      - "no_clustering"         : 不调用真正的聚类，每个文件自身为簇中心
      - "no_dynamic_threshold"  : 因果发现使用固定 top-k，而非 dynamic_lower_bound
      - "no_group_perturbation" : 保留 dynamic_lower_bound，但去掉 block_permute 验证
    """

    print(f"\n========== 运行消融实验：{ablation} ==========\n")

    # ---- 1) 备份原函数句柄 ----
    orig_compute = cd.compute_causal_strength_and_lag
    orig_compute_local = compute_causal_strength_and_lag  # 本模块里的别名
    orig_initial = initial_process
    orig_run_clustering = run_clustering
    orig_agg = aggregate_global_graph
    orig_diffusion = diffusion_train_eval_and_infer

    # ---- 2) 根据 ablation 类型做最小量 monkey-patch ----

    # 2.1 无因果图：S=I, L=0
    if ablation == "no_causal_graph":

        def compute_identity(X, params_local, gpu_id=0, seed=None):
            if isinstance(X, np.ndarray):
                F = X.shape[1]
            else:
                F = X.values.shape[1]
            S = np.eye(F, dtype=np.float32)
            L = np.zeros((F, F), dtype=np.int32)
            return S, L

        cd.compute_causal_strength_and_lag = compute_identity
        globals()["compute_causal_strength_and_lag"] = compute_identity

    # 2.2 无 Diffusion：只保留 initial_process 的结果
    elif ablation == "no_diffusion":
        globals()["diffusion_train_eval_and_infer"] = _diffusion_initial_only

    # 2.3 无初始插补器：使用最简单列均值插补
    elif ablation == "no_initial_imputer":

        def mean_initial_process(X, threshold, perturbation_prob, perturbation_scale):
            return _simple_mean_impute(X)

        globals()["initial_process"] = mean_initial_process

    # 2.4 无聚类：每个文件自身为簇中心（禁用真正的聚类算法）
    elif ablation == "no_clustering":

        def run_clustering_no_cluster(input_folder_local, output_folder_causal_local, seed: int = 42):
            csv_files = sorted(glob.glob(os.path.join(input_folder_local, "*.csv")))
            if not csv_files:
                raise FileNotFoundError(f"No CSV found in {input_folder_local}")
            clusters = [[f] for f in csv_files]
            centers = csv_files[:]  # 每个文件就是一个中心
            log("[Ablation no_clustering] 使用每个文件自身作为簇中心，跳过 utils.cluster.run")
            return clusters, centers

        globals()["run_clustering"] = run_clustering_no_cluster

    # 2.5 无动态阈值：用固定 top-k 代替 dynamic_lower_bound
    elif ablation == "no_dynamic_threshold":

        def compute_fixed_topk(X, params_local, gpu_id=0, seed=None):
            x, mask, _ = cd.prepare_data(X)
            device = torch.device(
                f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            )
            x = x.to(device)
            mask = mask.to(device)

            x_np = x.squeeze(0).cpu().numpy()
            F = x.shape[1]
            S = np.zeros((F, F), dtype=np.float32)
            L = np.zeros((F, F), dtype=np.int32)

            top_k = max(1, F // 2)

            for tgt in range(F):
                if seed is not None:
                    set_seed_all(seed + tgt)

                model = cd.ADDSTCN(
                    tgt, F,
                    params_local["layers"],
                    params_local["kernel_size"],
                    cuda=torch.cuda.is_available(),
                    dilation_c=params_local["dilation_c"],
                ).to(device)

                y = x[:, tgt, :].unsqueeze(-1).to(device)
                xx = x.to(device)
                msk = mask[:, tgt, :].to(device)

                opt = getattr(optim, params_local["optimizername"])(
                    model.parameters(), lr=params_local["lr"]
                )
                cd.train(xx, y, msk, model, opt, params_local["epochs"])

                scores = model.fs_attention.view(-1).detach().cpu().numpy()
                idx = np.argsort(-scores)[:top_k]

                for c in idx:
                    if c == tgt:
                        continue
                    S[tgt, c] = 1.0
                    L[tgt, c] = cd._estimate_best_lag(
                        x_np, tgt, c, params_local.get("max_lag", 10)
                    )

                if S[tgt].sum() == 0:
                    S[tgt, :] = 1.0
                    S[tgt, tgt] = 0.0
                    L[tgt, :] = 0

            return S, L

        cd.compute_causal_strength_and_lag = compute_fixed_topk
        globals()["compute_causal_strength_and_lag"] = compute_fixed_topk

    # 2.6 无分组扰动：保留动态阈值，去掉 block_permute 验证
    elif ablation == "no_group_perturbation":

        def compute_no_perturb(X, params_local, gpu_id=0, seed=None):
            x, mask, _ = cd.prepare_data(X)
            device = torch.device(
                f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            )
            x = x.to(device)
            mask = mask.to(device)

            x_np = x.squeeze(0).cpu().numpy()
            F = x.shape[1]
            S = np.zeros((F, F), dtype=np.float32)
            L = np.zeros((F, F), dtype=np.int32)

            for tgt in range(F):
                if seed is not None:
                    set_seed_all(seed + tgt)

                model = cd.ADDSTCN(
                    tgt, F,
                    params_local["layers"],
                    params_local["kernel_size"],
                    cuda=torch.cuda.is_available(),
                    dilation_c=params_local["dilation_c"],
                ).to(device)

                y = x[:, tgt, :].unsqueeze(-1).to(device)
                xx = x.to(device)
                msk = mask[:, tgt, :].to(device)

                opt = getattr(optim, params_local["optimizername"])(
                    model.parameters(), lr=params_local["lr"]
                )
                cd.train(xx, y, msk, model, opt, params_local["epochs"])

                scores = model.fs_attention.view(-1).detach().cpu().numpy()
                idx = np.argsort(-scores)
                alpha = params_local.get("alpha", 0.5)
                lb = cd.dynamic_lower_bound(scores, alpha=alpha)

                # 原来这里还会做 block_permute 验证；消融中直接用阈值筛选结果
                selected = [i for i in idx if scores[i] >= lb]

                for c in selected:
                    if c == tgt:
                        continue
                    S[tgt, c] = 1.0
                    L[tgt, c] = cd._estimate_best_lag(
                        x_np, tgt, c, params_local.get("max_lag", 10)
                    )

                if S[tgt].sum() == 0:
                    S[tgt, :] = 1.0
                    S[tgt, tgt] = 0.0
                    L[tgt, :] = 0

            return S, L

        cd.compute_causal_strength_and_lag = compute_no_perturb
        globals()["compute_causal_strength_and_lag"] = compute_no_perturb

    # ---- 3) 调用原始 main()（完整流程） ----
    try:
        main(
            input_folder=input_folder,
            output_folder=output_folder,
            params=params,
            **kwargs
        )
    finally:
        # ---- 4) 恢复所有被 patch 的函数，保证主流程不受影响 ----
        cd.compute_causal_strength_and_lag = orig_compute
        globals()["compute_causal_strength_and_lag"] = orig_compute_local
        globals()["initial_process"] = orig_initial
        globals()["run_clustering"] = orig_run_clustering
        globals()["aggregate_global_graph"] = orig_agg
        globals()["diffusion_train_eval_and_infer"] = orig_diffusion
        print(f"\n========== 消融 {ablation} 结束，已恢复原始函数 ==========\n")
if __name__ == "__main__":
    import sys
    import os
    import torch

    # ================== 命令行参数 ==================
    # 用法：python imputer_pipeline.py <input_folder> <output_folder>
    if len(sys.argv) != 3:
        print(
            f"Usage: python {os.path.basename(__file__)} <input_folder> <output_folder>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"[ERROR] input_folder does not exist or is not a directory: {input_folder}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    # ================== 硬件检查：强制要求 CUDA ==================
    if not torch.cuda.is_available():
        print(
            "[ERROR] CUDA is not available. "
            "This script is configured to run causal discovery on GPU only "
            "(no CPU fallback).",
            file=sys.stderr,
        )
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print(
            "[ERROR] No CUDA devices found. "
            "Please make sure at least one GPU is visible to PyTorch.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Detected {num_gpus} CUDA device(s). "
          f"Causal discovery will run with GPU parallelism.")

    # ================== 因果发现默认超参数 ==================
    # 注意：这里的 'layers' 必须是整数，表示 TCN 的层数(num_levels)，
    #      不能再是 list，否则会在 DepthwiseNet 里报 TypeError。
    #
    # 下面这份配置是相对稳妥的默认值，如果你在 utils/causal_discovery.py
    # 里有自己习惯的参数，可以按需改这里。
    params = {
        # TCN 层数（num_levels），用于 DepthwiseNet / ADDSTCN
        "layers": 4,              # <-- 关键：一定要是 int，而不是 [32,32,32]

        # 卷积核大小、扩张系数等结构超参
        "kernel_size": 3,
        "dilation_c": 2,

        # 训练相关
        "optimizername": "Adam",
        "lr": 1e-3,
        "epochs": 100,

        # 动态阈值和时滞估计相关
        "alpha": 0.5,             # dynamic_lower_bound 的系数
        "max_lag": 10,            # _estimate_best_lag 的最大滞后
    }

    print("[INFO] Causal discovery params:")
    for k, v in params.items():
        print(f"  - {k}: {v}")

    # ================== 聚类 / 因果 / Diffusion 结果目录 ==================
    # 你的 main 里有两个输出相关的路径：
    #   1) output_folder         ：总输出根目录
    #   2) output_folder_causal  ：聚类和因果发现的中间结果目录
    #
    # 在这里给 causal 单独弄一个子目录，避免和最终插补结果混在一起。
    output_folder_causal = os.path.join(output_folder, "causal_workdir")

    # ================== 调用原始 main ==================
    # main 会做：
    #   - Step 1: run_clustering
    #   - Step 2: initial_impute_centers
    #   - Step 3: causal_discovery_on_centers_multi_gpu（GPU 多进程）
    #   - Step 4: aggregate_global_graph
    #   - Step 5: initial_impute_all_files
    #   - Step 6: diffusion_train_eval_and_infer（含 GPU 训练与推理）
    #
    # 最终插补后的 CSV 会保存在：
    #   <output_folder>/diffusion_imputed/ 目录下
    main(
        input_folder=input_folder,
        output_folder=output_folder,
        params=params,
        output_folder_causal=output_folder_causal,
        # 如果你想在命令行层面统一指定训练轮数等，也可以在这里加：
        # diff_epochs=20,
        # drop_prob=0.2,
        # noise_scale=0.01,
        # diff_micro_bsz=2,
    )

    print("✅ All done. Final imputed CSVs are in:", os.path.join(output_folder, "diffusion_imputed"))
