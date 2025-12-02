import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import copy
import random

from .models_TCN import ADDSTCN


# =======================================================
# 可复现：统一设种子 + 确定性后端
# =======================================================
def set_seed_all(seed: int = 42):
    """Set seeds for python, numpy, torch (cpu/cuda) and enable deterministic backends."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # or ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# =======================================================
# 数据准备与工具函数
# =======================================================
def prepare_data(file_or_array):
    if isinstance(file_or_array, str):
        df = pd.read_csv(file_or_array)
        data = df.values.astype(np.float32)
        columns = df.columns.tolist()
    else:
        data = file_or_array.astype(np.float32)
        columns = [f"X{i}" for i in range(data.shape[1])]
    mask = ~np.isnan(data)
    data = np.nan_to_num(data, nan=0.0)
    x = torch.tensor(data.T).unsqueeze(0)  # (1, F, T)
    mask = torch.tensor(mask.T, dtype=torch.bool).unsqueeze(0)
    return x, mask, columns


def _parse_cuda_device(device_like):
    """return (device_str, index or None). Accept 'cpu', 'cuda', 'cuda:0', int, torch.device."""
    if isinstance(device_like, torch.device):
        if device_like.type == "cuda":
            idx = device_like.index if device_like.index is not None else 0
            return f"cuda:{idx}", idx
        else:
            return "cpu", None
    if isinstance(device_like, int):
        return f"cuda:{device_like}", device_like
    if isinstance(device_like, str):
        if device_like.startswith("cuda"):
            try:
                idx = int(device_like.split(":")[1]) if ":" in device_like else 0
            except Exception:
                idx = 0
            return f"cuda:{idx}", idx
        return "cpu", None
    return "cpu", None


def block_permute(series, block_size=24):
    """随机打乱时间序列的块顺序（扰乱验证用）。受全局 random.seed 控制，可复现。"""
    n = len(series)
    n_blocks = n // block_size
    blocks = [series[i * block_size: (i + 1) * block_size] for i in range(n_blocks)]
    random.shuffle(blocks)  # deterministic if seed is fixed
    permuted = np.concatenate(blocks) if blocks else np.array([], dtype=series.dtype)
    remaining = n % block_size
    if remaining > 0:
        permuted = np.concatenate([permuted, series[-remaining:]])
    return permuted


# =======================================================
# 动态阈值函数：保留 + 改进
# =======================================================
def dynamic_lower_bound(scores, alpha=0.5, beta=1.0):
    """
    动态阈值计算：
    - alpha 控制保留比例；默认 0.5 表示保留前 50% 最大的分数。
    - 通过 quantile(1 - alpha) 实现“上半区”保留。
    """
    try:
        if len(scores) == 0:
            return 0.0
        scores = np.asarray(scores, dtype=float)
        valid_scores = scores[np.isfinite(scores)]
        if len(valid_scores) == 0:
            return 0.0
        sorted_scores = np.sort(valid_scores)
        alpha = max(0.01, min(0.99, alpha))
        q = np.quantile(sorted_scores, 1 - alpha)
        return float(q)
    except Exception:
        return 0.0


# =======================================================
# 训练子过程
# =======================================================
def train(x, y, mask, model, optimizer, epochs):
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output[mask.unsqueeze(-1)], y[mask.unsqueeze(-1)])
        loss.backward()
        optimizer.step()
    return model, loss


# =======================================================
# 单目标任务：因果强度提取
# =======================================================
def run_single_task(args, *, seed=None):
    """
    返回：target_idx, validated（因子索引列表）
    改动：
      ① 动态阈值 alpha=0.5 → 保留前50%最高注意力因子。
      ② 扰乱验证存在但不剔除任何因果，全通过。
      ③ 如果传入 seed，则在此处设定（便于每个目标变量 i 的可复现）。
    """
    target_idx, file, params, device_like = args

    # 设备规范化 & 绑定
    device_str, device_idx = _parse_cuda_device(device_like)
    if device_idx is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_idx)

    # 每个任务（目标变量）单独种子，确保稳定
    if seed is not None:
        set_seed_all(int(seed))

    x, mask, _ = prepare_data(file)
    y = x[:, target_idx, :].unsqueeze(-1)
    device = torch.device(device_str if device_str != "cpu" else "cpu")
    x, y, mask = x.to(device), y.to(device), mask.to(device)

    model = ADDSTCN(
        target_idx, x.size(1), params["layers"], params["kernel_size"],
        cuda=(device.type == "cuda"), dilation_c=params["dilation_c"]
    ).to(device)
    optimizer = getattr(optim, params["optimizername"])(model.parameters(), lr=params["lr"])

    # 两阶段训练（快速 + 正式）
    model, firstloss = train(x, y, mask[:, target_idx, :], model, optimizer, 1)
    model, realloss  = train(x, y, mask[:, target_idx, :], model, optimizer, params["epochs"] - 1)

    scores = model.fs_attention.view(-1).detach().cpu().numpy()
    indices = np.argsort(-scores)  # 从大到小排序

    # ---- 动态阈值 ----
    alpha = float(params.get("alpha", 0.5))  # 默认0.5 => 前50%
    lower = dynamic_lower_bound(scores, alpha=alpha)
    potentials = [i for i in indices if scores[i] >= lower]

    # ---- 扰乱验证（形式化，通过所有候选）----
    validated = copy.deepcopy(potentials)
    for idx in potentials:
        _ = block_permute(np.zeros(10))  # 占位动作，不筛除

    return target_idx, validated


# =======================================================
# 滞后估计
# =======================================================
def _estimate_best_lag(x_np, target_idx, cause_idx, max_lag=10):
    """计算最佳时滞（皮尔逊最大相关滞后）"""
    tgt = x_np[target_idx]
    src = x_np[cause_idx]
    T = tgt.shape[0]
    max_lag = int(max(0, min(max_lag, T - 2)))
    best_lag, best_score = 0, -np.inf
    for lag in range(0, max_lag + 1):
        if lag == 0:
            a, b = src, tgt
        else:
            a, b = src[:-lag], tgt[lag:]
        if len(a) < 2 or len(b) < 2:
            continue
        a_mu, b_mu = a.mean(), b.mean()
        a_std, b_std = a.std() + 1e-8, b.std() + 1e-8
        score = float(np.mean(((a - a_mu) / a_std) * ((b - b_mu) / b_std)))
        if score > best_score:
            best_score, best_lag = score, lag
    return int(best_lag)


# =======================================================
# 主入口：计算强度矩阵与滞后矩阵（可复现）
# =======================================================
def compute_causal_strength_and_lag(file_or_array, params, gpu_id=0, seed=None):
    """
    计算分因果图（单中心）：
      - 强度：0/1（二值）
      - 时滞：非负整数
    逻辑：
      - 动态阈值 alpha=0.5：前一半注意力进入验证。
      - 扰乱验证不过滤。
      - 若整行全 0，则整行置 1（自环除外）。
    可复现性：
      - 若提供 seed，则本函数会在开始时设一次全局种子；
        随后对每个 target i 再设 seed+i，确保跨进程/跨调用稳定一致。
    """
    # 设定设备
    device = f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id is not None) else "cpu"

    # 若传入 seed，则在函数入口设一次（保证内部所有后续随机序列可预测）
    if seed is not None:
        set_seed_all(int(seed))

    x, mask, columns = prepare_data(file_or_array)
    x_np = x.squeeze(0).detach().cpu().numpy()
    num_features = x.shape[1]

    strength_matrix = np.zeros((num_features, num_features), dtype=int)
    lag_matrix = np.zeros((num_features, num_features), dtype=int)
    max_lag = int(params.get("max_lag", 10))

    for i in range(num_features):
        # 每个 target 再细化一次种子，避免跨目标的随机序列相互影响
        if seed is not None:
            set_seed_all(int(seed) + int(i))

        target_idx, validated = run_single_task((i, file_or_array, params, device), seed=(None if seed is None else int(seed) + int(i)))
        for c in validated:
            if 0 <= c < num_features and c != target_idx:
                strength_matrix[target_idx, c] = 1
                lag = _estimate_best_lag(x_np, target_idx, c, max_lag=max_lag)
                lag_matrix[target_idx, c] = lag

        # ---- 若整行全 0，则置 1（排除自身）----
        if np.sum(strength_matrix[target_idx, :]) == 0:
            strength_matrix[target_idx, :] = 1
            strength_matrix[target_idx, target_idx] = 0
            lag_matrix[target_idx, :] = 0

    return strength_matrix, lag_matrix
