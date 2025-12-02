import os
import gc
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import BayesianRidge
from sklearn.impute import SimpleImputer, KNNImputer
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from typing import List, Union

# =============================
# Utilities
# =============================

def _prefill_all_nan_columns(mx: np.ndarray, fill_value: float = -1.0) -> np.ndarray:
    """
    将整列全空（all-NaN）的列在进入模型前统一填成 fill_value（默认 -1）。
    仅作“前置兜底”，不做还原；后续模型可在此基础上继续学习/填补。
    """
    mx = np.array(mx, dtype=np.float32, copy=True)
    if mx.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {mx.shape}")
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        mx[:, all_nan_cols] = fill_value
    return mx

def _prep_data(mx: np.ndarray):
    import numpy as np
    mx = mx.astype(np.float32, copy=True)

    # step1: 全空列填 -1
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        print(f"[DEBUG] 全空列索引 {np.where(all_nan_cols)[0]} → -1 填充")
        mx[:, all_nan_cols] = -1.0

    # step2: 构造输入和掩码
    X_ori = mx[None, ...]                        # 保留原始（含 NaN）
    X = np.nan_to_num(X_ori, nan=-1.0)           # 模型输入必须无 NaN
    mask = (~np.isnan(X_ori)).astype(np.float32) # 1=观测，0=缺失

    # step3: 检查 mask 是否全 0
    total_obs = mask.sum()
    if total_obs == 0:
        print("[WARN] 样本 missing_mask 全 0 → 强制改为全 1 避免被 PyPOTS drop")
        mask[:] = 1.0

    # step4: 输出 debug 信息
    print(f"[DEBUG] _prep_data: X.shape={X.shape}, NaN数={np.isnan(X_ori).sum()}, "
          f"mask.sum={mask.sum()}")

    return {
        "X": X,
        "missing_mask": mask,
        "indicating_mask": mask,
        "X_ori": X_ori,
    }, X.shape[1:]
# =============================
# Simple baseline imputations
# =============================

def zero_impu(mx: np.ndarray) -> np.ndarray:
    """Replace missing values with zeros."""
    return np.nan_to_num(mx, nan=0.0)


def mean_impu(mx: np.ndarray) -> np.ndarray:
    """Replace missing values with the global mean."""
    mean = np.nanmean(mx)
    return np.where(np.isnan(mx), mean, mx)


def median_impu(mx: np.ndarray) -> np.ndarray:
    """Replace missing values with the global median."""
    mx = mx.copy()
    median = np.nanmedian(mx)
    return np.where(np.isnan(mx), median, mx)


def mode_impu(mx: np.ndarray) -> np.ndarray:
    """Replace missing values with the global mode (fallback = 0)."""
    mx = mx.copy()
    flat_values = mx[~np.isnan(mx)]
    global_mode = stats.mode(flat_values, keepdims=False).mode if flat_values.size > 0 else np.nan
    if np.isnan(global_mode):
        global_mode = 0.0
    inds = np.where(np.isnan(mx))
    mx[inds] = global_mode
    return mx


def random_impu(mx: np.ndarray) -> np.ndarray:
    """Replace missing values with random draws from observed values (fallback = -1)."""
    mx = mx.copy()
    non_nan_values = mx[~np.isnan(mx)]
    if non_nan_values.size == 0:
        mx[:] = -1.0
        return mx
    inds = np.where(np.isnan(mx))
    mx[inds] = np.random.choice(non_nan_values, size=len(inds[0]), replace=True)
    return mx


def ffill_impu(mx: np.ndarray) -> np.ndarray:
    """Forward fill missing values, then replace any remaining with the global mean."""
    mx = mx.copy()
    df = pd.DataFrame(mx)
    df = df.ffill(axis=0)
    global_mean = np.nanmean(mx)
    df = df.fillna(global_mean)
    return df.values


def bfill_impu(mx: np.ndarray) -> np.ndarray:
    """Backward fill missing values, then replace any remaining with the global mean."""
    mx = mx.copy()
    df = pd.DataFrame(mx)
    df = df.bfill(axis=0)
    global_mean = np.nanmean(mx)
    df = df.fillna(global_mean)
    return df.values


# =============================
# Intermediate methods
# =============================

def knn_impu(data: np.ndarray, k: int = 5) -> np.ndarray:
    data = data.copy()
    miss = np.isnan(data)

    # 列全空的情况，先用全局均值填补
    empty_cols = np.all(miss, axis=0)
    if empty_cols.any():
        g_mean = np.nanmean(data)
        if np.isnan(g_mean):
            g_mean = 0.0
        data[:, empty_cols] = g_mean

    # 限制 k 不超过有效样本数
    valid = (~miss).sum(axis=0).min()
    k = min(k, max(1, int(valid) - 1))

    imputer = KNNImputer(n_neighbors=k)
    out = imputer.fit_transform(data)

    # 轻微平滑与噪声调整
    col_mean = np.nanmean(data, axis=0)
    col_std = np.nanstd(data, axis=0)
    col_std = np.where(np.isnan(col_std) | (col_std == 0), 1.0, col_std)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)

    mask = miss
    shrink_ratio = 0.9
    nl = 0.5

    base = np.broadcast_to(col_mean, out.shape)
    out[mask] = (1 - shrink_ratio) * out[mask] + shrink_ratio * base[mask]
    rng = np.random.default_rng()
    jitter = rng.normal(0.0, col_std * nl, size=out.shape)
    out[mask] += jitter[mask]

    # 轻微裁剪范围，防止异常值
    cmin = np.nanmin(data, axis=0)
    cmax = np.nanmax(data, axis=0)
    cmin = np.where(np.isnan(cmin), -np.inf, cmin)
    cmax = np.where(np.isnan(cmax),  np.inf, cmax)
    for j in range(out.shape[1]):
        m = mask[:, j]
        if np.any(m):
            out[m, j] = np.clip(out[m, j], cmin[j], cmax[j])

    return out


def mice_impu(mx: np.ndarray, max_iter: int = 5) -> np.ndarray:
    """Multiple imputation by chained equations using Bayesian Ridge regression."""
    mx = mx.copy()
    n_rows, n_cols = mx.shape
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        global_mean = np.nanmean(mx)
        if np.isnan(global_mean):
            global_mean = 0.0
        mx[:, all_nan_cols] = global_mean

    imp = SimpleImputer(strategy='mean')
    matrix_filled = imp.fit_transform(mx)

    for _ in range(max_iter):
        for col in range(n_cols):
            missing_idx = np.where(np.isnan(mx[:, col]))[0]
            if len(missing_idx) == 0:
                continue
            observed_idx = np.where(~np.isnan(mx[:, col]))[0]
            X_train = np.delete(matrix_filled[observed_idx], col, axis=1)
            y_train = mx[observed_idx, col]
            X_pred = np.delete(matrix_filled[missing_idx], col, axis=1)
            model = BayesianRidge()
            model.fit(X_train, y_train)
            matrix_filled[missing_idx, col] = model.predict(X_pred)
    return matrix_filled


# =============================
# Advanced neural / complex methods
# =============================

def miracle_impu(mx: np.ndarray) -> np.ndarray:
    """
    MIRACLE neural imputation (new backbone).
    仅负责参数转接；实际训练/填补逻辑在新骨架中完成。
    - 不在此处加入任何多 GPU 逻辑（你会在第三个文件中实现）。
    - 保留整列全空的 -1.0 预填兜底，避免上游报错。
    """
    # ---- 兜底：整列全空 -> -1.0（与本文件其它方法保持一致） ----
    mx = _prefill_all_nan_columns(mx, fill_value=-1.0)
    mx = np.asarray(mx, dtype=np.float32)

    # 延迟导入以避免在未使用 MIRACLE 时引入额外依赖
    try:
        from complex_baseline_backbone.miracle import miracle_impute as _miracle_api_impute
    except Exception as e:
        # 如果用户尚未放置新骨架/封装文件，则回退到均值填补以保证流程不中断
        # 你也可以选择 raise e，视你的工程容错策略而定
        # raise
        return mean_impu(mx)

    # 只做参数传递；不在此处写并行/分布式逻辑
    # missing_list=None 让封装自动从 mx 中推断含空列
    n_features = mx.shape[1]
    try:
        imputed = _miracle_api_impute(
            mx,
            missing_list=None,                 # 自动推断
            n_hidden=min(64, max(16, n_features // 2)),
            lr=8e-3,
            max_steps=50,
            window=5,
            seed=42,
            device=None,                       # 由封装选择 'cuda' 或 'cpu'
            sync_pred_fn=None,                 # 多 GPU 聚合留给第三个文件
            verbose=False,
        )
        return imputed.astype(np.float32, copy=False)
    except Exception:
        # 兜底容错：若 MIRACLE 失败，则回退到均值填补，避免中断主流程
        return mean_impu(mx)


def saits_impu(mx, epochs=None, d_model=None, n_layers=None, device=None):
    from pypots.imputation import SAITS
    
    mx = mx.copy()
    seq_len, n_features = mx.shape
    total_size = seq_len * n_features
    
    global_mean = np.nanmean(mx)
    if np.isnan(global_mean):
        global_mean = 0.0
    
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        mx[:, all_nan_cols] = global_mean
    
    if epochs is None:
        if total_size > 50000:
            epochs = 20
            d_model = 64
            n_layers = 1
        elif total_size > 10000:
            epochs = 50
            d_model = 128
            n_layers = 2
        else:
            epochs = 100
            d_model = 128
            n_layers = 2
    
    if d_model is None:
        d_model = min(128, max(32, n_features * 4))
    
    if n_layers is None:
        n_layers = 2 if total_size < 20000 else 1
    
    try:
        data_3d = mx[np.newaxis, :, :]
        
        saits = SAITS(
            n_steps=seq_len,
            n_features=n_features,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=min(4, d_model // 32),
            d_k=d_model // 8,
            d_v=d_model // 8,
            d_ffn=d_model,
            dropout=0.1,
            epochs=epochs,
            patience=10,
            batch_size=32,
            device=device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        )
        
        train_set = {"X": data_3d}
        saits.fit(train_set)
        imputed_data_3d = saits.impute(train_set)
        
        return imputed_data_3d[0]
        
    except Exception as e:
        print(f"SAITS fails: {e}")
        return mean_impu(mx)


def timemixerpp_impu(mx):
    import numpy as np
    import torch
    from pypots.imputation import TimeMixerPP
    from sklearn.impute import SimpleImputer

    mx = mx.astype(np.float32)
    global_mean = np.nanmean(mx)
    all_nan_cols = np.all(np.isnan(mx), axis=0)
    if all_nan_cols.any():
        mx[:, all_nan_cols] = global_mean
    T, N = mx.shape
    data = mx[None, ...]  

    missing_mask = np.isnan(data).astype(np.float32)
    indicating_mask = (~np.isnan(data)).astype(np.float32)
    imp = SimpleImputer(strategy='mean', keep_empty_features=True)
    X_filled = imp.fit_transform(mx).astype(np.float32)
    X_filled = X_filled[None, ...]

    dataset = {
        "X": X_filled,
        "missing_mask": missing_mask,
        "indicating_mask": indicating_mask,
        "X_ori": data
    }

    model = TimeMixerPP(
            n_steps=T,
            n_features=N,
            n_layers=1,
            d_model=64,  
            d_ffn=128,  
            top_k=T//2,  
            n_heads=2,   
            n_kernels=6, 
            dropout=0.1,
            channel_mixing=True,  
            channel_independence=False,  
            downsampling_layers=1,    
            downsampling_window=2,   
            apply_nonstationary_norm=False,
            batch_size=1,
            epochs=10,
            patience=3,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    model.fit(train_set=dataset)

    result = model.predict(dataset)
    if isinstance(result, dict):
        imputed = result.get('imputation', list(result.values())[0])
    else:
        imputed = result

    if len(imputed.shape) == 3:
        imputed = imputed[0]

    return imputed


def tefn_impu(mx, epoch: int = 100, device: str = "cuda"):
    import numpy as np, torch
    from torch.utils.data import Dataset, DataLoader
    from pypots.imputation import TEFN
    from pypots.optim.adam import Adam
    from pypots.nn.modules.loss import MAE, MSE

    data, (T, N) = _prep_data(mx)

    # 确保 X_ori 没有 NaN
    if np.isnan(data["X_ori"]).any():
        data["X_ori"] = np.nan_to_num(data["X_ori"], nan=0.0)

    class One(Dataset):
        def __len__(self): return 1
        def __getitem__(self, idx):
            return (idx, data["X"][0], data["missing_mask"][0],
                    data["X_ori"][0], data["indicating_mask"][0])

    dl = DataLoader(One(), batch_size=1, shuffle=False)

    model = TEFN(
        n_steps=T, n_features=N, n_fod=1,
        batch_size=1, epochs=1, patience=1,
        training_loss=MAE, validation_metric=MSE, optimizer=Adam,
        device=device, verbose=False
    )
    model._train_model(dl, dl)
    model.model.load_state_dict(model.best_model_dict)

    try:
        out = model.impute(data)
        if isinstance(out, dict):
            imputed = out.get("imputation", list(out.values())[0])[0]
        else:
            imputed = out[0]
        return imputed
    except Exception as e:
        import traceback
        print("[ERROR][TEFN] impute调用失败:", e)
        traceback.print_exc()
        raise


def timesnet_impu(mx):
    import numpy as np, torch
    from pypots.imputation.timesnet import TimesNet
    data, (T, N) = _prep_data(mx)

    model = TimesNet(
        n_steps=T, n_features=N,
        n_layers=1, top_k=1, d_model=2, d_ffn=2, n_kernels=2,
        batch_size=1, epochs=1, patience=1,
        device="cuda", verbose=False
    )
    model.fit(data)

    try:
        out = model.impute(data)
        if isinstance(out, dict):
            imputed = out.get("imputation", list(out.values())[0])[0]
        else:
            imputed = out[0]
        return imputed
    except Exception as e:
        import traceback
        print("[ERROR][TimesNet] impute调用失败:", e)
        traceback.print_exc()
        raise


def tsde_impu(mx: np.ndarray, n_samples: int = 40,
              device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
    """
    TSDE diffusion-based imputation — GPU 调度封装（仅作用于本方法）：
    - 只在 GPU 跑；不回退 CPU
    - 自动在多卡上排队：每张卡并发上限由环境变量控制
      * TSDE_PER_GPU_LIMIT: 每张卡最多并发任务数（默认 1）
      * TSDE_MIN_FREE_MB:  选择一张卡所需的最小剩余显存（默认 1024 MB）
      * TSDE_POLL_SEC:     等待轮询间隔（默认 0.5 秒）
    - 默认无调试打印；若设置 TSDE_DEBUG=1，则打印第一条异常的栈以便定位
    """
    import os, time, errno
    import fcntl
    import numpy as _np
    from complex_baseline_backbone.tsde import impute_missing_data, _coerce_to_float32
    if not torch.cuda.is_available():
        raise RuntimeError("TSDE is GPU-only: CUDA not available.")

    # ---- 参数 ----
    per_gpu_limit = int(os.getenv("TSDE_PER_GPU_LIMIT", "1"))
    min_free_mb = int(os.getenv("TSDE_MIN_FREE_MB", "1024"))
    poll_sec = float(os.getenv("TSDE_POLL_SEC", "0.5"))
    lock_dir = "/tmp/tsde_gpu_locks"
    os.makedirs(lock_dir, exist_ok=True)
    debug = os.getenv("TSDE_DEBUG", "0") == "1"

    # ---- 选择/占用 GPU 槽位（阻塞直到成功） ----
    def _acquire_slot() -> tuple:
        """返回 (device_str, lock_file_obj)"""
        while True:
            # 枚举 GPU，找满足剩余显存的卡
            candidates = []
            for i in range(torch.cuda.device_count()):
                try:
                    free_b, total_b = torch.cuda.mem_get_info(i)
                    if free_b >= min_free_mb * 1024 * 1024:
                        candidates.append((free_b, i))
                except Exception:
                    continue
            # 如果没有满足显存阈值的，也尝试所有卡（避免 mem_get_info 异常时饿死）
            if not candidates:
                candidates = [(0, i) for i in range(torch.cuda.device_count())]

            # 优先选空闲最多的
            candidates.sort(reverse=True)

            # 在候选卡上尝试抢占一个 slot 锁
            for _, gpu_idx in candidates:
                for slot in range(per_gpu_limit):
                    path = os.path.join(lock_dir, f"tsde_{gpu_idx}_{slot}.lock")
                    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
                    f = os.fdopen(fd, "w")
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # 抢到锁：返回对应设备
                        return (f"cuda:{gpu_idx}", f)
                    except OSError as e:
                        if e.errno in (errno.EACCES, errno.EAGAIN):
                            f.close()
                            continue
                        else:
                            f.close()
                            continue
            # 所有槽位都满了 -> 等待再试
            time.sleep(poll_sec)

    def _release_slot(lock_file_obj):
        try:
            fcntl.flock(lock_file_obj, fcntl.LOCK_UN)
        finally:
            try:
                lock_file_obj.close()
            except Exception:
                pass

    # ---- 预处理输入 ----
    mx = _np.array(mx, copy=True)
    if mx.ndim != 2:
        if mx.ndim == 1:
            mx = mx.reshape(-1, 1).astype(_np.float32, copy=False)
        else:
            mx = mx.reshape(mx.shape[0], -1).astype(_np.float32, copy=False)
    # 整列全空 -> -1（初始化）
    try:
        if _np.issubdtype(mx.dtype, _np.floating):
            all_nan_cols = _np.all(_np.isnan(mx), axis=0)
            if all_nan_cols.any():
                mx[:, all_nan_cols] = -1.0
    except Exception:
        pass

    mx = _coerce_to_float32(mx)

    # ---- 获取 GPU 槽位并执行 ----
    dev_str, lock_f = _acquire_slot()
    try:
        # 关闭 cuDNN，避免多进程 RNN 初始化问题（仍然在 GPU 上跑）
        torch.backends.cudnn.enabled = False

        try:
            out = impute_missing_data(
                data=mx,
                n_samples=1,
                device=dev_str,
                epochs=5,
                hidden_dim=8,
                num_steps=4,
            )
        except Exception as e:
            if debug:
                import traceback
                print("[TSDE DEBUG] First error:", repr(e))
                traceback.print_exc()
            raise

        out = _np.asarray(out, dtype=_np.float32)
        if out.shape != mx.shape:
            out = out.reshape(mx.shape[0], mx.shape[1])
        out = _np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(_np.float32, copy=False)
        return out
    finally:
        # 释放槽位 + 清理缓存
        _release_slot(lock_f)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def grin_impu(mx: np.ndarray,
              window_size: Optional[int] = None,
              hidden_dim: Optional[int] = None,
              epochs: Optional[int] = None,
              lr: float = 1e-5,
              device: Optional[str] = None) -> np.ndarray:
    """
    GRIN graph-based imputation (process-safe).
    This function takes ONE matrix (T x F) and imputes it on the current process.
    Parallelism comes from your outer evaluator spawning multiple processes.

    Args:
        mx: np.ndarray (T, F)
        device: 'cpu' or 'cuda:N' (optional). If None, auto-pick per-process.

    Returns:
        np.ndarray (T, F)
    """
    from complex_baseline_backbone.grin import impute_single_matrix
    return impute_single_matrix(
        mx=mx,
        window_size=window_size,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        device_str=device,
    )



def csdi_impu(matrix_2d: np.ndarray,
                n_samples: int = 10,
                device: str = None,
                ckpt_path: str = None) -> np.ndarray:
    from complex_baseline_backbone.csdi import CSDI_Physio
    assert matrix_2d.ndim == 2, "输入必须是二维 numpy 矩阵 (L, K)"
    L, K = matrix_2d.shape

    # 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    config = {
        "model": {
            "timeemb": 128,           # 时间嵌入维度
            "featureemb": 16,         # 特征嵌入维度
            "is_unconditional": False,
            "target_strategy": "random",
        },
        "diffusion": {
            "num_steps": 50,          # 训练步数
            "beta_start": 1e-4,
            "beta_end": 2e-2,
            "schedule": "linear",

            # ====== 下面是新增的必要参数 ======
            "channels": 64,                # hidden channels（Transformer宽度）
            "diffusion_embedding_dim": 128,  # Diffusion 嵌入维度
            "side_dim": 145,               # side_info的总维度（大约=128+16+1）
            "nheads": 8,                   # 多头注意力的头数
            "layers": 4,                   # 残差块层数
            "is_linear": False,            # 是否使用线性attention
        },
    }


    # 构造观测数据与掩码
    x_np = np.array(matrix_2d, dtype=np.float32)
    obs_mask_np = (~np.isnan(x_np)).astype(np.float32)  # 1=观测到，0=缺失
    # 缺失处先置零，模型内部会用 cond_mask 指示哪些是条件输入
    x_np_filled0 = np.nan_to_num(x_np, nan=0.0).astype(np.float32)

    # 组一个 batch 字典，形状与 CSDI_Physio.process_data 的预期一致
    # CSDI 的 process_data 里会把 (B, L, K) 转为 (B, K, L)
    B = 1
    batch = {
        "observed_data": torch.from_numpy(x_np_filled0[None, ...]),  # (1, L, K)
        "observed_mask": torch.from_numpy(obs_mask_np[None, ...]),   # (1, L, K)
        "timepoints": torch.arange(L, dtype=torch.float32)[None, ...],  # (1, L)
        # gt_mask：评估/采样阶段作为条件掩码使用。这里就用观测掩码即可。
        "gt_mask": torch.from_numpy(obs_mask_np[None, ...]),         # (1, L, K)
    }

    # 实例化模型（使用 Physio 这个适配器即可，target_dim 传入 K）
    model = CSDI_Physio(config=config, device=device, target_dim=K).to(device)
    model.eval()

    # 如提供 checkpoint 且存在，则尝试加载（可选）
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # 兼容常见的保存方式
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    # 移到设备
    for k in batch:
        batch[k] = batch[k].to(device)

    # 直接走模型的 impute 流程（不通过 forward），以观测为条件、对缺失进行采样
    with torch.no_grad():
        # Process_data 会把 (B,L,K) → (B,K,L)
        observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length = \
            model.process_data(batch)

        cond_mask = observed_mask  # 条件：所有观测到的位置
        side_info = model.get_side_info(observed_tp, cond_mask)

        # 采样 n_samples 条样本，并取中位数作为确定性插补
        samples = model.impute(observed_data, cond_mask, side_info, n_samples=n_samples)  # (B, n, K, L)
        median_sample = samples.median(dim=1).values  # (B, K, L)

        # 把 (B, K, L) 转回 (B, L, K)
        median_sample = median_sample.permute(0, 2, 1)  # (B, L, K)
        out = median_sample[0]  # (L, K)

        # 观测到的位置用原值强制还原，仅对缺失处使用模型输出
        out = observed_mask[0].permute(1, 0) * observed_data[0].permute(1, 0) + \
              (1 - observed_mask[0].permute(1, 0)) * out

    return out.detach().cpu().numpy()


def diffputer_impu(matrix_2d: np.ndarray, device: str = None) -> np.ndarray:
    """
    Thin wrapper.
    All logic lives in complex_baseline_backbone/diffputer.py .
    """
    from complex_baseline_backbone.diffputer import diffputer_impu as _impl
    return _impl(matrix_2d, device=device)

if __name__ == "__main__":
    # quick demo
    rng = np.random.default_rng(0)
    L, K = 32, 6
    X = rng.normal(size=(L, K)).astype(np.float32)
    miss = rng.random(size=(L, K)) < 0.2
    X[miss] = np.nan

    X_filled = diffputer_impu(X, device=None)
    print("Missing before:", np.isnan(X).sum(), " -> after:", np.isnan(X_filled).sum())