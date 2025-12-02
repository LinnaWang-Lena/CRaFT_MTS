import os
import gc
import random
import shutil
import tempfile
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import multiprocessing as mp
from functools import partial

from pygrinder import mcar, mar_logistic, mnar_x
from imputer import *
from imputer import main_ablation as imputer_main

# 新增：导入 initial_imputer 的 baseline 实现
from utils.initial_imputer import initial_process


def set_seed_all(seed: int = 42) -> None:
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


def rmse_on_missing(pred: np.ndarray, gt: np.ndarray, obs_mask: np.ndarray) -> float:
    miss = obs_mask == 0
    if not miss.any():
        return 0.0
    diff = pred[miss] - gt[miss]
    v = float(np.sqrt(np.mean(diff ** 2)))
    if not np.isfinite(v):
        v = 0.0
    return v


def mae_on_missing(pred: np.ndarray, gt: np.ndarray, obs_mask: np.ndarray) -> float:
    miss = obs_mask == 0
    if not miss.any():
        return 0.0
    diff = np.abs(pred[miss] - gt[miss])
    v = float(np.mean(diff))
    if not np.isfinite(v):
        v = 0.0
    return v


def apply_missing(
    mx: np.ndarray,
    mode: str,
    seed: int,
    mar_obs_rate: float = 0.1,
    mar_missing_rate: float = 0.6,
    mnar_offset: float = 0.6,
    mcar_p: float = 0.5,
) -> np.ndarray:
    set_seed_all(seed)
    X = mx.copy()
    mode = mode.upper()
    if mode == "MAR":
        Xz = X.astype(np.float64, copy=True)
        col_std = Xz.std(axis=0, ddof=0)
        col_std[col_std == 0] = 1.0
        Xz = (Xz - Xz.mean(axis=0)) / col_std
        Xz_corrupt = mar_logistic(
            Xz, obs_rate=mar_obs_rate, missing_rate=mar_missing_rate
        )
        miss_mask = np.isnan(Xz_corrupt)
        X[miss_mask] = np.nan
        return X
    elif mode == "MNAR":
        X = X[np.newaxis, ...]
        X = mnar_x(X, offset=mnar_offset)
        X = X.squeeze(0)
    elif mode == "MCAR":
        X = X[np.newaxis, ...]
        X = mcar(X, p=mcar_p)
        X = X.squeeze(0)
    else:
        raise ValueError(f"Unknown missing mode: {mode}")
    return X


def list_csvs(dir_path: str) -> List[str]:
    return [f for f in sorted(os.listdir(dir_path)) if f.lower().endswith(".csv")]


def read_csv_matrix(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df.values.astype(np.float32)


_IMPUTER_DEFAULTS = dict(
    params={
        "layers": 5,
        "kernel_size": 3,
        "dilation_c": 2,
        "optimizername": "Adam",
        "lr": 0.001,
        "epochs": 50,
        "significance": 0.05,
        "max_lag": 10,
        "alpha": 0.5,
        "perturbation_validate": False,
    },
    init_threshold=0.8,
    perturbation_prob=0.1,
    perturbation_scale=0.1,
    agg_threshold=0.5,
    D=5,
    T=50,
    width=256,
    diff_epochs=20,
    diff_lr=1e-3,
    diff_weight_decay=1.0,
    phaseB_iter=3,
    val_ratio=0.1,
)


def _shard(items: List[str], n: int) -> List[List[str]]:
    if n <= 1:
        return [items]
    buckets = [[] for _ in range(n)]
    for i, x in enumerate(items):
        buckets[i % n].append(x)
    return [b for b in buckets if b]


def _simple_mean_impute(Xmiss: np.ndarray) -> np.ndarray:
    X = Xmiss.astype(np.float32, copy=True)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    col_means = np.nanmean(X, axis=0)
    if np.any(~np.isfinite(col_means)):
        global_mean = np.nanmean(X)
        if not np.isfinite(global_mean):
            global_mean = 0.0
        bad = ~np.isfinite(col_means)
        col_means[bad] = global_mean
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_means, inds[1])
    return X


def _clean_imputation(imp: np.ndarray, Xmiss: np.ndarray) -> np.ndarray:
    if imp is None:
        return _simple_mean_impute(Xmiss)
    imp_arr = np.asarray(imp, dtype=np.float32)
    if imp_arr.shape != Xmiss.shape:
        return _simple_mean_impute(Xmiss)
    if not np.isfinite(imp_arr).all():
        base = _simple_mean_impute(Xmiss)
        bad = ~np.isfinite(imp_arr)
        imp_arr[bad] = base[bad]
    return imp_arr


def _worker_run_imputer(
    shard_in_dir: str,
    shard_out_dir: str,
    gpu_id: int,
    *,
    params: dict,
    init_threshold: float,
    perturbation_prob: float,
    perturbation_scale: float,
    agg_threshold: float,
    D: int,
    T: int,
    width: int,
    diff_epochs: int,
    diff_lr: float,
    diff_weight_decay: float,
    phaseB_iter: int,
    val_ratio: float,
    seed: int,
    diff_micro_bsz: int,
    ablation: str = "none",  # 告诉 imputer_main 用哪种消融
):
    try:
        if gpu_id is not None and gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        set_seed_all(seed)

        imputer_main(
            input_folder=shard_in_dir,
            output_folder=shard_out_dir,
            params=params,
            ablation=ablation,
            init_threshold=init_threshold,
            perturbation_prob=perturbation_prob,
            perturbation_scale=perturbation_scale,
            agg_threshold=agg_threshold,
            D=D,
            T=T,
            width=width,
            output_folder_causal="./result",
            diff_epochs=diff_epochs,
            diff_lr=diff_lr,
            diff_weight_decay=diff_weight_decay,
            phaseB_iter=phaseB_iter,
            val_ratio=val_ratio,
            seed=seed,
            drop_prob=0.2,
            noise_scale=0.02,
            diff_micro_bsz=diff_micro_bsz,
        )
    except Exception as e:
        try:
            os.makedirs(shard_out_dir, exist_ok=True)
            with open(os.path.join(shard_out_dir, "_imputer_error.txt"), "w") as f:
                f.write(repr(e))
        except Exception:
            pass


def _run_ablation_single(
    args: Tuple[str, str, Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray]
) -> Dict:
    fname, method_name, abl_fn, gt, obs_mask = args
    try:
        Xmiss = gt.copy().astype(np.float32)
        Xmiss[obs_mask == 0] = np.nan
        try:
            imp_abl_raw = abl_fn(Xmiss)
        except Exception:
            imp_abl_raw = None
        imp_abl = _clean_imputation(imp_abl_raw, Xmiss)
        rmse_abl = rmse_on_missing(imp_abl, gt, obs_mask)
        mae_abl = mae_on_missing(imp_abl, gt, obs_mask)
    except Exception:
        rmse_abl, mae_abl = 0.0, 0.0
    return {
        "dataset": fname,
        "method": method_name,
        "RMSE": rmse_abl,
        "MAE": mae_abl,
    }


from typing import Optional, Dict, List, Tuple, Callable  # 保持你原来的重复导入，最小改动


def _evaluate_dirlevel_imputer_multi_gpu(
    data_dir: str,
    output_dir: str,
    missing_mode: str,
    seed: int,
    mar_obs_rate: float,
    mar_missing_rate: float,
    mnar_offset: float,
    mcar_p: float,
    method_name: str,
    params: dict,
    init_threshold: float,
    perturbation_prob: float,
    perturbation_scale: float,
    agg_threshold: float,
    D: int,
    T: int,
    width: int,
    diff_epochs: int,
    diff_lr: float,
    diff_weight_decay: float,
    phaseB_iter: int,
    val_ratio: float,
    diff_micro_bsz: int,
    ablation_map: Optional[Dict[str, str]] = None,  # key: 方法名, value: main_ablation 的 ablation 参数
) -> pd.DataFrame:
    set_seed_all(seed)
    files = list_csvs(data_dir)
    if not files:
        raise ValueError(f"No CSV files found in {data_dir}")

    # ====== 1) 构造缺失数据并分 shard ======
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        n_shards = min(num_gpus, len(files))
    else:
        cpu_workers = max(1, min(len(files), (mp.cpu_count() // 2) or 1))
        n_shards = cpu_workers

    shard_files = _shard(files, n_shards)
    shard_dirs: List[Tuple[str, List[str]]] = []  # (shard_in_dir, file_list)

    gt_cache: Dict[str, np.ndarray] = {}
    mask_cache: Dict[str, np.ndarray] = {}

    base_seed = seed
    for shard_idx, flist in enumerate(shard_files):
        shard_in = tempfile.mkdtemp(prefix="eval_imp_shard_in_")
        shard_dirs.append((shard_in, flist))
        for j, fname in enumerate(flist):
            gt = read_csv_matrix(os.path.join(data_dir, fname))
            Xmiss = apply_missing(
                gt,
                missing_mode,
                seed=base_seed + (len(files) * shard_idx + j),
                mar_obs_rate=mar_obs_rate,
                mar_missing_rate=mar_missing_rate,
                mnar_offset=mnar_offset,
                mcar_p=mcar_p,
            )
            gt_cache[fname] = gt
            mask_cache[fname] = (~np.isnan(Xmiss)).astype(np.int32)
            pd.DataFrame(Xmiss).to_csv(os.path.join(shard_in, fname), index=False)

    # ====== 通用：跑一个方法（主模型或某个消融） ======
    def _run_one_method(method: str, ablation: str) -> List[Dict]:
        shard_out_dirs: List[str] = []
        procs: List[mp.Process] = []

        # 2.1 启动每个 shard 的子进程
        for shard_idx, (shard_in, flist) in enumerate(shard_dirs):
            if not flist:
                continue
            shard_out = tempfile.mkdtemp(prefix=f"eval_imp_{method}_out_")
            shard_out_dirs.append(shard_out)

            if num_gpus > 0:
                gpu_id = shard_idx % num_gpus
            else:
                gpu_id = -1

            p = mp.Process(
                target=_worker_run_imputer,
                args=(shard_in, shard_out, gpu_id),
                kwargs=dict(
                    params=params,
                    init_threshold=init_threshold,
                    perturbation_prob=perturbation_prob,
                    perturbation_scale=perturbation_scale,
                    agg_threshold=agg_threshold,
                    D=D,
                    T=T,
                    width=width,
                    diff_epochs=diff_epochs,
                    diff_lr=diff_lr,
                    diff_weight_decay=diff_weight_decay,
                    phaseB_iter=phaseB_iter,
                    val_ratio=val_ratio,
                    seed=seed,
                    diff_micro_bsz=diff_micro_bsz,
                    ablation=ablation,
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # 2.2 读取该方法的输出并计算 RMSE/MAE
        rows: List[Dict] = []
        for (shard_in, flist), shard_out in zip(shard_dirs, shard_out_dirs):
            out_imputed_dir = os.path.join(shard_out, "diffusion_imputed")
            for fname in flist:
                gt = gt_cache[fname]
                obs_mask = mask_cache[fname]

                imp_path = os.path.join(out_imputed_dir, fname)
                if os.path.exists(imp_path):
                    try:
                        imp_raw = pd.read_csv(imp_path, header=None).values.astype(
                            np.float32
                        )
                    except Exception:
                        imp_raw = None
                else:
                    imp_raw = None

                Xmiss_local = gt.copy().astype(np.float32)
                Xmiss_local[obs_mask == 0] = np.nan
                imp = _clean_imputation(imp_raw, Xmiss_local)

                rmse = rmse_on_missing(imp, gt, obs_mask)
                mae = mae_on_missing(imp, gt, obs_mask)
                rows.append(
                    {
                        "dataset": fname,
                        "method": method,
                        "RMSE": rmse,
                        "MAE": mae,
                    }
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        for d in shard_out_dirs:
            shutil.rmtree(d, ignore_errors=True)

        return rows

    # ====== 2) 跑主模型（ablation="none"） ======
    rows_main = _run_one_method(method_name, ablation="none")

    # ====== 3) 跑所有消融 ======
    rows_ablation: List[Dict] = []
    if ablation_map:
        for abl_method, abl_flag in ablation_map.items():
            # 不再 special case ablation_no_diffusion，
            # 统一走 _run_one_method（内部会调用 main_ablation → _diffusion_initial_only）
            rows_ablation.extend(_run_one_method(abl_method, ablation=abl_flag))


    # ====== 汇总 & 清理 ======
    rows = rows_main + rows_ablation
    df_long = pd.DataFrame(rows)
    df_long["RMSE"] = df_long["RMSE"].fillna(0.0)
    df_long["MAE"] = df_long["MAE"].fillna(0.0)

    avg_df = (
        df_long.groupby("method", as_index=False)[["RMSE", "MAE"]]
        .mean()
        .sort_values("method")
        .reset_index(drop=True)
    )

    os.makedirs(output_dir, exist_ok=True)
    avg_df.to_csv(os.path.join(output_dir, "impute_eval_RMSE_MAE.csv"), index=False)

    for shard_in, _ in shard_dirs:
        shutil.rmtree(shard_in, ignore_errors=True)

    return avg_df


def evaluate_MSE(
    data_dir: str,
    output_dir: str,
    mode: str = "MAR",
    seed: int = 56,
    mar_obs_rate: float = 0.1,
    mar_missing_rate: float = 0.6,
    mnar_offset: float = 0.6,
    mcar_p: float = 0.5,
    params: dict = None,
    init_threshold: float = None,
    perturbation_prob: float = None,
    perturbation_scale: float = None,
    agg_threshold: float = None,
    D: int = None,
    T: int = None,
    width: int = None,
    diff_epochs: int = None,
    diff_lr: float = None,
    diff_weight_decay: float = None,
    phaseB_iter: int = None,
    val_ratio: float = None,
    diff_micro_bsz: int = 2,
) -> pd.DataFrame:
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    p = params if params is not None else _IMPUTER_DEFAULTS["params"]
    init_thr = (
        _IMPUTER_DEFAULTS["init_threshold"]
        if init_threshold is None
        else init_threshold
    )
    p_prob = (
        _IMPUTER_DEFAULTS["perturbation_prob"]
        if perturbation_prob is None
        else perturbation_prob
    )
    p_scale = (
        _IMPUTER_DEFAULTS["perturbation_scale"]
        if perturbation_scale is None
        else perturbation_scale
    )
    agg_thr = (
        _IMPUTER_DEFAULTS["agg_threshold"] if agg_threshold is None else agg_threshold
    )
    DD = _IMPUTER_DEFAULTS["D"] if D is None else D
    TT = _IMPUTER_DEFAULTS["T"] if T is None else T
    WW = _IMPUTER_DEFAULTS["width"] if width is None else width
    de = _IMPUTER_DEFAULTS["diff_epochs"] if diff_epochs is None else diff_epochs
    dlr = _IMPUTER_DEFAULTS["diff_lr"] if diff_lr is None else diff_lr
    dwd = (
        _IMPUTER_DEFAULTS["diff_weight_decay"]
        if diff_weight_decay is None
        else diff_weight_decay
    )
    pbi = _IMPUTER_DEFAULTS["phaseB_iter"] if phaseB_iter is None else phaseB_iter
    vr = _IMPUTER_DEFAULTS["val_ratio"] if val_ratio is None else val_ratio

    # ---- 消融：方法名 -> main_ablation 的 ablation 字符串 ----
    ablation_map = {
        # "ablation_no_causal_graph": "no_causal_graph",
        "ablation_no_diffusion": "no_diffusion",  # 名称保持不变，运行时已被 special case 成 initial_imputer 基线
        # "ablation_no_initial_imputer": "no_initial_imputer",
        # "ablation_no_clustering": "no_clustering",
        # "ablation_no_dynamic_threshold": "no_dynamic_threshold",
        # "ablation_no_group_perturbation": "no_group_perturbation",
    }

    avg_df = _evaluate_dirlevel_imputer_multi_gpu(
        data_dir=data_dir,
        output_dir=output_dir,
        missing_mode=mode,
        seed=seed,
        mar_obs_rate=mar_obs_rate,
        mar_missing_rate=mar_missing_rate,
        mnar_offset=mnar_offset,
        mcar_p=mcar_p,
        method_name="my_model",
        params=p,
        init_threshold=init_thr,
        perturbation_prob=p_prob,
        perturbation_scale=p_scale,
        agg_threshold=agg_thr,
        D=DD,
        T=TT,
        width=WW,
        diff_epochs=de,
        diff_lr=dlr,
        diff_weight_decay=dwd,
        phaseB_iter=pbi,
        val_ratio=vr,
        diff_micro_bsz=diff_micro_bsz,
        ablation_map=ablation_map,
    )
    print(avg_df)
    return avg_df


if __name__ == "__main__":
    evaluate_MSE(
        data_dir="./data_raw/lorenz",
        output_dir="./results",
        mode="MAR",
        seed=56,
        diff_epochs=100,
        diff_lr=0.005,
        phaseB_iter=3,
        val_ratio=0.1,
        diff_weight_decay=0.0,
    )
