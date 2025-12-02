# utils/diffusion_backbone.py
import torch
import torch.nn as nn

# ---------- 工具 ----------
def mask_from_nan(x: torch.Tensor) -> torch.Tensor:
    return (~torch.isnan(x)).float()


def feat_standardize(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6):
    """
    x: (T,F) or (B,T,F)
    m: same shape in {0,1}  —— 仅用于统计（m==1 的位置参与均值/方差）
    若某列无观测，回退 mu=0,std=1。
    """
    is_batched = (x.dim() == 3)
    if not is_batched:
        x = x.unsqueeze(0)
        m = m.unsqueeze(0)

    B, T, Fdim = x.shape
    obs = m.sum(dim=1).clamp_min(1.0)           # (B,F)
    mu = (x * m).sum(dim=1) / obs               # (B,F)
    var = ((x - mu.unsqueeze(1)) * m).pow(2).sum(dim=1) / obs
    std = var.clamp_min(eps).sqrt()             # (B,F)

    mu = torch.nan_to_num(mu, nan=0.0)
    std = torch.nan_to_num(std, nan=1.0)

    z = (x - mu.unsqueeze(1)) / std.unsqueeze(1)

    if not is_batched:
        z = z.squeeze(0); mu = mu.squeeze(0); std = std.squeeze(0)
    return z, mu, std


def observed_bounds(x: torch.Tensor, m: torch.Tensor,
                    p_lo: float = 0.01, p_hi: float = 0.99, pad_ratio: float = 0.05):
    """
    返回每列的上下限（基于 m==1 的统计）。对无观测列给默认范围。
    支持 (T,F) / (B,T,F)，输出 (F,) / (B,F)。
    """
    is_batched = (x.dim() == 3)
    if not is_batched:
        x = x.unsqueeze(0); m = m.unsqueeze(0)

    device = x.device
    B, T, Fdim = x.shape
    lo = torch.empty(B, Fdim, device=device)
    hi = torch.empty(B, Fdim, device=device)
    default_lo, default_hi = -10.0, 10.0

    for b in range(B):
        for d in range(Fdim):
            vals = x[b, :, d][m[b, :, d] > 0.5]
            if vals.numel() == 0:
                lo[b, d], hi[b, d] = default_lo, default_hi
                continue
            q = torch.quantile(vals, torch.tensor([p_lo, p_hi], device=device))
            l, h = q[0], q[1]
            span = (h - l).clamp_min(1e-3)
            lo[b, d] = l - pad_ratio * span
            hi[b, d] = h + pad_ratio * span

    if not is_batched:
        lo = lo.squeeze(0); hi = hi.squeeze(0)
    return lo, hi


# ---------- 可学习去噪器（与 F 无关，按特征展开为批次处理） ----------
class SmallTemporalDenoiser(nn.Module):
    """
    对每个特征，输入三通道：(z_i, neigh_i, z_i - neigh_i)，
    用 1D Conv 在时间维上建模，通道数=3，与特征维 F 无关。
    """
    def __init__(self, hidden: int = 64, k: int = 3):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=k, padding=pad)
        )

    def forward(self, Z: torch.Tensor, neigh: torch.Tensor) -> torch.Tensor:
        # Z, neigh: (B,T,F)
        B, T, F = Z.shape
        x = torch.stack([Z, neigh, Z - neigh], dim=3)         # (B,T, F, 3)
        x = x.permute(0, 2, 3, 1).contiguous()                # (B, F, 3, T)
        x = x.view(B * F, 3, T)                               # (B*F, 3, T)
        out = self.net(x)                                     # (B*F, 1, T)
        out = out.view(B, F, 1, T).permute(0, 3, 1, 2).squeeze(-1)  # (B,T,F)
        return out


# ---------- PhaseB：可微的邻域引导 + 可学习残差 ----------
class PhaseB(nn.Module):
    def __init__(self, D: int, T: int = 50, device: str = "cuda",
                 width: int = 256, T_refine: int = 12, trust_radius: float = 0.25,
                 range_pad: float = 0.05, step_size: float = 0.3,
                 hidden: int = 64, kernel: int = 3):
        super().__init__()
        self.device = torch.device(device)
        self.T_refine = max(1, int(T_refine))
        self.r = float(trust_radius)
        self.range_pad = range_pad
        self.step_size = step_size

        # 学习型去噪头
        self.denoiser = SmallTemporalDenoiser(hidden=hidden, k=kernel)

        # 可训练仿射用于最后微调
        self._w = nn.Parameter(torch.tensor(1.0))
        self._b = nn.Parameter(torch.tensor(0.0))

        # 运行时开关
        self.clip_each_step = False
        self.clip_final = True

    @staticmethod
    def _row_norm_no_diag(S: torch.Tensor) -> torch.Tensor:
        # print("shape of S in _row_norm_no_diag:", S.shape)
        S = S.float().clone()
        F = S.size(-1)
        I = torch.eye(F, device=S.device)
        # print("shape of I in _row_norm_no_diag:", I.shape)
        S = S * (1.0 - I)  # 去掉自环
        row_sum = S.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return S / row_sum

    @staticmethod
    def _shift_time(x: torch.Tensor, lag: int) -> torch.Tensor:
        """
        x: (B,T,F)。对时间维做右移（因果滞后：使用过去信息预测现在）。
        lag=0 → 原样；lag>0 → 前 lag 步用 0 填充，后 T-lag 对齐原始前缀。
        """
        if lag <= 0:
            return x
        B, T, F = x.shape
        out = x.new_zeros(B, T, F)
        if lag < T:
            out[:, lag:, :] = x[:, :-lag, :]
        # lag >= T 时保持全 0
        return out

    @staticmethod
    def _lag_aware_aggregate(Z: torch.Tensor,
                             W: torch.Tensor,
                             L: torch.Tensor) -> torch.Tensor:
        """
        滞后感知邻域聚合：
        将源特征按各自的滞后量 L[j,i] 做时间 shift，再通过对应权重 W[j,i] 聚合到目标 i。
        计算方式：按滞后值分桶，把 Z 做多份 shift_l，然后逐桶矩阵乘 @ M_l（仅保留该滞后的边的权重）。
        形状：
          Z: (B,T,F),  W: (F,F)  （与原有 neigh = Z @ W 的朝向保持一致）
          L: (F,F)     （与 W 对齐：索引 [src, tgt] 或 [tgt, src] 与原实现一致）
        返回：neigh (B,T,F)
        """
        device = Z.device
        B, T, F = Z.shape
        neigh = torch.zeros_like(Z)

        # 若 L 全 0，等价于原始 @W
        if L is None or L.numel() == 0 or torch.all(L <= 0):
            return torch.matmul(Z, W)

        # 统一到整型
        L_int = L.to(dtype=torch.int64)
        max_lag = int(L_int.max().item())
        if max_lag <= 0:
            return torch.matmul(Z, W)

        # 按 lag 分桶构造权重矩阵 M_l：只保留 L==l 的边，其余为 0
        # 注意：不改变原有方向约定，保持与 neigh = Z @ W 的一致性
        for l in range(0, max_lag + 1):
            mask_l = (L_int == l)
            if not torch.any(mask_l):
                continue
            Ml = torch.where(mask_l, W, torch.zeros_like(W))
            Z_shift = PhaseB._shift_time(Z, l)   # (B,T,F)
            # (B,T,F) @ (F,F) -> (B,T,F)
            neigh = neigh + torch.matmul(Z_shift, Ml)

        return neigh

    def refine_with_matrices(self,
        X_init: torch.Tensor,            # (T,F) or (B,T,F) 初始完整矩阵（观测位=原值，缺失位=初始填补）
        X_nan: torch.Tensor,             # 同形；仅用于“统计掩码”（缺失为 NaN）
        causal_strength_matrix: torch.Tensor,  # (F,F)
        causal_lag_matrix: torch.Tensor,       # (F,F) 将被用于滞后感知聚合
        max_iter: int = None, tol: float = 1e-4,
        lock_observed: bool = True,      # True: 推理锁观测；False: 训练不锁定
        override_mask: torch.Tensor = None # 若提供，覆盖“锁定掩码”（1=视作观测并锁定）
    ) -> torch.Tensor:
        dev = self.device
        batched = (X_init.dim() == 3)
        if not batched:
            X_init = X_init.unsqueeze(0); X_nan = X_nan.unsqueeze(0)

        B, T, F = X_init.shape
        X_init = X_init.to(dev)
        X_nan  = X_nan.to(dev)

        # 两套掩码：一个做“统计”（控制标准化与边界），一个做“锁定”（控制是否覆盖观测）
        M_stats = mask_from_nan(X_nan)                       # 统计掩码（由真实观测决定）
        if override_mask is not None:
            M_lock = override_mask.to(dev).float()
        else:
            M_lock = M_stats.clone()

        if not lock_observed:
            # 训练时不锁观测位：允许在所有位置输出预测
            M_lock = torch.zeros_like(M_lock)

        miss = 1.0 - M_lock

        # 标准化与边界（使用统计掩码）
        Z0, mu, std = feat_standardize(X_init, M_stats)
        lo, hi = observed_bounds(X_init, M_stats, pad_ratio=self.range_pad)

        Z = torch.nan_to_num(Z0, 0.0)

        # 因果权重（行归一化、无自环）
        # print("shape of S in refine_with_matrices before _row_norm_no_diag:", causal_strength_matrix.shape)
        W = self._row_norm_no_diag(causal_strength_matrix.to(dev))
        L = causal_lag_matrix.to(dev).float() if causal_lag_matrix is not None else None

        # 迭代步数：若未显式传入，则用构造时的 T_refine
        true_max_iter = int(max(1, self.T_refine if (max_iter is None) else max_iter))

        for it in range(true_max_iter):
            prev = Z

            # 滞后感知的邻域聚合（核心更新）
            neigh = self._lag_aware_aggregate(Z, W, L)      # (B,T,F)

            # 规则化步（图平滑）
            eps = Z - neigh
            z_rule = Z - self.step_size * eps

            # 学习残差（时间域去噪）
            delta = self.denoiser(Z, neigh)                 # (B,T,F)

            # 融合更新
            z_new = z_rule + self.r * torch.tanh(delta)

            # （可选）每步裁剪到观测分布范围
            if self.clip_each_step:
                X_step = z_new * std.unsqueeze(1) + mu.unsqueeze(1)
                X_step = miss * torch.clamp(X_step, lo.unsqueeze(1), hi.unsqueeze(1)) + M_lock * X_init
                z_new  = (X_step - mu.unsqueeze(1)) / std.unsqueeze(1)

            # 应用更新；锁定观测位仅在最终合成时使用，这里不强行拷贝
            Z = z_new

            if it > 0 and (Z - prev).abs().mean().item() < tol:
                break

        # 反标准化 + 最终裁剪
        X_hat = Z * std.unsqueeze(1) + mu.unsqueeze(1)
        if self.clip_final:
            X_hat = torch.clamp(X_hat, lo.unsqueeze(1), hi.unsqueeze(1))

        # 最终输出：推理锁定观测位
        out = M_lock * X_init + miss * (X_hat * self._w + self._b)

        if not batched:
            out = out.squeeze(0)
        return out


class DiffusionBackbone(nn.Module):
    """
    训练（lock_observed=False）：不锁定观测位，输出全位置预测；
    推理（lock_observed=True）：锁定观测位，仅在缺失位更新。
    """
    def __init__(self, D: int, T: int = 50, device: str = "cuda", width: int = 256,
                 T_refine: int = 12, trust_radius: float = 0.25, range_padB: float = 0.05,
                 step_size: float = 0.3, hidden: int = 64, kernel: int = 3):
        super().__init__()
        self.B = PhaseB(D, T, device, width, T_refine, trust_radius, range_padB,
                        step_size=step_size, hidden=hidden, kernel=kernel)
        self.device = torch.device(device)

    def forward(self,
        X_init_filled: torch.Tensor,        # (B,T,F) or (T,F)
        X_with_nans_for_mask: torch.Tensor, # same shape
        causal_strength_matrix: torch.Tensor,
        causal_lag_matrix: torch.Tensor,
        lock_observed: bool = True,
        override_mask: torch.Tensor = None,
        # 兼容：允许外部可选传入 max_iter；若不传则用构造时的 T_refine
        max_iter: int = None
    ) -> torch.Tensor:
        return self.B.refine_with_matrices(
            X_init=X_init_filled,
            X_nan=X_with_nans_for_mask,
            causal_strength_matrix=causal_strength_matrix,
            causal_lag_matrix=causal_lag_matrix,
            lock_observed=lock_observed,
            override_mask=override_mask,
            max_iter=max_iter
        )
