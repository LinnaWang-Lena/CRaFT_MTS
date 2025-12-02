# miracle_backbone.py
import gc
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


__all__ = ["MIRACLE"]


class MIRACLE(nn.Module):
    """
    A lightweight, self-contained imputer for 2D multivariate time series
    with missing values. Designed to be *parallel-friendly*:
      - No distributed logic is implemented here.
      - Exposes a `sync_pred_fn` callback you can provide from an external
        multi-GPU runner to aggregate predictions across ranks (e.g., all-reduce mean).

    Input/Output
    ------------
    - fit(X_miss: np.ndarray) -> np.ndarray
        X_miss: shape (seq_len, num_inputs), with NaNs marking missing entries.
        returns: imputed matrix of the same shape (no NaNs).

    Parallel Hooks (to be used by your 3rd file)
    -------------------------------------------
    - sync_pred_fn: Optional[Callable[[np.ndarray], np.ndarray]]
        Called at the end of each step on the *local* prediction (float32 numpy array
        of shape (seq_len, original_features)). Should return the aggregated prediction
        (same shape) if you want cross-device averaging, or just return the input for
        single-device/no-sync usage.

    Notes
    -----
    - The model predicts only the original feature dimensions.
    - Missing indicators are concatenated to inputs but not predicted.
    - The moving average window is kept per-process; you can replace it via
      sync_pred_fn to a global synchronized prediction if desired.
    """

    def __init__(
        self,
        num_inputs: int,
        missing_list: Sequence[int],
        n_hidden: int = 32,
        lr: float = 8e-3,
        max_steps: int = 50,
        window: int = 5,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        # ---- seeding (local) ----
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ---- device ----
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # ---- shapes & bookkeeping ----
        self.original_features: int = int(num_inputs)
        self.missing_list: Tuple[int, ...] = tuple(int(i) for i in missing_list)
        self.missing_features: int = len(self.missing_list)

        # input = original features + missing indicators
        self.total_input_dim: int = self.original_features + self.missing_features

        self.max_steps: int = int(max_steps)
        self.window: int = int(window)

        # ---- networks (one head per original feature) ----
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.total_input_dim, n_hidden),
                    nn.ELU(),
                    nn.Linear(n_hidden, n_hidden),
                    nn.ELU(),
                    nn.Linear(n_hidden, 1),
                )
                for _ in range(self.original_features)
            ]
        )

        self.to(self.device)
        self.opt = optim.Adam(self.parameters(), lr=lr)

    # ----------------------------
    # Internal utilities
    # ----------------------------
    @staticmethod
    def _nan_to_col_mean(X: np.ndarray) -> np.ndarray:
        """
        Fill NaNs column-wise with column means; if an entire column is NaN,
        fill with 0.0 as a last resort.
        """
        X = X.astype(np.float32, copy=True)
        col_mean = np.nanmean(X, axis=0)  # may be nan for all-NaN columns
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                val = 0.0 if np.isnan(col_mean[j]) else float(col_mean[j])
                X[mask, j] = val
        return X

    @staticmethod
    def _build_mask(X: np.ndarray) -> np.ndarray:
        """Observed mask: 1.0 for observed, 0.0 for missing."""
        return (~np.isnan(X)).astype(np.float32)

    def _build_indicators(self, X_mask: np.ndarray) -> np.ndarray:
        """
        Build missing-indicator columns *only* for columns specified by `missing_list`.
        Each indicator is 1.0 where X_miss was NaN for that column, else 0.0.
        """
        T, D = X_mask.shape
        if self.missing_features == 0:
            return np.zeros((T, 0), dtype=np.float32)
        out = np.zeros((T, self.missing_features), dtype=np.float32)
        for i, col_idx in enumerate(self.missing_list):
            if 0 <= col_idx < D:
                out[:, i] = 1.0 - X_mask[:, col_idx]
        return out

    def _concat_inputs(self, X: np.ndarray, indicators: np.ndarray) -> np.ndarray:
        X_all = np.concatenate([X, indicators], axis=1).astype(np.float32, copy=False)
        assert (
            X_all.shape[1] == self.total_input_dim
        ), f"Post-concat dim ({X_all.shape[1]}) != expected ({self.total_input_dim})"
        return X_all

    def forward(self, X_all: torch.Tensor) -> torch.Tensor:
        """
        X_all: (seq_len, total_input_dim)
        returns: (seq_len, original_features)
        """
        outputs = []
        for net in self.networks:
            y = net(X_all).squeeze(-1)  # (seq_len,)
            outputs.append(y)
        return torch.stack(outputs, dim=1)  # (seq_len, original_features)

    # ----------------------------
    # Public API
    # ----------------------------
    @torch.no_grad()
    def predict_numpy(self, X_all: np.ndarray) -> np.ndarray:
        """
        Convenience: run a forward pass on numpy input.
        """
        self.eval()
        xt = torch.tensor(X_all, dtype=torch.float32, device=self.device, non_blocking=True)
        pred = self(xt).detach().cpu().numpy().astype(np.float32, copy=False)
        return pred

    def fit(
        self,
        X_miss: np.ndarray,
        *,
        sync_pred_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Train in-place and return an imputed copy of X_miss (as numpy float32).

        Parameters
        ----------
        X_miss : np.ndarray, shape (seq_len, original_features)
            NaNs mark missing values.
        sync_pred_fn : Optional[Callable[[np.ndarray], np.ndarray]]
            External callback used to synchronize/aggregate predictions across devices.
            Signature: local_pred_np -> aggregated_pred_np. Must return the same shape.
            If None, no cross-device aggregation is performed here.
        verbose : bool
            If True, prints simple loss logs.

        Returns
        -------
        X_filled : np.ndarray, shape (seq_len, original_features)
        """
        assert X_miss.ndim == 2, f"X_miss must be 2D, got {X_miss.ndim}D"
        T, D = X_miss.shape
        assert (
            D == self.original_features
        ), f"Input features ({D}) != model original_features ({self.original_features})"

        # masks & naive fill
        X_mask = self._build_mask(X_miss)  # (T, D)
        X = self._nan_to_col_mean(X_miss)  # (T, D)
        indicators = self._build_indicators(X_mask)  # (T, |missing_list|)
        X_all = self._concat_inputs(X, indicators)  # (T, D + |missing_list|)

        # Training loop
        avg_preds = []  # moving window of local/aggregated predictions
        ones_ind = np.ones_like(indicators, dtype=np.float32)
        X_mask_all = np.concatenate([X_mask, ones_ind], axis=1).astype(np.float32, copy=False)

        for step in range(self.max_steps):
            # --- refine inputs from moving average ---
            if step > 0 and len(avg_preds) > 0:
                X_pred = np.mean(np.stack(avg_preds, axis=0), axis=0)  # (T, D)
                # update only missing positions
                X = X * X_mask + X_pred * (1.0 - X_mask)
                X_all = self._concat_inputs(X, indicators)

            # --- train step ---
            self.train()
            xt = torch.tensor(X_all, dtype=torch.float32, device=self.device, non_blocking=True)
            mt = torch.tensor(X_mask_all, dtype=torch.float32, device=self.device, non_blocking=True)

            pred_orig = self(xt)  # (T, D)
            # concatenate indicators (constant) to match xt for masked loss
            if indicators.shape[1] > 0:
                ind_t = torch.tensor(indicators, dtype=torch.float32, device=self.device, non_blocking=True)
                pred_all = torch.cat([pred_orig, ind_t], dim=1)  # (T, D + |missing_list|)
            else:
                pred_all = pred_orig

            loss = ((pred_all - xt) ** 2 * mt).mean()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            # --- make a prediction for refinement (and optional cross-device sync) ---
            with torch.no_grad():
                self.eval()
                local_pred = pred_orig.detach().cpu().numpy().astype(np.float32, copy=False)  # (T, D)

                if sync_pred_fn is not None:
                    # Your 3rd file can implement torch.distributed all_reduce here
                    # and return the aggregated numpy array (same shape).
                    refined_pred = sync_pred_fn(local_pred)
                    assert isinstance(refined_pred, np.ndarray) and refined_pred.shape == local_pred.shape
                    used_pred = refined_pred.astype(np.float32, copy=False)
                else:
                    used_pred = local_pred

                avg_preds.append(used_pred)
                if len(avg_preds) > self.window:
                    avg_preds.pop(0)

            if verbose and (step % max(1, self.max_steps // 10) == 0 or step == self.max_steps - 1):
                print(f"[MIRACLE] step={step:03d} loss={float(loss):.6f}")

            # memory hygiene for long sequences
            if step % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # final pass
        final_pred = np.mean(np.stack(avg_preds, axis=0), axis=0) if len(avg_preds) else self.predict_numpy(X_all)
        X_filled = (X * X_mask + final_pred * (1.0 - X_mask)).astype(np.float32, copy=False)
        return X_filled