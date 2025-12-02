import numpy as np
import pandas as pd
def FirstProcess(matrix, threshold=0.8):
    matrix = np.array(matrix, dtype=np.float32)

    # Stage 1: handle empty columns and high-duplication columns
    for col_idx in range(matrix.shape[1]):
        col_data = matrix[:, col_idx]

        if np.isnan(col_data).all():
            matrix[:, col_idx] = -1
            continue

        valid_mask = ~np.isnan(col_data)
        if not valid_mask.any():
            continue

        valid_data = col_data[valid_mask]
        unique_vals, counts = np.unique(valid_data, return_counts=True)
        max_count_idx = np.argmax(counts)
        mode_value = unique_vals[max_count_idx]
        mode_count = counts[max_count_idx]

        if mode_count >= threshold * len(valid_data):
            matrix[np.isnan(col_data), col_idx] = mode_value
    return matrix


def SecondProcess(matrix, perturbation_prob=0.3, perturbation_scale=0.3):
    for col_idx in range(matrix.shape[1]):
        col_data = matrix[:, col_idx]
        missing_mask = np.isnan(col_data)

        if not missing_mask.any():
            continue

        series = pd.Series(col_data)
        interpolated = series.interpolate(method="linear", limit_direction="both").values

        if np.isnan(interpolated).any():
            interpolated[np.isnan(interpolated)] = np.nanmean(col_data)

        # Add perturbation
        missing_indices = np.where(missing_mask)[0]
        if len(missing_indices) > 0 and perturbation_prob > 0:
            n_perturb = int(len(missing_indices) * perturbation_prob)
            if n_perturb > 0:
                perturb_indices = np.random.choice(missing_indices, n_perturb, replace=False)
                value_range = np.ptp(col_data[~missing_mask]) or 1.0
                perturbations = np.random.uniform(-1, 1, n_perturb) * perturbation_scale * value_range
                interpolated[perturb_indices] += perturbations

        matrix[:, col_idx] = interpolated

    return matrix.astype(np.float32)  #  Fix: move this outside the loop


def initial_process(matrix, threshold=0.8, perturbation_prob=0.1, perturbation_scale=0.1):
    matrix = FirstProcess(matrix, threshold)
    matrix = SecondProcess(matrix, perturbation_prob, perturbation_scale)
    return matrix