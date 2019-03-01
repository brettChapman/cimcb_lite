import numpy as np


def knnimpute(X, k, verbose=False, print_interval=100):
    """kNN missing value imputation."""
    
    missing_mask = np.isnan(X)
    n_rows, n_cols = X.shape
        
    X_row_major = X.copy("C")
    if missing_mask.sum() != np.isnan(X_row_major).sum():
        X_row_major[missing_mask] = np.nan
    
    n_rows, n_cols = X.shape

    # matrix of mean squared difference between between samples
    D = np.ones((n_rows, n_rows), dtype="float32", order="C") * np.inf

    # we can cheaply determine the number of columns that two rows share
    # by taking the dot product between their finite masks
    observed_elements = np.isfinite(X_row_major).astype(int)
    n_shared_features_for_pairs_of_rows = np.dot(
        observed_elements,
        observed_elements.T)
    no_overlapping_features_rows = n_shared_features_for_pairs_of_rows == 0
    number_incomparable_rows = no_overlapping_features_rows.sum(axis=1)
    row_overlaps_every_other_row = (number_incomparable_rows == 0)
    row_overlaps_no_other_rows = number_incomparable_rows == n_rows
    valid_rows_mask = ~row_overlaps_no_other_rows
    valid_row_indices = np.where(valid_rows_mask)[0]

    # preallocate all the arrays that we would otherwise create in the
    # following loop and pass them as "out" parameters to NumPy ufuncs
    diffs = np.zeros_like(X_row_major)
    missing_differences = np.zeros_like(diffs, dtype=bool)
    valid_rows = np.zeros(n_rows, dtype=bool)
    ssd = np.zeros(n_rows, dtype=X_row_major.dtype)

    for i in valid_row_indices:
        x = X_row_major[i, :]
        np.subtract(X_row_major, x.reshape((1, n_cols)), out=diffs)
        np.isnan(diffs, out=missing_differences)

        # zero out all NaN's
        diffs[missing_differences] = 0

        # square each difference
        diffs **= 2

        observed_counts_per_row = n_shared_features_for_pairs_of_rows[i]

        if row_overlaps_every_other_row[i]:
            # add up all the non-missing squared differences
            diffs.sum(axis=1, out=D[i, :])
            D[i, :] /= observed_counts_per_row
        else:
            np.logical_not(no_overlapping_features_rows[i], out=valid_rows)

            # add up all the non-missing squared differences
            diffs.sum(axis=1, out=ssd)
            ssd[valid_rows] /= observed_counts_per_row[valid_rows]
            D[i, valid_rows] = ssd[valid_rows]
    
    min_dist=1e-6
    max_dist_multiplier=1e6
    
    D_finite_flat = D[np.isfinite(D)]
    if len(D_finite_flat) > 0:
        max_dist = max_dist_multiplier * max(1, D_finite_flat.max())
    else:
        max_dist = max_dist_multiplier
    # set diagonal of distance matrix to a large value since we don't want
    # points considering themselves as neighbors
    np.fill_diagonal(D, max_dist)
    D[D < min_dist] = min_dist  # prevents 0s
    D[D > max_dist] = max_dist  # prevents infinities
    
    X_result = X_row_major
    D = D
    effective_infinity = max_dist
    
    for i in range(n_rows):
        for j in np.where(missing_mask[i, :])[0]:
            distances = D[i, :].copy()

            # any rows that don't have the value we're currently trying
            # to impute are set to infinite distances
            distances[missing_mask[:, j]] = effective_infinity
            neighbor_indices = np.argsort(distances)
            neighbor_distances = distances[neighbor_indices]

            # get rid of any infinite distance neighbors in the top k
            valid_distances = neighbor_distances < effective_infinity
            neighbor_distances = neighbor_distances[valid_distances][:k]
            neighbor_indices = neighbor_indices[valid_distances][:k]

            weights = 1.0 / neighbor_distances
            weight_sum = weights.sum()

            if weight_sum > 0:
                column = X[:, j]
                values = column[neighbor_indices]
                X_result[i, j] = np.dot(values, weights) / weight_sum
    return X_result
