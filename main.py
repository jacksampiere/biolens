import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import constants
from constraints import proximity, monotonicity, reproducibility
from embed import (
    encoder,
    fit_preproc_map,
    apply_preproc_map,
    fit_projection,
    apply_projection,
)
from spline import select_knots, fit_spline, extract_transition_points


if __name__ == "__main__":
    np.random.seed(seed=42)

    # --------------------------------------------
    # SETUP
    # --------------------------------------------

    # Instantiate evaluation interval + thresholds
    interval = [5, 22]
    taus = [0.5, 0.5]
    rho = 0.6
    alpha = 0.9
    thresholds = {"rho": rho, "taus": taus, "alpha": alpha}

    # Spoof 100 single-channel EEG recordings
    X = (
        np.random.random(size=(constants.N_RECORDINGS, constants.N_TIME_STEPS))
        * constants.UV_SCALING_FACTOR
    )
    # Spoof ordered, discrete labels in [1,30]
    outcomes = np.random.randint(low=1, high=31, size=constants.N_RECORDINGS)

    # --------------------------------------------
    # FIT SPLINE
    # --------------------------------------------

    # Generate embeddings
    embeddings = encoder(X=X, k=constants.EMBEDDING_DIM)
    # Train-test split so we can set aside a holdout set for the fit precondition
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        embeddings, outcomes, test_size=0.2
    )
    # Fit and apply preprocessing map
    preprocessing_map = fit_preproc_map(emb=X_train)
    X_train_pp = apply_preproc_map(preproc_map=preprocessing_map, emb=X_train)
    # Fit and apply projection
    projection = fit_projection(emb=X_train_pp, m=constants.PROJECTION_OUTPUT_DIM)
    X_train_projected = apply_projection(projection=projection, emb=X_train_pp)
    # Reframe x and y for clarity w.r.t paper
    outcomes_train = y_train
    embeddings_train = X_train_projected
    # Sort data for spline fitting
    train_order = np.argsort(outcomes_train)
    outcomes_train_sorted = outcomes_train[train_order]
    embeddings_train_sorted = embeddings_train[train_order]
    # Fit BioLENS g(y)
    knots = select_knots(
        x=outcomes_train_sorted,
        n_interior_knots=constants.N_INTERIOR_KNOTS,
        d=constants.SPLINE_DEGREE,
    )
    g_y = fit_spline(
        outcomes=outcomes_train_sorted,
        embeddings=embeddings_train_sorted,
        knots=knots,
        d=constants.SPLINE_DEGREE,
    )

    # --------------------------------------------
    # CONSTRAINTS
    # --------------------------------------------

    # Extract TIMEVIEW-like characteristics
    T_g = extract_transition_points(g_y)
    # Evaluate proximity + monotonicity + reproducibility for test interval
    prox = proximity(T=T_g, interval=interval, taus=taus)
    mono = monotonicity(
        g=g_y,
        interval=interval,
        rho=rho,
    )
    reproducible = reproducibility(
        embeddings_train=embeddings_train_sorted,
        outcomes_train=outcomes_train_sorted,
        interval=interval,
        thresholds=thresholds,
        n_iter=constants.N_ITER_REPRODUCIBILITY,
    )
    # Evaluate holdout MSE + compare to null
    X_holdout_pp = apply_preproc_map(preproc_map=preprocessing_map, emb=X_holdout)
    X_holdout_projected = apply_projection(projection=projection, emb=X_holdout_pp)
    outcomes_holdout = y_holdout
    embeddings_holdout = X_holdout_projected
    embeddings_holdout_pred = g_y(outcomes_holdout)
    mse = mean_squared_error(y_true=embeddings_holdout, y_pred=embeddings_holdout_pred)
    # Generate null distribution of MSE
    mse_nulls = []
    for _ in range(constants.N_ITER_REPRODUCIBILITY):
        # Shuffle training set
        outcomes_train_perm = np.random.permutation(outcomes_train)
        # Fit spline to shuffled data
        train_order_perm = np.argsort(outcomes_train_perm)
        outcomes_train_perm_sorted = outcomes_train_perm[train_order_perm]
        embeddings_train_perm_sorted = embeddings_train[
            train_order_perm
        ]  # sort in the order of the shuffled outcomes
        g_y_perm = fit_spline(
            outcomes=outcomes_train_perm_sorted,
            embeddings=embeddings_train_perm_sorted,
            knots=knots,
            d=constants.SPLINE_DEGREE,
        )
        # Calculate MSE on unshuffled heldout data
        embeddings_holdout_pred_perm = g_y_perm(outcomes_holdout)
        mse = mean_squared_error(
            y_true=embeddings_holdout, y_pred=embeddings_holdout_pred_perm
        )
        mse_nulls.append(mse)

    # One-sided MSE p-value (lower MSE is better)
    p = (1.0 + np.sum(np.array(mse_nulls) <= mse)) / (len(mse_nulls) + 1.0)
    fit_adequacy = p <= 0.05

    # Evaluation interval selected strategically to get base example to satisfy these
    print(f"Base example satisfies proximity: {prox}")
    print(f"Base example satisfies monotonicity: {mono}")
    # Don't expect these to pass per use of synthetic data
    print(f"Proximity and monotonicity both satisfy reproducibility: {reproducible}")
    print(f"Base example satisfies fit adequacy: {fit_adequacy} (p = {p:.4f})")
