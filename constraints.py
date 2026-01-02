import numpy as np

from constants import N_INTERIOR_KNOTS, SPLINE_DEGREE
from spline import select_knots, fit_spline, extract_transition_points, extract_motifs


def monotonicity(motifs_first_order, midpoints, interval, rho):
    """Monotonicity constraint: majority slope sign within interval exceeds rho.

    Args:
        motifs_first_order: Sign of first derivative per motif.
        midpoints: Motif midpoints.
        interval: Outcome interval to test.
        rho: Threshold for majority sign.
    Returns:
        Boolean indicating monotonicity satisfaction.
    """
    start, end = interval[0], interval[1]
    # Create mask + extract signs
    mask = (midpoints > start) & (
        midpoints < end
    )  # incorporates motifs with >=50% overlap
    signs = motifs_first_order[mask]
    # Calculate score + determine satisfaction
    K = len(signs)
    Np = (signs > 0).sum()
    Nn = (signs < 0).sum()
    score = max(Np, Nn) / K
    return score >= rho


def proximity(T, interval, taus):
    """Proximity constraint: transition points lie within tau of interval endpoints.

    Args:
        T: Transition points.
        interval: Outcome interval [y1, y2].
        taus: Distance tolerances (tau1, tau2).
    Returns:
        Boolean indicating proximity satisfaction.
    """
    y1, y2 = interval[0], interval[1]
    t1, t2 = taus[0], taus[1]
    # Distance from interval start to closest transition point
    dist_y1 = np.min(np.abs(T - y1))
    dist_y1_idx = np.argmin(np.abs(T - y1))
    # Distance from interval end to closest transition point
    dist_y2 = np.min(np.abs(T - y2))
    dist_y2_idx = np.argmin(np.abs(T - y2))
    # Ensure uniqueness
    if dist_y1_idx == dist_y2_idx:
        raise Exception("Identical start and end candidates for proximity evaluation")
    return (dist_y1 <= t1) and (dist_y2 <= t2)


def reproducibility(embeddings_train, outcomes_train, interval, thresholds, n_iter):
    """Bootstrap proximity/monotonicity to test reproducibility.

    Note: MVP starts from projected embeddings for simplicity; full
    version would resample end-to-end and refit scaler/projection.

    Args:
        embeddings_train: Projected embeddings for training.
        outcomes_train: Corresponding ordered outcomes.
        interval: Outcome interval to test.
        thresholds: Dict containing taus, rho, alpha.
        n_iter: Number of bootstrap iterations.
    Returns:
        Boolean indicating whether proximity and monotonicity meet reproducibility threshold.
    """
    rng = np.random.default_rng()
    n_samples_per_iter = embeddings_train.shape[0]
    mono, prox = [], []
    for i in range(n_iter):
        sample_indices = rng.choice(
            a=n_samples_per_iter, size=n_samples_per_iter, replace=True
        )
        embeddings_train_iter = embeddings_train[sample_indices]
        outcomes_train_iter = outcomes_train[sample_indices]
        prox_iter, mono_iter = pipeline_forward(
            embeddings=embeddings_train_iter,
            outcomes=outcomes_train_iter,
            interval=interval,
            taus=thresholds["taus"],
            rho=thresholds["rho"],
        )
        prox.append(prox_iter)
        mono.append(mono_iter)
        if (i + 1) % 10 == 0:
            print(f"Reproducibility bootstrapping: completed {i+1} iterations")

    pct_pass_prox = np.array(prox).sum() / n_iter
    pct_pass_mono = np.array(mono).sum() / n_iter
    prox_reproducible = pct_pass_prox >= thresholds["alpha"]
    mono_reproducible = pct_pass_mono >= thresholds["alpha"]

    return prox_reproducible and mono_reproducible


def pipeline_forward(embeddings, outcomes, interval, taus, rho):
    """Fit g(y) and run proximity + monotonicity checks, starting from projected embeddings.

    Args:
        embeddings: Projected embeddings.
        outcomes: Ordered outcomes.
        interval: Outcome interval to test.
        taus: Distance tolerances for proximity.
        rho: Threshold for monotonicity.
    Returns:
        Tuple (proximity_bool, monotonicity_bool).
    """
    order = np.argsort(outcomes)
    embeddings = embeddings[order]
    outcomes = outcomes[order]
    knots = select_knots(x=outcomes, n_interior_knots=N_INTERIOR_KNOTS, d=SPLINE_DEGREE)
    g_y = fit_spline(
        outcomes=outcomes, embeddings=embeddings, knots=knots, d=SPLINE_DEGREE
    )
    T_g = extract_transition_points(g_y)
    motifs, midpoints = extract_motifs(
        g=g_y, T=T_g, ymin=outcomes.min(), ymax=outcomes.max()
    )
    motifs_first_order = motifs[0, :]
    prox = proximity(T=T_g, interval=interval, taus=taus)
    mono = monotonicity(
        motifs_first_order=motifs_first_order,
        midpoints=midpoints,
        interval=interval,
        rho=rho,
    )
    return prox, mono
