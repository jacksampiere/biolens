import numpy as np
from scipy.interpolate import make_lsq_spline, PPoly


def select_knots(x, n_interior_knots, d):
    """Choose interior knots for spline g(y) via quantiles (arbitrary selection method for MVP).

    Args:
        x: Outcome values.
        n_interior_knots: Number of interior knots.
        d: Spline degree.
    Returns:
        Full knot vector with boundary repeats.
    """
    xu = np.unique(x)
    interior_quantiles = np.linspace(start=0.1, stop=0.9, num=n_interior_knots)
    knots_internal = np.quantile(a=np.unique(ar=x), q=interior_quantiles)
    knots_internal = np.unique(
        knots_internal[(knots_internal > xu.min()) & (knots_internal < xu.max())]
    )
    # Add the boundary knots
    lower_outer_knots = np.repeat(xu.min(), repeats=d + 1)
    upper_outer_knots = np.repeat(xu.max(), repeats=d + 1)
    knots = np.concatenate(
        [lower_outer_knots, knots_internal, upper_outer_knots], axis=0
    )
    return knots


def fit_spline(outcomes, embeddings, knots, d):
    """Fit g(y) as a cubic B-spline.

    Note: UnivariateSpline (~smoothing) cannot handle the duplicate y-values in this
    dataset. We fit with make_lsq_spline for demonstrative purposes, though it is not
    a true smoothing spline. Note that make_lsq_spline returns a BSpline object.

    Args:
        outcomes: Ordered outcome values.
        embeddings: Projected embeddings.
        knots: Knot vector for spline fit.
        d: Spline degree.
    Returns:
        Fitted BSpline object for g(y).
    """
    spl = make_lsq_spline(x=outcomes, y=embeddings, t=knots, k=d)
    return spl


def extract_transition_points(spl):
    """Find transition points T(g) (roots of first/second derivatives) for motif boundaries.

    Args:
        spl: Fitted BSpline g(y).
    Returns:
        Sorted array of transition points.
    """
    g = PPoly.from_spline(spl)
    gp1 = g.derivative(1)
    gp2 = g.derivative(2)
    T_g = np.sort(
        np.concatenate(
            [
                gp1.roots(extrapolate=False),
                gp2.roots(extrapolate=False),
            ],
            axis=0,
        )
    )
    return T_g
