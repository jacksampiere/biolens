import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import seaborn as sns

sns.set_theme(style="ticks", context="paper")

rng = np.random.default_rng(7)

# Spoofed parameters
y_min, y_max = 1.0, 30.0
y1, y2 = 12.0, 17.0
n_points = 90
k = 3
y_grid = np.linspace(y_min, y_max, 900)


def smoothstep(t):
    """C^1 smooth step: 0->1 with zero slope at both ends; inflection near t=0.5."""
    t = np.clip(t, 0.0, 1.0)
    return 3 * t**2 - 2 * t**3


def fit_smoothing_spline(y_data, g_obs, k=3, s=None):
    return UnivariateSpline(y_data, g_obs, k=k, s=s)


# ----------------------------
# LEFT PANEL: clear monotone transition within [y1, y2]
# This represents a viable candidate biomarker that exhibits the desired properties
# Steps:
#   - Start from a messy spline fit
#   - Enforce monotone increasing on [y1,y2] by replacing that segment with a
#     smoothstep transition (monotone + concave-up then concave-down + flat-ish endpoints).
# ----------------------------
noise_sigma_L = 0.55
smoothing_L = 0.50 * n_points * noise_sigma_L**2

y_data_L = np.sort(rng.uniform(y_min, y_max, size=n_points))
t_L = (y_data_L - y1) / (y2 - y1)

baseline_L = -1.1
amplitude_L = 2.2
g_true_L = baseline_L + amplitude_L * smoothstep(t_L)

g_obs_L = (
    g_true_L
    + 0.20 * np.sin(0.9 * y_data_L + 0.4)
    + 0.10 * np.sin(2.3 * y_data_L - 0.8)
    + rng.normal(0.0, noise_sigma_L, size=n_points)
)

spline_L = fit_smoothing_spline(y_data_L, g_obs_L, k=k, s=smoothing_L)
g_grid_L = spline_L(y_grid)

# Replace the transition segment with a guaranteed-monotone S shape (smoothstep)
mask = (y_grid >= y1) & (y_grid <= y2)
g_start = float(spline_L(y1))
g_end_raw = float(spline_L(y2))

# Ensure the transition is increasing
min_rise = 0.7
g_end = max(g_end_raw, g_start + min_rise)

t_seg = (y_grid[mask] - y1) / (y2 - y1)
g_seg = g_start + (g_end - g_start) * smoothstep(t_seg)

# Shift the right side so the curve stays continuous at y2
delta = g_end - g_end_raw
g_grid_L[y_grid > y2] = g_grid_L[y_grid > y2] + delta
g_grid_L[mask] = g_seg
g_y1_L = float(g_start)
g_y2_L = float(g_end)

# ----------------------------
# RIGHT PANEL: same highlighted interval, but no clear trend within it
# This is meant to represent a candidate biomarker that does not exhibit the desired properties
# ----------------------------
noise_sigma_R = 0.60
smoothing_R = 0.55 * n_points * noise_sigma_R**2

y_data_R = np.sort(rng.uniform(y_min, y_max, size=n_points))

# Oscillatory/no-trend structure (especially within [y1,y2])
g_true_R = (
    0.4 * np.sin(1.3 * y_data_R)
    + 0.35 * np.sin(2.8 * y_data_R + 0.8)
    + 0.15 * np.sin(5.0 * y_data_R - 0.5)
)

g_obs_R = g_true_R + rng.normal(0.0, noise_sigma_R, size=n_points)
spline_R = fit_smoothing_spline(y_data_R, g_obs_R, k=k, s=smoothing_R)
g_grid_R = spline_R(y_grid)

# ----------------------------
# Plot (1x2)
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=160, sharey=True)

# Styling helpers
arrow_kw = dict(arrowstyle="->", linewidth=1.4, color="black", shrinkA=0, shrinkB=6)

for ax in axes:
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(-3.2, 3.2)
    ax.axvspan(y1, y2, alpha=0.12)
    ax.axvline(y1, linestyle=":", linewidth=1)
    ax.axvline(y2, linestyle=":", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)
    ax.set_xlabel(r"$y$")

# Left: monotone transition + dots + annotations
(lineL,) = axes[0].plot(y_grid, g_grid_L, linewidth=2)
cL = lineL.get_color()

axes[0].plot(
    [y1, y2],
    [g_y1_L, g_y2_L],
    linestyle="none",
    marker="o",
    markersize=5,
    markerfacecolor=cL,
    markeredgecolor="black",
    markeredgewidth=0.6,
    zorder=6,
)

axes[0].annotate(
    r"transition point, ($y_1$, $g(y_1))$",
    xy=(y1, g_y1_L),
    xytext=(25, -30),
    textcoords="offset points",
    ha="right",
    va="bottom",
    arrowprops=arrow_kw,
    annotation_clip=False,
)
axes[0].set_title("Candidate 1", pad=6)

axes[0].annotate(
    r"transition point, ($y_2$, $g(y_2))$",
    xy=(y2, g_y2_L),
    xytext=(125, 30),
    textcoords="offset points",
    ha="right",
    va="bottom",
    arrowprops=arrow_kw,
    annotation_clip=False,
)
axes[1].set_title("Candidate 2", pad=6)


axes[0].set_ylabel(r"$g(y)$")

# Right: no trend + no text
axes[1].plot(y_grid, g_grid_R, linewidth=2)

plt.tight_layout()
plt.savefig("figure_1.png", dpi=150, bbox_inches="tight")
