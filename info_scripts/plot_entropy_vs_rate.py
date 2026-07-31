#!/usr/bin/env python3
"""
Phase-3 Defense Figure: Entropy vs. Adaptive Reduction Rate
============================================================
Generates a publication-quality scatter plot overlaying the theoretical curve
    r_target = r_max * (1 - H^alpha)
with the 19 empirically observed (H, r_target) points from ESAE-IS.

Key visual message: same formula, wildly different outcomes depending on
each dataset's intrinsic micro-cluster balance — the system is truly adaptive,
not a fixed-rate baseline in disguise.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# DATA — extracted from the ESAE-IS logs (r_max=0.50, alpha=15)
# ══════════════════════════════════════════════════════════════════════════════
# (dataset, H, r_target_pct, domain_category)
DATA = [
    ("20ng",          0.9596, 23.05, "Newsgroups"),
    ("acm",           0.9697, 18.48, "Academic"),
    ("books",         0.9727, 16.97, "Reviews"),
    ("dblp",          0.9714, 17.65, "Academic"),
    ("movie_review",  0.9439, 28.96, "Reviews"),
    ("mpqa",          0.9366, 31.29, "Opinion"),
    ("ohsumed",       0.9762, 15.18, "Medical"),
    ("pang_movie",    0.9455, 28.43, "Reviews"),
    ("reuters90",     0.9605, 22.69, "News"),
    ("sst1",          0.9310, 32.88, "Sentiment"),
    ("sst2",          0.9326, 32.44, "Sentiment"),
    ("subj",          0.9477, 27.65, "Subjectivity"),
    ("trec",          0.9607, 22.62, "QA"),
    ("twitter",       0.9441, 28.91, "Social"),
    ("vader_movie",   0.9471, 27.88, "Reviews"),
    ("webkb",         0.9417, 29.70, "Web"),
    ("wos11967",      0.9721, 17.28, "Science"),
    ("wos5736",       0.9567, 24.26, "Science"),
    ("yelp_reviews",  0.9295, 33.30, "Reviews"),
]

# ══════════════════════════════════════════════════════════════════════════════
# FORMULA PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
R_MAX = 0.50
ALPHA = 15.0

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN → MARKER / COLOR MAPPING
# ══════════════════════════════════════════════════════════════════════════════
DOMAIN_STYLE = {
    "Reviews":      {"color": "#E63946", "marker": "o"},   # red
    "Sentiment":    {"color": "#F4A261", "marker": "s"},   # amber
    "Opinion":      {"color": "#E9C46A", "marker": "D"},   # gold
    "Subjectivity": {"color": "#2A9D8F", "marker": "^"},   # teal
    "News":         {"color": "#264653", "marker": "v"},   # dark teal
    "Newsgroups":   {"color": "#457B9D", "marker": "<"},   # blue-grey
    "Academic":     {"color": "#1D3557", "marker": ">"},   # navy
    "Medical":      {"color": "#6A0572", "marker": "P"},   # purple
    "Science":      {"color": "#A855F7", "marker": "X"},   # violet
    "QA":           {"color": "#16A34A", "marker": "h"},   # green
    "Social":       {"color": "#0EA5E9", "marker": "p"},   # sky blue
    "Web":          {"color": "#78716C", "marker": "H"},   # stone
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,        # set True if LaTeX is available
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

fig, ax = plt.subplots(figsize=(7.0, 4.5))

# --- Theoretical curve ---
H_curve = np.linspace(0.0, 1.0, 500)
r_curve = R_MAX * (1.0 - H_curve ** ALPHA)
r_curve_pct = r_curve * 100.0
# Clamp as code does: max(5%, min(r, r_max))
r_curve_pct = np.clip(r_curve_pct, 5.0, R_MAX * 100.0)

ax.plot(H_curve, r_curve_pct, color="#94A3B8", linewidth=2.0, linestyle="-",
        zorder=1, label=r"$r = 0.50\,(1 - H^{15})$")

# Light fill under curve
ax.fill_between(H_curve, 0, r_curve_pct, color="#94A3B8", alpha=0.07, zorder=0)

# --- Scatter: empirical points ---
for name, H, r_pct, domain in DATA:
    style = DOMAIN_STYLE[domain]
    ax.scatter(H, r_pct, color=style["color"], marker=style["marker"],
               s=70, edgecolors="white", linewidths=0.6, zorder=3)

# --- Labels (avoid overlaps with manual nudges) ---
NUDGE = {
    # (dx, dy) offsets — manually tuned to minimise label overlap
    # Crowded high-rate region (H ≈ 0.93–0.95)
    "yelp_reviews":  (-0.008, +1.3),
    "sst1":          (+0.003, +1.2),
    "sst2":          (+0.003, -2.2),
    "mpqa":          (+0.003, +1.2),
    "movie_review":  (+0.003, -2.0),
    "twitter":       (+0.003, +1.3),
    "pang_movie":    (+0.003, +1.3),
    "vader_movie":   (-0.008, -2.0),
    "webkb":         (-0.008, +1.3),
    "subj":          (+0.003, -2.0),
    # Mid region
    "wos5736":       (+0.003, +1.3),
    "reuters90":     (+0.003, -2.0),
    "20ng":          (+0.003, +1.2),
    "trec":          (+0.003, -2.0),
    # Crowded low-rate region (H ≈ 0.97)
    "acm":           (-0.008, -2.0),
    "books":         (+0.003, -2.0),
    "dblp":          (+0.003, +1.2),
    "ohsumed":       (+0.003, +1.2),
    "wos11967":      (-0.008, +1.2),
    "wos5736":       (+0.003, -2.0),
}

for name, H, r_pct, domain in DATA:
    dx, dy = NUDGE.get(name, (0.003, 1.0))
    style = DOMAIN_STYLE[domain]
    # Use abbreviated names for space
    label = name.replace("_review", "").replace("_reviews", "").replace("_movie", "")
    ax.annotate(
        label, (H, r_pct),
        xytext=(H + dx, r_pct + dy),
        fontsize=6.5, color=style["color"], fontweight="bold",
        ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top",
    )

# --- Axes ---
ax.set_xlabel(r"Normalized Shannon Entropy  $H$  of Micro-Cluster Occupancy")
ax.set_ylabel(r"Adaptive Reduction Rate  $r_{\mathrm{target}}$  (%)")

# Zoom into the region where points actually live
ax.set_xlim(0.92, 0.985)
ax.set_ylim(10, 38)

# Grid
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)

# --- Annotation: formula box ---
formula_text = (
    r"$r_{\mathrm{target}} = r_{\max}\,(1 - H^{\alpha})$"
    "\n"
    r"$r_{\max}=0.50$,  $\alpha=15$"
)
ax.text(
    0.97, 0.96, formula_text,
    transform=ax.transAxes, fontsize=9,
    verticalalignment="top", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#CBD5E1", alpha=0.9),
)

# --- Annotation: spread range ---
rates = [r for _, _, r, _ in DATA]
r_min, r_max_val = min(rates), max(rates)
ax.annotate(
    "",
    xy=(0.925, r_min), xytext=(0.925, r_max_val),
    arrowprops=dict(arrowstyle="<->", color="#64748B", lw=1.2),
)
ax.text(
    0.9225, (r_min + r_max_val) / 2,
    f"Δ = {r_max_val - r_min:.1f} pp",
    fontsize=7, color="#64748B", ha="center", va="center",
    rotation=90,
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
              edgecolor="none", alpha=0.85),
)

# --- Legend by domain ---
legend_handles = []
for domain, style in DOMAIN_STYLE.items():
    # Only include domains present in data
    if any(d == domain for _, _, _, d in DATA):
        legend_handles.append(
            Line2D([0], [0], marker=style["marker"], color="w",
                   markerfacecolor=style["color"], markeredgecolor="white",
                   markeredgewidth=0.5, markersize=7, label=domain)
        )

# Add curve to legend
legend_handles.insert(0,
    Line2D([0], [0], color="#94A3B8", linewidth=1.5,
           linestyle="-", label=r"Theoretical $r(H)$")
)

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,
    framealpha=0.9,
    edgecolor="#CBD5E1",
    handletextpad=0.3,
    columnspacing=0.8,
)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_pdf = "/data/bernardolemos/info_scripts/fig_entropy_vs_rate.pdf"
out_png = "/data/bernardolemos/info_scripts/fig_entropy_vs_rate.png"

fig.savefig(out_pdf)
fig.savefig(out_png)
plt.close(fig)

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# Print summary statistics for the caption
print(f"\n--- Summary for caption ---")
print(f"H range:      [{min(h for _,h,_,_ in DATA):.4f}, {max(h for _,h,_,_ in DATA):.4f}]")
print(f"Rate range:   [{r_min:.2f}%, {r_max_val:.2f}%]")
print(f"Rate spread:  {r_max_val - r_min:.1f} percentage points")
print(f"Mean rate:    {np.mean(rates):.2f}%")
print(f"Std rate:     {np.std(rates):.2f}%")
