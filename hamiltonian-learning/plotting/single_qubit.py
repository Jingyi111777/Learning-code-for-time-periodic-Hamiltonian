"""Figure 2, using the archived single-qubit estimates."""
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


def coefficient_data(result):
    model = result["model"]
    truth = np.array([model[key] for key in ("A", "B", "C")])
    coarse = np.array([result["X"]["A_hat"], result["BC"]["B_hat"], result["BC"]["C_hat"]])
    final = np.array([coarse[0], result["refinement"]["B"]["estimate"],
                      result["refinement"]["C"]["estimate"]])
    return truth, coarse, final


def plot_single_qubit(result, output, font):
    truth, coarse, final = coefficient_data(result)
    coarse_error = 100 * (coarse - truth) / truth
    final_error = 100 * (final - truth) / truth

    plt.rcParams.update({
        "font.family": font,
        "font.size": 10,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.7,
        "axes.unicode_minus": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "savefig.facecolor": "white",
    })
    fig, (coefficients, errors) = plt.subplots(
        2, 1, sharex=True, figsize=(4.75, 4.45), gridspec_kw={"height_ratios": [1.05, 1]}
    )
    fig.subplots_adjust(left=0.185, right=0.966, top=0.867, bottom=0.176, hspace=0.18)

    blue, blue_dark = "#2587a7", "#19647e"
    gold, gold_dark = "#efb34c", "#98641c"
    ink, zero = "#25282a", "#657078"
    x = np.arange(3)
    width, offset = 0.19, 0.14
    for ax, left, right in ((coefficients, coarse, final), (errors, coarse_error, final_error)):
        ax.bar(x - offset, left, width, color=blue, edgecolor=blue_dark, linewidth=0.55)
        ax.bar(x + offset, right, width, color=gold, edgecolor=gold_dark, linewidth=0.55)
        ax.set_xlim(-0.54, 2.54)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="y", length=3.2, pad=3.5)
        ax.tick_params(axis="x", length=0, pad=5)

    for i, value in enumerate(truth):
        coefficients.hlines(value, i - 0.295, i + 0.295, color=ink,
                            linestyles=(0, (3.5, 2.3)), linewidth=1.05, zorder=4)
        coefficients.text(i, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=9.3)
    coefficients.set_ylim(0, 0.95)
    coefficients.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    coefficients.set_ylabel("Coefficient", labelpad=7)

    errors.axhline(0, color=zero, lw=0.7, zorder=0)
    errors.set_ylim(-0.59, 0.245)
    errors.set_yticks([-0.50, -0.25, 0, 0.25], labels=["−0.50", "−0.25", "0", "0.25"])
    errors.set_xticks(x, [
        "$A$\nStatic $X$", "$B$\nStatic $Z$", "$C$\nModulated $Z$",
    ])
    errors.set_ylabel("Relative deviation (%)", labelpad=7)
    errors.text(0, 0.017, "0", ha="center", va="bottom", fontsize=9.3, color=ink)
    for i in (1, 2):
        for sign, value, color in ((-1, coarse_error[i], blue_dark), (1, final_error[i], gold_dark)):
            shift = 0.017 if value >= 0 else -0.017
            label = f"{value:+.3g}".replace("-", "−")
            errors.text(i + sign * offset, value + shift, label, ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=9.15, color=color)
    for ax, label in ((coefficients, "(a)"), (errors, "(b)")):
        ax.text(-0.005, 1.035, label, transform=ax.transAxes,
                fontsize=10, weight="bold", ha="left", va="bottom")

    fig.legend(
        handles=[
            Patch(facecolor=blue, edgecolor=blue_dark, label="Initial estimate"),
            Patch(facecolor=gold, edgecolor=gold_dark, label="Final estimate"),
            Line2D([0], [0], color=ink, linestyle=(0, (3.5, 2.3)), label="True value"),
        ],
        loc="upper center", bbox_to_anchor=(0.565, 0.975), frameon=False,
        ncol=3, handlelength=1.7, columnspacing=1.4, fontsize=9.6,
    )

    # Match the article's raster crop of the original 600-dpi figure.
    with BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=600)
        buffer.seek(0)
        with Image.open(buffer) as canvas:
            canvas.crop((0, 0, canvas.width, canvas.height - 192)).save(output, dpi=(600, 600))
    plt.close(fig)
