"""Figures 3 and 4, using the archived three-qubit estimates."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
import numpy as np


def frequency_data(result):
    terms = result["model"]["terms"]
    records = [result["frequency"][term["label"]] for term in terms]
    true_frequency = np.array([term["omega"] for term in terms])
    estimate = np.array([record["omega_hat"] for record in records])
    absolute_error = np.abs(estimate - true_frequency)
    error_bound = np.array([record["frequency_error_bound"] for record in records])
    return true_frequency, estimate, absolute_error, error_bound


def relative_errors(coefficient, result, phase):
    values = []
    for term in result["model"]["terms"]:
        label = term["label"]
        truth = term[coefficient]
        if phase == "before":
            estimate = result["coefficients"][label][f"{coefficient}_hat"]
        else:
            estimate = result["refinement"][label][coefficient]["estimate"]
        values.append(100 * abs(estimate - truth) / abs(truth))
    return np.array(values)


def plot_three_qubit(baseline, finite_9, finite_10, output, font):
    terms = baseline["model"]["terms"]
    names = [term["label"] for term in terms]
    runs = (baseline, finite_9, finite_10)
    if any(run["model"]["terms"] != terms for run in runs):
        raise ValueError("The three-qubit runs must use the same Hamiltonian.")
    shots = tuple(run["model"]["ramsey_shots_per_quadrature"] for run in runs)
    if shots != (None, 10**9, 10**10):
        raise ValueError("Expected exact, 10^9-shot and 10^10-shot three-qubit runs.")

    plt.rcParams.update({
        "font.family": font,
        "font.size": 10.3,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.72,
        "axes.unicode_minus": True,
        "xtick.major.width": 0.72,
        "ytick.major.width": 0.72,
        "savefig.facecolor": "white",
    })
    blue = ("#2587a7", "#19647e")
    gold = ("#efb34c", "#98641c")
    purple = ("#9882b7", "#655080")
    green = ("#7ca891", "#406653")
    ink = "#25282a"
    dash = (0, (3.5, 2.3))


    def bars(ax, x, y, width, style, **kwargs):
        face, edge = style
        return ax.bar(x, y, width, color=face, edgecolor=edge, linewidth=0.55, **kwargs)


    def key(style, label):
        face, edge = style
        return Patch(facecolor=face, edgecolor=edge, linewidth=0.55, label=label)


    def style_panel(ax, panel):
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="y", length=3, pad=3.5)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.text(0, 1.055, f"({panel})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10.7, weight="bold")


    def truth_lines(ax, x, values, half_width):
        for center, value in zip(x, values):
            ax.hlines(value, center-half_width, center+half_width,
                      color=ink, linestyles=dash, linewidth=1.0, zorder=5)


    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=600)
        plt.close(fig)


    x = np.arange(len(names))
    true_frequency, est_frequency, freq_abs, freq_bound = frequency_data(baseline)
    freq_abs *= 1e3
    freq_bound *= 1e3
    truth_key = Line2D([], [], color=ink, linestyle=dash, lw=1.0, label="True value")

    fig, (freq_ax, err_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(4.95, 4.0), gridspec_kw={"height_ratios": [1, 1]}
    )
    fig.subplots_adjust(left=0.174, right=0.969, bottom=0.112, top=0.88, hspace=0.45)
    style_panel(freq_ax, "a")
    style_panel(err_ax, "b")
    bars(freq_ax, x, est_frequency, 0.18, blue)
    truth_lines(freq_ax, x, true_frequency, 0.18)
    for center, estimated in zip(x, est_frequency):
        freq_ax.text(center, estimated+0.06, f"{estimated:#.4g}",
                     ha="center", va="bottom", fontsize=9.4)
    freq_ax.set_ylabel("Frequency", labelpad=8)
    freq_ax.set_yticks([0, 0.5, 1, 1.5])
    freq_ax.set_ylim(0, 1.71)
    freq_ax.legend(handles=[key(blue, "Estimate"), truth_key], loc="lower center",
                   bbox_to_anchor=(0.56, 1.09), ncol=2, frameon=False, fontsize=9.7,
                   handlelength=1.55, columnspacing=1.35, borderaxespad=0)
    for offset, vals, color in ((-0.11, freq_abs, blue), (0.11, freq_bound, gold)):
        bars(err_ax, x+offset, vals, 0.15, color)
        for center, v in zip(x+offset, vals):
            err_ax.text(center, v+0.86, f"{v:#.4g}",
                        ha="center", va="bottom", color=color[1], fontsize=9.3)
    err_ax.set_ylabel(r"Frequency error ($\times 10^{-3}$)", labelpad=7)
    err_ax.set_yticks([0, 10, 20, 30])
    err_ax.set_ylim(0, 30.6)
    err_ax.set_xlim(-0.54, len(x)-0.46)
    err_ax.set_xticks(x, names)
    err_ax.set_xlabel("Pauli term", labelpad=3)
    err_ax.legend(handles=[key(blue, "Absolute error"), key(gold, "Error bound")],
                  loc="lower center", bbox_to_anchor=(0.56, 1.09), ncol=2,
                  frameon=False, fontsize=9.7, handlelength=1.55,
                  columnspacing=1.35, borderaxespad=0)
    save(fig, "three_qubit_frequency_bars")

    fig = plt.figure(figsize=(5.6, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.10], wspace=0.28, hspace=0.73)
    ax_top = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)
    fig.subplots_adjust(left=0.132, right=0.979, top=0.87, bottom=0.101)
    style_panel(ax_top, "a")
    style_panel(ax_b, "b")
    style_panel(ax_c, "c")

    coefficients = [(label, coef) for label in names for coef in ("B", "C")]
    truth = np.array([baseline["refinement"][label]["diagnostics"][f"{coef}_true"]
                      for label, coef in coefficients])
    initial = np.array([baseline["coefficients"][label][f"{coef}_hat"]
                        for label, coef in coefficients])
    refined = np.array([baseline["refinement"][label][coef]["estimate"]
                        for label, coef in coefficients])
    xx = np.arange(len(coefficients), dtype=float)
    xx += np.repeat([0.0, 0.35, 0.70], 2)
    bars(ax_top, xx-0.17, initial, 0.29, blue)
    bars(ax_top, xx+0.17, refined, 0.29, gold)
    truth_lines(ax_top, xx, truth, 0.38)
    for center, v in zip(xx, truth):
        ax_top.text(center, v+0.035, f"{v:#.4g}", ha="center", va="bottom", fontsize=9.15)
    ax_top.set_ylabel("Coefficient", labelpad=8)
    ax_top.set_ylim(0, 0.84)
    ax_top.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax_top.set_xlim(-0.53, xx[-1]+0.53)
    ax_top.set_xticks(xx, ["$" + rf"{coef}_{{\mathrm{{{label}}}}}$" for label, coef in coefficients])
    ax_top.legend(handles=[key(blue, "Initial"), key(gold, "Refined"), truth_key],
                  loc="lower center", bbox_to_anchor=(0.53, 1.08), frameon=False,
                  fontsize=9.8, ncol=3, columnspacing=1.5, borderaxespad=0)


    styles = (blue, gold, purple, green)
    floor = 0.006
    for ax, coef in ((ax_b, "B"), (ax_c, "C")):
        values = [relative_errors(coef, baseline, "before"),
                  relative_errors(coef, baseline, "after"),
                  relative_errors(coef, finite_9, "after"),
                  relative_errors(coef, finite_10, "after")]
        assert np.all(np.array(values) > floor)
        for offset, values_for_mode, color in zip([-0.288, -0.096, 0.096, 0.288], values, styles):
            bars(ax, x+offset, values_for_mode-floor, 0.165, color, bottom=floor)
        ax.set_yscale("log")
        ax.set_ylim(floor, 1.05)
        ax.set_yticks([0.01, 0.1, 1], labels=["0.01", "0.1", "1"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", which="minor", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", which="major", color="#d7dce0", linewidth=0.55)
        ax.set_xticks(x, names)
        ax.set_xlim(-0.51, len(x)-0.49)
        ax.text(0.5, 1.045, "$" + rf"{coef}_s$", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11.7)
    ax_b.set_ylabel("Relative error (%)", labelpad=7)
    ax_c.tick_params(axis="y", labelleft=False)
    fig.legend(handles=[key(blue, "Initial"), key(gold, "Refined"),
                        key(purple, r"Refined, $10^9$ shots"),
                        key(green, r"Refined, $10^{10}$ shots")],
               loc="center", bbox_to_anchor=(0.556, 0.497), ncol=4,
               frameon=False, fontsize=9.2, handlelength=1.33,
               handletextpad=0.42, columnspacing=0.85)
    save(fig, "three_qubit_coefficient_bars")
