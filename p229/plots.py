"""Grafiken für das Gesamt-Notebook. Jede Funktion gibt eine matplotlib-Figure zurück (Aufruf: plt.show())."""
import numpy as np
import matplotlib.pyplot as plt

from .config import SERIEN_NAMEN
from .stats import wil

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
DUNKEL, HELL = "#256abf", "#9ec5f4"


def _stil():
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID})


def _kopf(fig, titel, sub=None):
    fig.text(0.04, 0.96, titel, fontsize=15, fontweight="bold", ha="left", va="top")
    if sub:
        fig.text(0.04, 0.905, sub, fontsize=9.5, color=INK2, ha="left", va="top")


def _kommas(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def _name(s):
    return SERIEN_NAMEN.get(s, s)


def _achsen(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def konfusion_einzeln(ax, tp, fp, fn, tn, titel=None, gross=True):
    """Eine Konfusionsmatrix: Zeilen = Modell Kaufen/Warten, Spalten = Preis stieg / stieg nicht."""
    zellen = [[tp, fp], [fn, tn]]
    labels = [["richtig gekauft", "zu früh gekauft"], ["zu lange gewartet", "richtig gewartet"]]
    for i in range(2):
        for j in range(2):
            dunkel = i == j
            ax.add_patch(plt.Rectangle((j + 0.03, 1 - i + 0.03), 0.94, 0.94, color=DUNKEL if dunkel else HELL, zorder=1))
            ax.text(j + 0.5, 1 - i + (0.6 if gross else 0.55), str(int(zellen[i][j])), ha="center", va="center",
                    fontsize=22 if gross else 13, fontweight="bold", color="#ffffff" if dunkel else INK, zorder=2)
            ax.text(j + 0.5, 1 - i + (0.28 if gross else 0.22), labels[i][j], ha="center", va="center",
                    fontsize=9.5 if gross else 6.5, color="#ffffff" if dunkel else INK, zorder=2)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["Preis stieg", "stieg nicht"], fontsize=9 if gross else 7)
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Warten", "Kaufen"], fontsize=9 if gross else 7)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if titel:
        ax.set_title(titel, fontsize=10 if gross else 8.5, color=INK, pad=14 if gross else 12)


def konfusion_raster(df, titel="Fehlerübersicht aller Serien (Konfusionsmatrizen)", sub=None):
    """df: Spalten Serie, TP, FP, FN, TN, DA. Zeigt alle Serien als Raster."""
    _stil()
    n = len(df)
    spalten = 4
    zeilen = int(np.ceil(n / spalten))
    fig, axs = plt.subplots(zeilen, spalten, figsize=(12, 3.1 * zeilen + 1.0), dpi=110)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78 if zeilen == 2 else 0.7, bottom=0.04, hspace=0.55, wspace=0.35)
    axs = np.atleast_1d(axs).ravel()
    for ax, (_, r) in zip(axs, df.iterrows()):
        konfusion_einzeln(ax, r["TP"], r["FP"], r["FN"], r["TN"], f"{_name(r['Serie'])}: {_kommas(r['DA']*100)} % richtig", gross=False)
    for ax in axs[n:]:
        ax.axis("off")
    _kopf(fig, titel, sub or "Dunkelblau = richtig, hellblau = Fehler. Zeilen: was das Modell sagt. Spalten: was der Preis wirklich tat.")
    return fig


def trefferquote_mit_baselines(df, titel="Trefferquote je Serie", sub=None):
    """df: Serie, DA, Wilson_lo, Wilson_hi, Mehrheit, DA_NYMEX_Regel."""
    _stil()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=110)
    fig.subplots_adjust(left=0.2, right=0.96, top=0.76, bottom=0.26)
    _achsen(ax)
    y = np.arange(len(df))[::-1]
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.hlines(y, df["Wilson_lo"], df["Wilson_hi"], color=BLUE, lw=3, zorder=2, capstyle="round")
    ax.scatter(df["DA"], y, s=70, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5, label="Modell (XGBoost)")
    ax.scatter(df["DA_NYMEX_Regel"], y, s=42, marker="D", color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1, label="NYMEX-Regel (einfache Regel)")
    ax.scatter(df["Mehrheit"], y, s=42, marker="s", color=GREY, zorder=3, edgecolor=SURF, linewidth=1, label="Immer häufigere Richtung tippen")
    for yi, v in zip(y, df["DA"]):
        ax.text(v, yi + 0.33, f"{_kommas(v*100)} %", ha="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([_name(s) for s in df["Serie"]], fontsize=9.5)
    ax.set_ylim(-0.7, y.max() + 0.6)
    ax.text(0.502, -0.55, "50 % = Münzwurf", fontsize=8, color=INK2)
    lo = min(0.45, float(df["Wilson_lo"].min()) - 0.02)
    ax.set_xlim(lo, 0.75)
    ticks = [t for t in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75) if t >= lo]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{int(round(t*100))} %" for t in ticks])
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.24), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2)
    _kopf(fig, titel, sub or "Balken = 95-%-Konfidenzintervall (Wilson). Bei überlappenden 3-Tage-Labels nur Orientierung.")
    return fig


def precision_recall(df, titel="Precision, Recall und Basisrate", sub=None):
    """df: Serie, Precision, Recall, Basisrate."""
    _stil()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=110)
    fig.subplots_adjust(left=0.2, right=0.96, top=0.76, bottom=0.26)
    _achsen(ax)
    y = np.arange(len(df))[::-1]
    ax.scatter(df["Basisrate"], y, s=60, color=GREY, zorder=3, edgecolor=SURF, linewidth=1.5, label="Basisrate (beliebiger Tag)")
    ax.scatter(df["Precision"], y, s=70, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5, label="Precision (Kaufsignal stimmt)")
    ax.scatter(df["Recall"], y, s=60, marker="D", color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1, label="Recall (Anstieg erkannt)")
    ax.set_yticks(y); ax.set_yticklabels([_name(s) for s in df["Serie"]], fontsize=9.5)
    lo = min(0.4, float(df[["Precision", "Recall", "Basisrate"]].min().min()) - 0.02)
    hi = max(0.75, float(df[["Precision", "Recall", "Basisrate"]].max().max()) + 0.02)
    ax.set_xlim(lo, hi)
    ticks = [t for t in np.arange(0.4, 0.81, 0.05) if lo <= t <= hi]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{int(round(t*100))} %" for t in ticks])
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.24), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2)
    _kopf(fig, titel, sub or "Precision = von allen Kaufsignalen stieg der Preis wirklich. Recall = von allen Anstiegen erkannt.")
    return fig


def kalibrierung(p, y, titel="Wie verlässlich ist die Wahrscheinlichkeit des Modells?", sub=None):
    """Tatsächlicher Anteil steigender Preise je Bereich der vorhergesagten Wahrscheinlichkeit (alle Serien gepoolt)."""
    _stil()
    p, y = np.asarray(p, float), np.asarray(y, float)
    kanten = np.array([0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0])
    mitten, raten, los, his, ns = [], [], [], [], []
    for a, b in zip(kanten[:-1], kanten[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 20:
            k = int(y[m].sum()); lo, hi = wil(k, int(m.sum()))
            mitten.append(p[m].mean()); raten.append(y[m].mean()); los.append(lo); his.append(hi); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=110)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.78, bottom=0.14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.plot([0.25, 0.75], [0.25, 0.75], color=MUTED, lw=1, ls=(0, (3, 3)), label="perfekt verlässlich")
    ax.errorbar(mitten, raten, yerr=[np.array(raten) - np.array(los), np.array(his) - np.array(raten)], fmt="o", color=BLUE,
                ms=8, capsize=3, lw=2, label="Modell (Punkt = Bereich, Balken = 95-%-Intervall)")
    for x, n in zip(mitten, ns):
        ax.text(x, 0.205, f"n={n}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0.25, 0.75); ax.set_ylim(0.18, 0.78)
    ax.set_xlabel("Wahrscheinlichkeit laut Modell, dass der Preis in 3 Tagen höher ist")
    ax.set_ylabel("Tatsächlicher Anteil steigender Preise")
    ax.set_xticks(np.arange(0.3, 0.76, 0.1)); ax.set_xticklabels([f"{int(round(t*100))} %" for t in np.arange(0.3, 0.76, 0.1)])
    ax.set_yticks(np.arange(0.3, 0.76, 0.1)); ax.set_yticklabels([f"{int(round(t*100))} %" for t in np.arange(0.3, 0.76, 0.1)])
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    _kopf(fig, titel, sub or "Je höher die Wahrscheinlichkeit, desto öfter sollte der Preis wirklich steigen (Punkte nahe der gestrichelten Linie).")
    return fig


def abdeckung_kurve(df, titel="Weniger Signale, dafür zuverlässiger", sub=None):
    """df: Fenster, Abdeckung, DA_entschieden (eine Zeile je Anteil und Zeitraum)."""
    _stil()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=110)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.76, bottom=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    farben = {"Entwicklung": BLUE, "Test": ORANGE}
    for f, g in df.groupby("Fenster"):
        g = g.sort_values("Abdeckung")
        ax.plot(g["Abdeckung"] * 100, g["DA_entschieden"] * 100, "-o", color=farben.get(f, GREY), lw=2, ms=6, label=f)
    ax.axhline(50, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.annotate("50 % = Münzwurf", xy=(1, 50), xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points",
                ha="right", fontsize=8, color=INK2)
    ax.set_ylim(48, max(66, float(df["DA_entschieden"].max() * 100) + 2))
    ax.set_xlabel("Anteil der Tage mit Signal Kaufen oder Warten (Rest = Beobachten)")
    ax.set_ylabel("Trefferquote auf diesen Tagen")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f} %"); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f} %")
    ax.invert_xaxis()
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
    _kopf(fig, titel, sub or "Nach links: mehr Tage werden zu „Beobachten“. Mittel über alle Serien.")
    return fig


def ersparnis(df, titel="Ersparnis gegenüber „immer kaufen“", sub=None):
    """df: Serie, Ersparnis_vs_kaufen, Ersp_lo, Ersp_hi (Euro pro 100 l und Tag)."""
    _stil()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=110)
    fig.subplots_adjust(left=0.2, right=0.96, top=0.76, bottom=0.14)
    _achsen(ax)
    y = np.arange(len(df))[::-1]
    ax.axvline(0, color=INK, lw=1.2, zorder=1)
    ax.hlines(y, df["Ersp_lo"], df["Ersp_hi"], color=BLUE, lw=3, zorder=2, capstyle="round")
    ax.scatter(df["Ersparnis_vs_kaufen"], y, s=70, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5)
    for yi, v, lo in zip(y, df["Ersparnis_vs_kaufen"], df["Ersp_lo"]):
        ax.text(v, yi + 0.33, f"{_kommas(v, 2)} €", ha="center", fontsize=8, color=INK)
        if lo < 0:
            ax.text(df["Ersp_hi"].max() + 0.02, yi, "Intervall reicht unter 0", fontsize=7.5, color=INK2, va="center")
    ax.set_yticks(y); ax.set_yticklabels([_name(s) for s in df["Serie"]], fontsize=9.5)
    ax.set_xlim(min(-0.1, float(df["Ersp_lo"].min()) - 0.02), float(df["Ersp_hi"].max()) + 0.22)
    ticks = [t for t in np.arange(-0.2, 1.0, 0.1) if ax.get_xlim()[0] <= t <= df["Ersp_hi"].max() + 0.05]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{_kommas(t)} €" for t in ticks])
    _kopf(fig, titel, sub or "Euro pro 100 Liter und Tag. Rechts von der Null = Modell ist günstiger. Balken = 95-%-Bootstrap-Intervall.")
    return fig
