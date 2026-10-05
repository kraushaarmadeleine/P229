"""Erzeugt Präsentationsgrafiken (PNG, 16:9) aus den Ergebnissen von db_10 und db_11.

Die Zahlen sind aus den Ausgaben der Notebooks übernommen (db_11 Zelle 5 bis 7, db_10 Zelle 5).
Aufruf (Terminal im Ordner dieser Datei): python make_charts.py
Die PNG-Dateien landen im Unterordner "grafiken". Anderer Zielordner: python make_charts.py MEIN_ORDNER
Benötigt: pip install matplotlib numpy
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "grafiken"
os.makedirs(OUT, exist_ok=True)

# Farben (validiert, helle Fläche): Serie 1 blau, Serie 2 orange; Neutraltöne für Referenzen
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})

# ---- Daten: db_11 (XGBoost, S0, Test, n = 297 je Serie) ----
SER = ["Heizöl Süd", "Heizöl Südwest", "Heizöl Südost", "Heizöl Rhein-Main",
       "Diesel Süd", "Diesel Südwest", "Diesel Südost", "Diesel Rhein-Main"]
DA = [0.596, 0.606, 0.636, 0.596, 0.653, 0.606, 0.653, 0.633]
W_LO = [0.539, 0.549, 0.580, 0.539, 0.597, 0.549, 0.597, 0.577]
W_HI = [0.650, 0.660, 0.689, 0.650, 0.705, 0.660, 0.705, 0.686]
MEHRHEIT = [0.505, 0.542, 0.512, 0.525, 0.535, 0.549, 0.515, 0.542]
NYMEX = [0.582, 0.586, 0.593, 0.569, 0.606, 0.586, 0.599, 0.579]
PREC = [0.602, 0.569, 0.634, 0.562, 0.628, 0.556, 0.627, 0.586]
RECALL = [0.544, 0.574, 0.684, 0.674, 0.623, 0.634, 0.701, 0.676]
BASIS = [0.495, 0.458, 0.512, 0.475, 0.465, 0.451, 0.485, 0.458]
ERSP = [0.224, 0.246, 0.266, 0.242, 0.242, 0.202, 0.243, 0.221]
ERSP_LO = [0.058, 0.073, 0.099, 0.055, 0.030, -0.018, 0.008, -0.020]
ERSP_HI = [0.431, 0.451, 0.486, 0.480, 0.459, 0.426, 0.491, 0.478]
CM = np.array([[80, 53], [67, 97]])          # Heizöl Süd: Zeilen Modell Kaufen/Warten, Spalten Preis stieg / nicht (TP FP / FN TN)
# ---- Daten: db_10 (Mittel über 8 Serien) ----
MOD = ["XGBoost", "Logistische Regression", "Random Forest", "Logistisch klein\n(7 Merkmale)", "Mehrheitsklasse\n(Dummy)"]
DEV_DA = [0.607, 0.608, 0.595, 0.601, 0.477]
TEST_DA = [0.622, 0.604, 0.607, 0.586, 0.486]
RULE_DEV, RULE_TEST = 0.619, 0.588


def base(title, sub, w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.subplots_adjust(left=0.2, right=0.96, top=0.76, bottom=0.24)
    fig.text(0.04, 0.94, title, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.875, sub, fontsize=9.5, color=INK2, ha="left", va="top")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    return fig, ax


def note(fig, txt):
    fig.text(0.04, 0.02, txt, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)


# 1 Trefferquote je Serie
fig, ax = base("Trefferquote je Serie: 60–65 % im Test", "XGBoost, Szenario S0, Testzeitraum 2024-11-15 bis 2026-01-20, n = 297 je Serie.\nBalken = 95-%-Konfidenzintervall (Wilson)")
y = np.arange(len(SER))[::-1]
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
ax.hlines(y, W_LO, W_HI, color=BLUE, lw=3, zorder=2, capstyle="round")
ax.scatter(DA, y, s=70, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5, label="XGBoost")
ax.scatter(NYMEX, y, s=42, marker="D", color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1, label="NYMEX-Regel (einfache Regel)")
ax.scatter(MEHRHEIT, y, s=42, marker="s", color=GREY, zorder=3, edgecolor=SURF, linewidth=1, label="Immer häufigere Richtung tippen")
for yi, v in zip(y, DA):
    ax.text(v, yi + 0.33, f"{v*100:.1f} %".replace(".", ","), ha="center", fontsize=8, color=INK)
ax.set_yticks(y); ax.set_yticklabels(SER, fontsize=9.5)
ax.set_xlim(0.45, 0.72); ax.set_xticks([0.5, 0.55, 0.6, 0.65, 0.7]); ax.set_xticklabels([f"{int(t*100)} %" for t in [0.5, 0.55, 0.6, 0.65, 0.7]])
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_ylim(-0.7, y.max() + 0.6)
ax.text(0.502, -0.55, "50 % = Münzwurf", fontsize=8, color=INK2, ha="left")
ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.22), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2)
note(fig, "Quelle: db_11, Zelle 5. Bei überlappenden 3-Tage-Labels ist das Intervall nur Orientierung; Bootstrap-Untergrenze bei allen 8 Serien > 50 %.")
save(fig, "1_trefferquote_je_serie.png")

# 2 Precision vs Basisrate
fig, ax = base("Ein Kaufsignal trifft öfter als ein beliebiger Tag", "Precision = Anteil der „Kaufen“-Signale, nach denen der Preis wirklich stieg.\nBasisrate = Anteil aller Tage mit steigendem Preis")
ax.hlines(y, BASIS, PREC, color=GRID, lw=4, zorder=1)
ax.scatter(BASIS, y, s=70, color=GREY, zorder=3, edgecolor=SURF, linewidth=1.5, label="Basisrate (beliebiger Tag)")
ax.scatter(PREC, y, s=70, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5, label="Precision des Modells")
for yi, p, b in zip(y, PREC, BASIS):
    ax.text(p + 0.006, yi, f"{p*100:.0f} %  (+{(p-b)*100:.0f} Pkt.)", va="center", fontsize=8, color=INK)
ax.set_yticks(y); ax.set_yticklabels(SER, fontsize=9.5)
ax.set_xlim(0.42, 0.74); ax.set_xticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7]); ax.set_xticklabels([f"{int(round(t*100))} %" for t in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]])
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.23), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
note(fig, "Quelle: db_11, Zelle 6. Recall (erkannte Anstiege) liegt bei 54–70 %.")
save(fig, "2_precision_vs_basisrate.png")

# 3 Konfusionsmatrix Heizöl Süd
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
fig.subplots_adjust(left=0.27, right=0.9, top=0.72, bottom=0.14)
fig.text(0.04, 0.94, "Heizöl Süd: Wo das Modell richtig und falsch liegt", fontsize=15, fontweight="bold", ha="left", va="top")
fig.text(0.04, 0.875, "XGBoost, S0, Test, n = 297 Tage. Dunkelblau = richtig, hellblau = Fehler.\nTrefferquote 59,6 %, Precision 60 %, Recall 54 %", fontsize=9.5, color=INK2, ha="left", va="top")
shade = np.array([[3, 1], [0, 4]])
for i in range(2):
    for j in range(2):
        v = CM[i, j]
        col = SEQ[4] if (i == j) else SEQ[1]
        ax.add_patch(plt.Rectangle((j + 0.03, 1 - i + 0.03), 0.94, 0.94, color=col, zorder=1))
        ax.text(j + 0.5, 1 - i + 0.58, str(v), ha="center", va="center", fontsize=26, fontweight="bold", color=("#ffffff" if i == j else INK), zorder=2)
        lab = [["Richtig gekauft", "Zu früh gekauft"], ["Zu lange gewartet", "Richtig gewartet"]][i][j]
        ax.text(j + 0.5, 1 - i + 0.28, lab, ha="center", va="center", fontsize=10, color=("#ffffff" if i == j else INK), zorder=2)
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["Preis stieg", "Preis stieg nicht"], fontsize=10)
ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Modell: Warten", "Modell: Kaufen"], fontsize=10)
ax.xaxis.tick_top(); ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
note(fig, "Quelle: db_11, Zelle 6. Precision = 80 / (80 + 53), Recall = 80 / (80 + 67).")
save(fig, "3_konfusionsmatrix_heizoel_sued.png")

# 4 Ersparnis gegenüber immer kaufen
fig, ax = base("Ersparnis gegenüber „immer kaufen“", "Euro pro 100 Liter und Entscheidungstag. Rechts von der Null = Modell ist günstiger.\nBalken = 95-%-Bootstrap-Intervall")
ax.axvline(0, color=INK, lw=1.2, zorder=1)
ax.hlines(y, ERSP_LO, ERSP_HI, color=BLUE, lw=3, zorder=2, capstyle="round")
ax.scatter(ERSP, y, s=70, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5)
for yi, v, lo in zip(y, ERSP, ERSP_LO):
    ax.text(v, yi + 0.33, f"{v:.2f} €".replace(".", ","), ha="center", fontsize=8, color=INK)
    if lo < 0:
        ax.text(0.52, yi, "Intervall reicht unter 0", fontsize=7.5, color=INK2, va="center")
ax.set_yticks(y); ax.set_yticklabels(SER, fontsize=9.5)
ax.set_xlim(-0.1, 0.7); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_xticklabels([f"{t:.1f} €".replace(".", ",") for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]])
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
note(fig, "Quelle: db_11, Zelle 7. Bei 6 von 8 Serien liegt die untere Grenze über 0, bei Diesel Südwest und Rhein-Main nicht.")
save(fig, "4_ersparnis_vs_immer_kaufen.png")

# 5 Modellvergleich
fig, ax = base("Modellvergleich: kein klarer Sieger", "Mittlere Trefferquote über 8 Serien, S0.\nEntwicklung 2023-04 bis 2024-11, Test 2024-11 bis 2026-01", w=8, h=4.8)
fig.subplots_adjust(left=0.24, right=0.96, top=0.76, bottom=0.26)
y5 = np.arange(len(MOD))[::-1]
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
ax.hlines(y5, DEV_DA, TEST_DA, color=GRID, lw=4, zorder=1)
ax.scatter(DEV_DA, y5, s=70, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5, label="Entwicklungszeitraum")
ax.scatter(TEST_DA, y5, s=70, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1.5, label="Testzeitraum")
box = dict(boxstyle="square,pad=0.15", fc=SURF, ec="none")
for yi, a, b in zip(y5, DEV_DA, TEST_DA):
    for v, other in ((a, b), (b, a)):
        left = v < other or (v == other)
        ax.text(v + (-0.007 if left else 0.007), yi, f"{v*100:.1f} %".replace(".", ","), ha="right" if left else "left",
                va="center", fontsize=8, color=INK, bbox=box, zorder=5)
ax.set_yticks(y5); ax.set_yticklabels(MOD, fontsize=9.5)
ax.set_xlim(0.44, 0.67); ax.set_xticks([0.5, 0.55, 0.6, 0.65]); ax.set_xticklabels(["50 %", "55 %", "60 %", "65 %"])
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_ylim(-0.7, y5.max() + 0.6)
ax.text(0.502, -0.55, "50 % = Münzwurf", fontsize=8, color=INK2, ha="left")
ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.25), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
note(fig, f"Quelle: db_10, Zelle 5. NYMEX-Regel zum Vergleich: {str(round(RULE_DEV*100, 1)).replace('.', ',')} % (Entwicklung), {str(round(RULE_TEST*100, 1)).replace('.', ',')} % (Test). Unterschiede unter 2 Punkten sind Zufall.")
save(fig, "5_modellvergleich.png")
print("fertig:", sorted(os.listdir(OUT)))
