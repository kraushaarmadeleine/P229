# Databricks notebook source
# MAGIC %md
# MAGIC # db_12: Grafiken aus den echten Ergebnissen
# MAGIC
# MAGIC Liest die Ergebnisdateien aus dem Output-Ordner und zeichnet 5 Grafiken. Sie werden direkt angezeigt und als PNG in `output/grafiken` gespeichert:
# MAGIC 1. Trefferquote je Serie, 2. Precision gegen Basisrate, 3. Konfusionsmatrix (Heizöl Süd), 4. Ersparnis gegenüber „immer kaufen“, 5. Modellvergleich.
# MAGIC
# MAGIC **Voraussetzung:** `db11_ergebnisse.csv` (aus db_11, Szenario S0, Test) und für Grafik 5 `db10_ergebnisse.csv` (aus db_10) liegen im Output-Ordner. Es werden keine Zahlen von Hand eingetragen.
# MAGIC
# MAGIC Hinweis: Das ist nur Darstellung der bisherigen Ergebnisse (gleicher Zeitraum wie vorher), kein neuer Test.

# COMMAND ----------
# Zelle 1: Ergebnisse laden und vorbereiten.
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

BASE_PATH = "/Volumes/dev_workspace/p229/p229_files"
OUT = f"{BASE_PATH}/output"
GRAFIK = f"{OUT}/grafiken"
os.makedirs(GRAFIK, exist_ok=True)

NAMEN = {"heizoel_sued": "Heizöl Süd", "heizoel_suedwest": "Heizöl Südwest", "heizoel_suedost": "Heizöl Südost",
         "heizoel_rheinmain": "Heizöl Rhein-Main", "diesel_sued": "Diesel Süd", "diesel_suedwest": "Diesel Südwest",
         "diesel_suedost": "Diesel Südost", "diesel_rheinmain": "Diesel Rhein-Main"}

r11 = pd.read_csv(f"{OUT}/db11_ergebnisse.csv")
r11 = r11[(r11["Szenario"] == "S0") & (r11["Fenster"] == "Test")]
r11 = r11.set_index("Serie").loc[list(NAMEN)]              # feste Reihenfolge, Fehler wenn eine Serie fehlt
SER = [NAMEN[s] for s in r11.index]
DA, W_LO, W_HI = r11["DA"].tolist(), r11["Wilson_lo"].tolist(), r11["Wilson_hi"].tolist()
MEHRHEIT, NYMEX = r11["Mehrheit"].tolist(), r11["DA_NYMEX_Regel"].tolist()
PREC, RECALL, BASIS = r11["Precision"].tolist(), r11["Recall"].tolist(), r11["Basisrate"].tolist()
ERSP, ERSP_LO, ERSP_HI = r11["Ersparnis_vs_kaufen"].tolist(), r11["Ersp_lo"].tolist(), r11["Ersp_hi"].tolist()
z = r11.loc["heizoel_sued"]
CM = np.array([[z["TP"], z["FP"]], [z["FN"], z["TN"]]]).astype(int)   # Zeilen: Modell Kaufen/Warten, Spalten: Preis stieg / nicht
N = int(z["n"])
N_BOOT = int((r11["Boot_lo"] > 0.5).sum())
print("Geladen:", len(SER), "Serien, n =", N, "je Serie")

HAT10 = os.path.exists(f"{OUT}/db10_ergebnisse.csv")
if HAT10:
    r10 = pd.read_csv(f"{OUT}/db10_ergebnisse.csv")
    ordnung = ["XGBoost", "Logistische Regression", "Random Forest", "Logistisch klein (7 Merkmale)", "Mehrheitsklasse"]
    mm = r10.groupby(["Modell", "Fenster"])["DA"].mean().unstack().loc[ordnung]
    MOD = ["XGBoost", "Logistische Regression", "Random Forest", "Logistisch klein\n(7 Merkmale)", "Mehrheitsklasse\n(Dummy)"]
    DEV_DA, TEST_DA = mm["Entwicklung"].tolist(), mm["Test"].tolist()
    regel = r10[r10["Modell"] == "XGBoost"].groupby("Fenster")["DA_NYMEX_Regel"].mean()
    RULE_DEV, RULE_TEST = float(regel["Entwicklung"]), float(regel["Test"])
else:
    print("Hinweis: db10_ergebnisse.csv nicht gefunden, Grafik 5 wird übersprungen.")

# COMMAND ----------
# Zelle 2: Stil und Hilfsfunktionen.
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID})

def komma(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")

def base(title, sub, w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
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
    fig.savefig(f"{GRAFIK}/{name}", dpi=200)
    plt.show()
    plt.close(fig)

y = np.arange(len(SER))[::-1]

# COMMAND ----------
# Grafik 1: Trefferquote je Serie
fig, ax = base(f"Trefferquote je Serie: {round(min(DA)*100)}–{round(max(DA)*100)} % im Test",
               f"XGBoost, Szenario S0, Testzeitraum 2024-11-15 bis 2026-01-20, n = {N} je Serie.\nBalken = 95-%-Konfidenzintervall (Wilson)")
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
ax.hlines(y, W_LO, W_HI, color=BLUE, lw=3, zorder=2, capstyle="round")
ax.scatter(DA, y, s=70, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5, label="XGBoost")
ax.scatter(NYMEX, y, s=42, marker="D", color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1, label="NYMEX-Regel (einfache Regel)")
ax.scatter(MEHRHEIT, y, s=42, marker="s", color=GREY, zorder=3, edgecolor=SURF, linewidth=1, label="Immer häufigere Richtung tippen")
for yi, v in zip(y, DA):
    ax.text(v, yi + 0.33, f"{komma(v*100)} %", ha="center", fontsize=8, color=INK)
ax.set_yticks(y); ax.set_yticklabels(SER, fontsize=9.5)
ax.set_ylim(-0.7, y.max() + 0.6)
ax.text(0.502, -0.55, "50 % = Münzwurf", fontsize=8, color=INK2, ha="left")
ax.set_xlim(0.45, 0.72); ax.set_xticks([0.5, 0.55, 0.6, 0.65, 0.7]); ax.set_xticklabels([f"{int(round(t*100))} %" for t in [0.5, 0.55, 0.6, 0.65, 0.7]])
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.22), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2)
note(fig, f"Quelle: db_11. Bei überlappenden 3-Tage-Labels ist das Intervall nur Orientierung; Bootstrap-Untergrenze > 50 % bei {N_BOOT} von {len(SER)} Serien.")
save(fig, "1_trefferquote_je_serie.png")

# COMMAND ----------
# Grafik 2: Precision gegen Basisrate
fig, ax = base("Ein Kaufsignal trifft öfter als ein beliebiger Tag",
               "Precision = Anteil der „Kaufen“-Signale, nach denen der Preis wirklich stieg.\nBasisrate = Anteil aller Tage mit steigendem Preis")
ax.hlines(y, BASIS, PREC, color=GRID, lw=4, zorder=1)
ax.scatter(BASIS, y, s=70, color=GREY, zorder=3, edgecolor=SURF, linewidth=1.5, label="Basisrate (beliebiger Tag)")
ax.scatter(PREC, y, s=70, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5, label="Precision des Modells")
for yi, p, b in zip(y, PREC, BASIS):
    ax.text(p + 0.006, yi, f"{p*100:.0f} %  (+{(p-b)*100:.0f} Pkt.)", va="center", fontsize=8, color=INK)
ax.set_yticks(y); ax.set_yticklabels(SER, fontsize=9.5)
ax.set_xlim(0.42, 0.74); ax.set_xticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7]); ax.set_xticklabels([f"{int(round(t*100))} %" for t in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]])
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.23), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
note(fig, f"Quelle: db_11. Recall (erkannte Anstiege) liegt bei {min(RECALL)*100:.0f}–{max(RECALL)*100:.0f} %.")
save(fig, "2_precision_vs_basisrate.png")

# COMMAND ----------
# Grafik 3: Konfusionsmatrix Heizöl Süd
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
fig.subplots_adjust(left=0.27, right=0.9, top=0.72, bottom=0.14)
fig.text(0.04, 0.94, "Heizöl Süd: Wo das Modell richtig und falsch liegt", fontsize=15, fontweight="bold", ha="left", va="top")
fig.text(0.04, 0.875, f"XGBoost, S0, Test, n = {N} Tage. Dunkelblau = richtig, hellblau = Fehler.\n"
         f"Trefferquote {komma(DA[0]*100)} %, Precision {PREC[0]*100:.0f} %, Recall {RECALL[0]*100:.0f} %",
         fontsize=9.5, color=INK2, ha="left", va="top")
labels = [["Richtig gekauft", "Zu früh gekauft"], ["Zu lange gewartet", "Richtig gewartet"]]
for i in range(2):
    for j in range(2):
        v = CM[i, j]
        dunkel = (i == j)
        ax.add_patch(plt.Rectangle((j + 0.03, 1 - i + 0.03), 0.94, 0.94, color=SEQ[4] if dunkel else SEQ[1], zorder=1))
        ax.text(j + 0.5, 1 - i + 0.58, str(v), ha="center", va="center", fontsize=26, fontweight="bold",
                color="#ffffff" if dunkel else INK, zorder=2)
        ax.text(j + 0.5, 1 - i + 0.28, labels[i][j], ha="center", va="center", fontsize=10,
                color="#ffffff" if dunkel else INK, zorder=2)
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["Preis stieg", "Preis stieg nicht"], fontsize=10)
ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Modell: Warten", "Modell: Kaufen"], fontsize=10)
ax.xaxis.tick_top(); ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
note(fig, f"Quelle: db_11. Precision = {CM[0,0]} / ({CM[0,0]} + {CM[0,1]}), Recall = {CM[0,0]} / ({CM[0,0]} + {CM[1,0]}).")
save(fig, "3_konfusionsmatrix_heizoel_sued.png")

# COMMAND ----------
# Grafik 4: Ersparnis gegenüber "immer kaufen"
fig, ax = base("Ersparnis gegenüber „immer kaufen“",
               "Euro pro 100 Liter und Entscheidungstag. Rechts von der Null = Modell ist günstiger.\nBalken = 95-%-Bootstrap-Intervall")
ax.axvline(0, color=INK, lw=1.2, zorder=1)
ax.hlines(y, ERSP_LO, ERSP_HI, color=BLUE, lw=3, zorder=2, capstyle="round")
ax.scatter(ERSP, y, s=70, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5)
for yi, v, lo in zip(y, ERSP, ERSP_LO):
    ax.text(v, yi + 0.33, f"{komma(v, 2)} €", ha="center", fontsize=8, color=INK)
    if lo < 0:
        ax.text(0.52, yi, "Intervall reicht unter 0", fontsize=7.5, color=INK2, va="center")
ax.set_yticks(y); ax.set_yticklabels(SER, fontsize=9.5)
ax.set_xlim(-0.1, 0.7); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_xticklabels([f"{komma(t)} €" for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]])
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
note(fig, f"Quelle: db_11. Bei {sum(l > 0 for l in ERSP_LO)} von {len(ERSP_LO)} Serien liegt die untere Grenze über 0.")
save(fig, "4_ersparnis_vs_immer_kaufen.png")

# COMMAND ----------
# Grafik 5: Modellvergleich (braucht db10_ergebnisse.csv)
if HAT10:
    fig, ax = base("Modellvergleich: kein klarer Sieger",
                   "Mittlere Trefferquote über 8 Serien, S0.\nEntwicklung 2023-04 bis 2024-11, Test 2024-11 bis 2026-01", h=4.8)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.76, bottom=0.26)
    y5 = np.arange(len(MOD))[::-1]
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.hlines(y5, DEV_DA, TEST_DA, color=GRID, lw=4, zorder=1)
    ax.scatter(DEV_DA, y5, s=70, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5, label="Entwicklungszeitraum")
    ax.scatter(TEST_DA, y5, s=70, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1.5, label="Testzeitraum")
    box = dict(boxstyle="square,pad=0.15", fc=SURF, ec="none")
    for yi, a, b in zip(y5, DEV_DA, TEST_DA):
        for v, other in ((a, b), (b, a)):
            links = v <= other
            ax.text(v + (-0.007 if links else 0.007), yi, f"{komma(v*100)} %", ha="right" if links else "left",
                    va="center", fontsize=8, color=INK, bbox=box, zorder=5)
    ax.set_yticks(y5); ax.set_yticklabels(MOD, fontsize=9.5)
    ax.set_ylim(-0.7, y5.max() + 0.6)
    ax.text(0.502, -0.55, "50 % = Münzwurf", fontsize=8, color=INK2, ha="left")
    ax.set_xlim(0.44, 0.67); ax.set_xticks([0.5, 0.55, 0.6, 0.65]); ax.set_xticklabels(["50 %", "55 %", "60 %", "65 %"])
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.25), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    note(fig, f"Quelle: db_10. NYMEX-Regel zum Vergleich: {komma(RULE_DEV*100)} % (Entwicklung), {komma(RULE_TEST*100)} % (Test). Unterschiede unter 2 Punkten sind Zufall.")
    save(fig, "5_modellvergleich.png")
print("Gespeichert in:", GRAFIK)
