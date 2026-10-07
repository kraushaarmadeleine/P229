# Databricks notebook source
# MAGIC %md
# MAGIC # P229: Kaufsignal für Heizöl und Diesel – Gesamtprüfung
# MAGIC
# MAGIC **Ziel:** Ein tägliches Signal für den Einkauf: **Kaufen**, **Warten** oder **Beobachten**. Das Modell (XGBoost) schätzt, ob der Preis in **3 Tagen höher** ist als heute.
# MAGIC
# MAGIC **Dieses Notebook zeigt alles in einem Durchlauf. Einfach „Run all“:**
# MAGIC
# MAGIC | Teil | Inhalt |
# MAGIC |---|---|
# MAGIC | **A** | Welche Daten und welche Spalten nutzt das Modell? |
# MAGIC | **B** | Wie wurde getestet (ohne Blick in die Zukunft)? |
# MAGIC | **C** | Fehlermaße: Trefferquote, Precision, Recall, Konfusionsmatrix, Kalibrierung, Kosten |
# MAGIC | **D** | Dritte Stufe „Beobachten“ |
# MAGIC | **E** | **Neue Tage, die das Modell noch nie gesehen hat** |
# MAGIC | **F** | Was das Ergebnis nicht zeigt |
# MAGIC
# MAGIC **Dauer:** ca. 10 Minuten, wenn die gesicherten Vorhersagen aus db_10 im Output-Ordner liegen. Sonst werden sie neu berechnet (zusätzlich bis zu 30 Minuten). Teil E braucht ca. 5–10 Minuten.
# MAGIC
# MAGIC **Wichtig vorab:** Ergebnisse aus Teil C und D gelten für den Zeitraum bis 20.01.2026, der schon oft ausgewertet wurde. Teil E prüft auf Tagen danach.

# COMMAND ----------
# MAGIC %pip install xgboost scikit-learn openpyxl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# Setup: Pfade, Code, Daten laden. (Die Zelle davor legt den Projektcode an.)
import os, sys, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE_PATH = "/Volumes/dev_workspace/p229/p229_files"
OUT = f"{BASE_PATH}/output"
GRAFIK = f"{OUT}/grafiken_gesamt"
os.makedirs(GRAFIK, exist_ok=True)
ARGUS_XLSX = f"{BASE_PATH}/Heizoel und Diesel bis 2016 und ICE Daten.xlsx"
MARKT_XLSX = f"{BASE_PATH}/P229 - Daten von Brent, WTI, Heizöl, Wechselkurs.xlsx"

sys.path.insert(0, "/tmp/p229_code")
from p229.config import SERIES, EXO, DEV, TEST, LAST_ARGUS, REG, SERIEN_NAMEN
from p229.data import load_argus_excel, load_market_sheet, build_dataset, inspect_excel
from p229.features import build_features, merkmal_beschreibung
from p229.leakage import leak_test
from p229.walk_forward import walk_forward
from p229.evaluate import evaluate_run
from p229.stats import holm
from p229.beobachten import bewerte_dreistufig, signal, NAMEN
from p229.neue_daten import suche_neueste_api_csv, lade_neue_argus, kombiniere_argus, signale_neue_tage
from p229 import plots

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 90)

def zeige(fig, name):
    """Grafik anzeigen und als PNG speichern."""
    fig.savefig(f"{GRAFIK}/{name}", dpi=200)
    plt.show()
    plt.close(fig)

argus, argus_info = load_argus_excel(ARGUS_XLSX, return_info=True)
BLAETTER = [("Brent Rohöl", "brent"), ("WIT Rohöl", "wti"), ("Heizöl", "nymex_heating_oil"), ("USD_EUR Wechselkurs", "usd_eur")]
markt, markt_info = [], []
for blatt, name in BLAETTER:
    m, i = load_market_sheet(MARKT_XLSX, blatt, name, return_info=True)
    markt.append(m); markt_info.append(i)
d_all, fuell_bericht = build_dataset(argus, markt)
d = d_all[d_all["date"] <= LAST_ARGUS].reset_index(drop=True)
for H in SERIES:
    leak_test(d, H)
print(f"Datentabelle: {d.shape[0]} Tage von {d['date'].min().date()} bis {d['date'].max().date()} | Leakage-Test bestanden")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Teil A: Welche Daten nutzt das Modell?

# COMMAND ----------
# A1: Quellen. Welche Datei, welches Blatt, welche Spalte?
print("=== 1. Preise aus der Argus-Excel (Blatt 'Tabelle1') ===")
print("Argus O.M.R., prompt, London close, Index, Euro/100 l, fca Lkw. ICE Gasoil: Futures-Abrechnungspreis, USD/t.")
print(argus_info.drop(columns="Beschreibung").to_string(index=False))
print("\n=== 2. Marktdaten aus der Marktdaten-Excel ===")
print(pd.DataFrame(markt_info).drop(columns=["Datumsspalte"]).to_string(index=False))
print("\n=== Lücken (Feiertage u. ä.) und wie sie behandelt werden ===")
print("Fehlende Werte werden mit dem letzten bekannten Wert gefüllt (höchstens 5 Tage). Das nutzt nur Vergangenheit.")
print(fuell_bericht.to_string())
print("\nNicht genutzt: Nachrichten, Rheinfracht, Argus 'all regions', low/high-Werte, Platts, OMR-Übersicht.")

# COMMAND ----------
# A2: Die Eingabespalten des Modells (Merkmale), hier am Beispiel Heizöl Süd.
H0 = "heizoel_sued"
F0 = build_features(d, H0)
merkmale = [c for c in F0.columns if c != "date"]
tab = pd.DataFrame({"Nr": range(1, len(merkmale) + 1), "Merkmal (Spaltenname)": merkmale,
                    "Bedeutung": [merkmal_beschreibung(c, H0) for c in merkmale]})
print(f"Das Modell bekommt {len(merkmale)} Zahlen pro Tag. Alle sind Veränderungen (Renditen, Abstände, Schwankung), keine Preisniveaus.")
print(f"Beispiel für {SERIEN_NAMEN[H0]}; für die anderen Serien gilt dasselbe Muster mit dem jeweils eigenen Preis.\n")
print(tab.to_string(index=False))
print("\n=== Zielgröße (was vorhergesagt wird) ===")
print("Ziel = 1, wenn Preis(t+3 Handelstage) > Preis(t), sonst 0. Gleichstand zählt als 'nicht gestiegen'.")
print("\n=== Was das Modell NICHT sieht ===")
print("Preisniveaus, Monat/Jahreszeit, Nachrichten, Rheinfracht, Wetter, Zukunftsdaten, das Ergebnis selbst.")

# COMMAND ----------
# A3: Beispielzeilen und Wichtigkeit der Merkmale.
from xgboost import XGBClassifier
print("=== So sehen die Eingaben der letzten 5 Tage aus (Heizöl Süd) ===")
print(F0.set_index("date")[merkmale].tail(5).T.round(4).to_string())

# Wichtigkeit: ein Modell mit allen Daten bis zum Testbeginn (nur zur Darstellung)
ziel = (d[H0].shift(-3) > d[H0]).astype(float).where(d[H0].shift(-3).notna())
i_test = int(np.where(d["date"] >= TEST[0])[0][0])
letzte = i_test - 3
X0 = F0.loc[:letzte, merkmale]; y0 = ziel.loc[:letzte]; m0 = y0.notna()
modell0 = XGBClassifier(**REG).fit(X0[m0], y0[m0])
gain = pd.Series(modell0.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
anteil = (gain / gain.sum() * 100).head(12)
plots._stil()
fig, ax = plt.subplots(figsize=(9, 7), dpi=110)
fig.subplots_adjust(left=0.42, right=0.95, top=0.82, bottom=0.06)
yy = np.arange(len(anteil))[::-1]
ax.barh(yy, anteil.values, color=plots.BLUE, height=0.6)
import textwrap
ax.set_yticks(yy); ax.set_yticklabels([textwrap.fill(merkmal_beschreibung(k, H0), 46) for k in anteil.index], fontsize=7.5)
for yi, v in zip(yy, anteil.values):
    ax.text(v + 0.3, yi, f"{v:.1f} %".replace(".", ","), va="center", fontsize=8)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0); ax.xaxis.set_visible(False)
plots._kopf(fig, "Welche Eingaben nutzt das Modell am stärksten?", f"Anteil am Gewinn je Eingabe, {SERIEN_NAMEN[H0]}, Modell mit Daten bis {d['date'].iloc[letzte].date()}. Zeigt Nutzung, keine Ursache.")
zeige(fig, "A_wichtigkeit.png")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Teil B: Wie wurde getestet?
# MAGIC
# MAGIC - **Walk-Forward:** Für jeden Tag im Testzeitraum wird das Modell nur mit Daten trainiert, deren Ergebnis an diesem Tag **schon bekannt** ist (alle Tage bis t − 3). Danach sagt es den Tag voraus. Alle 5 Tage wird neu trainiert. Es wird nichts aus der Zukunft benutzt.
# MAGIC - **Zeiträume:** Entwicklung 2023-04-21 bis 2024-11-01, Test 2024-11-15 bis 2026-01-20. Beide wurden vorher nie zum Anpassen des Modells benutzt, nur zur Auswahl von Regeln (siehe Teil D).
# MAGIC - **Modell:** XGBoost mit festen, stark vorsichtigen Einstellungen (keine Feinabstimmung).
# MAGIC - **Szenario S0:** Entscheidung nach Marktschluss am Tag t, Kauf zum Argus-Preis von Tag t.
# MAGIC - **Kontrollen:** Leakage-Test (Merkmale ändern sich nicht, wenn man die Zukunft abschneidet), Zufallstest (verschobene Antworten fallen auf ca. 48 %, siehe db_13).

# COMMAND ----------
# B1: Vorhersagen laden (gesichert aus db_10) oder neu berechnen.
FENSTER = {"Entwicklung": DEV, "Test": TEST}
PROBA = {}
t0 = time.time()
for H in SERIES:
    for fenster, win in FENSTER.items():
        pfad = f"{OUT}/db10_proba_XGBoost_{H}_{fenster}.csv"
        if os.path.exists(pfad):
            PROBA[(H, fenster)] = pd.read_csv(pfad, parse_dates=["date"])
        else:
            print("Rechne neu:", SERIEN_NAMEN[H], fenster, f"({time.time() - t0:.0f}s)")
            res = walk_forward(d, H, build_features(d, H), win, entry_lag=0, feat_lag=0)
            res.to_csv(pfad, index=False)
            PROBA[(H, fenster)] = res
print("Vorhersagen bereit für", len(PROBA), "Läufe | Test: n =", sorted({len(v) for (h, f), v in PROBA.items() if f == 'Test'}))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Teil C: Fehlermaße (Testzeitraum, Szenario S0)

# COMMAND ----------
# C1: Alle Kennzahlen je Serie.
zeilen = []
for H in SERIES:
    F = build_features(d, H)
    r = evaluate_run(d, H, F, PROBA[(H, "Test")], 0, 0)
    r["Basisrate"] = PROBA[(H, "Test")]["y_true"].mean()
    zeilen.append(r)
R = pd.DataFrame(zeilen)
R["F1"] = 2 * R["Precision"] * R["Recall"] / (R["Precision"] + R["Recall"])
R["Spezifitaet"] = R["TN"] / (R["TN"] + R["FP"])            # richtig erkannte "nicht gestiegen"-Tage
R["Fehlalarmrate"] = R["FP"] / (R["FP"] + R["TN"])           # Kaufsignal, obwohl Preis nicht stieg
R["p_Holm"] = holm(R["p_nicht_ueberl"].values)
R["p_Holm_NYMEX"] = holm(R["p_McNemar_NYMEX"].values)
R["Serie_Name"] = R["Serie"].map(SERIEN_NAMEN)

print("=== 1. Trefferquote und wie sicher sie ist ===")
print(R[["Serie_Name", "n", "DA", "Wilson_lo", "Wilson_hi", "Boot_lo", "Boot_hi", "p_nicht_ueberl", "p_Holm"]]
      .rename(columns={"Serie_Name": "Serie", "DA": "Trefferquote"}).round(3).to_string(index=False))
print("\n=== 2. Vergleich mit einfachen Regeln ===")
print(R[["Serie_Name", "DA", "Mehrheit", "DA_NYMEX_Regel", "DA_Momentum", "p_Holm_NYMEX", "Brier_Skill"]]
      .rename(columns={"Serie_Name": "Serie", "DA": "Modell", "Mehrheit": "Immer häufigere Richtung", "DA_NYMEX_Regel": "NYMEX-Regel",
                       "DA_Momentum": "Momentum", "p_Holm_NYMEX": "p (Modell besser als NYMEX, Holm)"}).round(3).to_string(index=False))
print("\n=== 3. Fehlerarten ===")
print(R[["Serie_Name", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Spezifitaet", "Fehlalarmrate", "Basisrate"]]
      .rename(columns={"Serie_Name": "Serie", "TP": "Richtig gekauft", "FP": "Zu früh gekauft", "FN": "Zu lange gewartet",
                       "TN": "Richtig gewartet"}).round(3).to_string(index=False))
print(f"\nMittel über 8 Serien: Trefferquote {R['DA'].mean():.3f} | Precision {R['Precision'].mean():.3f} | Recall {R['Recall'].mean():.3f} | "
      f"Basisrate {R['Basisrate'].mean():.3f} | Brier-Skill {R['Brier_Skill'].mean():.3f}")
print("Lesehilfe: Precision = von den Kaufsignalen stieg der Preis wirklich. Recall = von den Anstiegen wurden so viele erkannt.")
print("Brier-Skill > 0: Wahrscheinlichkeiten besser als eine konstante Basisrate. p_nicht_ueberl: strenger Test gegen überlappende Labels.")

# COMMAND ----------
# C2: Konfusionsmatrizen (Fehlerübersicht).
fig = plots.konfusion_raster(R[["Serie", "TP", "FP", "FN", "TN", "DA"]], "Fehlerübersicht aller 8 Serien (Konfusionsmatrizen)")
zeige(fig, "C_konfusion_alle.png")

# Eine große Matrix zum Zeigen
plots._stil()
z = R[R["Serie"] == "heizoel_sued"].iloc[0]
fig, ax = plt.subplots(figsize=(7, 4.6), dpi=110)
fig.subplots_adjust(left=0.2, right=0.95, top=0.72, bottom=0.08)
plots.konfusion_einzeln(ax, z["TP"], z["FP"], z["FN"], z["TN"], gross=True)
plots._kopf(fig, "Heizöl Süd: Wo das Modell richtig und falsch liegt",
            f"Test, n = {int(z['n'])} Tage. Trefferquote {z['DA']*100:.1f} %, Precision {z['Precision']*100:.0f} %, Recall {z['Recall']*100:.0f} %".replace(".", ","))
zeige(fig, "C_konfusion_heizoel_sued.png")

# COMMAND ----------
# C3: Trefferquote im Vergleich zu einfachen Regeln.
fig = plots.trefferquote_mit_baselines(R[["Serie", "DA", "Wilson_lo", "Wilson_hi", "Mehrheit", "DA_NYMEX_Regel"]],
                                       "Trefferquote je Serie im Test")
zeige(fig, "C_trefferquote.png")

# COMMAND ----------
# C4: Precision, Recall, Basisrate.
fig = plots.precision_recall(R[["Serie", "Precision", "Recall", "Basisrate"]], "Ein Kaufsignal trifft öfter als ein beliebiger Tag")
zeige(fig, "C_precision_recall.png")

# COMMAND ----------
# C5: Kalibrierung. Kann man der Wahrscheinlichkeit des Modells trauen?
p_alle = np.concatenate([PROBA[(H, "Test")]["proba"].values for H in SERIES])
y_alle = np.concatenate([PROBA[(H, "Test")]["y_true"].values for H in SERIES])
fig = plots.kalibrierung(p_alle, y_alle, sub="Alle 8 Serien zusammen, Test. Die Serien sind verwandt, die Balken sind daher zu optimistisch.")
zeige(fig, "C_kalibrierung.png")

# COMMAND ----------
# C6: Kosten gegenüber "immer kaufen".
print(R[["Serie_Name", "Regret_Modell", "Regret_immer_kaufen", "Ersparnis_vs_kaufen", "Ersp_lo", "Ersp_hi",
         "davon_zu_frueh_gekauft", "davon_zu_lange_gewartet"]]
      .rename(columns={"Serie_Name": "Serie", "Regret_Modell": "Mehrkosten Modell", "Regret_immer_kaufen": "Mehrkosten immer kaufen",
                       "Ersparnis_vs_kaufen": "Ersparnis", "davon_zu_frueh_gekauft": "davon zu früh gekauft",
                       "davon_zu_lange_gewartet": "davon zu lange gewartet"}).round(3).to_string(index=False))
print("\nEuro pro 100 l und Tag gegenüber dem besten Zeitpunkt (heute oder in 3 Tagen). Ersparnis > 0: Modell ist günstiger als immer kaufen.")
print(f"Serien mit abgesicherter Ersparnis (untere Grenze > 0): {int((R['Ersp_lo'] > 0).sum())} von {len(R)}")
fig = plots.ersparnis(R[["Serie", "Ersparnis_vs_kaufen", "Ersp_lo", "Ersp_hi"]])
zeige(fig, "C_ersparnis.png")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Teil D: Dritte Stufe „Beobachten“
# MAGIC
# MAGIC **Regel:** Kaufen, wenn p ≥ 0,5 + δ. Warten, wenn p < 0,5 − δ. **Beobachten** dazwischen (das Modell ist unsicher). δ wird nur aus dem **Entwicklungszeitraum** abgeleitet, danach einmalig im Test geprüft. Aus 20, 30 und 40 % Beobachten wählt die Regel den Anteil mit der besten Trefferquote auf der Entwicklung.

# COMMAND ----------
# D1: Anteil Beobachten wählen und prüfen.
Q_GRID, Q_ERLAUBT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4]
p_dev = np.concatenate([PROBA[(H, "Entwicklung")]["proba"].values for H in SERIES])
absdev = np.abs(p_dev - 0.5)
zeilen = []
for q in Q_GRID:
    delta = float(np.quantile(absdev, q))
    for fenster in FENSTER:
        r = pd.DataFrame([bewerte_dreistufig(d, H, PROBA[(H, fenster)], delta) for H in SERIES])
        zeilen.append({"Anteil_Beobachten": q, "delta": delta, "Fenster": fenster, "Abdeckung": r["Abdeckung"].mean(),
                       "DA_entschieden": r["DA_entschieden"].mean(), "Precision_Kaufen": r["Precision_Kaufen"].mean(),
                       "Precision_Warten": r["Precision_Warten"].mean(), "Ersparnis_A": r["Ersparnis_A"].mean(),
                       "Ersparnis_B": r["Ersparnis_B"].mean()})
T = pd.DataFrame(zeilen)
for fenster in FENSTER:
    print(f"=== {fenster} (Mittel über 8 Serien) ===")
    print(T[T["Fenster"] == fenster].drop(columns="Fenster").round(3).to_string(index=False))
dev = T[(T["Fenster"] == "Entwicklung") & (T["Anteil_Beobachten"].isin(Q_ERLAUBT))]
q_star = float(dev.sort_values("DA_entschieden", ascending=False)["Anteil_Beobachten"].iloc[0])
delta_star = float(T[T["Anteil_Beobachten"] == q_star]["delta"].iloc[0])
test0 = T[(T["Anteil_Beobachten"] == 0.0) & (T["Fenster"] == "Test")].iloc[0]
testq = T[(T["Anteil_Beobachten"] == q_star) & (T["Fenster"] == "Test")].iloc[0]
print(f"\nAuf der Entwicklung gewählt: {q_star:.0%} Beobachten (δ = {delta_star:.3f})")
print(f"Test: Trefferquote {test0.DA_entschieden:.3f} -> {testq.DA_entschieden:.3f} auf {testq.Abdeckung:.0%} der Tage. "
      f"Ersparnis (A) {test0.Ersparnis_A:.3f} -> {testq.Ersparnis_A:.3f} Euro pro 100 l und Tag.")
fig = plots.abdeckung_kurve(T[["Fenster", "Abdeckung", "DA_entschieden"]])
zeige(fig, "D_beobachten_kurve.png")

# COMMAND ----------
# D2: So sieht das tägliche Signal aus (letzte 10 Tage des Testzeitraums, Heizöl).
REGION = {"heizoel_sued": "Süd", "heizoel_suedwest": "Südwest", "heizoel_suedost": "Südost", "heizoel_rheinmain": "Rhein-Main"}
zl = []
for H, reg in REGION.items():
    res = PROBA[(H, "Test")].tail(10)
    for dt, p, s in zip(res["date"], res["proba"], signal(res["proba"].values, delta_star)):
        zl.append({"Datum": dt.date(), "Region": "Heizöl " + reg, "Signal": NAMEN[int(s)]})
print(pd.DataFrame(zl).pivot(index="Datum", columns="Region", values="Signal").to_string())
print("\nBeispiel aus dem Testzeitraum, kein aktuelles Signal. Die Regionen bewegen sich fast gleich: im Kern ein Marktsignal.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Teil E: Neue Tage, die das Modell noch nie gesehen hat
# MAGIC
# MAGIC **So funktioniert es:** Das Modell wird mit allen bisherigen Daten trainiert und sagt dann die neuen Tage voraus, Tag für Tag, ohne deren Ergebnis zu kennen. Danach wird verglichen, was wirklich passiert ist.
# MAGIC
# MAGIC **Was du eingeben kannst:** Eine Datei mit Argus-Preisen für neuere Tage (CSV mit den Spalten `date`, den 8 Preisreihen und `ice_gasoil`, oder eine Argus-Excel im selben Format wie die bisherige). Standardmäßig wird die neueste Datei `argus_api_*.csv` aus dem Output-Ordner genommen. Eigene Datei: weiter unten bei `NEU_ARGUS_DATEI` den Pfad eintragen. Die Marktdaten (Brent, WTI, Heizöl, Dollar-Euro) kommen aus der Marktdaten-Excel und müssen die neuen Tage abdecken.
# MAGIC
# MAGIC **Hinweis zur Einordnung:** Dieser Bereich ist ein **erster Blick auf neue Tage**. Die formale Prüfung (Pre-Registration, Holdout A) ist noch nicht eingefroren. Wer das Ergebnis streng belegen will, setzt `NEU_AUSWERTEN = False`, dann werden nur Signale erzeugt und keine Trefferquote berechnet.

# COMMAND ----------
# E1: Einstellungen und neue Daten laden.
NEU_ARGUS_DATEI = suche_neueste_api_csv(OUT) or ""     # eigenen Pfad eintragen, z. B. f"{BASE_PATH}/meine_neuen_daten.csv"
NEU_AUSWERTEN = True                                    # False = nur Signale, keine Bewertung
SERIEN_NEU = list(SERIES)                               # z. B. ["heizoel_sued"] für einen schnellen Lauf
DELTA_NEU = delta_star                                  # Breite des Beobachten-Bereichs aus Teil D

NEU_OK = False
if not NEU_ARGUS_DATEI or not os.path.exists(NEU_ARGUS_DATEI):
    print("Keine Datei mit neuen Daten gefunden.")
    print(f"Lege eine CSV oder Excel in {OUT} ab (CSV: Spalten date, die 8 Serien und ice_gasoil) und trage den Pfad oben bei NEU_ARGUS_DATEI ein.")
else:
    try:
        neu = lade_neue_argus(NEU_ARGUS_DATEI)
        argus_komb, ber = kombiniere_argus(argus, neu)
        d_neu_all, _ = build_dataset(argus_komb, markt)
        ende_markt = min(m["date"].max() for m in markt)          # letzter Tag, an dem ALLE Marktreihen echte Werte haben
        d_neu = d_neu_all[d_neu_all["date"] <= ende_markt].reset_index(drop=True)
        print(f"Marktdaten enden am {ende_markt.date()}. Spätere Tage werden nicht benutzt (sonst wären die Marktwerte veraltet).")
        START_NEU = LAST_ARGUS + pd.Timedelta(days=1)
        n_neu = int((d_neu["date"] >= START_NEU).sum())
        print("Datei:", os.path.basename(NEU_ARGUS_DATEI))
        print(f"Überlappung mit bisherigen Daten: {ber['Tage_ueberlappung']} Tage, größte Abweichung {ber['max_Abweichung_Ueberlappung']}")
        print(f"Neue Tage mit Argus-Daten: {ber['Tage_neu_dazu']} | davon mit vollständigen Marktdaten nutzbar: {n_neu}")
        print(f"Neue Tage im Modell: {d_neu.loc[d_neu['date'] >= START_NEU, 'date'].min().date() if n_neu else '-'} bis {d_neu['date'].max().date()}")
        if n_neu < 10:
            print("Zu wenige neue Tage mit Marktdaten. Prüfe, ob die Marktdaten-Excel die neuen Tage abdeckt.")
        else:
            for H in SERIES:
                leak_test(d_neu, H)
            print("Leakage-Test auf den erweiterten Daten bestanden.")
            NEU_OK = True
    except Exception as e:
        print("Die neuen Daten konnten nicht geladen werden:", e)

# COMMAND ----------
# E2: Signale für die neuen Tage (ohne deren Ergebnis zu benutzen).
if NEU_OK:
    ENDE_NEU = d_neu["date"].max()
    ALLE_NEU, F_NEU, t0 = {}, {}, time.time()
    for H in SERIEN_NEU:
        res, F = signale_neue_tage(d_neu, H, START_NEU, ENDE_NEU, DELTA_NEU)
        ALLE_NEU[H], F_NEU[H] = res, F
        print(f"{SERIEN_NAMEN[H]} fertig ({time.time() - t0:.0f}s)")
    sig_alle = pd.concat(ALLE_NEU.values())
    sig_alle.to_csv(f"{OUT}/db15_signale_neue_tage.csv", index=False)
    print("\nVerteilung der Signale auf den neuen Tagen (alle Serien):")
    print(sig_alle["Signal"].value_counts().to_string())
    print("\nLetzte 12 Tage mit Signal je Serie (Ergebnis steht bei den letzten 3 Tagen noch aus):")
    letzte = sig_alle.groupby("Serie").tail(12).pivot(index="date", columns="Serie", values="Signal")
    letzte.columns = [SERIEN_NAMEN[c] for c in letzte.columns]
    print(letzte.to_string())
else:
    print("Übersprungen (keine nutzbaren neuen Daten, siehe oben).")

# COMMAND ----------
# E3: Bewertung auf den neuen Tagen: wie viele Signale waren richtig?
if NEU_OK and NEU_AUSWERTEN:
    zeilen = []
    for H in SERIEN_NEU:
        res = ALLE_NEU[H]
        bekannt = res[res["Ergebnis_bekannt"]][["date", "y_true", "proba"]].reset_index(drop=True)
        r = evaluate_run(d_neu, H, F_NEU[H], bekannt, 0, 0)
        r["Basisrate"] = bekannt["y_true"].mean()
        dr = bewerte_dreistufig(d_neu, H, bekannt, DELTA_NEU)
        r["Abdeckung_3Stufen"], r["DA_sichere_Tage"] = dr["Abdeckung"], dr.get("DA_entschieden", np.nan)
        zeilen.append(r)
    RN = pd.DataFrame(zeilen)
    RN["Serie_Name"] = RN["Serie"].map(SERIEN_NAMEN)
    n_neu_bew = int(RN["n"].iloc[0])
    print("ERSTER BLICK AUF NEUE TAGE (explorativ, nicht die formale Prüfung)")
    print(f"Bewertet: {n_neu_bew} Tage je Serie ({bekannt['date'].min().date()} bis {bekannt['date'].max().date()}). Bei so wenigen Tagen ist die Unsicherheit groß.\n")
    print(RN[["Serie_Name", "n", "DA", "Wilson_lo", "Wilson_hi", "Boot_lo", "Boot_hi", "Mehrheit", "DA_NYMEX_Regel", "Precision", "Recall",
              "Basisrate", "Brier_Skill", "Ersparnis_vs_kaufen", "Ersp_lo"]]
          .rename(columns={"Serie_Name": "Serie", "DA": "Trefferquote"}).round(3).to_string(index=False))
    vergleich = pd.DataFrame({"Test (alt)": [R["DA"].mean(), R["Precision"].mean(), R["Recall"].mean()],
                              "Neue Tage": [RN["DA"].mean(), RN["Precision"].mean(), RN["Recall"].mean()]},
                             index=["Trefferquote", "Precision", "Recall"]).round(3)
    print("\nVergleich (Mittel über die Serien):")
    print(vergleich.to_string())
    print(f"\nMit Beobachten-Stufe (δ = {DELTA_NEU:.3f}): Trefferquote auf den sicheren Tagen {RN['DA_sichere_Tage'].mean():.3f} "
          f"bei {RN['Abdeckung_3Stufen'].mean():.0%} der Tage.")
    fig = plots.konfusion_raster(RN[["Serie", "TP", "FP", "FN", "TN", "DA"]], "Neue Tage: Fehlerübersicht (Konfusionsmatrizen)",
                                 f"Tage, die das Modell nie gesehen hat. n = {n_neu_bew} je Serie.")
    zeige(fig, "E_konfusion_neue_tage.png")
    fig = plots.trefferquote_mit_baselines(RN[["Serie", "DA", "Wilson_lo", "Wilson_hi", "Mehrheit", "DA_NYMEX_Regel"]],
                                           "Neue Tage: Trefferquote je Serie", f"n = {n_neu_bew} Tage je Serie, erster Blick (explorativ). Balken = 95-%-Intervall (Wilson).")
    zeige(fig, "E_trefferquote_neue_tage.png")
    print("\nEinordnung: Bei", n_neu_bew, "Tagen kann ein Test nur große Unterschiede zeigen. Ein Ergebnis unter dem alten Test heißt nicht automatisch, dass das Modell nicht funktioniert.")
    print("Liegt die Trefferquote auch hier klar über der Basisrate und den einfachen Regeln, ist das ein gutes Zeichen. Ein Beleg im strengen Sinn ist es erst nach der Pre-Registration.")
elif NEU_OK:
    print("Bewertung ist ausgeschaltet (NEU_AUSWERTEN = False). Es wurden nur Signale erzeugt.")
else:
    print("Übersprungen.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Teil F: Was das Ergebnis nicht zeigt
# MAGIC
# MAGIC - **Zeitpunkt des Einkaufs:** Alles gilt für Szenario S0 (Kauf zum Preis von Tag t). Kann der Einkäufer erst am Folgetag kaufen, liegt die Trefferquote bei ca. 50 %. Offen ist auch, woher am Entscheidungstag die US-Preise kommen (die Marktdaten sehen nach verzögert veröffentlichten Spotpreisen aus).
# MAGIC - **Statistik:** Die Serien sind stark verwandt und zählen nicht als 8 unabhängige Belege. Die 3-Tage-Ergebnisse überlappen, deshalb sind einfache Konfidenzintervalle zu optimistisch.
# MAGIC - **Vergleich mit Regeln:** Die einfache NYMEX-Regel erreicht schon 57–61 %. Der Vorsprung des Modells ist klein und nach Holm-Korrektur nicht signifikant.
# MAGIC - **Beobachten:** Macht die Signale zuverlässiger, spart aber im Durchschnitt nicht mehr Geld.
# MAGIC - **Neue Tage (Teil E):** Wenige Tage, daher große Unsicherheit. Keine Ursachen-Aussage, nur Vorhersage-Güte.
# MAGIC - **Nicht enthalten:** Nachrichten, Rheinfracht, andere Horizonte (1 und 14 Tage).

# COMMAND ----------
# Zusammenfassung zum Mitnehmen.
print("ZUSAMMENFASSUNG")
print(f"- Daten: {d.shape[0]} Tage bis {d['date'].max().date()}, 8 Preisreihen, {len(merkmale)} Eingaben je Serie.")
print(f"- Test (n = {int(R['n'].iloc[0])} Tage je Serie): Trefferquote {R['DA'].mean():.1%} im Mittel ({R['DA'].min():.1%} bis {R['DA'].max():.1%}); einfache NYMEX-Regel {R['DA_NYMEX_Regel'].mean():.1%}.")
print(f"- Precision {R['Precision'].mean():.1%} (Basisrate {R['Basisrate'].mean():.1%}), Recall {R['Recall'].mean():.1%}.")
print(f"- Ersparnis gegenüber 'immer kaufen': {R['Ersparnis_vs_kaufen'].mean():.2f} Euro pro 100 l und Tag, bei {int((R['Ersp_lo'] > 0).sum())} von 8 Serien abgesichert.")
print(f"- Mit Beobachten ({q_star:.0%} der Tage): Trefferquote auf den übrigen Tagen {testq.DA_entschieden:.1%}.")
if NEU_OK and NEU_AUSWERTEN:
    print(f"- Neue Tage (explorativ, {n_neu_bew} Tage): Trefferquote {RN['DA'].mean():.1%} im Mittel.")
print(f"- Grafiken gespeichert in: {GRAFIK}")
