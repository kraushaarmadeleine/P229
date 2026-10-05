# Befundnotiz db_10: Modellvergleich (05.10.2026)

Notebook: `notebooks/db_10_modellvergleich.py` / `db_10_komplett.ipynb`. Ausführung durch Leni in Databricks (Laufzeit 56 Minuten).
Zahlen unten stammen aus der Ausgabe von Zelle 5 bis 7 des Laufs. Die Ergebnisdatei `db10_ergebnisse.csv` liegt im Volume (`output/`), nicht im Repo.

## Frage
Welches Modell sagt aus den Excel-Daten die 3-Tage-Richtung der Argus-Preise (Heizöl, Diesel, 4 Regionen) am besten voraus, und ist es besser als eine einfache Regel?

## Design
- Daten: Argus-Excel + Marktdaten-Excel, 2016-01-04 bis 2026-01-20, 8 Serien (Aufbau identisch zur Gold-Tabelle, siehe db_09).
- Szenario S0 (Kauf zum Preis von Tag t, Daten bis t). Gepurgter Walk-Forward, Neutraining alle 5 Tage, expanding window.
- Zeiträume: Entwicklung 2023-04-21…2024-11-01, Test 2024-11-15…2026-01-20 (n = 297 je Serie im Test).
- Modelle (feste Einstellungen, **kein Tuning**): XGBoost (REG aus db_08), Logistische Regression (L2, C = 0,05), Random Forest (200 Bäume, Tiefe 4), Logistisch klein (7 Merkmale), Mehrheitsklasse (Dummy). Median-Imputation und Skalierung nur auf Trainingsdaten.
- Vergleich: NYMEX-Vorzeichenregel, Momentum. Paarvergleich mit McNemar je Serie, Holm über 32 Vergleiche.

## Ergebnis (Mittel über 8 Serien)
| Modell | DA Entwicklung | DA Test | Serien Boot_lo > 0,5 (Test) | Serien Ersparnis_lo > 0 (Test) |
|---|---|---|---|---|
| XGBoost | 0,607 | 0,622 | 8 | 6 |
| Logistische Regression | 0,608 | 0,604 | 8 | 5 |
| Random Forest | 0,595 | 0,607 | 8 | 4 |
| Logistisch klein | 0,601 | 0,586 | 7 | 4 |
| Mehrheitsklasse | 0,477 | 0,486 | 0 | 0 |
| NYMEX-Regel | 0,619 | 0,588 | | |
| Momentum | 0,511 | 0,537 | | |

- Rangfolge wechselt zwischen den Zeiträumen (Entwicklung: Logistische Regression vor XGBoost mit 0,1 Prozentpunkten Abstand; Test: XGBoost vorn).
- XGBoost gegen Logistische Regression und Random Forest: Differenz −1,9 bzw. −1,6 Prozentpunkte (Mittel, Test), kleinster p_holm = 1,0. **Kein nachweisbarer Unterschied.**
- Logistisch klein ist in beiden Zeiträumen das schwächste der vier. In einer Serie nach Holm signifikant schlechter als XGBoost (p_holm = 0,04).
- Mehrheitsklasse liegt unter 50 % und ist signifikant schlechter (erwartbar, Kontrollwert).
- Die NYMEX-Regel schlägt im Entwicklungszeitraum jedes Modell (0,619), im Test unterliegt sie den drei guten Modellen. Der Vorsprung der Modelle gegenüber der einfachen Regel ist damit **nicht stabil**.

## Was die Ergebnisse nicht zeigen
- Kein neuer Zeitraum: beide Fenster wurden in früheren Notebooks bereits ausgewertet.
- Nur S0. Ob der Einkäufer am Tag t noch zum Argus-Preis von t kaufen kann, ist offen (Frage an Robin).
- Kein Tuning: andere Einstellungen könnten die Rangfolge ändern.
- Die 8 Serien sind stark korreliert. Ca. 300 Tage mit überlappenden 3-Tage-Labels reichen nicht, um Unterschiede von 1 bis 2 Prozentpunkten zu erkennen.
- Die Auswahl „bestes Modell nach Entwicklung = Logistische Regression“ (Zelle 7) ist wegen 0,1 Prozentpunkten Vorsprung nicht aussagekräftig.

## Empfehlung (vorläufig)
- XGBoost bleibt das Modell (Kickoff-Vorgabe, in beiden Zeiträumen vorn dabei, im Test beste Kosten gegenüber „immer kaufen“). Kein Beleg für ein besseres Modell.
- Logistische Regression als einfacher Vergleichswert und Plan B (ähnlich gut, leichter erklärbar).
- Belastbarer Test auf neuen Tagen steht aus (`docs/PRAEREGISTRIERUNG_holdout_A.md`).
