# Pre-Registration Holdout A (ENTWURF – gilt erst nach Bestätigung und Commit „FINAL“)

Stand: 05.10.2026. Dieses Dokument wird eingecheckt, **bevor** irgendetwas auf dem Holdout-Zeitraum ausgewertet wird.
Der Zeitstempel des Commits ist der Nachweis. Nach dem FINAL-Commit werden Regeln nur noch durch einen neuen, datierten Abschnitt am Ende ergänzt, nie rückwirkend geändert.

## 1. Frage (ein Satz)
Sagt das bestehende XGBoost-Vollmodell die 3-Tage-Richtung des Heizölpreises Süd (Argus, Szenario S0) auf bisher nicht ausgewerteten Tagen besser als Zufall voraus?

## 2. Was bereits ausgewertet wurde (und deshalb kein Beleg mehr ist)
Alle Tage bis 2026-01-20 (db_03 bis db_08). Entwicklung 2023-04-21…2024-11-01, Test 2024-11-15…2026-01-20.

## 3. Holdout A
- Entscheidungstage t: **2026-01-21 bis 2026-06-01**, alle Tage, an denen Argus-Heizöl-Süd UND alle Marktdaten (ICE, Brent, WTI, NYMEX, EUR/USD) vorhanden sind.
- Label: Preis(t+3) > Preis(t) (Handelstage, Argus Heizöl Süd, Szenario S0). Gleichstand zählt als „nicht gestiegen“ (wie bisher).
- Erwartete Größe: ca. 85–90 Tage (**Annahme**, nach Datenzusammenführung zu bestätigen; die Zahl wird in der Ergebnisdatei berichtet, nicht angepasst).
- Tage nach 2026-06-01 gehören **nicht** zu Holdout A. Sie bilden einen eigenen späteren Holdout B, sobald Marktdaten vorliegen. A und B werden nie vermischt oder nachträglich zusammengelegt, um ein Ergebnis zu retten.

## 4. Daten (eingefroren)
- Argus-Preise: gold-Tabelle bis 2026-01-20, danach `argus_api_2025-10-01_bis_2026-10-02.csv`. Überlappung validiert (77 Tage, Abweichung 0,0).
- ICE Gasoil nach 2026-01-20: echte API-Werte, **nicht** der fortgeschriebene Wert aus der OMR-Phase.
- Brent, WTI, NYMEX, EUR/USD: Excel „P229 – Daten von Brent, WTI, Heizöl, Wechselkurs“, dieselbe Quelle wie die Gold-Tabelle. EUR/USD **nicht** aus der EZB-API (Abweichung bis 0,0058, nicht dieselbe Reihe). **Annahme**, zu bestätigen.
- Tage, an denen ICE vorhanden ist, Argus Deutschland aber nicht (Feiertage): keine Entscheidung, Features und Preisreihe werden wie bisher auf den Argus-Handelstagen geführt (**Annahme**: gleiche Behandlung wie in db_08).

## 5. Modell (eingefroren)
- `p229.config.REG`, Features `build_features` (44 stationäre Merkmale), keine Änderung, kein Tuning, Zufallsstartwert 42.
- Gepurgter Walk-Forward, expanding window ab Datenbeginn, Neutraining alle 5 Tage, `last = i − 3 − feat_lag`. Training schließt Tage von Holdout A ein, sobald deren Label bekannt ist.
- Vor der Auswertung: `leak_test` und `pytest` müssen grün sein; Commit-Hash des Codes wird in der Ergebnisdatei gespeichert.

## 6. Hauptkriterium (eines, vorab festgelegt)
Heizöl Süd, Vollmodell, S0: **Trefferquote > 50 %**, erfüllt, wenn die untere Grenze des 95-%-Block-Bootstrap-Intervalls (Block 5) **> 0,50** liegt.
Das entspricht dem Kickoff-Ziel „Treffsicherheit > 50 %“ (Robin Schoenheinz, 10.04.2026) in einer Form, die die Überlappung der 3-Tage-Labels berücksichtigt.

Ergebnislogik:
- Untergrenze > 0,50: „Kickoff-Ziel auf Holdout A erreicht.“
- sonst: „Nicht bestätigt.“ Das heißt **nicht** „widerlegt“ (siehe Abschnitt 8).

## 7. Nebenbefunde (nur beschreibend, keine Beweise)
Alle 8 Serien mit Pflichtmetriken: DA mit Wilson-95-%-KI und Block-Bootstrap, `p_nicht_ueberl` (Holm über 8 Serien), Konfusionsmatrix mit Precision/Recall, Brier und Brier-Skill, Regret in beide Richtungen (Ersparnis gegenüber „immer kaufen“), McNemar gegen NYMEX-Regel und Momentum (Holm). Zusätzlich: Argus-only-Variante (ICE + EUR/USD). Einzelne Serien werden nicht herausgegriffen, wenn das Hauptkriterium verfehlt wird.
Die 8 Serien sind stark korreliert und zählen nicht als 8 unabhängige Bestätigungen.
Ergebnisse getrennt vor und nach 2026-03-31 berichten (Preisbruch im Frühjahr 2026; Datum **Annahme**, vor der Auswertung zu bestätigen).

## 8. Was dieser Test nicht zeigen kann
- Aussagekraft ist gering: bei ca. 87 Tagen findet das Kriterium eine wahre Trefferquote von 60 % nur in etwa 3 von 10 Fällen, bei 65 % in etwa 5 von 10 (grobe Simulation, `scripts/power_holdout.py`, vereinfachte Annahmen, nicht an den Daten gerechnet). Fälschlich „bestanden“ bei wahrem 50 %: ca. 5–6 %.
- Ein „Nicht bestätigt“ ist daher bei kleiner Stichprobe der häufige Ausgang, auch wenn das Modell funktioniert.
- Nicht geklärt, ob der Einkäufer in S0 wirklich noch zum Argus-Preis von t kaufen kann (Frage an Robin).
- Kein Beleg für Kausalität, kein Beleg für Verhalten in einem anderen Preisregime.
- Das Kosten-Kriterium (Ersparnis gegenüber „immer kaufen“) ist Nebenbefund, nicht Hauptkriterium.

## 9. Offene Bestätigungen vor FINAL
- [ ] Leni: Annahmen aus Abschnitt 3, 4 und 7 bestätigen.
- [ ] Robin: Entscheidungszeitpunkt und Kaufpreis (t oder t+1), Argus-Fixzeit.
