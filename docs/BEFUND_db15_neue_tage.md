# Befundnotiz db_15 Teil E: Erster Blick auf neue Tage (07.10.2026)

Ausführung durch Leni in Databricks (Run all). Zahlen aus der Ausgabe von Teil C bis E.
**Explorativ.** Die Pre-Registration (Holdout A) war noch nicht FINAL. Dieser Lauf zählt daher nicht als formaler Test.

## Daten
- Neue Tage: 2026-01-21 bis 2026-05-27 bewertet (88 Tage je Serie). Marktdaten enden am 2026-06-01, die letzten 3 Tage haben noch kein bekanntes Ergebnis.
- Argus-Preise: API-Datei `argus_api_2025-10-01_bis_2026-10-05.csv`. Überlappung mit der Excel 77 Tage, größte Abweichung 0,0. Leakage-Test auf den erweiterten Daten bestanden.
- Modell unverändert (XGBoost, feste Einstellungen), Walk-Forward ab 2026-01-21 mit allen davor bekannten Daten.

## Ergebnis (Mittel über 8 Serien)
| | Test alt (n = 297) | Neue Tage (n = 88) |
|---|---|---|
| Trefferquote | 62,2 % | **58,2 %** (54,5–62,5 %) |
| Immer häufigere Richtung tippen | 52,8 % | 52,7 % |
| NYMEX-Regel | 58,8 % | 56,7 % |
| Precision | 59,5 % | 58,7 % |
| Recall | 63,9 % | 64,4 % |
| Basisrate (Anteil steigender Tage) | 47,5 % | 51,6 % |
| Brier-Skill | 0,071 | 0,004 |

- Bootstrap-Untergrenze über 50 %: nur **2 von 8** Serien (Diesel Südwest, Diesel Rhein-Main). Hauptkriterium der Pre-Registration (Heizöl Süd): 54,5 %, Untergrenze 44,3 %, also nicht erfüllt (Wert: „nicht bestätigt“, nicht „widerlegt“).
- Einfacher z-Wert für 58,2 % gegen 50 %: 1,55 bei 88 Tagen, 0,89 bei etwa 29 unabhängigen Tagen (überlappende Labels). Nicht signifikant.
- Brier-Skill nahe 0: die Wahrscheinlichkeiten sind auf diesen Tagen nicht besser als eine konstante Basisrate.
- Kosten gegenüber „immer kaufen“: Mittel −0,04 € pro 100 l und Tag. Heizöl-Serien negativ (−0,17 bis −0,40), Diesel-Serien positiv (+0,06 bis +0,44). Intervalle sehr breit (untere Grenzen −1,0 bis −2,1), nicht aussagekräftig. Die Preise schwanken im neuen Zeitraum viel stärker als im alten (Preisniveau Heizöl Süd Jan. 2026 ca. 73, Okt. 2026 ca. 132), absolute Euro-Beträge sind nicht mit dem alten Test vergleichbar.
- Beobachten (δ = 0,086 aus der Entwicklung): 27 % der Tage statt 42 % im Test, Trefferquote auf den „sicheren“ Tagen 57,9 % bei 73 % Abdeckung, also **kein Zugewinn**. Das Modell war auf den neuen Tagen sicherer, aber nicht treffsicherer.

## Einordnung
- Die Richtung passt zum alten Test (Trefferquote über Zufall und Mehrheitsklasse), der Wert ist aber niedriger. Der Unterschied (4 Punkte) liegt innerhalb der Stichprobenschwankung bei 88 Tagen.
- Aussagekraft: Bei 88 Tagen findet das Kriterium (Bootstrap-Untergrenze > 50 %) eine wahre Trefferquote von 60 % nur in etwa 3 von 10 Fällen (grobe Simulation, `scripts/power_holdout.py`).
- Realistische Erwartung auf Basis beider Zeiträume: eher 55–60 % als 62–65 %. Der alte Test ist vermutlich optimistisch.
- Mögliche Ursache für das Beobachten-Ergebnis: Regimewechsel im Frühjahr 2026 (Hypothese, nicht geprüft).

## Was das nicht zeigt
- Kein formaler Beleg und keine Widerlegung. Holdout B (Tage nach 2026-06-01) ist noch ungenutzt und braucht Marktdaten bis Oktober.
- Die Ersparnis-Kennzahl in Euro ist bei so unterschiedlichen Preisniveaus nicht vergleichbar. Besser: Ersparnis in Prozent des Preises.
- Offen bleibt der Kaufzeitpunkt (Tag t oder t+1) und die Verfügbarkeit der US-Preise am Entscheidungstag.
