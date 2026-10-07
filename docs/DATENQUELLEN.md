# Datenquellen (Stand 05.10.2026)

Nach Zelle 2 von `db_11_xgboost_test` (Datenprotokoll, Lauf von Leni). **Fakten** stammen aus der Ausgabe, **Schlüsse** sind markiert und nicht bestätigt.

## Argus-Excel (`Heizoel und Diesel bis 2016 und ICE Daten.xlsx`, Blatt „Tabelle1“)
- Fakt: 2016-01-04 bis 2026-01-20, 2578 Zeilen. Argus O.M.R. vDIP, prompt, London close, **index**, Euro/100 l, fca Lkw (ein Wert pro Tag, nicht low/high).
- 8 Serien: Heizöl 50 ppm und Diesel EN 590 10 ppm je Süd, Südwest, Südost, Rhein-Main. Dazu ICE Gasoil NWE Monat 1, London close, settlement, USD/t.
- Lücken je Serie 24 bis 80 Tage (vermutlich Feiertage), Vorwärtsfüllen bis 5 Tage.
- Nicht genutzt: „all regions“ (2 Spalten, nur 1429 Tage).

## Marktdaten-Excel (`P229 - Daten von Brent, WTI, Heizöl, Wechselkurs.xlsx`)
| Variable | Blatt | Zeitraum | Min / Max | letzter Wert |
|---|---|---|---|---|
| brent | Brent Rohöl | bis 2026-06-01 | 9,10 / 143,95 | 98,29 |
| wti | WIT Rohöl | bis 2026-06-01 | −36,98 / 145,31 | 95,96 |
| nymex_heating_oil | Heizöl | bis 2026-06-01 | 0,284 / 5,152 | 3,551 |
| usd_eur | USD_EUR Wechselkurs | bis 2026-06-05 | 0,827 / 1,601 | 1,1533 |

- Fakt: Reihenname des Wechselkurses in der Excel ist `DEXUSEU`. Das ist die FRED-Bezeichnung für „US-Dollar je 1 Euro“ (Werte um 1,15, also **USD pro EUR**).
- Fakt: WTI hat einen negativen Wert (April 2020). `build_features` setzt nicht-positive Preise auf fehlend, XGBoost verarbeitet das.
- **Schluss, nicht bestätigt:** Maxima (Brent 143,95, WTI 145,31) und das WTI-Minimum (−36,98) entsprechen den EIA-Spotpreisen, wie sie bei FRED geführt werden. Dann ist `nymex_heating_oil` vermutlich ein **EIA/FRED-Spotpreis (NY Harbor, USD/gal)** und kein NYMEX-Future. Bestätigen: Original-Reihennamen in der Excel (Zelle 2, „Rohe Kopfzeilen“) und Herkunft der Datei klären.
- **Offene Folgefrage:** EIA/FRED-Reihen werden mit Verzögerung veröffentlicht. Für den Betrieb muss geklärt werden, woher die US-Preise von Tag t am Entscheidungstag kommen (Robin/IT). Das Ergebnis in Szenario S0 setzt voraus, dass sie am Tag t schon vorliegen.
- Zeitzonenhypothese (nicht geprüft): US-Preise schließen nach dem Argus-London-Fix, ein Teil des Vorsprungs in S0 könnte daher stammen.

## Nachtrag 07.10.2026: Ursprung der Marktdaten (aus Datenprotokoll db_15, aktuelle Kopfzeilen)
- Fakt: Anfangsdaten der Reihen in der Marktdaten-Excel: WTI 1986-01-02, NY-Harbor-Heizöl 1986-06-02, Brent 1987-05-20, Wechselkurs 1999-01-04 (Reihenname `DEXUSEU`). Das sind genau die Startdaten der EIA-Tagesreihen (Cushing WTI, NY Harbor No. 2 Heating Oil, Brent Europe) bzw. der FRED-Reihe DEXUSEU.
- **Schluss (stark, nicht bestätigt durch die Quelle selbst):** „Heizöl“ = NY-Harbor-Spotpreis (EIA), kein NYMEX-Future. Die interne Bezeichnung `nymex_heating_oil` und der Name „NYMEX-Regel“ sind damit vermutlich irreführend. Bestätigen: Herkunft der Excel klären (FRED/EIA-Download?).
- Konsequenz: Diese Reihen erscheinen mit Verzögerung. Für den Betrieb muss ein Datenlieferant mit Daten am selben Tag gefunden werden (z. B. Börsendaten).
