# Befundnotiz db_14: Dritte Stufe „Beobachten“ (07.10.2026)

Notebook: `notebooks/db_14_beobachten.py` / `db_14_komplett.ipynb`. Ausführung durch Leni in Databricks. Zahlen aus der Ausgabe von Zelle 4 und 5.

## Frage
Wird das Signal zuverlässiger, wenn das Modell an unsicheren Tagen weder „Kaufen“ noch „Warten“ sagt, sondern „Beobachten“?

## Design (vorab festgelegt)
- XGBoost, Szenario S0, Vorhersagen aus db_10 (gleiche Läufe wie im Modellvergleich).
- Regel: Kaufen bei p ≥ 0,5 + δ, Warten bei p < 0,5 − δ, sonst Beobachten.
- δ wird aus der **Entwicklung** abgeleitet: Anteil der unsichersten Tage (|p − 0,5|), gepoolt über 8 Serien. Aus 20, 30 und 40 % wählt die Regel den Anteil mit der höchsten Trefferquote auf der Entwicklung. Gewählt: **40 %** (δ = 0,086). Danach einmalige Prüfung im Test (2024-11-15 bis 2026-01-20, n = 297 je Serie).
- Kosten: Beobachten einmal wie Warten (A), einmal wie Kaufen (B) gerechnet.

## Ergebnis
**Trefferquote auf den Tagen mit Signal steigt, in beiden Zeiträumen und in jeder Stufe** (Mittel über 8 Serien):

| Anteil Beobachten | Entwicklung | Test | Tage mit Signal (Test) |
|---|---|---|---|
| 0 % | 60,7 % | 62,3 % | 100 % |
| 20 % | 62,8 % | 63,6 % | 79 % |
| 30 % | 63,1 % | 65,4 % | 68 % |
| 40 % (gewählt) | 64,2 % | 66,5 % | 58 % |
| 50 % | 65,3 % | 68,0 % | 49 % |

Test, gewählte Regel (40 %), je Serie: Trefferquote auf Signal-Tagen 60,9–72,4 %, Bootstrap-Untergrenze bei allen 8 Serien über 50 % (0,552 bis 0,659). Höher als ohne Beobachten bei **8 von 8 Serien**, mittlerer Zugewinn **+4,3 Prozentpunkte** (Spanne +0,3 Heizöl Südwest bis +8,7 Heizöl Südost).
Precision „Warten“ steigt stärker (65 % → 72 %) als Precision „Kaufen“ (60 % → 62 %).

**Kosten steigen nicht:** Ersparnis gegenüber „immer kaufen“ im Test von 0,236 € (kein Beobachten) auf 0,196 € (A) bzw. 0,209 € (B) pro 100 l und Tag. Serien mit abgesicherter Ersparnis (Untergrenze > 0) fallen von 6 auf 2.

## Einordnung
- Die Sicherheit des Modells ist informativ: Je größer der Abstand von 50 %, desto zuverlässiger das Signal. Das gilt in zwei getrennten Zeiträumen.
- „Beobachten“ ist deshalb eine Zuverlässigkeits-Ampel, aber **kein Kostenhebel**. An Beobachten-Tagen entsteht keine Ersparnis aus dem Modell.
- Der Nutzen hängt davon ab, was der Einkäufer an Beobachten-Tagen tut (Frage an Robin).

## Was das nicht zeigt
- Der Zugewinn von +4,3 Punkten ist nicht statistisch abgesichert (ca. 170 Signal-Tage je Serie, überlappende 3-Tage-Labels). „8 von 8“ sind keine 8 unabhängigen Bestätigungen, die Serien sind stark korreliert.
- Gilt nur für S0 (Kauf zum Preis von Tag t) und die beiden bereits ausgewerteten Zeiträume. Kein Test auf neuen Tagen.
- Der Anteil 40 % wurde auf der Entwicklung gewählt, die Tabelle zeigt aber alle Anteile; Anteile über 40 % (weniger als 60 % Signal-Tage) wurden bewusst nicht zugelassen.
