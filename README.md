# P229 – BayWa: Kauf/Warten-Signal für Heizöl und Diesel

XGBoost-Signal, Horizont 3 Tage, 4 Regionen, 8 Serien. Methodik und Ergebnisse: siehe `notebooks/db_08_xgboost_sauber.ipynb`.

## Struktur

| Pfad | Inhalt |
|---|---|
| `p229/config.py` | Konstanten (Serien, Fenster, Modellparameter, Szenarien) |
| `p229/features.py` | `build_features`, `london_features` (Argus-only) |
| `p229/walk_forward.py` | gepurgter Walk-Forward |
| `p229/evaluate.py` | Pflichtmetriken (`evaluate_run`) |
| `p229/stats.py` | Wilson, Block-Bootstrap, McNemar, Holm |
| `p229/leakage.py` | `leak_test` |
| `tests/` | Leakage-, Statistik- und Konsistenztests |
| `scripts/check_modul_gegen_db08.py` | Reproduktionscheck in Databricks |
| `notebooks/` | db_08 (aktuell) und db_06 (nur Verlauf, Werte durch Label-Leak verfälscht) – Ausgaben entfernt |

Die Datentabelle `d` wird immer explizit übergeben, es gibt keine globale Variable.

## Tests lokal (Mac, Terminal im Repo-Ordner)

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

## Regeln

- Keine Zugangsdaten in Code, Notebooks oder Chat (nur `getpass`/Databricks-Secrets).
- Keine Argus-Rohdaten im Repo.
- Kein Ergebnis ohne Out-of-Sample-Prüfung; Mehrfachtests mit Holm-Korrektur.
