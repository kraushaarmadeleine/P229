import numpy as np
import pandas as pd

from p229.config import EXO, SERIES
from p229.data import _serie_name, build_dataset, load_argus_excel, load_market_sheet


def test_serienname_zuordnung():
    t = "Gasoil heating oil 50ppm southwest Germany fca truck Argus O.M.R. vDIP prompt, London close, index"
    assert _serie_name(t) == "heizoel_suedwest"
    assert _serie_name(t.replace("southwest", "south")) == "heizoel_sued"
    assert _serie_name("Gasoil diesel EN 590 10ppm southeast Germany fca truck") == "diesel_suedost"
    assert _serie_name("Gasoil diesel EN 590 10ppm Rhine-Main Germany fca truck") == "diesel_rheinmain"
    assert _serie_name("Gasoil Ice NWE month 1, London close, settlement, USD/t, cif") == "ice_gasoil"
    assert _serie_name("Gasoil heating oil 50ppm all regions Germany fca truck") is None


def _mini_argus(path):
    desc = {1: "Gasoil diesel EN 590 10ppm south Germany fca truck", 2: "Gasoil heating oil 50ppm south Germany fca truck"}
    k = 3
    for r in ["southwest", "southeast", "Rhine-Main"]:
        desc[k] = f"Gasoil diesel EN 590 10ppm {r} Germany fca truck"; desc[k + 1] = f"Gasoil heating oil 50ppm {r} Germany fca truck"; k += 2
    desc[k] = "Gasoil Ice NWE month 1, London close, settlement, USD/t, cif"
    dates = pd.bdate_range("2020-01-01", periods=30)
    rows = [["Copyright"] + [None] * k, ["combined description"] + [desc[j] for j in range(1, k + 1)]]
    rng = np.random.default_rng(0)
    for dt in dates:
        rows.append([dt] + list(100 + rng.normal(size=k)))
    pd.DataFrame(rows).to_excel(path, sheet_name="Tabelle1", header=False, index=False)
    return dates


def test_argus_loader_und_build(tmp_path):
    p = tmp_path / "argus.xlsx"
    dates = _mini_argus(p)
    a = load_argus_excel(p)
    assert len(a) == 30 and set(SERIES + ["ice_gasoil"]) <= set(a.columns)
    # Marktblatt mit Kopfzeilen-Text und Datum/Wert
    m = tmp_path / "markt.xlsx"
    with pd.ExcelWriter(m) as w:
        for sheet in ["Brent Rohöl", "WIT Rohöl", "Heizöl", "USD_EUR Wechselkurs"]:
            df = pd.DataFrame({"Datum": dates[:-1], "Preis": np.linspace(50, 60, 29)})
            df.to_excel(w, sheet_name=sheet, index=False, startrow=2)
    ms = [load_market_sheet(m, "Brent Rohöl", "brent"), load_market_sheet(m, "WIT Rohöl", "wti"),
          load_market_sheet(m, "Heizöl", "nymex_heating_oil"), load_market_sheet(m, "USD_EUR Wechselkurs", "usd_eur")]
    assert all(len(x) == 29 for x in ms)
    d, bericht = build_dataset(a, ms)
    assert set(SERIES + EXO) <= set(d.columns)
    assert bericht["fehlend_vor_ffill"]["brent"] == 1          # letzter Tag fehlt, wird vorwärts gefüllt
    assert len(d) == 30 and d[SERIES + EXO].notna().all().all()


def test_protokoll_info(tmp_path):
    p = tmp_path / "argus.xlsx"
    dates = _mini_argus(p)
    a, info = load_argus_excel(p, return_info=True)
    assert len(info) == 9 and set(info["Serie"]) >= set(SERIES) and (info["Tage"] == 30).all()
    m = tmp_path / "markt.xlsx"
    pd.DataFrame({"Datum": dates, "Preis USD/bbl": np.linspace(50, 60, 30)}).to_excel(m, sheet_name="Brent", index=False, startrow=2)
    out, i = load_market_sheet(m, "Brent", "brent", return_info=True)
    assert i["Tage"] == 30 and i["Wertspalte"] == 2 and "Preis" in i["Kopfzeile Wert"]
