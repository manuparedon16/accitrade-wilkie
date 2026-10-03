"""
Valida el universo invertible de AcciTrade (Portafolio Universitario) contra Yahoo Finance.

No basta con preguntar "¿existe el ticker?": muchas emisoras del catálogo de AcciTrade
ya no cotizan en BMV y Yahoo devuelve una serie congelada. Ese es el modo de falla
peligroso, porque yfinance no lanza error: entrega datos muertos y las estimaciones
de mu y sigma salen inservibles.

Por eso el criterio de validez es: existe Y tiene dato reciente Y suficientes observaciones.

Salida: data/mapeo_tickers.csv
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------
# Universo tal como aparece en el selector de AcciTrade (renta variable)
# --------------------------------------------------------------------------

TICKERS_ACCITRADE = [
    "AC", "ACCELSA.B", "ACTINVR.B", "AGUA", "AGUILAS.CPO", "ALPEK.A", "ALSEA",
    "ALTERNA.B", "AMX.B", "ANGELD.10", "ARA", "ARISTOS.A", "ASUR.B", "AUTLAN.B",
    "AXTEL.CPO", "BAFAR.B", "BBAJIO.O", "BEVIDES.A", "BEVIDES.B", "BIMBO.A",
    "BOLSA.A", "CABLE.CPO", "CADU.A", "CEMEX.CPO", "CHDRAUI.B", "CIDMEGA",
    "CIE.B", "CMOCTEZ", "CMR.B", "COLLADO", "CONVER.A", "CTAXTEL.A", "CUERVO",
    "CULTIBA.B", "CYDSASA.A", "DANHOS.13", "DIABLOI.10", "DIABLOS.A", "DIABLOS.O",
    "DINE.A", "DINE.B", "DLRTRAC.15", "EDUCA.18", "ESGMEX.ISHRS", "FCFE.18",
    "FEMSA.UB", "FEMSA.UBD", "FHIPO.14", "FIBRAHD.15", "FIBRAMQ.12", "FIBRAPL.14",
    "FIBRATC.14", "FIBRAUP.18", "FIDEAL.20", "FIHO.12", "FINAMEX.O", "FINDEP",
    "FINN.13", "FMTY.14", "FNOVA.17", "FPLUS.16", "FRAGUA.B", "FSHOP.13",
    "FSITES.20", "FUNO.11", "FVIA.16", "GAP.B", "GBM.O", "GCARSO.A1", "GCC",
    "GENTERA", "GFINBUR.O", "GFMULTI.O", "GFNORTE.O", "GICSA.B", "GIGANTE",
    "GISSA.A", "GMD", "GMEXICO.B", "GNP", "GPH.1", "GPROFUT", "GRUMA.B",
    "HCITY", "HERDEZ", "HOTEL", "ICH.B", "IDEAL.B1", "INFRAEX.18", "INVEX.A",
    "IVVPESO.ISHRS", "KIMBER.A", "KIMBER.B", "KOF.UBL", "KUO.A", "KUO.B",
    "LAB.B", "LACOMER.UBC", "LAMOSA", "LASEG", "LASITE", "LIVEPOL.1",
    "LIVEPOL.C1", "MEDICA.B", "MEGA.CPO", "MEXTRAC.09", "MFRISCO.A1", "MINSA.B",
    "NAFTRAC.ISHRS", "NEMAK.A", "OMA.B", "ORBIA", "PASA.B", "PENOLES", "PINFRA",
    "PINFRA.L", "PLANI", "POCHTEC.B", "POSADAS.A", "PROCORP.B", "PSOTRAC.15",
    "PV", "Q", "R.A", "RLH.A", "SIGMAF.A", "SIMEC.B", "SITES1.A1", "SMARTRC.14",
    "SORIANA.B", "SPORT.S", "STORAGE.18", "TEAK.CPO", "TLEVISA.CPO", "TMM.A",
    "TRAXION.A", "VASCONI", "VESTA", "VINTE", "VISTA.A", "VITRO.A", "VOLAR.A",
    "WALMEX",
]

# Casos donde la regla general falla. Yahoo codifica la enie como '&'.
EXCEPCIONES = {
    "PENOLES": ["PE&OLES.MX", "PENOLES.MX"],
    "LIVEPOL.C1": ["LIVEPOLC-1.MX", "LIVEPOLC1.MX"],
    "MFRISCO.A1": ["MFRISCOA-1.MX", "MFRISCOA1.MX"],
    "IDEAL.B1": ["IDEALB-1.MX", "IDEALB1.MX"],
    "SITES1.A1": ["SITES1A-1.MX", "SITESB-1.MX"],
}

# Umbrales de aceptacion
MAX_DIAS_SIN_DATO = 10       # mas que esto = probablemente deslistada
MIN_OBS_ANUALES = 200        # ~252 dias habiles; menos indica serie con huecos
PERIODO = "3y"


def candidatos_yahoo(ticker):
    """Genera tickers Yahoo plausibles para una clave de AcciTrade, en orden de preferencia.

    Regla general: Yahoo concatena emisora y serie sin punto y agrega '.MX'.
        GFNORTE.O  -> GFNORTEO.MX
        CEMEX.CPO  -> CEMEXCPO.MX
        ALSEA      -> ALSEA.MX
    Fallback: soltar la serie (AMX.B -> AMX.MX), porque algunas emisoras
    consolidaron series y Yahoo conserva la clave corta.
    """
    if ticker in EXCEPCIONES:
        return list(EXCEPCIONES[ticker])

    concatenado = ticker.replace(".", "") + ".MX"
    cands = [concatenado]

    if "." in ticker:
        emisora = ticker.split(".")[0] + ".MX"
        if emisora not in cands:
            cands.append(emisora)

    return cands


def descargar_lote(simbolos, periodo=PERIODO, tam_lote=40, pausa=1.0):
    """Descarga precios de cierre ajustados en lotes. Devuelve dict simbolo -> Series."""
    series = {}
    simbolos = list(dict.fromkeys(simbolos))

    for i in range(0, len(simbolos), tam_lote):
        lote = simbolos[i:i + tam_lote]
        print(f"  lote {i // tam_lote + 1}: {len(lote)} simbolos")
        try:
            data = yf.download(
                lote, period=periodo, auto_adjust=True,
                progress=False, threads=True, group_by="column",
            )
        except Exception as exc:
            print(f"  fallo el lote: {exc}")
            continue

        if data is None or data.empty:
            continue

        cierres = data["Close"]
        if isinstance(cierres, pd.Series):          # caso de un solo simbolo
            cierres = cierres.to_frame(name=lote[0])

        for s in lote:
            if s in cierres.columns:
                serie = cierres[s].dropna()
                if len(serie) > 0:
                    series[s] = serie

        time.sleep(pausa)

    return series


def diagnosticar(serie, hoy):
    """Metricas de salud de una serie de precios."""
    ultima_fecha = serie.index[-1].tz_localize(None)
    dias_sin_dato = (hoy - ultima_fecha).days
    ultimo_anio = serie[serie.index >= serie.index[-1] - pd.Timedelta(days=365)]
    retornos = np.log(serie / serie.shift(1)).dropna()

    return {
        "n_obs": len(serie),
        "n_obs_1a": len(ultimo_anio),
        "primera_fecha": serie.index[0].date(),
        "ultima_fecha": ultima_fecha.date(),
        "dias_sin_dato": dias_sin_dato,
        "ultimo_precio": round(float(serie.iloc[-1]), 4),
        "vol_anual": round(float(retornos.std() * np.sqrt(252)), 4) if len(retornos) > 30 else np.nan,
        "pct_dias_planos": round(float((retornos == 0).mean()), 4) if len(retornos) > 30 else np.nan,
    }


def validar_universo(tickers=None, salida=None):
    tickers = list(TICKERS_ACCITRADE) if tickers is None else list(tickers)
    hoy = pd.Timestamp.today().normalize()

    mapa = {t: candidatos_yahoo(t) for t in tickers}

    print(f"Validando {len(tickers)} claves de AcciTrade contra Yahoo Finance\n")

    # Primera pasada: candidato preferente de cada clave
    print("Pasada 1 (candidato principal):")
    principales = [c[0] for c in mapa.values()]
    series = descargar_lote(principales)

    # Segunda pasada: solo para las claves que no resolvieron
    pendientes = [t for t in tickers if mapa[t][0] not in series and len(mapa[t]) > 1]
    if pendientes:
        print(f"\nPasada 2 (fallback, {len(pendientes)} claves):")
        alternos = [mapa[t][1] for t in pendientes]
        series.update(descargar_lote(alternos))

    filas = []
    for t in tickers:
        elegido, serie = None, None
        for cand in mapa[t]:
            if cand in series:
                elegido, serie = cand, series[cand]
                break

        fila = {"accitrade": t, "yahoo": elegido, "candidatos": "|".join(mapa[t])}

        if serie is None:
            fila.update({"estatus": "SIN_DATOS", "n_obs": 0, "n_obs_1a": 0,
                         "primera_fecha": None, "ultima_fecha": None,
                         "dias_sin_dato": None, "ultimo_precio": None,
                         "vol_anual": None, "pct_dias_planos": None})
        else:
            diag = diagnosticar(serie, hoy)
            fila.update(diag)
            if diag["dias_sin_dato"] > MAX_DIAS_SIN_DATO:
                fila["estatus"] = "RANCIA"           # deslistada o suspendida
            elif diag["n_obs_1a"] < MIN_OBS_ANUALES:
                fila["estatus"] = "ILIQUIDA"         # cotiza pero con huecos
            else:
                fila["estatus"] = "OK"

        filas.append(fila)

    df = pd.DataFrame(filas)
    orden = ["OK", "ILIQUIDA", "RANCIA", "SIN_DATOS"]
    df["estatus"] = pd.Categorical(df["estatus"], categories=orden, ordered=True)
    df = df.sort_values(["estatus", "accitrade"]).reset_index(drop=True)

    salida = Path("data/mapeo_tickers.csv") if salida is None else Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False, encoding="utf-8")

    print("\n" + "=" * 58)
    print(df["estatus"].value_counts().reindex(orden).fillna(0).astype(int).to_string())
    print("=" * 58)
    print(f"\nGuardado en {salida.resolve()}")

    invertibles = df.loc[df["estatus"] == "OK", "yahoo"].tolist()
    print(f"\nUniverso invertible depurado ({len(invertibles)}):")
    print(invertibles)

    descartadas = df.loc[df["estatus"].isin(["RANCIA", "SIN_DATOS"]), "accitrade"].tolist()
    if descartadas:
        print(f"\nDescartadas ({len(descartadas)}): {', '.join(descartadas)}")

    return df


if __name__ == "__main__":
    validar_universo()