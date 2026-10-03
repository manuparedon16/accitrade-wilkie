"""
Depura el universo por LIQUIDEZ REAL y produce la tabla de candidatos.

Motivo
------
validar_universo.py responde "existe y tiene dato reciente". Eso no basta.
Yahoo arrastra el ultimo precio de las emisoras sin operacion: publica una barra
diaria con cierre identico y volumen cero. La serie luce viva pero son retornos
cero, y eso sesga sigma A LA BAJA.

El sesgo es grave para una estrategia de torneo, donde el activo objetivo es
precisamente el de mayor sigma: una emisora ilquida se veria tranquila cuando en
realidad salta con violencia cada vez que opera.

El discriminante es la proporcion de dias con retorno exactamente cero.

Salida: data/candidatos.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from metricas import excedente_de_perdida, retornos_log, retornos_simples, var_historico

DIR_DATOS = Path("data")

# Dias habiles del horizonte: 05-sep-2026 a 05-dic-2026
DIAS_HORIZONTE = 63
DIAS_ANUALES = 252

# Umbrales de dias planos (retorno diario exactamente cero)
LIQUIDA_MAX = 0.05
MEDIA_MAX = 0.20

# Emisoras que no reconozco y conviene identificar antes de operar.
# No las descarto: las marco.
POR_IDENTIFICAR = {
    "AGUILAS.CPO", "ANGELD.10", "DIABLOI.10", "DIABLOS.A", "DIABLOS.O",
    "PLANI", "LASEG", "GPH.1", "PROCORP.B", "ALTERNA.B",
}


def clasificar_liquidez(pct_planos):
    if pd.isna(pct_planos):
        return "SIN_MEDIR"
    if pct_planos < LIQUIDA_MAX:
        return "LIQUIDA"
    if pct_planos < MEDIA_MAX:
        return "MEDIA"
    return "ILIQUIDA"


def cargar_mapeo(ruta=None):
    ruta = DIR_DATOS / "mapeo_tickers.csv" if ruta is None else Path(ruta)
    df = pd.read_csv(ruta)
    df = df[df["estatus"] == "OK"].copy()
    df["liquidez"] = df["pct_dias_planos"].apply(clasificar_liquidez)
    return df


def descargar_precios(simbolos, periodo="3y", tam_lote=40):
    marcos = []
    simbolos = list(dict.fromkeys(simbolos))
    for i in range(0, len(simbolos), tam_lote):
        lote = simbolos[i:i + tam_lote]
        data = yf.download(lote, period=periodo, auto_adjust=True,
                           progress=False, threads=True, group_by="column")
        if data is None or data.empty:
            continue
        cierres = data["Close"]
        if isinstance(cierres, pd.Series):
            cierres = cierres.to_frame(name=lote[0])
        marcos.append(cierres)
    if not marcos:
        raise RuntimeError("No se descargo ningun precio.")
    return pd.concat(marcos, axis=1).sort_index()


def tabla_candidatos(precios, mapeo, rf_anual, alpha=0.95):
    """Metricas por activo, escaladas al horizonte del torneo."""
    r_log = retornos_log(precios)
    r_simple = retornos_simples(precios)
    perdidas = excedente_de_perdida(r_simple)

    escala = np.sqrt(DIAS_HORIZONTE)
    filas = []

    for col in precios.columns:
        serie = precios[col].dropna()
        rl = r_log[col].dropna()
        if len(rl) < 100:
            continue

        mu_d = float(rl.mean())
        sd_d = float(rl.std(ddof=1))
        planos = float((rl == 0).mean())

        # Estimador robusto de escala, inmune a los ceros de arrastre:
        # sigma ~ IQR / 1.349 bajo normalidad, calculado solo sobre dias operados
        rl_activos = rl[rl != 0]
        iqr = float(rl_activos.quantile(0.75) - rl_activos.quantile(0.25)) if len(rl_activos) > 30 else np.nan
        sd_robusta = iqr / 1.349 if not np.isnan(iqr) else np.nan

        filas.append({
            "yahoo": col,
            "mu_anual": mu_d * DIAS_ANUALES,
            "sigma_anual": sd_d * np.sqrt(DIAS_ANUALES),
            "sigma_anual_robusta": sd_robusta * np.sqrt(DIAS_ANUALES) if not np.isnan(sd_robusta) else np.nan,
            "mu_horizonte": mu_d * DIAS_HORIZONTE,
            "sigma_horizonte": sd_d * escala,
            "sharpe": (mu_d * DIAS_ANUALES - rf_anual) / (sd_d * np.sqrt(DIAS_ANUALES)),
            "pct_dias_planos": planos,
            "VaR95_diario": var_historico(perdidas[col], alpha),
            "precio_actual": float(serie.iloc[-1]),
            "n_obs": int(len(rl)),
        })

    tabla = pd.DataFrame(filas)
    tabla = tabla.merge(mapeo[["accitrade", "yahoo", "liquidez"]], on="yahoo", how="left")
    tabla["por_identificar"] = tabla["accitrade"].isin(POR_IDENTIFICAR)

    # Brecha entre sigma cruda y robusta: si la robusta es mucho mayor, los
    # ceros de arrastre estaban ocultando el riesgo verdadero.
    tabla["sesgo_ceros"] = tabla["sigma_anual_robusta"] / tabla["sigma_anual"]

    return tabla


def main(rf_anual=0.075):
    mapeo = cargar_mapeo()

    print("Distribucion de liquidez sobre las claves con estatus OK:")
    print(mapeo["liquidez"].value_counts().to_string())

    operables = mapeo[mapeo["liquidez"].isin(["LIQUIDA", "MEDIA"])]
    print(f"\nOperables: {len(operables)} de {len(mapeo)}")
    print(f"Descartadas por iliquidez: {len(mapeo) - len(operables)}")

    simbolos = operables["yahoo"].dropna().tolist()
    print(f"\nDescargando {len(simbolos)} series...")
    precios = descargar_precios(simbolos)

    tabla = tabla_candidatos(precios, mapeo, rf_anual)

    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    destino = DIR_DATOS / "candidatos.csv"
    tabla.sort_values("sigma_horizonte", ascending=False).to_csv(
        destino, index=False, encoding="utf-8"
    )

    cols = ["accitrade", "mu_horizonte", "sigma_horizonte", "sharpe",
            "pct_dias_planos", "sesgo_ceros", "liquidez", "por_identificar"]

    print("\n" + "=" * 70)
    print("TOP 15 POR VOLATILIDAD AL HORIZONTE (estrategia de torneo)")
    print("=" * 70)
    print(tabla.nlargest(15, "sigma_horizonte")[cols].round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 15 POR SHARPE (estrategia de exencion por umbral bajo)")
    print("=" * 70)
    print(tabla.nlargest(15, "sharpe")[cols].round(4).to_string(index=False))

    sesgadas = tabla[tabla["sesgo_ceros"] > 1.30].sort_values("sesgo_ceros", ascending=False)
    if not sesgadas.empty:
        print("\n" + "=" * 70)
        print("ALERTA: sigma subestimada por dias de arrastre (robusta / cruda > 1.30)")
        print("=" * 70)
        print(sesgadas[["accitrade", "sigma_anual", "sigma_anual_robusta",
                        "sesgo_ceros", "pct_dias_planos"]].round(4).to_string(index=False))

    print(f"\nGuardado en {destino.resolve()}")
    return tabla


if __name__ == "__main__":
    main()