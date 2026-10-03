"""
Metricas de riesgo y retorno, siguiendo la convencion de excedente de perdida
de Erdely (2017).

Convencion central
------------------
Erdely define el excedente de perdida como L := X - M(X), donde X es la variable
de perdida y M(X) su mediana. Trabajando con retornos R, la perdida es X = -R,
de modo que:

    L = -R - M(-R) = M(R) - R

Se usa la MEDIANA y no la media porque, como argumenta Erdely (seccion 2), la
mediana siempre existe para variables continuas y es robusta, mientras que la
media puede no existir o ser numericamente inestable bajo colas pesadas. Para
una Pareto con delta cercano a 1 se cumple E(X) > VaR_alpha(X) para cualquier
alpha < 1, lo que la vuelve inservible como estimador puntual.

Todas las funciones de VaR y TVaR reciben o construyen PERDIDAS, no retornos.
Confundir el signo es el error mas comun en este tipo de codigo.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DIAS_HABILES = 252


# --------------------------------------------------------------------------
# Retornos
# --------------------------------------------------------------------------

def retornos_log(precios):
    """Retornos logaritmicos diarios. Aditivos en el tiempo."""
    precios = pd.DataFrame(precios).astype(float)
    return np.log(precios / precios.shift(1)).dropna(how="all")


def retornos_simples(precios):
    """Retornos aritmeticos diarios. Aditivos en la seccion transversal.

    Para agregacion de portafolio (R_p = suma de w_i R_i) el retorno simple es
    el correcto; el logaritmico NO es aditivo entre activos. Para anualizar una
    sola serie, el logaritmico es el correcto. Usa cada uno donde corresponde.
    """
    precios = pd.DataFrame(precios).astype(float)
    return precios.pct_change().dropna(how="all")


# --------------------------------------------------------------------------
# Excedente de perdida (Erdely, Def. 2.1)
# --------------------------------------------------------------------------

def excedente_de_perdida(retornos):
    """L = M(R) - R, columna por columna.

    Nota importante para el caso de portafolio: al agregar se usa
    L_p = suma_i w_i L_i, cuya constante de centrado es suma_i w_i M(R_i) y NO
    M(R_p), porque la mediana no es aditiva. Esto replica la ecuacion (2) de
    Erdely, donde L = S - c con c = suma de las medianas individuales.
    """
    retornos = pd.DataFrame(retornos).astype(float)
    return retornos.median(axis=0) - retornos


def perdida_portafolio(perdidas, pesos):
    """Agrega excedentes de perdida individuales en un portafolio: L_p = L @ w."""
    perdidas = pd.DataFrame(perdidas)
    pesos = np.asarray(pesos, dtype=float)
    if pesos.shape[0] != perdidas.shape[1]:
        raise ValueError(
            f"Dimension incompatible: {perdidas.shape[1]} activos, {pesos.shape[0]} pesos."
        )
    return pd.Series(perdidas.to_numpy() @ pesos, index=perdidas.index)


# --------------------------------------------------------------------------
# Medidas de riesgo
# --------------------------------------------------------------------------

def var_historico(perdidas, alpha=0.95):
    """VaR_alpha(L) = F_L^{-1}(alpha), por simulacion historica.

    Sin supuesto distribucional. Es la estimacion preferible cuando hay
    dependencia en cola, precisamente porque no la destruye.
    """
    L = np.asarray(perdidas, dtype=float)
    L = L[~np.isnan(L)]
    if L.size == 0:
        return np.nan
    return float(np.quantile(L, alpha))


def var_parametrico(perdidas, alpha=0.95):
    """VaR_alpha bajo supuesto Normal. Se incluye solo como contraste.

    La brecha contra var_historico es evidencia directa de que la Normal
    subestima el riesgo agregado (McNeil et al., citado por Erdely en la
    seccion 4): margenes no normales y ceguera ante dependencia en cola.
    """
    from scipy.stats import norm

    L = np.asarray(perdidas, dtype=float)
    L = L[~np.isnan(L)]
    if L.size == 0:
        return np.nan
    return float(L.mean() + L.std(ddof=1) * norm.ppf(alpha))


def tvar_historico(perdidas, alpha=0.95):
    """TVaR_alpha(L) = E[L | L > VaR_alpha(L)]. Coherente, por tanto subaditiva.

    Se calcula para contrastarla con el VaR: por construccion su indice de
    diversificacion nunca es negativo, lo que ilustra el punto final de Erdely
    de que una medida coherente no puede detectar combinaciones perniciosas.
    """
    L = np.asarray(perdidas, dtype=float)
    L = L[~np.isnan(L)]
    if L.size == 0:
        return np.nan
    v = np.quantile(L, alpha)
    cola = L[L > v]
    return float(cola.mean()) if cola.size > 0 else float(v)


def maximo_drawdown(precios):
    """Caida maxima desde un maximo previo. Riesgo de trayectoria, no terminal."""
    p = pd.Series(precios).astype(float).dropna()
    if p.empty:
        return np.nan
    return float((p / p.cummax() - 1.0).min())


# --------------------------------------------------------------------------
# Retorno y riesgo anualizados
# --------------------------------------------------------------------------

def resumen_activos(precios, rf_anual, alpha=0.95, dias=DIAS_HABILES):
    """Tabla por activo: mu, sigma, Sharpe, VaR, TVaR y drawdown.

    rf_anual debe ser la tasa libre de riesgo EN LA MISMA MONEDA que los
    activos. Para instrumentos en pesos es CETES, no 4%.
    """
    precios = pd.DataFrame(precios).astype(float)
    r_log = retornos_log(precios)
    r_simple = retornos_simples(precios)
    perdidas = excedente_de_perdida(r_simple)

    mu = r_log.mean() * dias
    sigma = r_log.std(ddof=1) * np.sqrt(dias)
    sharpe = (mu - rf_anual) / sigma

    filas = []
    for col in precios.columns:
        filas.append({
            "activo": col,
            "mu_anual": float(mu[col]),
            "sigma_anual": float(sigma[col]),
            "sharpe": float(sharpe[col]),
            f"VaR_{alpha:.3f}_diario": var_historico(perdidas[col], alpha),
            f"TVaR_{alpha:.3f}_diario": tvar_historico(perdidas[col], alpha),
            "max_drawdown": maximo_drawdown(precios[col]),
            "n_obs": int(r_log[col].notna().sum()),
        })

    return pd.DataFrame(filas).set_index("activo").sort_values(
        "sigma_anual", ascending=False
    )


def metricas_portafolio(retornos, pesos, rf_anual, dias=DIAS_HABILES):
    """mu, sigma y Sharpe del portafolio bajo el enfoque media-varianza."""
    retornos = pd.DataFrame(retornos).astype(float)
    pesos = np.asarray(pesos, dtype=float)

    mu_vec = retornos.mean().to_numpy() * dias
    cov = retornos.cov().to_numpy() * dias

    mu_p = float(pesos @ mu_vec)
    sigma_p = float(np.sqrt(pesos @ cov @ pesos))
    return {
        "mu_anual": mu_p,
        "sigma_anual": sigma_p,
        "sharpe": (mu_p - rf_anual) / sigma_p if sigma_p > 0 else np.nan,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1500
    idx = pd.bdate_range("2020-01-01", periods=n)
    precios = pd.DataFrame({
        "TRANQUILA": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, n))),
        "VOLATIL":   100 * np.exp(np.cumsum(rng.normal(0.0002, 0.035, n))),
    }, index=idx)

    print("Resumen por activo (rf = 8% anual, proxy CETES):\n")
    print(resumen_activos(precios, rf_anual=0.08).round(4).to_string())

    r = retornos_simples(precios)
    print("\nPortafolio 50/50:")
    print({k: round(v, 4) for k, v in metricas_portafolio(r, [0.5, 0.5], 0.08).items()})