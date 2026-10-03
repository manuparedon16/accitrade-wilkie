"""
Indice de beneficio de diversificacion, siguiendo Erdely (2017).

Definicion
----------
Para pesos w > 0 y excedentes de perdida L_1, ..., L_n:

    D_alpha(w) = suma_i VaR_alpha(w_i L_i) - VaR_alpha(suma_i w_i L_i)

Como el VaR es positivamente homogeneo (Erdely, Prop. 3.1c), VaR(w_i L_i) =
w_i VaR(L_i) para w_i > 0, y el primer termino se reduce a w . VaR_ind.

Interpretacion
--------------
    D_alpha > 0   la diversificacion reduce el capital de riesgo requerido
    D_alpha = 0   caso comonotono (Teorema 5.1): beneficio exactamente nulo
    D_alpha < 0   SUPERADITIVIDAD: la combinacion es perniciosa

El tercer caso es el que importa. Erdely argumenta en las observaciones finales
que la no subaditividad del VaR es una ventaja y no un defecto: cuando el VaR de
la suma excede la suma de los VaR individuales, se esta detectando una
combinacion de riesgos especialmente danina. Bajo una medida coherente como el
TVaR, la subaditividad es axiomatica y D_alpha >= 0 SIEMPRE, por lo que esa
combinacion resulta indetectable.

Por eso este modulo calcula ambos indices en paralelo: el contraste es la
evidencia.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from metricas import (
    excedente_de_perdida,
    perdida_portafolio,
    retornos_simples,
    tvar_historico,
    var_historico,
)

DIR_GRAFICAS = Path("graficas")


# --------------------------------------------------------------------------
# Nucleo: opera directamente sobre perdidas
# --------------------------------------------------------------------------

def indice_diversificacion(perdidas, pesos, alpha=0.95, medida="var"):
    """D_alpha(w) para una matriz de excedentes de perdida.

    perdidas : DataFrame (n_obs x n_activos) de excedentes de perdida L_i
    pesos    : vector w >= 0. No se exige que sume 1: la homogeneidad positiva
                del VaR hace valida la formula para cualquier w > 0, lo que
                permite replicar el caso agregado S = X + Y del paper con w=(1,1).
    """
    f = var_historico if medida == "var" else tvar_historico

    perdidas = pd.DataFrame(perdidas)
    pesos = np.asarray(pesos, dtype=float)
    if np.any(pesos < 0):
        raise ValueError("La homogeneidad positiva exige pesos no negativos.")

    riesgo_individual = np.array([f(perdidas[c], alpha) for c in perdidas.columns])
    suma_individual = float(pesos @ riesgo_individual)
    riesgo_agregado = float(f(perdida_portafolio(perdidas, pesos), alpha))

    return {
        "alpha": alpha,
        "medida": medida.upper(),
        "suma_individual": suma_individual,
        "agregado": riesgo_agregado,
        "D": suma_individual - riesgo_agregado,
        "D_relativo": (suma_individual - riesgo_agregado) / suma_individual
        if suma_individual != 0 else np.nan,
    }


def perfil_diversificacion(perdidas, pesos, alphas=None):
    """Barre alpha y devuelve D_alpha para VaR y TVaR lado a lado.

    Si D_VaR cambia de signo dentro del rango, existe un umbral de alpha a
    partir del cual conviene o deja de convenir diversificar. Ese es el
    fenomeno del Ejemplo 6.3 de Erdely.
    """
    if alphas is None:
        alphas = np.round(np.arange(0.50, 0.996, 0.005), 4)

    filas = []
    for a in alphas:
        dv = indice_diversificacion(perdidas, pesos, a, medida="var")
        dt = indice_diversificacion(perdidas, pesos, a, medida="tvar")
        filas.append({
            "alpha": a,
            "VaR_suma": dv["suma_individual"],
            "VaR_agregado": dv["agregado"],
            "D_VaR": dv["D"],
            "TVaR_suma": dt["suma_individual"],
            "TVaR_agregado": dt["agregado"],
            "D_TVaR": dt["D"],
        })
    return pd.DataFrame(filas)


def umbral_de_signo(perfil):
    """Localiza el alpha donde D_VaR cruza cero, por interpolacion lineal."""
    a = np.asarray(perfil["alpha"], dtype=float)
    d = np.asarray(perfil["D_VaR"], dtype=float)
    cruces = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if cruces.size == 0:
        return None
    i = cruces[0]
    return float(a[i] - d[i] * (a[i + 1] - a[i]) / (d[i + 1] - d[i]))


# --------------------------------------------------------------------------
# Envoltura para trabajar desde precios
# --------------------------------------------------------------------------

def perfil_desde_precios(precios, pesos, alphas=None):
    perdidas = excedente_de_perdida(retornos_simples(precios))
    return perfil_diversificacion(perdidas, pesos, alphas)


def graficar_perfil(perfil, titulo="Beneficio de diversificacion", archivo=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(perfil["alpha"], perfil["D_VaR"], lw=2, label=r"$D_\alpha$ con VaR")
    ax.plot(perfil["alpha"], perfil["D_TVaR"], lw=2, ls="--",
            label=r"$D_\alpha$ con TVaR (coherente)")
    ax.axhline(0.0, color="black", lw=1)

    u = umbral_de_signo(perfil)
    if u is not None:
        ax.axvline(u, color="crimson", ls=":", lw=1.5)
        ax.annotate(f"umbral $\\alpha \\approx$ {u:.4f}", xy=(u, 0),
                    xytext=(8, 22), textcoords="offset points", color="crimson")

    ax.fill_between(perfil["alpha"], perfil["D_VaR"], 0,
                    where=(perfil["D_VaR"] < 0), alpha=0.18, color="crimson")
    ax.set_xlabel(r"nivel de confianza $\alpha$")
    ax.set_ylabel(r"$D_\alpha = \sum_i w_i\,\rho(L_i) - \rho(\sum_i w_i L_i)$")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=0.25)

    DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)
    destino = DIR_GRAFICAS / (archivo or "perfil_diversificacion.png")
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destino


# --------------------------------------------------------------------------
# Comprobacion: Ejemplo 6.3 de Erdely (2017)
# --------------------------------------------------------------------------

def comprobar_ejemplo_63(n=2_000_000, semilla=7):
    """Replica el Ejemplo 6.3: X, Y iid Exponencial(1), S = X + Y ~ Gamma(2,1).

    Resultado analitico del paper:
        g(alpha) = 1 - (1-alpha)^2 [1 - 2 log(1-alpha)]
        g(alpha) = alpha  si y solo si  alpha ~ 0.7153319

        alpha < 0.7153319 :  VaR(X) + VaR(Y) < VaR(X+Y)  ->  D_alpha < 0
        alpha > 0.7153319 :  VaR(X) + VaR(Y) > VaR(X+Y)  ->  D_alpha > 0

    Si la simulacion recupera ese umbral, el estimador de D_alpha esta bien.
    """
    from scipy.optimize import brentq

    def g(a):
        return 1.0 - (1.0 - a) ** 2 * (1.0 - 2.0 * np.log(1.0 - a)) - a

    umbral_teorico = brentq(g, 0.50, 0.999)

    rng = np.random.default_rng(semilla)
    perdidas = pd.DataFrame({
        "X": rng.exponential(1.0, n),
        "Y": rng.exponential(1.0, n),
    })

    alphas = np.round(np.arange(0.60, 0.90, 0.0025), 5)
    perfil = perfil_diversificacion(perdidas, pesos=[1.0, 1.0], alphas=alphas)
    umbral_sim = umbral_de_signo(perfil)

    print("Comprobacion contra el Ejemplo 6.3 de Erdely (2017)")
    print("-" * 52)
    print(f"  umbral analitico    : {umbral_teorico:.7f}")
    print(f"  umbral simulado     : {umbral_sim:.7f}")
    print(f"  error absoluto      : {abs(umbral_sim - umbral_teorico):.2e}")

    for a in (0.65, 0.7153319, 0.80):
        d = indice_diversificacion(perdidas, [1.0, 1.0], a)["D"]
        signo = "D<0 (no conviene)" if d < 0 else "D>0 (si conviene)"
        print(f"  alpha={a:<9.7f} D={d:+.5f}   {signo}")

    ok = abs(umbral_sim - umbral_teorico) < 5e-3
    print("-" * 52)
    print("RESULTADO:", "coincide" if ok else "NO coincide, revisar")
    return ok


if __name__ == "__main__":
    comprobar_ejemplo_63()

    print("\n\nCaso comonotono (Teorema 5.1): Y = g(X) estrictamente creciente")
    print("-" * 52)
    rng = np.random.default_rng(11)
    x = rng.exponential(1.0, 400_000)
    comon = pd.DataFrame({"X": x, "Y": x ** 2})     # g(x) = x^2, creciente en x>0
    for a in (0.90, 0.95, 0.99):
        r = indice_diversificacion(comon, [1.0, 1.0], a)
        print(f"  alpha={a:.2f}  suma={r['suma_individual']:9.4f}  "
            f"agregado={r['agregado']:9.4f}  D={r['D']:+.2e}")
    print("  -> D ~ 0: sin beneficio de diversificacion, como predice el Thm 5.1")