"""
Analisis de pares: contrasta un par casi comonotono contra pares que si
diversifican.

Por que hacen falta los controles
---------------------------------
Obtener D_alpha ~ 0 en AXTEL/CTAXTEL por si solo no prueba nada: podria ser que
el estimador siempre de valores chicos con datos reales. El resultado solo
significa algo si, con la MISMA metodologia y la MISMA ventana, otros pares si
producen D_alpha claramente positivo.

Metricas de dependencia reportadas
----------------------------------
tau de Kendall : dependencia de concordancia, invariante a transformaciones
                 monotonas crecientes. Para variables comonotonas vale
                 exactamente 1, mientras que la correlacion de Pearson NO
                 (Pearson solo llega a 1 bajo relacion lineal). Por eso tau es
                 la medida correcta para detectar comonotonicidad.
lambda_U       : dependencia en la cola superior. La Normal multivariada la
                 fija en cero, que es la ceguera que Erdely senala en la
                 seccion 4.

Salida: graficas/perfil_pares.png y data/perfil_pares.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import kendalltau

from diversificacion import indice_diversificacion, perfil_diversificacion, umbral_de_signo
from metricas import excedente_de_perdida, retornos_simples

DIR_GRAFICAS = Path("graficas")
DIR_DATOS = Path("data")

PARES = {
    "AXTEL + CTAXTEL (mismo grupo)": ("AXTELCPO.MX", "CTAXTELA.MX"),
    "AXTEL + IVVPESO (control)":     ("AXTELCPO.MX", "IVVPESOISHRS.MX"),
    "PENOLES + FIBRAMQ (control)":   ("PE&OLES.MX", "FIBRAMQ12.MX"),
    "GMEXICO + PENOLES (mineras)":   ("GMEXICOB.MX", "PE&OLES.MX"),
}

ALPHAS_CLAVE = (0.90, 0.95, 0.99)
PESOS = [0.5, 0.5]


def lambda_superior(u, v, q=0.95):
    """Estimador no parametrico de dependencia en cola superior.

        lambda_U(q) = P(V > q | U > q) = (1 - 2q + C_n(q,q)) / (1 - q)

    con C_n la copula empirica. Se evalua en q alto; el valor limite cuando
    q -> 1 es el coeficiente teorico, inalcanzable con muestra finita, asi que
    se reporta el estimador a un q fijo y comparable entre pares.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(u)
    # Pseudo-observaciones: rangos normalizados
    ru = pd.Series(u).rank().to_numpy() / (n + 1)
    rv = pd.Series(v).rank().to_numpy() / (n + 1)
    c_qq = float(np.mean((ru <= q) & (rv <= q)))
    return (1.0 - 2.0 * q + c_qq) / (1.0 - q)


def descargar_par(a, b, periodo="3y"):
    data = yf.download([a, b], period=periodo, auto_adjust=True,
                       progress=False, threads=True, group_by="column")
    cierres = data["Close"]
    if isinstance(cierres, pd.Series):
        raise RuntimeError(f"Solo se descargo una serie para {a}, {b}.")
    faltantes = [s for s in (a, b) if s not in cierres.columns]
    if faltantes:
        raise RuntimeError(f"Sin datos para: {faltantes}")
    return cierres[[a, b]].dropna()


def analizar_par(nombre, simbolos, pesos=PESOS):
    a, b = simbolos
    precios = descargar_par(a, b)
    retornos = retornos_simples(precios)
    perdidas = excedente_de_perdida(retornos)

    tau, p_tau = kendalltau(retornos[a], retornos[b])
    lam_u = lambda_superior(perdidas[a], perdidas[b], q=0.95)
    pearson = float(retornos[a].corr(retornos[b]))

    perfil = perfil_diversificacion(perdidas, pesos)
    perfil["par"] = nombre

    resumen = {
        "par": nombre,
        "n_obs": len(retornos),
        "pearson": pearson,
        "tau_kendall": float(tau),
        "p_valor_tau": float(p_tau),
        "lambda_U_q95": lam_u,
        "umbral_alpha": umbral_de_signo(perfil),
    }
    for a_ in ALPHAS_CLAVE:
        d_var = indice_diversificacion(perdidas, pesos, a_, "var")
        d_tvar = indice_diversificacion(perdidas, pesos, a_, "tvar")
        resumen[f"D_VaR_{a_}"] = d_var["D"]
        resumen[f"Drel_VaR_{a_}"] = d_var["D_relativo"]
        resumen[f"Drel_TVaR_{a_}"] = d_tvar["D_relativo"]

    return resumen, perfil


def graficar(perfiles, archivo="perfil_pares.png"):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    for nombre, perfil in perfiles.items():
        # Se grafica D relativo para que los pares sean comparables entre si
        rel_var = perfil["D_VaR"] / perfil["VaR_suma"]
        rel_tvar = perfil["D_TVaR"] / perfil["TVaR_suma"]
        ax1.plot(perfil["alpha"], rel_var, lw=2, label=nombre)
        ax2.plot(perfil["alpha"], rel_tvar, lw=2, label=nombre)

    for ax, titulo in ((ax1, "VaR (no subaditivo)"), (ax2, "TVaR (coherente)")):
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xlabel(r"nivel de confianza $\alpha$")
        ax.set_title(titulo)
        ax.grid(alpha=0.25)

    ax1.set_ylabel(r"$D_\alpha$ relativo")
    ax1.legend(fontsize=8)
    fig.suptitle(r"Beneficio de diversificacion $D_\alpha$, pares 50/50",
                 fontsize=13)
    fig.tight_layout()

    DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)
    destino = DIR_GRAFICAS / archivo
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destino


def main():
    resumenes, perfiles = [], {}

    for nombre, simbolos in PARES.items():
        print(f"Procesando: {nombre}")
        try:
            resumen, perfil = analizar_par(nombre, simbolos)
        except Exception as exc:
            print(f"  fallo: {exc}")
            continue
        resumenes.append(resumen)
        perfiles[nombre] = perfil

    if not resumenes:
        raise RuntimeError("Ningun par se pudo procesar.")

    tabla = pd.DataFrame(resumenes)

    print("\n" + "=" * 74)
    print("DEPENDENCIA")
    print("=" * 74)
    print(tabla[["par", "n_obs", "pearson", "tau_kendall",
                 "lambda_U_q95"]].round(4).to_string(index=False))

    print("\n" + "=" * 74)
    print("BENEFICIO DE DIVERSIFICACION RELATIVO (fraccion del capital ahorrada)")
    print("=" * 74)
    cols = ["par"] + [f"Drel_VaR_{a}" for a in ALPHAS_CLAVE] \
                   + [f"Drel_TVaR_{a}" for a in ALPHAS_CLAVE]
    print(tabla[cols].round(4).to_string(index=False))

    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(DIR_DATOS / "perfil_pares.csv", index=False, encoding="utf-8")
    pd.concat(perfiles.values()).to_csv(
        DIR_DATOS / "perfil_pares_completo.csv", index=False, encoding="utf-8")

    destino = graficar(perfiles)
    print(f"\nGrafica: {destino.resolve()}")
    print(f"Tablas : {(DIR_DATOS / 'perfil_pares.csv').resolve()}")

    return tabla


if __name__ == "__main__":
    main()