# src/bootstrap_d.py
"""IC bootstrap para D_alpha. Sin esto, los puntos no son interpretables."""
import numpy as np, pandas as pd
from analisis_pares import PARES, descargar_par
from diversificacion import indice_diversificacion
from metricas import excedente_de_perdida, retornos_simples

ALPHAS = (0.90, 0.95)   # 0.99 se omite: solo 7.5 obs en diario
PESOS = [0.5, 0.5]
B = 1000

def ic_bootstrap(L, alpha, b=B, semilla=0):
    """Remuestreo por RENGLONES: preserva la dependencia entre activos."""
    rng = np.random.default_rng(semilla)
    n = len(L)
    muestras = [
        indice_diversificacion(L.iloc[rng.integers(0, n, n)], PESOS, alpha)["D_relativo"]
        for _ in range(b)
    ]
    return np.mean(muestras), np.std(muestras), np.percentile(muestras, [2.5, 97.5])

def main():
    filas = []
    for nombre, simbolos in PARES.items():
        L = excedente_de_perdida(retornos_simples(descargar_par(*simbolos)))
        for a in ALPHAS:
            punto = indice_diversificacion(L, PESOS, a)["D_relativo"]
            _, ee, (lo, hi) = ic_bootstrap(L, a)
            filas.append({"par": nombre, "alpha": a, "D_rel": punto,
                          "EE": ee, "IC_inf": lo, "IC_sup": hi,
                          "signif": lo > 0})
    print(pd.DataFrame(filas).round(4).to_string(index=False))

if __name__ == "__main__":
    main()