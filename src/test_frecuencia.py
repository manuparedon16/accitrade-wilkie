# src/test_frecuencia.py
"""Verifica si la dependencia AXTEL/CTAXTEL esta atenuada por no sincronia."""
import pandas as pd
from scipy.stats import kendalltau
from analisis_pares import descargar_par, lambda_superior
from metricas import excedente_de_perdida, retornos_simples

PAR = ("AXTELCPO.MX", "CTAXTELA.MX")

def main():
    precios = descargar_par(*PAR)
    for etiqueta, regla in [("diaria", None), ("semanal", "W-FRI"), ("quincenal", "2W-FRI")]:
        p = precios if regla is None else precios.resample(regla).last().dropna()
        r = retornos_simples(p)
        if len(r) < 40:
            continue
        tau, pv = kendalltau(r.iloc[:, 0], r.iloc[:, 1])
        L = excedente_de_perdida(r)
        lam = lambda_superior(L.iloc[:, 0], L.iloc[:, 1], q=0.90)
        planos = float((r.iloc[:, 0] == 0).mean())
        print(f"{etiqueta:<10} n={len(r):>4}  tau={tau:6.4f}  p={pv:.2e}  "
              f"lambda_U={lam:6.4f}  planos={planos:.3f}")

if __name__ == "__main__":
    main()