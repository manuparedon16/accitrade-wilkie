# src/reordenar_por_tope.py
"""Re-rankea el universo bajo el tope de 9,999,999 titulos por orden."""
import numpy as np, pandas as pd

TOPE = 9_999_999
K = 1_122_334_464

t = pd.read_csv("data/candidatos.csv")
t["capital_max"] = TOPE * t["precio_actual"]
t["w_max"] = np.minimum(1.0, t["capital_max"] / K)
t["sigma_efectiva"] = t["w_max"] * t["sigma_horizonte"]
t["mu_efectiva"] = t["w_max"] * t["mu_horizonte"]

# Probabilidad de superar el umbral c, contando el efectivo ocioso como 0%
from scipy.stats import norm
for c in (0.10, 0.20, 0.30):
    t[f"P(R>{c:.0%})"] = 1 - norm.cdf((c - t["mu_efectiva"]) / t["sigma_efectiva"].replace(0, np.nan))

cols = ["accitrade","precio_actual","w_max","mu_horizonte","sigma_horizonte",
        "sigma_efectiva","P(R>10%)","P(R>20%)","P(R>30%)","liquidez"]
print(t.nlargest(20, "sigma_efectiva")[cols].round(4).to_string(index=False))
print(f"\nPrecio minimo para desplegar 100% en una orden: ${K/TOPE:,.2f}")