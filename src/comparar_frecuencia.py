# src/comparar_frecuencia.py
"""Contrasta D_alpha diario vs semanal: cuanto del beneficio es artefacto."""
import pandas as pd
from analisis_pares import PARES, descargar_par
from diversificacion import indice_diversificacion
from metricas import excedente_de_perdida, retornos_simples

ALPHAS = (0.90, 0.95, 0.99)
PESOS = [0.5, 0.5]

def main():
    filas = []
    for nombre, simbolos in PARES.items():
        precios = descargar_par(*simbolos)
        for etiqueta, regla in [("diaria", None), ("semanal", "W-FRI")]:
            p = precios if regla is None else precios.resample(regla).last().dropna()
            L = excedente_de_perdida(retornos_simples(p))
            fila = {"par": nombre, "frecuencia": etiqueta, "n": len(L)}
            for a in ALPHAS:
                fila[f"Drel_{a}"] = indice_diversificacion(L, PESOS, a)["D_relativo"]
            filas.append(fila)

    t = pd.DataFrame(filas)
    print(t.round(4).to_string(index=False))

    # Cuanto se encoge el beneficio al eliminar la atenuacion
    piv = t.pivot(index="par", columns="frecuencia", values="Drel_0.95")
    piv["contraccion"] = 1 - piv["semanal"] / piv["diaria"]
    print("\nContraccion del beneficio en alpha=0.95:")
    print(piv.round(4).to_string())

if __name__ == "__main__":
    main()