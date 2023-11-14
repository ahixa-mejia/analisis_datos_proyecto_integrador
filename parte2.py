import numpy as np
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("mstz/heart_failure")
data = dataset ["train"]

df = pd.DataFrame(data)

pacientes_fallecidos = df[df["is_dead"] == 1]
pacientes_no_fallecidos = df[df["is_dead"] == 0]

prom_edad_pacientes_fallecidos = round(pacientes_fallecidos["age"].mean())
prom_edad_pacientes_no_fallecidos = round(pacientes_no_fallecidos["age"].mean())

print("\n =============================")
print(f"Promedio de edad de pacientes fallecidos: {int(prom_edad_pacientes_fallecidos)} años")
print("\n =============================")
print(f"Promedio de edad de pacientes No fallecidos: {int(prom_edad_pacientes_no_fallecidos)} años")
print("\n =============================")
