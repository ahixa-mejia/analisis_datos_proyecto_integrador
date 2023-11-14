import pandas as pd 
from datasets import load_dataset

dataset = load_dataset("mstz/heart_failure")
data = dataset ["train"]

df = pd.DataFrame(data)

total_mujeres = df[df['is_male'] == False].shape[0]
total_hombres = df[df['is_male'] == True].shape[0]

fumadores_mujeres = df[(df['is_male'] == False) & (df['is_smoker'] == True)].shape[0]
fumadores_hombres = df[(df['is_male'] == True) & (df['is_smoker'] == True)].shape[0]

print("\n =============================")
print(f"total de mujeres: {total_mujeres}, fumadoras: {fumadores_mujeres}")
print(f"total de hombres: {total_hombres}, fumadores: {fumadores_hombres}")
print("\n =============================")