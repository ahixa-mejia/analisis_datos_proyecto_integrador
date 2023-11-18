import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("datos_categorizados.csv")
df=pd.DataFrame(df)

x = ["si","no"]

y1 = df['anaemia'].value_counts()
y2 = df['diabetes'].value_counts()
y3 = df['smoking'].value_counts()
y4 = df['DEATH_EVENT'].value_counts()

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
axes[0].pie(y1, labels=x, autopct="%1.1f%%",startangle=90)
axes[0].set_title("Anemicos")

axes[1].pie(y2, labels=x, autopct="%1.1f%%",startangle=90)
axes[1].set_title("Diabeticos")

axes[2].pie(y3, labels=x, autopct="%1.1f%%",startangle=90)
axes[2].set_title("Fumadores")

axes[3].pie(y4, labels=x, autopct="%1.1f%%",startangle=90)
axes[3].set_title("Muertes")

plt.tight_layout()
plt.show()