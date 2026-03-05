import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/diamonds.csv")

print(df.head())
print(df.info())

plt.figure(figsize=(8, 4))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Распределение цен на бриллианты")
plt.show()
