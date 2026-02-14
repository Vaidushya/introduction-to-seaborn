import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path="USA_Housing.csv"):
	if os.path.exists(path):
		try:
			return pd.read_csv(path)
		except Exception as e:
			print(f"Failed to read '{path}': {e}")
	print("Using seaborn's 'iris' dataset as fallback.")
	return sns.load_dataset('iris')


df = load_data()
print(df.head(10))

num = df.select_dtypes(include='number')
if num.empty:
	raise SystemExit("No numeric columns available for plotting.")

sns.pairplot(num)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(num.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation heatmap')
plt.show()
