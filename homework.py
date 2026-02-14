import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_tips(local='tips.csv'):
    if os.path.exists(local):
        for sep in (',', '\t', ';'):
            try:
                df = pd.read_csv(local, sep=sep)
                print(f"Loaded {local} using sep='{sep}'")
                return df
            except Exception:
                pass
        print(f"Could not read {local}; falling back to seaborn 'tips'.")
    return sns.load_dataset('tips')


def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


df = load_tips()
df.columns = [c.strip().lower() for c in df.columns]

tb = find_col(df, ['total_bill', 'total bill', 'totalbill', 'total'])
tip = find_col(df, ['tip', 'tips'])
size = find_col(df, ['size'])

print(df.head())
print(df.describe())

if tb and tip:
    x = pd.to_numeric(df[tb], errors='coerce')
    y = pd.to_numeric(df[tip], errors='coerce')
    plt.figure()
    sns.histplot(x.dropna(), kde=True, bins=20)
    plt.title('Total bill distribution')
    plt.show()

    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel('total_bill')
    plt.ylabel('tip')
    plt.title('Total bill vs tip')
    plt.show()

nums = df.select_dtypes(include='number')
corr_cols = [c for c in (tb, tip, size) if c and c in nums.columns]
if len(corr_cols) >= 2:
    print(df[corr_cols].corr())
else:
    print('Not enough numeric columns for correlation:', corr_cols)
