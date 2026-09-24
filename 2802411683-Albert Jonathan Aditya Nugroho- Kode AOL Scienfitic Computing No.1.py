import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("MATH6192031_AoL Data.csv", delimiter=';')
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Month'], format='%b-%y')
df = df.sort_values('Date').reset_index(drop=True)

y = df['Number of bags'].values
x = np.arange(len(y))

log_y = np.log(y)
A = np.vstack([np.ones_like(x), x]).T
params = np.linalg.lstsq(A, log_y, rcond=None)[0]
ln_a, ln_b = params[0], params[1]
a = np.exp(ln_a)
b = np.exp(ln_b)

y_pred = a * (b ** x)

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, 'o', label='Data Aktual', markersize=4, alpha=0.7)
plt.plot(df['Date'], y_pred, '-', label='Model Eksponensial y = a·b^x', color='red', linewidth=2)
plt.xlabel('Tanggal')
plt.ylabel('Produksi Tas')
plt.title('Eksponensial: y = a·b^x')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"y = {a:.4f} * ({b:.4f})^x")
print(f"R^2 = {r2:.4f}")

n_future = 60
x_future = np.arange(len(y), len(y) + n_future)
dates_future = pd.date_range(df['Date'].iloc[-1] + pd.offsets.MonthBegin(1), periods=n_future, freq='MS')
y_pred_future = a * (b ** x_future)

threshold = 25000
idx_above = np.where(y_pred_future >= threshold)[0]
