import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("MATH6192031_AoL Data.csv", delimiter=';')
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Month'], format='%b-%y')
df = df.sort_values('Date').reset_index(drop=True)

y = df['Number of bags'].values
x = np.arange(len(y))
threshold = 25000
n_future = 36

# Taylor Orde 4
log_y = np.log(y)
A = np.vstack([np.ones_like(x), x]).T
ln_a, ln_b = np.linalg.lstsq(A, log_y, rcond=None)[0]
a, k = np.exp(ln_a), ln_b

y_taylor = a * (1 + k*x + (k**2)*x**2/2 + (k**3)*x**3/6 + (k**4)*x**4/24)
r2_taylor = 1 - np.sum((y - y_taylor)**2) / np.sum((y - np.mean(y))**2)

x_future = np.arange(len(y), len(y) + n_future)
dates_future = pd.date_range(df['Date'].iloc[-1] + pd.offsets.MonthBegin(1), periods=n_future, freq='MS')
y_taylor_future = a * (1 + k*x_future + (k**2)*x_future**2/2 + (k**3)*x_future**3/6 + (k**4)*x_future**4/24)

idx_taylor = np.where(y_taylor_future >= threshold)[0]
if len(idx_taylor) > 0:
    t_pred = dates_future[idx_taylor[0]]
    t_start = t_pred - pd.DateOffset(months=13)
    print(f"[Taylor] >25.000: {t_pred.strftime('%B %Y')}, pembangunan: {t_start.strftime('%B %Y')}")
else:
    print("[TAYLOR] Tidak melebihi 25.000 dalam 3 tahun ke depan.")

# Eksponensial
y_exp = a * (np.exp(k) ** x)
r2_exp = 1 - np.sum((y - y_exp)**2) / np.sum((y - np.mean(y))**2)
y_exp_future = a * (np.exp(k) ** x_future)

idx_exp = np.where(y_exp_future >= threshold)[0]
if len(idx_exp) > 0:
    t_pred_exp = dates_future[idx_exp[0]]
    t_start_exp = t_pred_exp - pd.DateOffset(months=13)
    print(f"[Eksponensiall] >25.000: {t_pred_exp.strftime('%B %Y')}, pembangunan: {t_start_exp.strftime('%B %Y')}")
else:
    print("[EKSPONENSIAL] Tidak melebihi 25.000 dalam 3 tahun ke depan.")

print(f"R^2 Taylor: {r2_taylor:.3f}")
print(f"R^2 Eksponensial: {r2_exp:.3f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, 'o', color='blue', label='Data Aktual', markersize=4, alpha=0.7)
plt.plot(df['Date'], y_taylor, '-', color='red', label='Taylor Orde 4 (Data Asli)', linewidth=2)
plt.plot(dates_future, y_taylor_future, '--', color='green', label='Taylor Orde 4 (Prediksi)', linewidth=2)
plt.plot(df['Date'], y_exp, '-', color='orange', label='Eksponensial (Data Asli)', linewidth=2)
plt.plot(dates_future, y_exp_future, '--', color='purple', label='Eksponensial (Prediksi)', linewidth=2)
plt.axhline(threshold, color='red', linestyle=':', label='Batas 25.000')
plt.xlabel('Tanggal')
plt.ylabel('Produksi Tas')
plt.title('Prediksi 3 Tahun ke Depan (Taylor & Eksponensial)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()