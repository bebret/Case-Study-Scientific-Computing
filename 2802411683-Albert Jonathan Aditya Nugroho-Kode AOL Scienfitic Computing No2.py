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
k = ln_b

y_taylor = a * (
    1
    + k * x
    + (k**2) * x**2 / 2
    + (k**3) * x**3 / 6
    + (k**4) * x**4 / 24
)

ss_res_taylor = np.sum((y - y_taylor) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_taylor = 1 - (ss_res_taylor / ss_tot)

print(f"y ~ {a:.3f} * (1 + {k:.3f}x + {k**2/2:.3f}x^2 + {k**3/6:.3f}x^3 + {k**4/24:.3f}x^4)")
print("R^2 Taylor Orde 4:", f"{r2_taylor:.3f}")

n_future = 3
x_future = np.arange(len(y), len(y) + n_future)
dates_future = pd.date_range(df['Date'].iloc[-1] + pd.offsets.MonthBegin(1), periods=n_future, freq='MS')
y_taylor_future = a * (
    1
    + k * x_future
    + (k**2) * x_future**2 / 2
    + (k**3) * x_future**3 / 6
    + (k**4) * x_future**4 / 24
)

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, 'o', color='blue', label='Data Aktual', markersize=4, alpha=0.7)
plt.plot(df['Date'], y_taylor, '-', color='red', label='Taylor Orde 4 (Data Asli)', linewidth=2)
plt.plot(dates_future, y_taylor_future, '--', color='green', label='Taylor Orde 4 (Prediksi)', linewidth=2)
plt.xlabel('Tanggal')
plt.ylabel('Produksi Tas')
plt.title('Prediksi 3 Tahun ke Depan (Taylor Orde 4)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()