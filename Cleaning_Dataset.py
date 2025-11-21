import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df_raw = pd.read_parquet('data.parquet')
df_raw.index = pd.to_datetime(df_raw.index)

print(f"Raw data: {len(df_raw)} rows, {df_raw.isnull().sum().sum()} missing values")


df = df_raw.copy()
df['trading_day'] = df.index.date


df['banknifty'] = df.groupby('trading_day')['banknifty'].transform(
    lambda x: x.interpolate(method='linear', limit_area='inside')
)
df['nifty'] = df.groupby('trading_day')['nifty'].transform(
    lambda x: x.interpolate(method='linear', limit_area='inside')
)


df['banknifty'] = df.groupby('trading_day')['banknifty'].transform(lambda x: x.ffill().bfill())
df['nifty'] = df.groupby('trading_day')['nifty'].transform(lambda x: x.ffill().bfill())
df = df.between_time('09:15', '15:30')


data = df[['banknifty', 'nifty']].resample('10T').last().dropna()
data = data.rename(columns={'banknifty': 'banknifty_iv', 'nifty': 'nifty_iv'})

print(f"Clean dataset ready: {len(data):,} 10-min bars")
print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
print(f"Final missing values: {data.isnull().sum().sum()}")

# Visualisation
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(data['banknifty_iv'][:1000], label='BankNifty IV', alpha=0.8)
plt.plot(data['nifty_iv'][:1000], label='Nifty IV', alpha=0.8)
plt.title('First 1000 Bars – Clean 10-min Data')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(data['banknifty_iv'][-1000:], label='BankNifty IV', alpha=0.8)
plt.plot(data['nifty_iv'][-1000:], label='Nifty IV', alpha=0.8)
plt.title('Last 1000 Bars')
plt.legend()

plt.subplot(2, 2, 4)
missing_dates = df[df.isnull().any(axis=1)].index
plt.plot(df.index, [1] * len(df), 'g-', alpha=0.3, label='Data Present')
plt.plot(missing_dates, [1] * len(missing_dates), 'ro', alpha=0.5, markersize=2, label='Missing')
plt.title('Data Points')
plt.legend()
plt.yticks([])

plt.tight_layout()
plt.show()
