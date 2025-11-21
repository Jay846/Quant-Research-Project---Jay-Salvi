import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('data.parquet')

print(df.info())
print(df.describe())

missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing_Count': missing_count,
    'Missing_Percentage': missing_pct
})

print("Missing Values Summary:")
print(missing_summary)

def analyze_missing_pattern(series, name):
    missing_streaks = (series.isnull() != series.isnull().shift()).cumsum()
    streak_lengths = series.groupby(missing_streaks).size()
    missing_streaks = streak_lengths[series.groupby(missing_streaks).apply(lambda x: x.isnull().all())]
    
    print(f"\n{name} Missing Pattern:")
    print(f"Total missing: {series.isnull().sum()}")
    print(f"Longest streak: {missing_streaks.max() if len(missing_streaks)>0 else 0}")
    print(f"Average streak: {missing_streaks.mean():.1f}" if len(missing_streaks)>0 else " - Average streak: 0")
    return missing_streaks

banknifty_streaks = analyze_missing_pattern(df['banknifty'], 'BankNifty IV')
nifty_streaks = analyze_missing_pattern(df['nifty'], 'Nifty IV')


# Visualisation
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(df.index, df['banknifty'], alpha=0.7, label='BankNifty IV')
plt.plot(df.index, df['nifty'], alpha=0.7, label='Nifty IV')
plt.title('Overall Implied Volatility Trend')
plt.ylabel('IV Level')
plt.legend()

plt.subplot(2, 2, 2)
plt.hist(df['banknifty'].dropna(), bins=50, alpha=0.7, label='BankNifty IV')
plt.hist(df['nifty'].dropna(), bins=50, alpha=0.7, label='Nifty IV')
plt.title('IV Distribution')
plt.legend()

plt.subplot(2, 2, 3)
sample = df.loc['2021-03-01':'2021-03-07']
plt.plot(sample.index, sample['banknifty'], label='BankNifty IV')
plt.plot(sample.index, sample['nifty'], label='Nifty IV')
plt.title('Sample Week (Mar 1–7, 2021)')
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
