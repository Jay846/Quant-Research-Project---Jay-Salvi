import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

df = pd.read_csv('implied_volatility_data.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df = df.rename(columns={'banknifty': 'banknifty_iv', 'nifty': 'nifty_iv'})
df = df.between_time('09:15', '15:30')
data = df[['banknifty_iv', 'nifty_iv', 'tte']].resample('10min').last().dropna()
print(f"Dataset ready: {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")



COST_PER_IV_POINT_ROUNDTRIP = 0.35


def base_model_raw_spread(df):
    df = df.copy()
    df['spread'] = df['banknifty_iv'] - df['nifty_iv']
    df['beta'] = np.nan  
    return df

def run_base_strategy(df, window_min, entry_z, exit_z):
    df = df.copy()
    periods = max(20, int(window_min / 10))
    min_p = int(periods * 0.7)
    
    df['mean'] = df['spread'].rolling(periods, min_periods=min_p).mean()
    df['std']  = df['spread'].rolling(periods, min_periods=min_p).std()
    df['z'] = (df['spread'] - df['mean']) / df['std']
    
   
    position = 0
    positions = []
    for i in range(len(df)):
        z_val = df['z'].iloc[i]
        if np.isnan(z_val):
            positions.append(position)
            continue
        if z_val > entry_z and position == 0:
            position = -1
        elif z_val < -entry_z and position == 0:
            position = 1
        elif abs(z_val) < exit_z and position != 0:
            position = 0
        positions.append(position)
    df['position'] = positions
    
    
    df['pnl_gross'] = df['position'].shift(1) * df['spread'].diff() * (df['tte'] ** 0.7)
    turnover = df['position'].diff().abs()
    spread_move = df['spread'].diff().abs() * turnover
    df['tc'] = spread_move * (df['tte'] ** 0.7) * COST_PER_IV_POINT_ROUNDTRIP
    df['tc'] = df['tc'].fillna(0)
    df['pnl_net'] = df['pnl_gross'] - df['tc']
    df['cum_pnl'] = df['pnl_net'].cumsum()
    
    # Metrics
    net_pnl = df['cum_pnl'].iloc[-1]
    max_dd = (df['cum_pnl'].cummax() - df['cum_pnl']).max()
    returns = df['pnl_net']
    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 37.5) if returns.std() > 0 else 0
    total_trades = int(turnover.sum() / 2)
    
    return {
        'window': window_min,
        'entry_z': entry_z,
        'exit_z': exit_z,
        'sharpe': round(sharpe, 4),
        'net_pnl': round(net_pnl, 3),
        'max_dd': round(max_dd, 3),
        'trades': total_trades,
        'df': df.dropna(subset=['z'])
    }

# Grid search 
windows = [30, 60, 375, 750, 1125, 1500] 
entries = [4.0, 5.0, 6.0, 7.0, 8.0]
exits   = [1.0, 2.0, 3.0]

results_base = []
df_base = base_model_raw_spread(data)

print("Running Base Model grid search")
for w in windows:
    for e in entries:
        for x in exits:
            if x >= e: continue
            res = run_base_strategy(df_base, window_min=w, entry_z=e, exit_z=x)
            results_base.append(res)

base_results_df = pd.DataFrame([{
    'Window': r['window'],
    'Entry_Z': r['entry_z'],
    'Exit_Z': r['exit_z'],
    'Sharpe': r['sharpe'],
    'Net_PnL': r['net_pnl'],
    'Max_DD': r['max_dd'],
    'Trades': r['trades']
} for r in results_base])

print("\nBASE MODEL – Top 5 Configurations")
print(base_results_df.sort_values('Sharpe', ascending=False).head(5).to_string(index=False))


best_base = base_results_df.loc[base_results_df['Sharpe'].idxmax()]
best_base_run = next(r for r in results_base 
                     if r['window'] == best_base['Window'] and 
                        r['entry_z'] == best_base['Entry_Z'] and 
                        r['exit_z'] == best_base['Exit_Z'])
df_plot_base = best_base_run['df']

print(f"\nBest Base Model: Window {int(best_base['Window'])} min | Entry ±{best_base['Entry_Z']:.1f}σ | Exit ±{best_base['Exit_Z']:.1f}σ")
print(f"Sharpe: {best_base['Sharpe']:.4f} | Net P&L: {best_base['Net_PnL']:.3f} | Trades: {best_base['Trades']}")

# Visualisation
plt.figure(figsize=(15, 10))

plt.subplot(3,1,2)
plt.plot(df_plot_base.index, df_plot_base['z'], color='green', alpha=0.8)
plt.axhline(best_base['Entry_Z'], color='red', linestyle='--', label='Entry Threshold')
plt.axhline(-best_base['Entry_Z'], color='red', linestyle='--')
plt.axhline(best_base['Exit_Z'], color='orange', linestyle='--', label='Exit Threshold')
plt.axhline(-best_base['Exit_Z'], color='orange', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.title('Z-Score on Raw Spread')
plt.ylabel('Z-Score')

plt.subplot(3,1,3)
plt.plot(df_plot_base.index, df_plot_base['cum_pnl'], color='purple', linewidth=1.5)
plt.title(f'Base Model Equity Curve – Net P&L = {best_base["Net_PnL"]:.3f}')
plt.ylabel('Cumulative P&L')
plt.xlabel('Date')

plt.tight_layout()
plt.show()
