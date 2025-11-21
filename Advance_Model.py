import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

df = pd.read_csv('implied_volatility_data.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df = df.rename(columns={'banknifty': 'banknifty_iv', 'nifty': 'nifty_iv'})

# Trading hours only + 10-min bars
df = df.between_time('09:15', '15:30')
data = df[['banknifty_iv', 'nifty_iv', 'tte']].resample('10min').last().dropna()

print(f"Dataset ready: {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")


COST_PER_IV_POINT_ROUNDTRIP = 0.35   


def apply_kalman_hedge(df):
    df = df.copy()
    
    # Adaptive parameters estimated from data
    simple_beta = df['banknifty_iv'] / df['nifty_iv']
    beta_change = simple_beta.pct_change().dropna()
    trans_cov = max(beta_change.var() * 252 * 37.5, 0.001)
    
    residuals = df['banknifty_iv'] - simple_beta * df['nifty_iv']
    obs_cov = max(residuals.var(), 0.1)
    
    
    x = df['nifty_iv'].values
    y = df['banknifty_iv'].values
    beta = simple_beta.mean()
    P = 1.0
    betas = []
    
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            betas.append(beta)
            continue
        P = P + trans_cov
        residual = y[i] - beta * x[i]
        S = x[i]**2 * P + obs_cov
        K = P * x[i] / S if S > 1e-8 else 0
        beta = beta + K * residual
        P = (1 - K * x[i]) * P
        betas.append(beta)
    
    df['beta'] = pd.Series(betas, index=df.index).clip(0.7, 1.9).ffill().bfill()
    df['hedged_spread'] = df['banknifty_iv'] - df['beta'] * df['nifty_iv']
    return df

def run_kalman_strategy(df, window_min, entry_z, exit_z):
    df = df.copy()
    periods = max(20, int(window_min / 10))
    min_p = int(periods * 0.7)
    
    df['mean'] = df['hedged_spread'].rolling(periods, min_periods=min_p).mean()
    df['std'] = df['hedged_spread'].rolling(periods, min_periods=min_p).std()
    df['z'] = (df['hedged_spread'] - df['mean']) / df['std']
    
    
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
    
    
    df['pnl_gross'] = df['position'].shift(1) * df['hedged_spread'].diff() * (df['tte'] ** 0.7)
    
    
    turnover = df['position'].diff().abs()
    spread_move = df['hedged_spread'].diff().abs() * turnover
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
    
    
    trade_pnls = []
    current = 0.0
    in_trade = False
    for i in range(len(df)):
        if turnover.iloc[i] == 1:
            if in_trade:
                trade_pnls.append(current)
            current = 0.0
            in_trade = True
        if in_trade:
            current += df['pnl_net'].iloc[i]
    if in_trade:
        trade_pnls.append(current)
    win_rate = np.mean(np.array(trade_pnls) > 0) if trade_pnls else 0
    
    return {
        'window': window_min, 'entry_z': entry_z, 'exit_z': exit_z,
        'sharpe': round(sharpe, 4), 'net_pnl': round(net_pnl, 3),
        'max_dd': round(max_dd, 3), 'trades': total_trades,
        'win_rate': round(win_rate, 4), 'df': df.dropna(subset=['z'])
    }

# Grid search
windows = [375, 750, 1125, 1500, 1875]
entries = [1.0, 1.5, 2.0, 2.5, 3.0]
exits = [0.0, 0.25, 0.5, 0.75, 1.0]

results_kalman = []
df_kalman = apply_kalman_hedge(data)

print("Running Kalman model grid search")
for w in windows:
    for e in entries:
        for x in exits:
            if x >= e: continue
            res = run_kalman_strategy(df_kalman, window_min=w, entry_z=e, exit_z=x)
            results_kalman.append(res)

kalman_df = pd.DataFrame([{
    'Window': r['window'], 'Entry_Z': r['entry_z'], 'Exit_Z': r['exit_z'],
    'Sharpe': r['sharpe'], 'Net_PnL': r['net_pnl'], 'Max_DD': r['max_dd'],
    'Trades': r['trades'], 'WinRate': f"{r['win_rate']:.1%}"
} for r in results_kalman])

print("\nKALMAN MODEL – Top 5 Configurations")
print(kalman_df.sort_values('Sharpe', ascending=False).head(5).to_string(index=False))

best_kalman = kalman_df.loc[kalman_df['Sharpe'].idxmax()]
best_kalman_run = next(r for r in results_kalman 
                       if r['window'] == best_kalman['Window'] and 
                          r['entry_z'] == best_kalman['Entry_Z'] and 
                          r['exit_z'] == best_kalman['Exit_Z'])
df_plot_k = best_kalman_run['df']

print(f"\nBest Kalman: Window {int(best_kalman['Window'])} | Entry ±{best_kalman['Entry_Z']:.1f}σ | Exit ±{best_kalman['Exit_Z']:.1f}σ")
print(f"Sharpe: {best_kalman['Sharpe']:.4f} | Net P&L: {best_kalman['Net_PnL']:.3f} | Win Rate: {best_kalman['WinRate']}")

# # Visualisation
plt.figure(figsize=(15, 10))

plt.subplot(3,1,1)
plt.plot(df_plot_k.index, df_plot_k['beta'], color='blue')
plt.title('Kalman Filter Dynamic Beta')
plt.ylabel('Beta')

plt.subplot(3,1,2)
plt.plot(df_plot_k.index, df_plot_k['z'], color='green', alpha=0.8)
plt.axhline(best_kalman['Entry_Z'], color='red', linestyle='--', label='Entry')
plt.axhline(-best_kalman['Entry_Z'], color='red', linestyle='--')
plt.axhline(best_kalman['Exit_Z'], color='orange', linestyle='--', label='Exit')
plt.axhline(-best_kalman['Exit_Z'], color='orange', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.title('Z-Score on Hedged Spread')
plt.ylabel('Z-Score')

plt.subplot(3,1,3)
plt.plot(df_plot_k.index, df_plot_k['cum_pnl'], color='purple', linewidth=1.5)
plt.title(f'Kalman Model Equity Curve – Net P&L = {best_kalman["Net_PnL"]:.3f}')
plt.ylabel('Cumulative P&L')

plt.tight_layout()
plt.show()
