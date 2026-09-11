import json
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import requests
import yfinance as yf


def calculate_indicators(df):
    """計算 RSI(14)、EMA(50) 與 布林通道(20, 2)"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['Std20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA20'] + (df['Std20'] * 2)
    df['Lower_BB'] = df['SMA20'] - (df['Std20'] * 2)

    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    return df


def generate_chart(df, filename='zinc_chart.png'):
    """使用 mplfinance 繪製包含布林通道與 RSI 的 K 線圖"""
    plot_df = df.tail(60).copy()

    add_plots = [
        mpf.make_addplot(
            plot_df['Upper_BB'], color='crimson', linestyle='--', width=1
        ),
        mpf.make_addplot(plot_df['SMA20'], color='orange', width=1),
        mpf.make_addplot(
            plot_df['Lower_BB'], color='forestgreen', linestyle='--', width=1
        ),
        mpf.make_addplot(plot_df['EMA50'], color='royalblue', width=1.2),
        mpf.make_addplot(
            plot_df['RSI'],
            panel=1,
            color='purple',
            ylabel='RSI (14)',
            ylim=(0, 100),
        ),
    ]

    custom_style = mpf.make_mpf_style(
        base_mpf_style='charles',
        rc={
            'font.size': 9,
            'axes.labelsize': 10,
            'figure.titlesize': 12,
        },
    )

    mpf.plot(
        plot_df,
        type='candle',
        style=custom_style,
        addplot=add_plots,
        title='LME Zinc (ZNC=F) Daily K-Line Chart',
        figratio=(12, 8),
        panel_ratios=(3, 1),
        savefig=dict(fname=filename, dpi=120, bbox_inches='tight'),
    )


def analyze_and_notify():
    webhook_url = os.environ.get('WEBHOOK_URL')
    if not webhook_url:
        print('錯誤：未設置 WEBHOOK_URL')
        return

    ticker = 'ZNC=F'
    df = yf.Ticker(ticker).history(period='6m')
    if df.empty:
        print('無法抓取數據。')
        return

    df = calculate_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close_price = latest['Close']
    rsi = latest['RSI']
    ema50 = latest['EMA50']
    upper_bb = latest['Upper_BB']
    lower_bb = latest['Lower_BB']
    price_change_pct = ((close_price - prev['Close']) / prev['Close']) * 100

    chart_file = 'zinc_chart.png'
    generate_chart(df, chart_file)

    alerts = []
    if rsi >= 70:
        alerts.append(f'⚠️ **RSI (14) 過熱**：{rsi:.1f} (>70 超買區)')
    elif rsi <= 30:
        alerts.append(f'🟢 **RSI (14) 超賣**：{rsi:.1f} (<30 超賣區)')

    if close_price >= upper_bb:
        alerts.append(
            f'🚀 **突破布林上軌**：收盤 ${close_price:.1f} 衝破上軌 ${upper_bb:.1f}'
        )
    elif close_price <= lower_bb:
        alerts.append(
            f'📉 **跌破布林下軌**：收盤 ${close_price:.1f} 低於下軌 ${lower_bb:.1f}'
        )

    change_emoji = '📈' if price_change_pct >= 0 else '📉'
    message_lines = [
        '【**LME 鋅價 K 線圖與技術指標每日自動監控**】',
        f'📊 **最新價**：**${close_price:.1f} / 噸** ({change_emoji} {price_change_pct:+.2f}%)',
        f'• **RSI (14)**：{rsi:.1f} | **50日 EMA**：${ema50:.1f}',
        f'• **布林通道**：[${lower_bb:.1f} ~ ${upper_bb:.1f}]',
    ]

    if alerts:
        message_lines.append('\n🚨 **指標預警**：')
        message_lines.extend([f'• {a}' for a in alerts])

    full_message = '\n'.join(message_lines)

    with open(chart_file, 'rb') as f:
        payload = {'payload_json': json.dumps({'content': full_message})}
        files = {'file': (chart_file, f, 'image/png')}
        res = requests.post(webhook_url, data=payload, files=files)

    print('通知與圖表發送完成，回應碼：', res.status_code)


if __name__ == '__main__':
    analyze_and_notify()
