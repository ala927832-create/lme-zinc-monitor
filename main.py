import json
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import requests
import yfinance as yf


def fetch_zinc_data():
    """使用 yf.download 搭配多代碼自動備援，解決 GitHub Actions IP 阻擋與代碼失效問題"""
    # 備援代碼清單：ZNC=F (LME鋅期貨), ZNC.L (倫敦鋅), SZI=F (上海鋅期貨折算)
    tickers = ['ZNC=F', 'ZNC.L', 'SZI=F']

    for ticker in tickers:
        print(f'正在嘗試抓取 {ticker} 最新數據...')
        try:
            # yf.download 比 Ticker().history 更穩定且自動處理抓取 Header
            df = yf.download(
                ticker, period='6m', progress=False, auto_adjust=True
            )

            # 處理 yfinance 新版可能產生的 MultiIndex 雙層欄位
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 確保欄位包含 Open, High, Low, Close
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not df.empty and all(col in df.columns for col in required_cols):
                # 剔除全零或空值的無效列
                df = df.dropna(subset=required_cols)
                if len(df) >= 20:
                    print(
                        f'成功獲取 {ticker} 數據！(共 {len(df)} 筆交易日紀錄)'
                    )
                    return df, ticker
        except Exception as e:
            print(f'{ticker} 抓取失敗: {e}')

    return pd.DataFrame(), None


def calculate_all_indicators(df):
    """計算 5 大技術指標：RSI(14)、上影線率、布林通道(20,2)、EMA50、ATR(14)"""
    # 1. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. 布林通道 (20, 2)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['Std20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA20'] + (df['Std20'] * 2)
    df['Lower_BB'] = df['SMA20'] - (df['Std20'] * 2)

    # 3. 50日 EMA
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # 4. ATR (14)
    high_low = df['High'] - df['Low']
    high_pc = (df['High'] - df['Close'].shift(1)).abs()
    low_pc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(window=14).mean()

    # 5. 上影線佔比 與 10日最高價
    upper_body = df[['Open', 'Close']].max(axis=1)
    upper_shadow = df['High'] - upper_body
    total_range = df['High'] - df['Low'] + 1e-9
    df['Shadow_Ratio'] = upper_shadow / total_range
    df['Is_10D_High'] = df['High'] == df['High'].rolling(window=10).max()

    return df


def generate_chart(df, ticker, filename='zinc_chart.png'):
    """繪製帶有布林通道、EMA50 與 RSI 的 K 線圖"""
    plot_df = df.tail(60).copy()

    add_plots = [
        mpf.make_addplot(
            plot_df['Upper_BB'], color='crimson', linestyle='--', width=1
        ),
        mpf.make_addplot(plot_df['SMA20'], color='orange', width=1),
        mpf.make_addplot(
            plot_df['Lower_BB'],
            color='forestgreen',
            linestyle='--',
            width=1,
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
        title=f'LME Zinc ({ticker}) Technical Analysis',
        figratio=(12, 8),
        panel_ratios=(3, 1),
        savefig=dict(fname=filename, dpi=120, bbox_inches='tight'),
    )


def analyze_and_notify():
    webhook_url = os.environ.get('WEBHOOK_URL')
    if not webhook_url:
        raise ValueError('錯誤：未設置 WEBHOOK_URL 環境變數。')

    # 多代碼與多機制抓取數據
    df, active_ticker = fetch_zinc_data()
    if df.empty or active_ticker is None:
        raise RuntimeError(
            '錯誤：備用代碼庫全數失敗，無法獲取鋅期貨數據。請檢查網路或 Yahoo'
            ' 服務狀態。'
        )

    df = calculate_all_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close_price = float(latest['Close'])
    high_price = float(latest['High'])
    low_price = float(latest['Low'])
    rsi = float(latest['RSI'])
    ema50 = float(latest['EMA50'])
    upper_bb = float(latest['Upper_BB'])
    lower_bb = float(latest['Lower_BB'])
    atr14 = float(latest['ATR14'])
    shadow_ratio = float(latest['Shadow_Ratio'])
    is_10d_high = bool(latest['Is_10D_High'])
    day_range = high_price - low_price
    price_change_pct = (
        (close_price - float(prev['Close'])) / float(prev['Close'])
    ) * 100

    chart_file = 'zinc_chart.png'
    generate_chart(df, active_ticker, chart_file)

    indicator_notes = []
    if rsi > 70:
        indicator_notes.append(
            f'• **RSI (14)**：{rsi:.1f} ⚠️ (進入 >70 超買區，過熱警戒)'
        )
    elif rsi < 30:
        indicator_notes.append(
            f'• **RSI (14)**：{rsi:.1f} 🟢 (進入 <30 超賣區，築底機會)'
        )
    else:
        indicator_notes.append(f'• **RSI (14)**：{rsi:.1f} (中立區間)')

    if shadow_ratio > 0.6 and is_10d_high:
        indicator_notes.append(
            f'• **K線型態**：上影線佔比 **{shadow_ratio*100:.1f}%** ⚠️'
            ' (創10日新高後急拉回，流星線特徵)'
        )

    if close_price > upper_bb:
        indicator_notes.append(
            f'• **布林通道**：突破上軌 ${upper_bb:.1f} 🚀 (極端軋空或爆發點)'
        )
    elif close_price < lower_bb:
        indicator_notes.append(
            f'• **布林通道**：跌破下軌 ${lower_bb:.1f} 📉 (尋求超跌支撐)'
        )

    ema_diff = abs(close_price - ema50)
    if ema_diff <= 20:
        indicator_notes.append(
            f'• **50日 EMA 支撐**：目前價格 (${close_price:.1f}) 接近 EMA50'
            f' 支撐 (${ema50:.1f} ±$20) 🎯'
        )
    else:
        indicator_notes.append(
            f'• **50日 EMA 支撐**：${ema50:.1f} (距離當前'
            f' ${close_price - ema50:+.1f})'
        )

    if day_range > 2.0 * atr14:
        indicator_notes.append(
            f'• **ATR 波動率**：單日振幅 ${day_range:.1f} > 2.0 × ATR'
            f' (${atr14:.1f}) 💥 (劇烈洗盤行情)'
        )

    composite_alerts = []
    is_bull_trap = close_price > upper_bb and rsi > 70 and shadow_ratio > 0.6
    is_golden_dip = ema_diff <= 20 and rsi < 40

    if is_bull_trap:
        composite_alerts.append(
            '🚨 **【複合警報：高位假突破 / 主力派發】**\n'
            '突破布林上軌 + RSI>70 + 長上影線流星線觸發！極高機率為 Bull'
            ' Trap。'
        )

    if is_golden_dip:
        composite_alerts.append(
            '🟢 **【複合警報：黃金補庫點 / 支撐確認】**\n'
            '價格拉回至 EMA50 支撐區且 RSI<40，下檔承接力道強。'
        )

    if is_bull_trap or (rsi > 70 and shadow_ratio > 0.5):
        advice = (
            '💡 **綜合操作建議**：\n'
            '• **鍍鋅廠**：高位假突破機率高，建議僅執行 JIT'
            ' 隨用隨買，切勿囤積高價庫存。\n'
            '• **積極投資者**：動能衰竭浮現，可評估阻力區 Bear Put Spread'
            ' 或高位做空策略。'
        )
    elif is_golden_dip or rsi < 35:
        advice = (
            '💡 **綜合操作建議**：\n'
            '• **鍍鋅廠**：價格進入關鍵支撐區，可果斷分批購入 30%–50%'
            ' 安全庫存。\n'
            '• **積極投資者**：觀察支撐區止跌訊號，可尋求做多或 Call Spread'
            ' 佈局。'
        )
    else:
        advice = (
            '💡 **綜合操作建議**：\n'
            '• **鍍鋅廠**：行情處於區間震盪，維繫 10–15 天常態營運庫存即可。\n'
            '• **積極投資者**：維持觀望或使用無方向性期權組合（如區間賣出價差）。'
        )

    change_emoji = '📈' if price_change_pct >= 0 else '📉'
    message_lines = [
        '【**LME 鋅價 & K線技術面每日自動警報**】\n',
        f'📊 **價格速報 (標的: {active_ticker})**：',
        (
            f'• 最新收盤價：**${close_price:.1f} / 噸** ({change_emoji}'
            f' {price_change_pct:+.2f}%)'
        ),
        f'• 今日高 / 低價：${high_price:.1f} / ${low_price:.1f}\n',
        '📈 **5 大技術指標動態分析**：',
    ]
    message_lines.extend(indicator_notes)

    if composite_alerts:
        message_lines.append('\n' + '\n'.join(composite_alerts))

    message_lines.append('\n' + advice)
    full_message = '\n'.join(message_lines)

    with open(chart_file, 'rb') as f:
        payload = {'payload_json': json.dumps({'content': full_message})}
        files = {'file': (chart_file, f, 'image/png')}
        res = requests.post(webhook_url, data=payload, files=files)

    res.raise_for_status()
    print(f'成功使用 {active_ticker} 推送通知與圖表！回應碼：{res.status_code}')


if __name__ == '__main__':
    analyze_and_notify()
