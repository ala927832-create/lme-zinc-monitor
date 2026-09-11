from io import StringIO
import json
import os
import traceback
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests


def fetch_silver_price():
    """抓取國際白銀期貨價格 (USD/oz) 作為副產品補貼參考"""
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            ' (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        )
    }
    try:
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/SI=F?range=5d&interval=1d'
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            quote = data['chart']['result'][0]['indicators']['quote'][0]
            closes = [c for c in quote.get('close', []) if c is not None]
            if closes:
                return float(closes[-1])
    except Exception as e:
        print(f'⚠️ 白銀價格自動抓取提示: {e}')
    return 31.5  # 預設基準價 ($31.5 USD/oz)


def fetch_lme_zinc_data():
    """多通道抓取 LME 倫敦鋅價 (USD/噸) 數據"""
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            ' (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        ),
        'Referer': 'https://finance.sina.com.cn/',
    }

    # 通道 1：新浪環球期貨 API (hf_ZM)
    try:
        url = 'https://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getGlobalFuturesDailyKLine?symbol=hf_ZM'
        res = requests.get(url, headers=headers, timeout=12)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) >= 10:
                records = []
                for item in data:
                    if isinstance(item, dict):
                        d, o, h, l, c = (
                            item.get('date') or item.get('d'),
                            float(item.get('open') or item.get('o')),
                            float(item.get('high') or item.get('h')),
                            float(item.get('low') or item.get('l')),
                            float(item.get('close') or item.get('c')),
                        )
                    elif isinstance(item, list) and len(item) >= 5:
                        d, o, h, l, c = (
                            item[0],
                            float(item[1]),
                            float(item[2]),
                            float(item[3]),
                            float(item[4]),
                        )
                    else:
                        continue
                    records.append(
                        {'Date': d, 'Open': o, 'High': h, 'Low': l, 'Close': c}
                    )

                df = pd.DataFrame(records)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index().dropna()

                latest_price = float(df['Close'].iloc[-1])
                if len(df) >= 10 and 1000 <= latest_price <= 6000:
                    return df, 'LME Zinc (hf_ZM)'
    except Exception as e:
        print(f'⚠️ [通道 1 失敗]: {e}')

    # 通道 2：Stooq 金融數據源 (zn.f)
    try:
        url = 'https://stooq.com/q/d/l/?s=zn.f&i=d'
        stooq_headers = headers.copy()
        stooq_headers['Referer'] = 'https://stooq.com/'
        res = requests.get(url, headers=stooq_headers, timeout=12)
        if res.status_code == 200 and 'Date,Open,High,Low,Close' in res.text:
            df = pd.read_csv(StringIO(res.text))
            if not df.empty and len(df) >= 10:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                df = df[['Open', 'High', 'Low', 'Close']].dropna().tail(120)
                return df, 'LME Zinc (Stooq ZN.F)'
    except Exception as e:
        print(f'⚠️ [通道 2 失敗]: {e}')

    return pd.DataFrame(), None


def calculate_all_indicators(df, silver_price, tc_base=50.0):
    """計算 5 大技術指標與冶煉綜合利潤/成本線"""
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

    # 5. 上影線佔比與 10 日高價
    upper_body = df[['Open', 'Close']].max(axis=1)
    upper_shadow = df['High'] - upper_body
    total_range = df['High'] - df['Low'] + 1e-9
    df['Shadow_Ratio'] = upper_shadow / total_range
    df['Is_10D_High'] = df['High'] == df['High'].rolling(window=10).max()

    # 6. Cash/3M 價差擬合 (以高低價與動能做模擬價差)
    df['Spread'] = (
        (df['Close'] - df['SMA20']) * 0.15 + (df['RSI'] - 50) * 0.5
    ).round(1)

    # 7. 冶煉綜合利潤指標 (結合 TC 與白銀補貼)
    # 利潤 = 鋅價*0.85 + TC + 白銀補貼 - 冶煉邊際成本基線($2,650)
    silver_credit = (silver_price - 25.0) * 8.0 if silver_price > 25.0 else 0
    df['Smelter_Margin'] = (
        (df['Close'] * 0.85 + tc_base + silver_credit) - 2650
    ).round(1)

    return df


def generate_4panel_chart(df, ticker, filename='zinc_chart.png'):
    """繪製 4-Panel 法人雙核決策 K 線圖"""
    plot_df = df.tail(60).copy()

    # 價差柱狀圖顏色 (正數紅，負數綠)
    spread_colors = np.where(plot_df['Spread'] >= 0, 'crimson', 'forestgreen')

    add_plots = [
        # Panel 0: 價格主圖
        mpf.make_addplot(
            plot_df['Upper_BB'],
            panel=0,
            color='crimson',
            linestyle='--',
            width=1,
        ),
        mpf.make_addplot(plot_df['SMA20'], panel=0, color='orange', width=1),
        mpf.make_addplot(
            plot_df['Lower_BB'],
            panel=0,
            color='forestgreen',
            linestyle='--',
            width=1,
        ),
        mpf.make_addplot(plot_df['EMA50'], panel=0, color='royalblue', width=1.2),
        # Panel 1: RSI (14)
        mpf.make_addplot(
            plot_df['RSI'],
            panel=1,
            color='purple',
            ylabel='RSI (14)',
            ylim=(0, 100),
        ),
        # Panel 2: Cash/3M Spread
        mpf.make_addplot(
            plot_df['Spread'],
            panel=2,
            type='bar',
            color=spread_colors,
            ylabel='Spread ($)',
        ),
        # Panel 3: Smelter Margin
        mpf.make_addplot(
            plot_df['Smelter_Margin'],
            panel=3,
            color='darkcyan',
            ylabel='Smelter Margin',
        ),
    ]

    custom_style = mpf.make_mpf_style(
        base_mpf_style='charles',
        rc={'font.size': 8, 'axes.labelsize': 9, 'figure.titlesize': 11},
    )

    mpf.plot(
        plot_df,
        type='candle',
        style=custom_style,
        addplot=add_plots,
        title=f'{ticker} Institutional 4-Panel Analysis (USD/MT)',
        figratio=(12, 10),
        panel_ratios=(3, 1, 1, 1),
        savefig=dict(fname=filename, dpi=120, bbox_inches='tight'),
    )


def analyze_and_notify():
    webhook_url = os.environ.get('WEBHOOK_URL')
    if not webhook_url:
        raise ValueError('錯誤：未設置 WEBHOOK_URL 環境變數。')

    silver_price = fetch_silver_price()
    tc_base = 50.0  # SMM 鋅精礦 TC 基準 ($50 USD/dmt)

    df, active_ticker = fetch_lme_zinc_data()
    if df.empty or active_ticker is None:
        raise RuntimeError('錯誤：數據源無法獲取 LME 鋅價。')

    df = calculate_all_indicators(df, silver_price=silver_price, tc_base=tc_base)
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
    spread = float(latest['Spread'])
    smelter_margin = float(latest['Smelter_Margin'])
    day_range = high_price - low_price
    price_change_pct = (
        (close_price - float(prev['Close'])) / float(prev['Close'])
    ) * 100

    # 台灣鍍鋅廠進口預估成本 (TWD/kg)
    usdtwd_rate = 31.8
    tw_cost_per_kg = (close_price * usdtwd_rate * 1.05) / 1000.0

    # === 4 級燈號判定邏輯 ===
    # 1. 🚀 強買 Strong Buy: 跌破下軌/接近EMA50 + RSI<35 + 冶煉利潤虧損 (成本底線堅實)
    # 2. 🟢 補庫 Buy: 接近EMA50 + RSI<45
    # 3. 🔴 減量 Reduce: 突破上軌 + RSI>68 + 流星線或高升水
    # 4. 🟡 觀望 Hold: 區間震盪
    if (
        (close_price <= ema50 or close_price <= lower_bb)
        and rsi < 35
        and smelter_margin < 50
    ):
        signal_badge = '🚀【強買 Strong Buy】'
        signal_color_desc = '極度超賣 + 冶煉廠成本底線支撐強勁！最佳大額補庫/做多時機。'
        best_direction = '【本日首選方向】：⚡ 積極型做多 / 🏭 鍍鋅廠重倉建庫 (60%–80%)'
    elif (close_price <= ema50 * 1.01 or close_price <= lower_bb) and rsi < 45:
        signal_badge = '🟢【補庫 Buy】'
        signal_color_desc = '回檔至關鍵均線支撐區，下檔有承接力道。'
        best_direction = '【本日首選方向】：🛡️ 穩健型分批佈局 / 🏭 鍍鋅廠常態補庫 (30%–50%)'
    elif close_price >= upper_bb and rsi > 68 and shadow_ratio > 0.5:
        signal_badge = '🔴【減量 Reduce JIT】'
        signal_color_desc = '短線過熱 + 長上影線假突破！主力高位派發風險極高。'
        best_direction = '【本日首選方向】：☕ 持平不投資 (觀望) / 🛡️ 穩健型賣出高位價差 / 🏭 鍍鋅廠嚴格 JIT 隨用隨買'
    else:
        signal_badge = '🟡【觀望 Hold】'
        signal_color_desc = '行情於布林通道內區間震盪，動能中性，靜待突破。'
        best_direction = '【本日首選方向】：☕ 持平不投資 (觀望) / 🛡️ 穩健型觀望 / 🏭 鍍鋅廠保持 10–15 天常態備貨'

    # === 思路與邏輯教學解析 ===
    teaching_logic = (
        f'• **動能與區間判讀**：RSI 目前為 **{rsi:.1f}**，價格距離 50日 EMA (${ema50:.1f}) 差額為 **${close_price - ema50:+.1f}**。'
        f'當前布林帶寬度為 ${upper_bb - lower_bb:.1f}，波動率 ATR 為 ${atr14:.1f}。\n'
        f'• **基本面支撐判讀**：目前 SMM TC 為 **${tc_base:.0f}/dmt**，白銀價格 **${silver_price:.2f}/oz**。'
        f'冶煉估算利潤為 **${smelter_margin:.1f}/噸**。當冶煉利潤跌破 $50 時，冶煉廠減產預期將為鋅價提供強大的成本防線。'
    )

    chart_file = 'zinc_chart.png'
    generate_4panel_chart(df, active_ticker, chart_file)

    message_lines = [
        '【**LME 鋅價 & 法人雙核決策面板**】\n',
        f'🚦 **當前市場燈號：{signal_badge}**',
        f'📝 **燈號診斷**：{signal_color_desc}\n',
        '📊 **雙核數據速報**：',
        f'• LME 最新收盤價：**${close_price:.1f} 美元/噸** ({price_change_pct:+.2f}%)',
        f'• Cash/3M 價差估算：**${spread:+.1f} 美元/噸**',
        f'• 冶煉加工費 (TC)：**${tc_base:.0f} USD/dmt** | 國際白銀：**${silver_price:.2f} USD/oz**',
        f'• 台灣鍍鋅廠預估成本：**NT$ {tw_cost_per_kg:.2f} / kg** (匯率: {usdtwd_rate})\n',
        '🧠 **核心判斷思路與教學解析**：',
        teaching_logic + '\n',
        '💡 **三軌操作與投資指南**：',
        '🏭 **鍍鋅廠採購**：'
        + (
            '建議僅執行 JIT 隨用隨買，切勿囤高價庫存。'
            if '減量' in signal_badge
            else '可購入 30%–50% 安全庫存。'
        ),
        '🛡️ **穩健型投資者**：'
        + (
            '建議觀望不建倉，或採取 Sell Call / Bear Put Spread 鎖定收益。'
            if '減量' in signal_badge
            else '可在 EMA50 附近分批建構現貨低吸頭寸。'
        ),
        '⚡ **積極型投資者**：'
        + (
            '動能衰竭浮現，可評估阻力區佈局做空或 Bear Spread。'
            if '減量' in signal_badge
            else '可順勢建立 Call 買權或期貨多單。'
        ),
        f'\n🎯 **{best_direction}**',
    ]

    full_message = '\n'.join(message_lines)

    with open(chart_file, 'rb') as f:
        payload = {'payload_json': json.dumps({'content': full_message})}
        files = {'file': (chart_file, f, 'image/png')}
        res = requests.post(webhook_url, data=payload, files=files)

    res.raise_for_status()
    print(f'🎉 成功推送法人雙核面板與 4-Panel 圖表！回應碼：{res.status_code}')


if __name__ == '__main__':
    try:
        analyze_and_notify()
    except Exception as e:
        print('❌ 程式執行失敗，詳細報錯如下：')
        traceback.print_exc()
        raise e
