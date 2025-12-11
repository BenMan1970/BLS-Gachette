import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Configuration
st.set_page_config(page_title="Bluestar Scanner", layout="wide", page_icon="📡")

# ==========================================
# 1. LISTE DES ACTIFS
# ==========================================
ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "CHF_JPY",
    "XAU_USD", "XPT_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD"
]

# ==========================================
# 2. CLIENT API OANDA
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            
            if "OANDA_ENVIRONMENT" in st.secrets:
                self.environment = st.secrets["OANDA_ENVIRONMENT"]
            else:
                self.environment = "practice" 

            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except Exception:
            st.error("⚠️ Erreur Secrets : Vérifiez 'OANDA_ACCESS_TOKEN' et 'OANDA_ACCOUNT_ID'.")
            st.stop()

    def get_candles(self, instrument, granularity, count=150):
        params = {"count": count, "granularity": granularity, "price": "M"}
        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
        except Exception:
            return pd.DataFrame()
        
        data = []
        for candle in r.response['candles']:
            if candle['complete']:
                data.append({
                    'time': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
        df = pd.DataFrame(data)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
        return df

# ==========================================
# 3. INDICATEURS
# ==========================================

def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def get_rsi_ohlc4(df, length=7):
    # RSI (O+H+L+C)/4
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    delta = ohlc4.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_colored_hma(df, length=20):
    src = df['close']
    wma1 = calculate_wma(src, int(length / 2))
    wma2 = calculate_wma(src, length)
    raw_hma = 2 * wma1 - wma2
    hma = calculate_wma(raw_hma, int(np.round(np.sqrt(length))))
    
    hma_prev = hma.shift(1)
    trend_array = np.where(hma > hma_prev, 1, -1)
    trend_series = pd.Series(trend_array, index=df.index)
    return hma, trend_series

def get_bluestar_trend(df):
    if df.empty: return 0
    length = 70
    src = df['close']
    lag = int((length - 1) / 2)
    src_lagged = src.shift(lag)
    zlema_input = src + (src - src_lagged)
    zlema = calculate_ema(zlema_input, length)
    
    current_close = src.iloc[-1]
    current_zlema = zlema.iloc[-1]
    return 1 if current_close > current_zlema else -1

# ==========================================
# 4. LOGIQUE D'ANALYSE
# ==========================================

def get_mtf_status(api, symbol):
    # Helper pour le debug pour voir les détails MTF
    df_h1 = api.get_candles(symbol, "H1")
    df_h4 = api.get_candles(symbol, "H4")
    df_d1 = api.get_candles(symbol, "D")
    
    if df_h1.empty or df_h4.empty or df_d1.empty:
        return 0, {}
        
    t_h1 = get_bluestar_trend(df_h1)
    t_h4 = get_bluestar_trend(df_h4)
    t_d1 = get_bluestar_trend(df_d1)
    
    score = t_h1 + t_h4 + t_d1
    details = {"H1": t_h1, "H4": t_h4, "D1": t_d1}
    return score, details

def analyze_single_asset_debug(api, symbol):
    st.markdown(f"### 🔍 Analyse détaillée : {symbol}")
    
    col1, col2 = st.columns(2)
    
    # 1. Données M15
    df_m15 = api.get_candles(symbol, "M15")
    if df_m15.empty:
        st.error("Pas de données M15")
        return

    rsi_m15 = get_rsi_ohlc4(df_m15)
    _, hma_trend_m15 = get_colored_hma(df_m15)
    
    m15_curr = rsi_m15.iloc[-1]
    m15_prev = rsi_m15.iloc[-2]
    hma_m15_val = hma_trend_m15.iloc[-1]
    
    # 2. Données H1
    df_h1 = api.get_candles(symbol, "H1")
    rsi_h1 = get_rsi_ohlc4(df_h1)
    _, hma_trend_h1 = get_colored_hma(df_h1)
    
    h1_curr = rsi_h1.iloc[-1]
    h1_prev = rsi_h1.iloc[-2]
    hma_h1_val = hma_trend_h1.iloc[-1]
    
    # 3. MTF
    mtf_score, mtf_details = get_mtf_status(api, symbol)
    
    # --- AFFICHAGE DEBUG M15 ---
    with col1:
        st.info("📊 Données M15")
        st.write(f"**RSI Actuel :** {m15_curr:.2f}")
        st.write(f"**RSI Précédent :** {m15_prev:.2f}")
        
        cross_up = m15_prev < 50 and m15_curr > 50
        cross_down = m15_prev > 50 and m15_curr < 50
        
        if cross_up: st.success("✅ CROISEMENT HAUSSIER (50)")
        elif cross_down: st.error("✅ CROISEMENT BAISSIER (50)")
        else: st.warning("❌ Pas de croisement RSI récent")
        
        st.write(f"**HMA Couleur :** {'🟢 Verte' if hma_m15_val==1 else '🔴 Rouge'}")
    
    # --- AFFICHAGE DEBUG MTF ---
    with col2:
        st.info("🌍 Alignement MTF (H1 + H4 + D1)")
        st.write(f"H1 : {'🔼 Bull' if mtf_details.get('H1')==1 else '🔽 Bear'}")
        st.write(f"H4 : {'🔼 Bull' if mtf_details.get('H4')==1 else '🔽 Bear'}")
        st.write(f"D1 : {'🔼 Bull' if mtf_details.get('D1')==1 else '🔽 Bear'}")
        st.metric("Score Global", mtf_score, help="Doit être >= 2 pour Buy ou <= -2 pour Sell")
    
    # --- VERDICT ---
    st.markdown("---")
    st.subheader("Verdict du Bot :")
    
    # Test BUY M15
    if cross_up:
        if hma_m15_val == 1:
            if mtf_score >= 2:
                st.success("🚀 SIGNAL BUY VALIDE !")
            else:
                st.error(f"❌ Rejeté : RSI et HMA OK, mais Score MTF insuffisant ({mtf_score} < 2)")
        else:
            st.error("❌ Rejeté : HMA est Rouge (Contre tendance locale)")
    
    # Test SELL M15
    elif cross_down:
        if hma_m15_val == -1:
            if mtf_score <= -2:
                st.success("📉 SIGNAL SELL VALIDE !")
            else:
                st.error(f"❌ Rejeté : RSI et HMA OK, mais Score MTF insuffisant ({mtf_score} > -2)")
        else:
            st.error("❌ Rejeté : HMA est Verte (Contre tendance locale)")
            
    else:
        st.write("⚪ Aucun signal car le RSI ne croise pas 50 actuellement.")


def scan_market(api):
    # Version Scan Global (Silencieuse)
    valid_signals = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(ASSETS)
    
    for i, symbol in enumerate(ASSETS):
        status_text.text(f"Scanning {symbol}...")
        progress_bar.progress((i + 1) / total)
        
        df_m15 = api.get_candles(symbol, "M15")
        df_h1 = api.get_candles(symbol, "H1")
        if df_m15.empty or df_h1.empty: continue

        # RSI Checks
        rsi_m15 = get_rsi_ohlc4(df_m15)
        rsi_h1 = get_rsi_ohlc4(df_h1)
        
        m15_curr, m15_prev = rsi_m15.iloc[-1], rsi_m15.iloc[-2]
        h1_curr, h1_prev = rsi_h1.iloc[-1], rsi_h1.iloc[-2]
        
        m15_cross_up = m15_prev < 50 and m15_curr > 50
        m15_cross_down = m15_prev > 50 and m15_curr < 50
        h1_cross_up = h1_prev < 50 and h1_curr > 50
        h1_cross_down = h1_prev > 50 and h1_curr < 50
        
        if not (m15_cross_up or m15_cross_down or h1_cross_up or h1_cross_down):
            continue

        # MTF
        mtf_score, _ = get_mtf_status(api, symbol)
        is_bull_trend = (mtf_score >= 2) 
        is_bear_trend = (mtf_score <= -2)
        
        # M15 Logic
        if m15_cross_up or m15_cross_down:
            _, hma_trend = get_colored_hma(df_m15)
            hma_val = hma_trend.iloc[-1]
            if m15_cross_up and hma_val == 1 and is_bull_trend:
                valid_signals.append({"Symbol": symbol, "TF": "M15", "Type": "BUY", "RSI": round(m15_curr, 2)})
            elif m15_cross_down and hma_val == -1 and is_bear_trend:
                valid_signals.append({"Symbol": symbol, "TF": "M15", "Type": "SELL", "RSI": round(m15_curr, 2)})

        # H1 Logic
        if h1_cross_up or h1_cross_down:
            _, hma_trend = get_colored_hma(df_h1)
            hma_val = hma_trend.iloc[-1]
            if h1_cross_up and hma_val == 1 and is_bull_trend:
                valid_signals.append({"Symbol": symbol, "TF": "H1", "Type": "BUY", "RSI": round(h1_curr, 2)})
            elif h1_cross_down and hma_val == -1 and is_bear_trend:
                valid_signals.append({"Symbol": symbol, "TF": "H1", "Type": "SELL", "RSI": round(h1_curr, 2)})

    progress_bar.empty()
    status_text.empty()
    return valid_signals

# ==========================================
# 5. INTERFACE
# ==========================================

st.title("📡 BLUESTAR Scanner Forex & Indices")

# Sidebar Menu
mode = st.sidebar.radio("Mode", ["SCAN GLOBAL", "TEST UNITAIRE (DEBUG)"])

if mode == "SCAN GLOBAL":
    st.write(f"Surveillance de **{len(ASSETS)} actifs**.")
    if st.button("LANCER LE SCAN", type="primary"):
        api = OandaClient()
        with st.spinner("Analyse en cours..."):
            results = scan_market(api)
        
        if not results:
            st.info("✅ Scan terminé. Aucun signal confirmé (Logique stricte).")
        else:
            st.success(f"🎯 {len(results)} Signal(aux) trouvé(s) !")
            st.dataframe(pd.DataFrame(results))

elif mode == "TEST UNITAIRE (DEBUG)":
    st.warning("Ce mode permet de comprendre pourquoi un signal est validé ou rejeté.")
    selected_asset = st.selectbox("Choisir un actif à vérifier :", ASSETS)
    
    if st.button("VÉRIFIER CET ACTIF"):
        api = OandaClient()
        analyze_single_asset_debug(api, selected_asset)
