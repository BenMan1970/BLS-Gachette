import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Configuration de la page Streamlit
st.set_page_config(page_title="Bot Trading M15/H1", layout="wide", page_icon="📈")

# ==========================================
# 1. LISTE DES ACTIFS (OANDA SYMBOLS)
# ==========================================
ASSETS = [
    # --- MAJEURS ---
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    # --- MINEURS / CROISÉS ---
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CHF_JPY",
    # --- MÉTAUX ---
    "XAU_USD", "XPT_USD",
    # --- INDICES (Syntaxe Oanda) ---
    "US30_USD",     # Wall Street 30
    "NAS100_USD",   # Nasdaq 100
    "SPX500_USD"    # S&P 500
]

# ==========================================
# 2. GESTION API OANDA
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            
            # Gestion environnement
            if "OANDA_ENVIRONMENT" in st.secrets:
                self.environment = st.secrets["OANDA_ENVIRONMENT"]
            else:
                self.environment = "practice" 

            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except Exception as e:
            st.error("❌ Erreur de connexion aux secrets.")
            st.info("Le code attend les clés : 'OANDA_ACCESS_TOKEN' et 'OANDA_ACCOUNT_ID' dans les secrets Streamlit.")
            st.stop()

    def get_candles(self, instrument, granularity, count=300):
        params = {"count": count, "granularity": granularity, "price": "M"}
        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
        except Exception as e:
            st.error(f"Erreur API Oanda sur {instrument} : {e}")
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
# 3. BIBLIOTHÈQUE D'INDICATEURS (CORRIGÉE)
# ==========================================

def calculate_wma(series, length):
    """Weighted Moving Average"""
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_ema(series, length):
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

# --- A. RSI 7 sur OHLC4 (Trigger) ---
def get_rsi_ohlc4(df, length=7):
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    delta = ohlc4.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- B. HMA 20 Colorée (CORRIGÉ: Retourne Series et non Array) ---
def get_colored_hma(df, length=20):
    src = df['close']
    wma1 = calculate_wma(src, int(length / 2))
    wma2 = calculate_wma(src, length)
    raw_hma = 2 * wma1 - wma2
    hma = calculate_wma(raw_hma, int(np.round(np.sqrt(length))))
    
    # Correction ici : conversion explicite en Series pandas pour utiliser .iloc plus tard
    hma_prev = hma.shift(1)
    trend_array = np.where(hma > hma_prev, 1, -1)
    trend_series = pd.Series(trend_array, index=df.index)
    
    return hma, trend_series

# --- C. Bluestar ZLEMA Trend (MTF Alignment) ---
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
    
    if current_close > current_zlema:
        return 1
    else:
        return -1

# ==========================================
# 4. INTERFACE UTILISATEUR
# ==========================================

st.title("🤖 Trading Bot Dashboard (Oanda)")
st.caption("Stratégie M15/H1 avec RSI OHLC4, HMA et Alignement MTF")

# Sidebar
st.sidebar.header("Paramètres")

# Liste déroulante des actifs
symbol = st.sidebar.selectbox("Choisir l'actif", ASSETS, index=0)

run_btn = st.sidebar.button("Scanner le Marché")

if run_btn:
    api = OandaClient()
    
    with st.spinner(f"Analyse de {symbol} en cours..."):
        try:
            # 1. Récupération des données multi-frames
            df_m15 = api.get_candles(symbol, "M15")
            df_h1 = api.get_candles(symbol, "H1")
            df_h4 = api.get_candles(symbol, "H4")
            df_d1 = api.get_candles(symbol, "D")
            
            if df_m15.empty or df_h1.empty:
                st.warning(f"Données insuffisantes pour {symbol}. Le marché est peut-être fermé.")
                st.stop()

            # 2. Calcul des indicateurs
            
            # --- M15 ---
            rsi_m15_series = get_rsi_ohlc4(df_m15)
            hma_m15_series, hma_trend_m15_series = get_colored_hma(df_m15)
            
            # --- H1 ---
            rsi_h1_series = get_rsi_ohlc4(df_h1)
            hma_h1_series, hma_trend_h1_series = get_colored_hma(df_h1)
            
            # --- MTF (H1, H4, D1) ---
            trend_h1 = get_bluestar_trend(df_h1)
            trend_h4 = get_bluestar_trend(df_h4)
            trend_d1 = get_bluestar_trend(df_d1)
            
            mtf_score = trend_h1 + trend_h4 + trend_d1 
            
            global_trend = "NEUTRE"
            if mtf_score >= 2: global_trend = "HAUSSIER"
            if mtf_score <= -2: global_trend = "BAISSIER"

            # 3. Moteur de Décision
            
            # --- SIGNAL M15 ---
            curr_rsi_m15 = rsi_m15_series.iloc[-1]
            prev_rsi_m15 = rsi_m15_series.iloc[-2]
            curr_hma_trend_m15 = hma_trend_m15_series.iloc[-1]
            
            cross_up_m15 = prev_rsi_m15 < 50 and curr_rsi_m15 > 50
            cross_down_m15 = prev_rsi_m15 > 50 and curr_rsi_m15 < 50
            
            signal_m15 = "WAIT"
            if cross_up_m15 and curr_hma_trend_m15 == 1 and global_trend == "HAUSSIER":
                signal_m15 = "BUY"
            elif cross_down_m15 and curr_hma_trend_m15 == -1 and global_trend == "BAISSIER":
                signal_m15 = "SELL"

            # --- SIGNAL H1 ---
            curr_rsi_h1 = rsi_h1_series.iloc[-1]
            prev_rsi_h1 = rsi_h1_series.iloc[-2]
            curr_hma_trend_h1 = hma_trend_h1_series.iloc[-1]
            
            cross_up_h1 = prev_rsi_h1 < 50 and curr_rsi_h1 > 50
            cross_down_h1 = prev_rsi_h1 > 50 and curr_rsi_h1 < 50
            
            signal_h1 = "WAIT"
            if cross_up_h1 and curr_hma_trend_h1 == 1 and global_trend == "HAUSSIER":
                signal_h1 = "BUY"
            elif cross_down_h1 and curr_hma_trend_h1 == -1 and global_trend == "BAISSIER":
                signal_h1 = "SELL"
            
            # 4. Affichage
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("M15")
                if signal_m15 == "BUY": st.success("🚀 BUY")
                elif signal_m15 == "SELL": st.error("📉 SELL")
                else: st.info("WAIT")
                st.write(f"RSI: {curr_rsi_m15:.2f}")
                st.write(f"HMA: {'🟢' if curr_hma_trend_m15 == 1 else '🔴'}")

            with col2:
                st.subheader("H1")
                if signal_h1 == "BUY": st.success("🚀 BUY")
                elif signal_h1 == "SELL": st.error("📉 SELL")
                else: st.info("WAIT")
                st.write(f"RSI: {curr_rsi_h1:.2f}")
                st.write(f"HMA: {'🟢' if curr_hma_trend_h1 == 1 else '🔴'}")

            with col3:
                st.subheader("MTF Trend")
                if global_trend == "HAUSSIER": st.success("BULLISH")
                elif global_trend == "BAISSIER": st.error("BEARISH")
                else: st.warning("NEUTRE")
                st.caption(f"Bluestar Score: {mtf_score}")
                st.write(f"H1:{'🔼' if trend_h1==1 else '🔽'} | H4:{'🔼' if trend_h4==1 else '🔽'} | D1:{'🔼' if trend_d1==1 else '🔽'}")

        except Exception as e:
            st.error(f"Erreur d'analyse (Check Code): {e}")

else:
    st.info("👈 Sélectionnez un actif et lancez le scan.")
