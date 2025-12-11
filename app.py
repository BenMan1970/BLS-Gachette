import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Configuration de la page Streamlit
st.set_page_config(page_title="Bot Trading M15/H1", layout="wide", page_icon="📈")

# ==========================================
# 1. GESTION API OANDA (Format Secrets Spécifique)
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            # Modification : Utilisation de vos noms de variables exacts
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            
            # Gestion de l'environnement (Par défaut "practice" si non spécifié)
            # Si vous avez une variable OANDA_ENVIRONMENT dans vos secrets, elle sera utilisée
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
            st.error(f"Erreur API Oanda : {e}")
            return pd.DataFrame() # Retourne vide en cas d'erreur
        
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
# 2. BIBLIOTHÈQUE D'INDICATEURS (MATHS)
# ==========================================

def calculate_wma(series, length):
    """Weighted Moving Average (pour HMA)"""
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_ema(series, length):
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

# --- A. RSI 7 sur OHLC4 (Trigger) ---
def get_rsi_ohlc4(df, length=7):
    # Formule OHLC4 : (O+H+L+C)/4
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    delta = ohlc4.diff()
    # Utilisation de l'alpha=1/length pour imiter le RMA (Wilder) de TradingView
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- B. HMA 20 Colorée (Filtre Local) ---
def get_colored_hma(df, length=20):
    src = df['close']
    wma1 = calculate_wma(src, int(length / 2))
    wma2 = calculate_wma(src, length)
    raw_hma = 2 * wma1 - wma2
    hma = calculate_wma(raw_hma, int(np.round(np.sqrt(length))))
    
    # 1 = Vert (Hausse), -1 = Rouge (Baisse)
    hma_prev = hma.shift(1)
    trend = np.where(hma > hma_prev, 1, -1)
    
    return hma, trend

# --- C. Bluestar ZLEMA Trend (MTF Alignment) ---
def get_bluestar_trend(df):
    """
    Simplification logique Python du Bluestar Tableau.
    Retourne 1 (Bull) ou -1 (Bear) basé sur la position ZLEMA.
    """
    if df.empty: return 0
    
    length = 70
    src = df['close']
    
    # Calcul ZLEMA (Zero Lag EMA)
    lag = int((length - 1) / 2)
    src_lagged = src.shift(lag)
    # Formule: EMA of (Close + (Close - Close[lag]))
    zlema_input = src + (src - src_lagged)
    zlema = calculate_ema(zlema_input, length)
    
    # Détermination tendance
    current_close = src.iloc[-1]
    current_zlema = zlema.iloc[-1]
    
    if current_close > current_zlema:
        return 1
    else:
        return -1

# ==========================================
# 3. LOGIQUE PRINCIPALE & INTERFACE
# ==========================================

st.title("🤖 Trading Bot Dashboard (Oanda)")
st.caption("Signaux M15 & H1 | Stratégie: RSI OHLC4 + HMA 20 + MTF Alignment")

# Sidebar
st.sidebar.header("Paramètres")
symbol = st.sidebar.text_input("Symbole", value="EUR_USD")
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
                st.error("Pas de données reçues. Vérifiez le symbole.")
                st.stop()

            # 2. Calcul des indicateurs
            
            # --- Indicateurs M15 ---
            rsi_m15_series = get_rsi_ohlc4(df_m15)
            hma_m15_series, hma_trend_m15_series = get_colored_hma(df_m15)
            
            # --- Indicateurs H1 ---
            rsi_h1_series = get_rsi_ohlc4(df_h1)
            hma_h1_series, hma_trend_h1_series = get_colored_hma(df_h1)
            
            # --- Tendance de Fond (MTF) ---
            trend_h1 = get_bluestar_trend(df_h1)
            trend_h4 = get_bluestar_trend(df_h4)
            trend_d1 = get_bluestar_trend(df_d1)
            
            mtf_score = trend_h1 + trend_h4 + trend_d1 
            
            global_trend = "NEUTRE"
            if mtf_score >= 2: global_trend = "HAUSSIER"
            if mtf_score <= -2: global_trend = "BAISSIER"

            # 3. Moteur de Décision (Logique RSI Cross + Filtres)
            
            # --- ANALYSE M15 ---
            curr_rsi_m15 = rsi_m15_series.iloc[-1]
            prev_rsi_m15 = rsi_m15_series.iloc[-2]
            curr_hma_trend_m15 = hma_trend_m15_series.iloc[-1] # 1 ou -1
            
            cross_up_m15 = prev_rsi_m15 < 50 and curr_rsi_m15 > 50
            cross_down_m15 = prev_rsi_m15 > 50 and curr_rsi_m15 < 50
            
            signal_m15 = "WAIT"
            
            if cross_up_m15:
                if curr_hma_trend_m15 == 1 and global_trend == "HAUSSIER":
                    signal_m15 = "BUY"
            
            elif cross_down_m15:
                if curr_hma_trend_m15 == -1 and global_trend == "BAISSIER":
                    signal_m15 = "SELL"

            # --- ANALYSE H1 ---
            curr_rsi_h1 = rsi_h1_series.iloc[-1]
            prev_rsi_h1 = rsi_h1_series.iloc[-2]
            curr_hma_trend_h1 = hma_trend_h1_series.iloc[-1]
            
            cross_up_h1 = prev_rsi_h1 < 50 and curr_rsi_h1 > 50
            cross_down_h1 = prev_rsi_h1 > 50 and curr_rsi_h1 < 50
            
            signal_h1 = "WAIT"
            
            if cross_up_h1:
                if curr_hma_trend_h1 == 1 and global_trend == "HAUSSIER":
                    signal_h1 = "BUY"
            
            elif cross_down_h1:
                if curr_hma_trend_h1 == -1 and global_trend == "BAISSIER":
                    signal_h1 = "SELL"
            
            # 4. Affichage
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Signal M15")
                if signal_m15 == "BUY":
                    st.success("🚀 ACHAT CONFIRMÉ")
                elif signal_m15 == "SELL":
                    st.error("📉 VENTE CONFIRMÉE")
                else:
                    st.info("⏳ ATTENTE")
                st.write(f"**RSI (7):** {curr_rsi_m15:.2f}")
                st.write(f"**HMA (20):** {'🟢 Verte' if curr_hma_trend_m15 == 1 else '🔴 Rouge'}")

            with col2:
                st.subheader("Signal H1")
                if signal_h1 == "BUY":
                    st.success("🚀 ACHAT CONFIRMÉ")
                elif signal_h1 == "SELL":
                    st.error("📉 VENTE CONFIRMÉE")
                else:
                    st.info("⏳ ATTENTE")
                st.write(f"**RSI (7):** {curr_rsi_h1:.2f}")
                st.write(f"**HMA (20):** {'🟢 Verte' if curr_hma_trend_h1 == 1 else '🔴 Rouge'}")

            with col3:
                st.subheader("Tendance MTF")
                if global_trend == "HAUSSIER":
                    st.success("BULLISH (H1+H4+D1)")
                elif global_trend == "BAISSIER":
                    st.error("BEARISH (H1+H4+D1)")
                else:
                    st.warning("NEUTRE")
                st.caption(f"Score: {mtf_score} (H1/H4/D1)")

        except Exception as e:
            st.error(f"Erreur d'analyse: {e}")

else:
    st.info("Cliquez sur 'Scanner le Marché' pour lancer l'analyse.")
