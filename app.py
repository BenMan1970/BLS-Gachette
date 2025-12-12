import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime

# ==========================================
# CONFIGURATION & DESIGN
# ==========================================
st.set_page_config(page_title="Bluestar M15 Sniper", layout="centered", page_icon="⚡")

# CSS pour un look épuré
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #30333F;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
# 2. MOTEUR API
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
            st.error("⚠️ Clés API manquantes dans les secrets.")
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
# 4. LOGIQUE SCANNER M15
# ==========================================

def run_sniper_scan(api):
    signals = []
    
    # Barre de progression minimaliste
    progress_bar = st.progress(0)
    total = len(ASSETS)
    
    for i, symbol in enumerate(ASSETS):
        progress_bar.progress((i + 1) / total)
        
        # 1. Données M15
        df_m15 = api.get_candles(symbol, "M15")
        if df_m15.empty: continue

        # 2. RSI OHLC4 (7) - Détection Croisement
        rsi_series = get_rsi_ohlc4(df_m15)
        curr_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]
        
        cross_up = prev_rsi < 50 and curr_rsi > 50
        cross_down = prev_rsi > 50 and curr_rsi < 50
        
        # Optimisation : Si pas de croisement, on passe direct
        if not (cross_up or cross_down):
            continue

        # 3. HMA M15
        hma, hma_trend = get_colored_hma(df_m15)
        hma_val = hma_trend.iloc[-1] 
        
        # 4. MTF (Alignement H1/H4/D1)
        df_h1 = api.get_candles(symbol, "H1")
        df_h4 = api.get_candles(symbol, "H4")
        df_d1 = api.get_candles(symbol, "D")
        
        if df_h1.empty or df_h4.empty or df_d1.empty: continue
        
        score = get_bluestar_trend(df_h1) + get_bluestar_trend(df_h4) + get_bluestar_trend(df_d1)
        
        # Validation Finale
        current_price = df_m15['close'].iloc[-1]
        
        if cross_up and hma_val == 1 and score >= 2:
            signals.append({
                "symbol": symbol, "type": "BUY", 
                "price": current_price, "rsi": curr_rsi, 
                "score": score
            })
        elif cross_down and hma_val == -1 and score <= -2:
            signals.append({
                "symbol": symbol, "type": "SELL", 
                "price": current_price, "rsi": curr_rsi, 
                "score": score
            })

    progress_bar.empty()
    return signals

# ==========================================
# 5. INTERFACE UTILISATEUR
# ==========================================

st.title("⚡ Bluestar M15 Sniper")
st.caption(f"Stratégie M15 | Assets: {len(ASSETS)}")

if st.button("SCANNER LE MARCHÉ", type="primary"):
    
    api = OandaClient()
    
    with st.spinner("Recherche d'opportunités..."):
        results = run_sniper_scan(api)
    
    st.divider()
    
    if not results:
        # Affichage sympa si rien n'est trouvé
        st.info("😴 Le marché est calme.")
        st.markdown("**Aucun signal valide détecté pour le moment.**")
        st.caption("Critères : RSI(7) croise 50 + HMA(20) alignée + Tendance de fond (H1/H4/D1) confirmée.")
    
    else:
        # Affichage des Cartes de Signaux
        st.success(f"🎯 {len(results)} opportunité(s) détectée(s) !")
        
        for sig in results:
            # Choix des couleurs et icônes
            if sig['type'] == 'BUY':
                icon = "🚀"
                color = "green"
                bg_color = "#d4edda" # Vert clair doux
                border_color = "#c3e6cb"
                text_color = "#155724"
            else:
                icon = "📉"
                color = "red"
                bg_color = "#f8d7da" # Rouge clair doux
                border_color = "#f5c6cb"
                text_color = "#721c24"

            # Création de la "Carte" Visuelle
            with st.container():
                st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    border: 1px solid {border_color};
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 10px;
                    color: {text_color};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; color: {text_color};">{icon} {sig['symbol']}</h3>
                            <span style="font-weight: bold; font-size: 1.2em;">{sig['type']} SIGNAL</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.5em; font-weight: bold;">{sig['price']:.5f}</div>
                            <small>RSI: {sig['rsi']:.1f} | MTF Score: {sig['score']}/3</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    # Petit timestamp discret
    st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
