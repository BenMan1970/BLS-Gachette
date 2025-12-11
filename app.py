import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import time

# Configuration de la page
st.set_page_config(page_title="Auto Scanner Oanda", layout="wide", page_icon="📡")

# ==========================================
# 1. LISTE DES ACTIFS A SCANNER
# ==========================================
ASSETS = [
    # Forex Majeurs
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    # Forex Cross
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "CAD_JPY", "NZD_JPY",
    # Métaux
    "XAU_USD", "XPT_USD",
    # Indices
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
        except Exception as e:
            st.error("❌ Erreur Secrets Streamlit (OANDA_ACCESS_TOKEN / OANDA_ACCOUNT_ID manquants).")
            st.stop()

    def get_candles(self, instrument, granularity, count=100):
        # On réduit le count à 100 pour accélérer le scan global
        params = {"count": count, "granularity": granularity, "price": "M"}
        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
        except Exception:
            return pd.DataFrame() # Retour vide si erreur (marché fermé etc)
        
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
        return df

# ==========================================
# 3. INDICATEURS (Logique Mathématique)
# ==========================================

def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def get_rsi_ohlc4(df, length=7):
    # RSI sur (O+H+L+C)/4
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    delta = ohlc4.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_colored_hma(df, length=20):
    # HMA 20
    src = df['close']
    wma1 = calculate_wma(src, int(length / 2))
    wma2 = calculate_wma(src, length)
    raw_hma = 2 * wma1 - wma2
    hma = calculate_wma(raw_hma, int(np.round(np.sqrt(length))))
    
    # Tendance HMA (1=Vert, -1=Rouge)
    hma_prev = hma.shift(1)
    # Important : conversion en Series pour usage avec .iloc
    trend_array = np.where(hma > hma_prev, 1, -1)
    trend_series = pd.Series(trend_array, index=df.index)
    
    return hma, trend_series

def get_bluestar_trend(df):
    # ZLEMA Trend (MTF Alignment)
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
# 4. LOGIQUE D'ANALYSE D'UN ACTIF
# ==========================================

def analyze_asset(api, symbol):
    """
    Retourne une liste de signaux [Dict] pour cet actif si conditions remplies.
    """
    # 1. Récupération Data
    df_m15 = api.get_candles(symbol, "M15")
    df_h1 = api.get_candles(symbol, "H1")
    
    # Si pas de données M15 ou H1, on zappe
    if df_m15.empty or df_h1.empty:
        return []

    # Pour MTF, on a besoin de H4 et D1. 
    # Optimisation : On ne les charge que si M15 ou H1 a un potentiel croisement RSI.
    # (Cela économise des appels API).
    
    # Calcul RSI M15 et H1 pour pré-filtrage
    rsi_m15 = get_rsi_ohlc4(df_m15)
    rsi_h1 = get_rsi_ohlc4(df_h1)
    
    # Check Cross M15
    m15_curr = rsi_m15.iloc[-1]
    m15_prev = rsi_m15.iloc[-2]
    m15_cross_up = m15_prev < 50 and m15_curr > 50
    m15_cross_down = m15_prev > 50 and m15_curr < 50
    
    # Check Cross H1
    h1_curr = rsi_h1.iloc[-1]
    h1_prev = rsi_h1.iloc[-2]
    h1_cross_up = h1_prev < 50 and h1_curr > 50
    h1_cross_down = h1_prev > 50 and h1_curr < 50
    
    # Si aucun croisement nulle part, on arrête là pour cet actif
    if not (m15_cross_up or m15_cross_down or h1_cross_up or h1_cross_down):
        return []
    
    # Si on a un croisement, on charge le contexte MTF (H4, D1)
    df_h4 = api.get_candles(symbol, "H4")
    df_d1 = api.get_candles(symbol, "D")
    
    if df_h4.empty or df_d1.empty: return [] # Sécurité

    # Calcul MTF Global Trend
    t_h1 = get_bluestar_trend(df_h1)
    t_h4 = get_bluestar_trend(df_h4)
    t_d1 = get_bluestar_trend(df_d1)
    score = t_h1 + t_h4 + t_d1
    
    global_bull = (score >= 2)
    global_bear = (score <= -2)
    
    found_signals = []
    
    # --- VERIFICATION M15 ---
    if m15_cross_up or m15_cross_down:
        _, hma_trend_m15 = get_colored_hma(df_m15)
        hma_val = hma_trend_m15.iloc[-1]
        
        if m15_cross_up and hma_val == 1 and global_bull:
            found_signals.append({
                "Symbol": symbol, "TF": "M15", "Type": "BUY", 
                "RSI": round(m15_curr, 2), "MTF Score": score
            })
        elif m15_cross_down and hma_val == -1 and global_bear:
            found_signals.append({
                "Symbol": symbol, "TF": "M15", "Type": "SELL", 
                "RSI": round(m15_curr, 2), "MTF Score": score
            })

    # --- VERIFICATION H1 ---
    if h1_cross_up or h1_cross_down:
        _, hma_trend_h1 = get_colored_hma(df_h1)
        hma_val = hma_trend_h1.iloc[-1]
        
        if h1_cross_up and hma_val == 1 and global_bull:
            found_signals.append({
                "Symbol": symbol, "TF": "H1", "Type": "BUY", 
                "RSI": round(h1_curr, 2), "MTF Score": score
            })
        elif h1_cross_down and hma_val == -1 and global_bear:
            found_signals.append({
                "Symbol": symbol, "TF": "H1", "Type": "SELL", 
                "RSI": round(h1_curr, 2), "MTF Score": score
            })
            
    return found_signals

# ==========================================
# 5. INTERFACE DASHBOARD
# ==========================================

st.title("📡 Scanner Automatique de Signaux")
st.write(f"Surveillance de **{len(ASSETS)} actifs**. Stratégie : RSI(7) OHLC4 Cross 50 + HMA(20) + Alignement MTF.")

if st.button("LANCER LE SCAN GLOBAL"):
    api = OandaClient()
    results = []
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_assets = len(ASSETS)
    
    for i, asset in enumerate(ASSETS):
        # Mise à jour UI
        status_text.text(f"Analyse en cours : {asset} ({i+1}/{total_assets})")
        progress_bar.progress((i + 1) / total_assets)
        
        # Scan de l'actif
        signals = analyze_asset(api, asset)
        if signals:
            results.extend(signals)
            
    status_text.text("Scan terminé !")
    progress_bar.empty()
    
    # Affichage des résultats
    st.divider()
    
    if not results:
        st.warning("Aucun signal détecté pour le moment sur les marchés scannés.")
    else:
        st.success(f"🎯 {len(results)} Signal(aux) trouvé(s) !")
        
        # Création DataFrame pour affichage propre
        df_res = pd.DataFrame(results)
        
        # Coloration conditionnelle pour le style
        def color_type(val):
            color = 'green' if val == 'BUY' else 'red'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            df_res.style.map(color_type, subset=['Type']),
            use_container_width=True,
            hide_index=True
        )
