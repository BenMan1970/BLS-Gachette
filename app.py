import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Configuration
st.set_page_config(page_title="Scanner H1 Diagnostic", layout="wide", page_icon="🔍")

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
            st.error("⚠️ Erreur Secrets Oanda.")
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
    # OHLC4
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
    trend_array = np.where(hma > hma_prev, 1, -1) # 1=Vert, -1=Rouge
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
# 4. LOGIQUE DIAGNOSTIC (SCAN H1)
# ==========================================

def scan_h1_debug(api):
    overview_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(ASSETS)
    
    for i, symbol in enumerate(ASSETS):
        status_text.text(f"Analyse H1 : {symbol}...")
        progress_bar.progress((i + 1) / total)
        
        # 1. Données
        df_h1 = api.get_candles(symbol, "H1")
        if df_h1.empty: continue

        # 2. RSI OHLC4 (7)
        rsi_series = get_rsi_ohlc4(df_h1)
        curr_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]
        
        # Détection Cross Strict
        cross_up = prev_rsi < 50 and curr_rsi > 50
        cross_down = prev_rsi > 50 and curr_rsi < 50
        
        cross_status = "Aucun"
        if cross_up: cross_status = "⬆️ CROSS UP"
        if cross_down: cross_status = "⬇️ CROSS DOWN"

        # 3. HMA (20) Couleur
        _, hma_trend = get_colored_hma(df_h1)
        hma_val = hma_trend.iloc[-1] # 1 ou -1
        hma_str = "🟢 Verte" if hma_val == 1 else "🔴 Rouge"

        # 4. MTF (H1 + H4 + D1)
        df_h4 = api.get_candles(symbol, "H4")
        df_d1 = api.get_candles(symbol, "D")
        
        mtf_status = "N/A"
        mtf_valid = False
        
        if not df_h4.empty and not df_d1.empty:
            t_h1 = get_bluestar_trend(df_h1)
            t_h4 = get_bluestar_trend(df_h4)
            t_d1 = get_bluestar_trend(df_d1)
            score = t_h1 + t_h4 + t_d1
            
            if score >= 2: mtf_status = "🐂 BULLISH"
            elif score <= -2: mtf_status = "🐻 BEARISH"
            else: mtf_status = "⚪ NEUTRE"
            
            # Validation Logic
            if cross_up and hma_val == 1 and score >= 2:
                mtf_valid = True
            elif cross_down and hma_val == -1 and score <= -2:
                mtf_valid = True
        
        # Résultat final pour cette paire
        signal_final = "❌"
        if mtf_valid:
            signal_final = "✅ BUY" if cross_up else "✅ SELL"

        overview_data.append({
            "Symbole": symbol,
            "RSI Préc": round(prev_rsi, 2),
            "RSI Act": round(curr_rsi, 2),
            "Etat RSI": cross_status,
            "HMA": hma_str,
            "Tendance MTF": mtf_status,
            "SIGNAL": signal_final
        })

    progress_bar.empty()
    status_text.empty()
    return overview_data

# ==========================================
# 5. AFFICHAGE
# ==========================================

st.title("Scanner H1 - Diagnostic Complet")
st.write("Ce tableau affiche l'état de **toutes** les paires, même s'il n'y a pas de signal, pour vérifier la logique.")

if st.button("LANCER LE DIAGNOSTIC H1", type="primary"):
    api = OandaClient()
    
    with st.spinner("Récupération des données H1/H4/D1..."):
        data = scan_h1_debug(api)
    
    if data:
        df = pd.DataFrame(data)
        
        # MISE EN FORME DU TABLEAU (Correction Visuelle)
        def style_dataframe(row):
            # 1. Signaux Valides (Fond Vert/Rouge, Texte Noir/Blanc)
            if "BUY" in row["SIGNAL"]:
                return ['background-color: #28a745; color: white; font-weight: bold'] * len(row)
            elif "SELL" in row["SIGNAL"]:
                return ['background-color: #dc3545; color: white; font-weight: bold'] * len(row)
            
            # 2. Croisement RSI seul (Fond Jaune, TEXTE NOIR FORCE)
            elif "CROSS" in row["Etat RSI"]:
                return ['background-color: #fff3cd; color: black; font-weight: bold'] * len(row)
            
            else:
                return [''] * len(row)

        st.dataframe(
            df.style.apply(style_dataframe, axis=1), 
            use_container_width=True # Correction du paramètre
        )
        
        # Résumé
        signals = [d for d in data if "✅" in d["SIGNAL"]]
        if signals:
            st.success(f"🎉 {len(signals)} Signal(aux) confirmé(s) !")
        else:
            st.warning("Aucun signal validé. Regardez les lignes jaunes : ce sont des croisements RSI qui ont échoué à cause de la HMA ou du MTF.")
    else:
        st.error("Aucune donnée récupérée.")
