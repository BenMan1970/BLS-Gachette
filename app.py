import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime, timezone
import time
import logging
from typing import Optional, Dict, List

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
st.set_page_config(page_title="Bluestar M15 Sniper Pro", layout="centered", page_icon="⚡")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# CSS MODERNE avec COULEURS VIBRANTES
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: #ffffff;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        height: 3.5em;
        font-weight: 700;
        font-size: 1.1em;
        border: none;
        background: #dc143c;
        color: white;
        box-shadow: 0 6px 20px rgba(220, 20, 60, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(220, 20, 60, 0.7);
        background: #c41230;
    }
    
    /* Signal cards */
    .signal-card {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        animation: slideIn 0.4s ease-out;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        background: #ffffff;
        border: 2px solid #e0e0e0;
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0; 
            transform: translateX(-20px);
        }
        to { 
            opacity: 1; 
            transform: translateX(0);
        }
    }
    
    /* Badges */
    .score-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 1.1em;
        font-weight: 700;
        margin: 5px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }
    
    /* Premium badge */
    .badge-premium {
        background: linear-gradient(135deg, #ff0080 0%, #ff8c00 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Excellent badge */
    .badge-excellent {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Fort badge */
    .badge-fort {
        background: linear-gradient(135deg, #00ff88 0%, #00cc99 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Bon badge */
    .badge-bon {
        background: linear-gradient(135deg, #ffaa00 0%, #ff6600 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Metrics - couleurs douces */
    div[data-testid="stMetricValue"] {
        font-size: 1.8em;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #555;
    }
    
    /* Expander - Actifs en GROS et GRAS */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white !important;
        border-radius: 12px;
        font-weight: 900;
        font-size: 1.4em;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        border: 2px solid #00ff88;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Style pour le nom de l'actif en TRÈS GROS */
    .streamlit-expanderHeader strong {
        font-size: 1.6em !important;
        font-weight: 900 !important;
        letter-spacing: 2px;
    }
    
    /* Success/Error boxes */
    .element-container div[data-baseweb="notification"] {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, #5b6ff5, transparent);
        margin: 20px 0;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #dc143c 0%, #b01030 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 2px 4px rgba(220, 20, 60, 0.3));
    }
    
    h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: #dc143c;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #dc143c;
    }
    
    /* Info boxes with gradient borders */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
        border-image: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%) 1;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Caption text */
    .caption, [data-testid="stCaptionContainer"] {
        color: #555 !important;
        font-weight: 500;
    }
    
    /* Better text contrast */
    p, span, div {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LISTE DES ACTIFS
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

# Liste des pairs Forex pour Currency Strength
FOREX_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "CHF_JPY"
]

# Cache réduit pour alertes rapides
if 'cache' not in st.session_state:
    st.session_state.cache = {}
    st.session_state.cache_time = {}
    st.session_state.currency_strength_cache = None
    st.session_state.currency_strength_time = 0

CACHE_DURATION = 30
CURRENCY_STRENGTH_CACHE_DURATION = 300  # 5 minutes pour Currency Strength

# ==========================================
# MOTEUR API ROBUSTE
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
            self.request_count = 0
            self.last_request_time = time.time()
        except KeyError as e:
            st.error(f"⚠️ Clé manquante dans les secrets: {e}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Erreur d'initialisation API: {str(e)}")
            st.stop()

    def _rate_limit(self):
        """Gestion du rate limiting (max 20 req/sec pour OANDA)"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)
        
        self.last_request_time = time.time()
        self.request_count += 1

    def get_candles(self, instrument: str, granularity: str, count: int = 150) -> pd.DataFrame:
        """Récupération des données avec cache"""
        
        # Vérifier le cache
        cache_key = f"{instrument}_{granularity}"
        if cache_key in st.session_state.cache:
            cache_age = time.time() - st.session_state.cache_time.get(cache_key, 0)
            if cache_age < CACHE_DURATION:
                return st.session_state.cache[cache_key].copy()
        
        self._rate_limit()
        
        params = {"count": count, "granularity": granularity, "price": "M"}
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                r = instruments.InstrumentsCandles(instrument=instrument, params=params)
                self.client.request(r)
                
                if 'candles' not in r.response:
                    return pd.DataFrame()
                
                data = []
                for candle in r.response['candles']:
                    if candle['complete']:
                        try:
                            data.append({
                                'time': candle['time'],
                                'open': float(candle['mid']['o']),
                                'high': float(candle['mid']['h']),
                                'low': float(candle['mid']['l']),
                                'close': float(candle['mid']['c']),
                                'volume': int(candle['volume'])
                            })
                        except (KeyError, ValueError):
                            continue
                
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                
                if len(df) < 50:
                    return pd.DataFrame()
                
                # Mise en cache
                st.session_state.cache[cache_key] = df.copy()
                st.session_state.cache_time[cache_key] = time.time()
                
                return df
                
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(0.3)
                    continue
                return pd.DataFrame()
        
        return pd.DataFrame()

# ==========================================
# INDICATEURS OPTIMISÉS
# ==========================================

def calculate_wma(series: pd.Series, length: int) -> pd.Series:
    """WMA optimisé"""
    if len(series) < length:
        return pd.Series(index=series.index, dtype=float)
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum() if len(x) == length else np.nan, 
        raw=True
    )

def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    """EMA standard"""
    return series.ewm(span=length, adjust=False).mean()

def calculate_sma(series: pd.Series, length: int) -> pd.Series:
    """SMA standard"""
    return series.rolling(window=length).mean()

def calculate_zlema(series: pd.Series, length: int) -> pd.Series:
    """ZLEMA - Zero Lag EMA"""
    if len(series) < length:
        return pd.Series(index=series.index, dtype=float)
    lag = int((length - 1) / 2)
    src_adj = series + (series - series.shift(lag))
    return src_adj.ewm(span=length, adjust=False).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR - Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """ADX - Average Directional Index"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx, plus_di, minus_di

def get_rsi_ohlc4(df: pd.DataFrame, length: int = 7) -> pd.Series:
    """RSI sur OHLC4"""
    if len(df) < length + 10:
        return pd.Series(index=df.index, dtype=float)
    
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    delta = ohlc4.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def get_colored_hma(df: pd.DataFrame, length: int = 20) -> tuple:
    """HMA coloré"""
    if len(df) < length + 10:
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=int)
    
    src = df['close']
    wma1 = calculate_wma(src, int(length / 2))
    wma2 = calculate_wma(src, length)
    raw_hma = 2 * wma1 - wma2
    hma = calculate_wma(raw_hma, int(np.round(np.sqrt(length))))
    
    hma_prev = hma.shift(1)
    trend_array = np.where(hma > hma_prev, 1, -1)
    trend_series = pd.Series(trend_array, index=df.index)
    
    return hma, trend_series

# ==========================================
# CURRENCY STRENGTH ENGINE (Market Map Pro)
# ==========================================

def calculate_currency_strength(api: OandaClient, lookback_days: int = 1) -> Dict[str, float]:
    """
    Calcule le score de force pour chaque devise (logique Market Map Pro)
    Returns: Dict[currency, weighted_score]
    """
    # Vérifier le cache
    cache_age = time.time() - st.session_state.currency_strength_time
    if st.session_state.currency_strength_cache and cache_age < CURRENCY_STRENGTH_CACHE_DURATION:
        return st.session_state.currency_strength_cache
    
    # Récupérer les données Forex
    forex_data = {}
    for pair in FOREX_PAIRS:
        try:
            df = api.get_candles(pair, "D", count=lookback_days + 5)
            if df is not None and len(df) > lookback_days:
                now = df['close'].iloc[-1]
                past = df['close'].shift(lookback_days).iloc[-1]
                pct = (now - past) / past * 100
                forex_data[pair] = pct
        except:
            continue
    
    # Construire les données par devise
    data = {}
    for symbol, pct in forex_data.items():
        parts = symbol.split('_')
        if len(parts) != 2:
            continue
        base, quote = parts[0], parts[1]
        
        if base not in data:
            data[base] = []
        if quote not in data:
            data[quote] = []
        
        data[base].append({'pct': pct, 'other': quote})
        data[quote].append({'pct': -pct, 'other': base})
    
    # Algorithme "Smart Weighted Score" (de Market Map Pro)
    currency_scores = {}
    
    for curr, items in data.items():
        score = 0
        valid_items = 0
        
        for item in items:
            opponent = item['other']
            val = item['pct']
            
            # Pondération : USD, EUR, JPY comptent double
            weight = 2.0 if opponent in ['USD', 'EUR', 'JPY'] else 1.0
            
            score += (val * weight)
            valid_items += weight
        
        # Score final normalisé
        final_score = score / valid_items if valid_items > 0 else 0
        currency_scores[curr] = final_score
    
    # Mise en cache
    st.session_state.currency_strength_cache = currency_scores
    st.session_state.currency_strength_time = time.time()
    
    return currency_scores

def calculate_currency_strength_score(api: OandaClient, symbol: str, direction: str) -> Dict:
    """
    Score Currency Strength : 0-2 points
    Analyse la force relative des devises base/quote
    """
    # Ignorer les non-Forex
    if symbol not in FOREX_PAIRS:
        return {
            'score': 0,
            'details': 'Non-Forex',
            'base_score': 0,
            'quote_score': 0,
            'rank_info': 'N/A'
        }
    
    parts = symbol.split('_')
    if len(parts) != 2:
        return {'score': 0, 'details': 'Format invalide', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
    
    base, quote = parts[0], parts[1]
    
    # Récupérer les scores de force
    try:
        strength_scores = calculate_currency_strength(api)
    except:
        return {'score': 0, 'details': 'Erreur calcul', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
    
    if base not in strength_scores or quote not in strength_scores:
        return {'score': 0, 'details': 'Données manquantes', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
    
    base_score = strength_scores[base]
    quote_score = strength_scores[quote]
    
    # Classement des devises
    sorted_currencies = sorted(strength_scores.items(), key=lambda x: x[1], reverse=True)
    base_rank = next(i for i, (curr, _) in enumerate(sorted_currencies, 1) if curr == base)
    quote_rank = next(i for i, (curr, _) in enumerate(sorted_currencies, 1) if curr == quote)
    total_currencies = len(sorted_currencies)
    
    score = 0
    details = []
    
    if direction == 'BUY':
        # BUY base/quote : on veut base forte et quote faible
        if base_rank <= 3 and quote_rank >= total_currencies - 2:
            score = 2
            details.append(f"✅ {base} TOP3 (#{base_rank}) & {quote} BOTTOM3 (#{quote_rank})")
        elif base_score > quote_score:
            score = 1
            details.append(f"📊 {base} > {quote} (Δ: {base_score - quote_score:+.2f}%)")
        else:
            score = 0
            details.append(f"⚠️ Divergence : {quote} plus fort que {base}")
    
    else:  # SELL
        # SELL base/quote : on veut quote forte et base faible
        if quote_rank <= 3 and base_rank >= total_currencies - 2:
            score = 2
            details.append(f"✅ {quote} TOP3 (#{quote_rank}) & {base} BOTTOM3 (#{base_rank})")
        elif quote_score > base_score:
            score = 1
            details.append(f"📊 {quote} > {base} (Δ: {quote_score - base_score:+.2f}%)")
        else:
            score = 0
            details.append(f"⚠️ Divergence : {base} plus fort que {quote}")
    
    rank_info = f"{base}:#{base_rank} vs {quote}:#{quote_rank}"
    
    return {
        'score': score,
        'details': ' | '.join(details),
        'base_score': base_score,
        'quote_score': quote_score,
        'rank_info': rank_info
    }

# ==========================================
# NOUVELLE LOGIQUE MTF GPS
# ==========================================

def analyze_timeframe_gps(df: pd.DataFrame, timeframe: str) -> Dict:
    """
    Analyse GPS d'un timeframe
    Returns: {trend, score, details, atr}
    """
    if df.empty or len(df) < 50:
        return {'trend': 'Neutral', 'score': 0, 'details': 'Données insuffisantes', 'atr': 0}
    
    close = df['close']
    curr_price = close.iloc[-1]
    
    # Calcul ATR
    atr_val = calculate_atr(df, 14).iloc[-1]
    
    # Pour H4/D1 : utiliser logique macro (SMA 200)
    if timeframe in ['H4', 'D1']:
        sma50 = calculate_sma(close, 50)
        sma200 = calculate_sma(close, 200)
        
        curr_sma50 = sma50.iloc[-1] if len(df) >= 50 else curr_price
        has_200 = len(df) >= 200
        curr_sma200 = sma200.iloc[-1] if has_200 else curr_sma50
        
        if has_200:
            if curr_price > curr_sma200:
                trend = "Bullish"
                score = 60
                if curr_price > curr_sma50: score += 20
                if curr_sma50 > curr_sma200: score += 20
                details = f"Prix > SMA200 ({curr_sma200:.5f})"
            else:
                trend = "Bearish"
                score = 60
                if curr_price < curr_sma50: score += 20
                if curr_sma50 < curr_sma200: score += 20
                details = f"Prix < SMA200 ({curr_sma200:.5f})"
        else:
            if curr_price > curr_sma50:
                trend = "Bullish"
                score = 50
                details = f"Prix > SMA50 ({curr_sma50:.5f})"
            else:
                trend = "Bearish"
                score = 50
                details = f"Prix < SMA50 ({curr_sma50:.5f})"
    
    # Pour H1/M15 : utiliser logique intraday (ZLEMA + ADX)
    else:
        zlema_val = calculate_zlema(close, 50)
        baseline = calculate_sma(close, 200)
        adx_val, _, _ = calculate_adx(df, 14)
        
        curr_zlema = zlema_val.iloc[-1]
        curr_adx = adx_val.iloc[-1]
        
        has_base = len(df) >= 200
        curr_base = baseline.iloc[-1] if has_base else curr_zlema
        
        trend = "Range"
        score = curr_adx
        
        # Logique de Retracement vs Tendance
        if curr_price > curr_zlema:
            if has_base and curr_price > curr_base:
                trend = "Bullish"
                details = f"Prix > ZLEMA & Baseline (ADX: {curr_adx:.1f})"
            elif has_base and curr_price < curr_base:
                trend = "Retracement"
                details = f"Hausse sous Baseline (ADX: {curr_adx:.1f})"
            else:
                trend = "Bullish"
                details = f"Prix > ZLEMA (ADX: {curr_adx:.1f})"
        elif curr_price < curr_zlema:
            if has_base and curr_price < curr_base:
                trend = "Bearish"
                details = f"Prix < ZLEMA & Baseline (ADX: {curr_adx:.1f})"
            elif has_base and curr_price > curr_base:
                trend = "Retracement"
                details = f"Baisse au-dessus Baseline (ADX: {curr_adx:.1f})"
            else:
                trend = "Bearish"
                details = f"Prix < ZLEMA (ADX: {curr_adx:.1f})"
        else:
            details = f"Range (ADX: {curr_adx:.1f})"
        
        if curr_adx < 20 and trend == "Retracement":
            trend = "Range"
            details = f"ADX faible ({curr_adx:.1f})"
    
    return {
        'trend': trend,
        'score': min(100, score),
        'details': details,
        'atr': atr_val
    }

# ==========================================
# SYSTÈME DE SCORING AMÉLIORÉ
# ==========================================

def calculate_rsi_score(rsi_series: pd.Series, direction: str) -> Dict:
    """Score RSI : 0-3 points"""
    if len(rsi_series) < 3:
        return {'score': 0, 'details': 'Données insuffisantes'}
    
    curr_rsi = rsi_series.iloc[-1]
    prev_rsi = rsi_series.iloc[-2]
    
    score = 0
    details = []
    
    if direction == 'BUY':
        if prev_rsi < 50 and curr_rsi > 50:
            score = 3
            details.append("✅ Croisement haussier confirmé")
        elif 45 < curr_rsi < 50 and curr_rsi > prev_rsi:
            score = 2
            details.append("⚠️ Approche haussière (momentum +)")
        elif curr_rsi < 50 and curr_rsi > prev_rsi:
            score = 1
            details.append("📊 Zone basse, momentum positif")
    
    else:  # SELL
        if prev_rsi > 50 and curr_rsi < 50:
            score = 3
            details.append("✅ Croisement baissier confirmé")
        elif 50 < curr_rsi < 55 and curr_rsi < prev_rsi:
            score = 2
            details.append("⚠️ Approche baissière (momentum -)")
        elif curr_rsi > 50 and curr_rsi < prev_rsi:
            score = 1
            details.append("📊 Zone haute, momentum négatif")
    
    return {
        'score': score,
        'value': curr_rsi,
        'details': ' | '.join(details) if details else 'Pas de signal'
    }

def calculate_hma_score(hma_trend: pd.Series, direction: str) -> Dict:
    """Score HMA : 0-2 points"""
    if len(hma_trend) < 2:
        return {'score': 0, 'details': 'Données insuffisantes'}
    
    curr = hma_trend.iloc[-1]
    prev = hma_trend.iloc[-2]
    
    score = 0
    details = []
    
    if direction == 'BUY':
        if prev == -1 and curr == 1:
            score = 2
            details.append("✅ Changement VERT")
        elif curr == 1:
            score = 1
            details.append("📈 Déjà VERT")
    
    else:  # SELL
        if prev == 1 and curr == -1:
            score = 2
            details.append("✅ Changement ROUGE")
        elif curr == -1:
            score = 1
            details.append("📉 Déjà ROUGE")
    
    return {
        'score': score,
        'color': 'VERT' if curr == 1 else 'ROUGE',
        'details': ' | '.join(details) if details else 'Neutre'
    }

def calculate_mtf_score_gps(api: OandaClient, symbol: str, direction: str) -> Dict:
    """Score MTF GPS amélioré : 0-3 points + Qualité A+/A/B/C"""
    timeframes = {'D1': 'D', 'H4': 'H4', 'H1': 'H1'}
    analysis = {}
    
    for tf_name, tf_code in timeframes.items():
        df = api.get_candles(symbol, tf_code, count=300)
        if df.empty or len(df) < 50:
            analysis[tf_name] = {'trend': 'Neutral', 'score': 0, 'details': 'N/A', 'atr': 0}
        else:
            analysis[tf_name] = analyze_timeframe_gps(df, tf_name)
    
    score = 0
    details = []
    expected = 'Bullish' if direction == 'BUY' else 'Bearish'
    
    weights = {'D1': 2.0, 'H4': 1.0, 'H1': 0.5}
    aligned_weight = 0
    
    for tf in ['D1', 'H4', 'H1']:
        if analysis[tf]['trend'] == expected:
            aligned_weight += weights[tf]
    
    total_weight = sum(weights.values())
    alignment_pct = (aligned_weight / total_weight) * 100
    
    if alignment_pct >= 85:
        score = 3
        details.append("✅ Alignement FORT")
    elif alignment_pct >= 57:
        score = 2
        details.append("⚠️ Alignement MOYEN")
    elif alignment_pct >= 28:
        score = 1
        details.append("📊 Alignement FAIBLE")
    
    quality = 'C'
    if analysis['D1']['trend'] == analysis['H4']['trend']:
        quality = 'B'
    if analysis['D1']['trend'] == analysis['H4']['trend'] == analysis['H1']['trend']:
        quality = 'A'
    if quality == 'A' and analysis['D1']['score'] > 70:
        quality = 'A+'
    
    return {
        'score': score,
        'quality': quality,
        'analysis': analysis,
        'alignment': f"{alignment_pct:.0f}%",
        'details': ' | '.join(details) if details else 'Pas d\'alignement'
    }

# ==========================================
# SCANNER
# ==========================================

def run_sniper_scan(api: OandaClient, min_score: int = 4) -> List[Dict]:
    signals = []
    skipped = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(ASSETS)
    
    status_text.text("🌍 Calcul de la force des devises...")
    try:
        calculate_currency_strength(api)
        status_text.text("✅ Currency Strength calculé")
    except:
        status_text.text("⚠️ Erreur Currency Strength")
    
    time.sleep(1)
    
    for i, symbol in enumerate(ASSETS):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"🔍 {symbol}... ({i+1}/{total})")
        
        try:
            df_m15 = api.get_candles(symbol, "M15", count=150)
            if df_m15.empty or len(df_m15) < 50:
                skipped += 1
                continue

            rsi_series = get_rsi_ohlc4(df_m15)
            if rsi_series.empty:
                continue
            
            hma, hma_trend = get_colored_hma(df_m15)
            if hma_trend.empty:
                continue
            
            current_price = df_m15['close'].iloc[-1]
            atr_m15 = calculate_atr(df_m15, 14).iloc[-1]
            signal_time_utc = df_m15['time'].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)
            
            # Test BUY
            rsi_buy = calculate_rsi_score(rsi_series, 'BUY')
            if rsi_buy['score'] > 0:
                hma_buy = calculate_hma_score(hma_trend, 'BUY')
                mtf_buy = calculate_mtf_score_gps(api, symbol, 'BUY')
                cs_buy = calculate_currency_strength_score(api, symbol, 'BUY')
                
                total_score = rsi_buy['score'] + hma_buy['score'] + mtf_buy['score'] + cs_buy['score']
                
                if total_score >= min_score:
                    if total_score >= 9 and mtf_buy['quality'] in ['A+', 'A'] and cs_buy['score'] == 2:
                        quality = "🔥 PREMIUM"
                        quality_color = "#ff0080"
                    elif total_score >= 8 and mtf_buy['quality'] in ['A+', 'A']:
                        quality = "💎 EXCELLENT"
                        quality_color = "#00d4ff"
                    elif total_score >= 6:
                        quality = "⭐ FORT"
                        quality_color = "#00ff88"
                    elif total_score >= 5:
                        quality = "✓ BON"
                        quality_color = "#ffaa00"
                    else:
                        quality = "⚠️ MOYEN"
                        quality_color = "#666666"
                    
                    warning = ""
                    if cs_buy['score'] == 0 and symbol in FOREX_PAIRS:
                        if quality in ["🔥 PREMIUM", "💎 EXCELLENT"]:
                            quality = "⭐ FORT"
                            quality_color = "#00ff88"
                        warning = "⚠️ Divergence devise"
                    
                    signals.append({
                        "symbol": symbol,
                        "type": "BUY",
                        "price": current_price,
                        "atr_m15": atr_m15,
                        "total_score": total_score,
                        "quality": quality,
                        "quality_color": quality_color,
                        "warning": warning,
                        "rsi": rsi_buy,
                        "hma": hma_buy,
                        "mtf": mtf_buy,
                        "currency_strength": cs_buy,
                        "timestamp_utc": signal_time_utc
                    })
            
            # Test SELL
            rsi_sell = calculate_rsi_score(rsi_series, 'SELL')
            if rsi_sell['score'] > 0:
                hma_sell = calculate_hma_score(hma_trend, 'SELL')
                mtf_sell = calculate_mtf_score_gps(api, symbol, 'SELL')
                cs_sell = calculate_currency_strength_score(api, symbol, 'SELL')
                
                total_score = rsi_sell['score'] + hma_sell['score'] + mtf_sell['score'] + cs_sell['score']
                
                if total_score >= min_score:
                    if total_score >= 9 and mtf_sell['quality'] in ['A+', 'A'] and cs_sell['score'] == 2:
                        quality = "🔥 PREMIUM"
                        quality_color = "#f093fb"
                    elif total_score >= 8 and mtf_sell['quality'] in ['A+', 'A']:
                        quality = "💎 EXCELLENT"
                        quality_color = "#4facfe"
                    elif total_score >= 6:
                        quality = "⭐ FORT"
                        quality_color = "#43e97b"
                    elif total_score >= 5:
                        quality = "✓ BON"
                        quality_color = "#fa709a"
                    else:
                        quality = "⚠️ MOYEN"
                        quality_color = "#a8a8a8"
                    
                    warning = ""
                    if cs_sell['score'] == 0 and symbol in FOREX_PAIRS:
                        if quality in ["🔥 PREMIUM", "💎 EXCELLENT"]:
                            quality = "⭐ FORT"
                            quality_color = "#43e97b"
                        warning = "⚠️ Divergence devise"
                    
                    signals.append({
                        "symbol": symbol,
                        "type": "SELL",
                        "price": current_price,
                        "atr_m15": atr_m15,
                        "total_score": total_score,
                        "quality": quality,
                        "quality_color": quality_color,
                        "warning": warning,
                        "rsi": rsi_sell,
                        "hma": hma_sell,
                        "mtf": mtf_sell,
                        "currency_strength": cs_sell,
                        "timestamp_utc": signal_time_utc
                    })
        
        except Exception:
            skipped += 1
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if skipped > 0:
        st.caption(f"ℹ️ {skipped} actif(s) ignoré(s)")
    
    return signals

# ==========================================
# AFFICHAGE DES SIGNAUX - DESIGN VIBRANT
# ==========================================

def display_signal(sig: Dict):
    """Affiche un signal avec design moderne et coloré"""
    if sig['type'] == 'BUY':
        icon = "🚀"
        emoji_direction = "📈"
        card_gradient = "linear-gradient(135deg, #00ff88 0%, #00d4ff 100%)"
        border_color = "#00ff88"
    else:
        icon = "📉"
        emoji_direction = "📉"
        card_gradient = "linear-gradient(135deg, #ff0080 0%, #ff4500 100%)"
        border_color = "#ff0080"
    
    signal_utc = sig['timestamp_utc']
    time_utc_str = signal_utc.strftime("%H:%M:%S")
    date_str = signal_utc.strftime("%Y-%m-%d")
    
    now_utc = datetime.now(timezone.utc)
    elapsed = now_utc - signal_utc
    elapsed_seconds = int(elapsed.total_seconds())
    
    if elapsed_seconds > 1800:
        freshness = f"Bougie du {date_str} à {time_utc_str} UTC"
        freshness_emoji = "⚪"
        freshness_label = "(Marché fermé)"
    elif elapsed_seconds < 60:
        freshness = f"il y a {elapsed_seconds}s"
        freshness_emoji = "🟢"
        freshness_label = ""
    elif elapsed_seconds < 300:
        freshness = f"il y a {elapsed_seconds // 60}min"
        freshness_emoji = "🟢"
        freshness_label = ""
    elif elapsed_seconds < 900:
        freshness = f"il y a {elapsed_seconds // 60}min"
        freshness_emoji = "🟡"
        freshness_label = ""
    else:
        freshness = f"il y a {elapsed_seconds // 60}min"
        freshness_emoji = "🔴"
        freshness_label = "(Signal ancien)"
    
    expander_title = f"{icon} **{sig['symbol']}** {emoji_direction} {sig['type']} • Score: **{sig['total_score']}/10** • {sig['quality']}"
    
    with st.expander(expander_title, expanded=True):
        # Header simple sans néon
        st.markdown(f"""
        <div style="background: {card_gradient}; padding: 15px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);">
            <h3 style="margin: 0; color: white;">🕐 {date_str} à {time_utc_str} UTC</h3>
            <p style="margin: 5px 0 0 0; font-size: 1.1em;">{freshness_emoji} {freshness} {freshness_label}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1.5])
        
        with col1:
            st.metric("💰 Prix", f"{sig['price']:.5f}")
        with col2:
            st.metric("📊 Score", f"{sig['total_score']}/10")
        with col3:
            st.markdown(f"""
            <div style="background: {sig['quality_color']}; padding: 12px; border-radius: 12px; text-align: center; color: white; font-weight: 800; font-size: 0.95em; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                {sig['quality']}
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffa500 0%, #ff6347 100%); padding: 12px; border-radius: 12px; text-align: center; color: white; font-weight: 800; font-size: 0.95em; box-shadow: 0 4px 12px rgba(255, 165, 0, 0.3);">
                GPS: {sig['mtf']['quality']}
            </div>
            """, unsafe_allow_html=True)
        with col5:
            atr_display = f"{sig['atr_m15']:.5f}" if sig['atr_m15'] < 1 else f"{sig['atr_m15']:.2f}"
            st.metric("⚡ ATR M15", atr_display)
        
        if sig.get('warning'):
            st.warning(f"⚠️ {sig['warning']}")
        
        st.divider()
        
        # Indicateurs
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"### 📊 RSI(7) - **{sig['rsi']['score']}/3** pts")
            
            # Jauge RSI avec couleur douce
            rsi_val = sig['rsi']['value']
            if rsi_val < 30:
                rsi_color = "#ff6b9d"  # Rose doux
            elif rsi_val > 70:
                rsi_color = "#ffa07a"  # Orange doux
            else:
                rsi_color = "#87ceeb"  # Bleu ciel doux
            
            st.markdown(f"""
            <div style="background: {rsi_color}; padding: 10px; border-radius: 10px; text-align: center; color: white; font-weight: 700; margin-bottom: 10px;">
                RSI: {rsi_val:.1f}
            </div>
            """, unsafe_allow_html=True)
            
            if sig['rsi']['score'] >= 2:
                st.success(sig['rsi']['details'])
            elif sig['rsi']['score'] == 1:
                st.info(sig['rsi']['details'])
            else:
                st.caption(sig['rsi']['details'])
            
            st.write("")
            
            st.markdown(f"### 📈 HMA(20) - **{sig['hma']['score']}/2** pts")
            if sig['hma']['color'] == 'VERT':
                st.markdown("""
                <div style="background: linear-gradient(135deg, #98fb98 0%, #90ee90 100%); padding: 15px; border-radius: 12px; color: white; font-weight: 800; text-align: center; box-shadow: 0 4px 12px rgba(152, 251, 152, 0.3);">
                    🟢 VERT
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ffb6c1 0%, #ffb3d9 100%); padding: 15px; border-radius: 12px; color: white; font-weight: 800; text-align: center; box-shadow: 0 4px 12px rgba(255, 182, 193, 0.3);">
                    🔴 ROUGE
                </div>
                """, unsafe_allow_html=True)
            
            st.write("")
            if sig['hma']['score'] >= 1:
                st.success(sig['hma']['details'])
            else:
                st.caption(sig['hma']['details'])
        
        with col_right:
            st.markdown(f"### 🌍 MTF GPS - **{sig['mtf']['score']}/3** pts")
            st.metric("Qualité GPS", sig['mtf']['quality'])
            st.metric("Alignement", sig['mtf']['alignment'])
            
            if sig['mtf']['score'] >= 2:
                st.success(sig['mtf']['details'])
            elif sig['mtf']['score'] == 1:
                st.info(sig['mtf']['details'])
            else:
                st.caption(sig['mtf']['details'])
            
            st.write("")
            
            st.markdown(f"### 💪 Currency Strength - **{sig['currency_strength']['score']}/2** pts")
            if sig['currency_strength']['score'] == 2:
                st.success(sig['currency_strength']['details'])
            elif sig['currency_strength']['score'] == 1:
                st.info(sig['currency_strength']['details'])
            else:
                st.warning(sig['currency_strength']['details'])
            
            st.caption(f"📊 {sig['currency_strength']['base_score']:.2f}% vs {sig['currency_strength']['quote_score']:.2f}% | {sig['currency_strength']['rank_info']}")
        
        st.divider()
        
        # MTF Analysis
        st.markdown("### 📅 Analyse Multi-Timeframe")
        
        col_d1, col_h4, col_h1 = st.columns(3)
        
        timeframes = ['D1', 'H4', 'H1']
        columns = [col_d1, col_h4, col_h1]
        
        for tf, col in zip(timeframes, columns):
            tf_data = sig['mtf']['analysis'][tf]
            
            with col:
                if tf_data['trend'] == 'Bullish':
                    gradient = "linear-gradient(135deg, #98fb98 0%, #90ee90 100%)"
                    emoji = "🟢"
                    shadow = "rgba(152, 251, 152, 0.3)"
                elif tf_data['trend'] == 'Bearish':
                    gradient = "linear-gradient(135deg, #ffb6c1 0%, #ffb3d9 100%)"
                    emoji = "🔴"
                    shadow = "rgba(255, 182, 193, 0.3)"
                elif tf_data['trend'] == 'Retracement':
                    gradient = "linear-gradient(135deg, #ffd700 0%, #ffb347 100%)"
                    emoji = "🟠"
                    shadow = "rgba(255, 215, 0, 0.3)"
                else:
                    gradient = "linear-gradient(135deg, #d3d3d3 0%, #c0c0c0 100%)"
                    emoji = "⚪"
                    shadow = "rgba(211, 211, 211, 0.3)"
                
                st.markdown(f"""
                <div style="background: {gradient}; padding: 15px; border-radius: 12px; color: white; text-align: center; margin-bottom: 10px; box-shadow: 0 4px 12px {shadow};">
                    <h3 style="margin: 0; color: white; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{emoji} {tf}</h3>
                    <p style="margin: 5px 0 0 0; font-weight: 700; font-size: 1.1em;">{tf_data['trend']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                atr_display = f"{tf_data['atr']:.5f}" if tf_data['atr'] < 1 else f"{tf_data['atr']:.2f}"
                st.caption(f"ATR: {atr_display}")
                st.caption(f"{tf_data['details']}")

# ==========================================
# INTERFACE PRINCIPALE
# ==========================================

st.title("⚡ Bluestar M15 Sniper Pro")
st.markdown("""
<div style="background: #dc143c; padding: 20px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px; box-shadow: 0 6px 20px rgba(220, 20, 60, 0.4);">
    <h3 style="margin: 0; color: white; font-weight: 800;">🎯 Scanner GPS + Currency Strength</h3>
    <p style="margin: 5px 0 0 0; font-weight: 600; font-size: 1.1em;">{} actifs • Score max: 10/10</p>
</div>
""".format(len(ASSETS)), unsafe_allow_html=True)

# Controls
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    min_score = st.slider("Score minimum", 3, 10, 5, help="Score total sur 10 (RSI:3 + HMA:2 + MTF:3 + CS:2)")
with col2:
    scan_button = st.button("🎯 LANCER LE SCAN", type="primary", use_container_width=True)
with col3:
    if st.button("🗑️", use_container_width=True):
        st.session_state.cache = {}
        st.session_state.cache_time = {}
        st.session_state.currency_strength_cache = None
        st.session_state.currency_strength_time = 0
        st.success("✓")

if scan_button:
    api = OandaClient()
    
    start_time = time.time()
    
    with st.spinner("🔎 Analyse GPS + Currency Strength en cours..."):
        results = run_sniper_scan(api, min_score=min_score)
    
    scan_duration = time.time() - start_time
    
    st.divider()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Signaux", len(results))
    with col2:
        st.metric("Score min", f"{min_score}/10")
    with col3:
        st.metric("Scan", f"{scan_duration:.1f}s")
    with col4:
        st.metric("Requêtes", api.request_count)
    
    st.divider()
    
    if not results:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d3d3d3 0%, #c0c0c0 100%); padding: 25px; border-radius: 12px; color: white; text-align: center; box-shadow: 0 6px 20px rgba(0,0,0,0.2);">
            <h3 style="margin: 0; color: white; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">😴 Aucun signal ≥ {}/10 points</h3>
            <p style="margin: 10px 0 0 0; font-weight: 600; font-size: 1.1em;">💡 Essaie de baisser le score minimum</p>
        </div>
        """.format(min_score), unsafe_allow_html=True)
    
    else:
        results_sorted = sorted(results, key=lambda x: (x['total_score'], x['mtf']['quality'], x['currency_strength']['score']), reverse=True)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #98fb98 0%, #90ee90 100%); padding: 25px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px; box-shadow: 0 6px 20px rgba(152, 251, 152, 0.4);">
            <h3 style="margin: 0; color: white; font-weight: 800;">🎯 {len(results)} signal(aux) détecté(s) !</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for sig in results_sorted:
            display_signal(sig)
    
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Score: RSI(3) + HMA(2) + MTF GPS(3) + CS(2) | 🔥 PREMIUM = 9-10/10")
