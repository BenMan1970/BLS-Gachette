import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime
import time
import logging
from typing import Optional, Dict, List

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
st.set_page_config(page_title="Bluestar M15 Sniper Pro", layout="centered", page_icon="⚡")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# CSS amélioré
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .signal-card {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        animation: fadeIn 0.3s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .score-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 1em;
        font-weight: bold;
        margin: 2px;
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
    """
    Score MTF GPS amélioré : 0-3 points + Qualité A+/A/B/C
    Analyse D1, H4, H1
    """
    timeframes = {'D1': 'D', 'H4': 'H4', 'H1': 'H1'}
    analysis = {}
    
    for tf_name, tf_code in timeframes.items():
        df = api.get_candles(symbol, tf_code, count=300)
        if df.empty or len(df) < 50:
            analysis[tf_name] = {'trend': 'Neutral', 'score': 0, 'details': 'N/A', 'atr': 0}
        else:
            analysis[tf_name] = analyze_timeframe_gps(df, tf_name)
    
    # Calcul du score MTF
    score = 0
    details = []
    expected = 'Bullish' if direction == 'BUY' else 'Bearish'
    
    # Poids: D1=2, H4=1, H1=0.5
    weights = {'D1': 2.0, 'H4': 1.0, 'H1': 0.5}
    aligned_weight = 0
    
    for tf in ['D1', 'H4', 'H1']:
        if analysis[tf]['trend'] == expected:
            aligned_weight += weights[tf]
    
    # Score sur 3 points basé sur l'alignement pondéré
    total_weight = sum(weights.values())  # 3.5
    alignment_pct = (aligned_weight / total_weight) * 100
    
    if alignment_pct >= 85:  # ~3 TF alignés
        score = 3
        details.append("✅ Alignement FORT")
    elif alignment_pct >= 57:  # ~2 TF alignés
        score = 2
        details.append("⚠️ Alignement MOYEN")
    elif alignment_pct >= 28:  # ~1 TF aligné
        score = 1
        details.append("📊 Alignement FAIBLE")
    
    # Déterminer la qualité GPS
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
# SCANNER AVEC SCORING GPS + CURRENCY STRENGTH
# ==========================================

def run_sniper_scan(api: OandaClient, min_score: int = 4) -> List[Dict]:
    """
    Scanner avec système de scoring GPS + Currency Strength
    Score total : 0-10 points
    - RSI : 0-3 points
    - HMA : 0-2 points
    - MTF GPS : 0-3 points
    - Currency Strength : 0-2 points
    """
    signals = []
    skipped = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(ASSETS)
    
    # Pré-calculer Currency Strength une seule fois
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
            # 1. Données M15
            df_m15 = api.get_candles(symbol, "M15", count=150)
            if df_m15.empty or len(df_m15) < 50:
                skipped += 1
                continue

            # 2. Calcul des indicateurs
            rsi_series = get_rsi_ohlc4(df_m15)
            if rsi_series.empty:
                continue
            
            hma, hma_trend = get_colored_hma(df_m15)
            if hma_trend.empty:
                continue
            
            current_price = df_m15['close'].iloc[-1]
            atr_m15 = calculate_atr(df_m15, 14).iloc[-1]
            
            # 3. Test BUY
            rsi_buy = calculate_rsi_score(rsi_series, 'BUY')
            if rsi_buy['score'] > 0:
                hma_buy = calculate_hma_score(hma_trend, 'BUY')
                mtf_buy = calculate_mtf_score_gps(api, symbol, 'BUY')
                cs_buy = calculate_currency_strength_score(api, symbol, 'BUY')
                
                total_score = rsi_buy['score'] + hma_buy['score'] + mtf_buy['score'] + cs_buy['score']
                
                if total_score >= min_score:
                    # Déterminer la qualité
                    if total_score >= 9 and mtf_buy['quality'] in ['A+', 'A'] and cs_buy['score'] == 2:
                        quality = "🔥 PREMIUM"
                        quality_color = "#ff6b00"
                    elif total_score >= 8 and mtf_buy['quality'] in ['A+', 'A']:
                        quality = "💎 EXCELLENT"
                        quality_color = "#28a745"
                    elif total_score >= 6:
                        quality = "⭐ FORT"
                        quality_color = "#ffc107"
                    elif total_score >= 5:
                        quality = "✓ BON"
                        quality_color = "#17a2b8"
                    else:
                        quality = "⚠️ MOYEN"
                        quality_color = "#6c757d"
                    
                    # Déclassement si divergence Currency Strength
                    warning = ""
                    if cs_buy['score'] == 0 and symbol in FOREX_PAIRS:
                        if quality in ["🔥 PREMIUM", "💎 EXCELLENT"]:
                            quality = "⭐ FORT"
                            quality_color = "#ffc107"
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
                        "currency_strength": cs_buy
                    })
            
            # 4. Test SELL
            rsi_sell = calculate_rsi_score(rsi_series, 'SELL')
            if rsi_sell['score'] > 0:
                hma_sell = calculate_hma_score(hma_trend, 'SELL')
                mtf_sell = calculate_mtf_score_gps(api, symbol, 'SELL')
                cs_sell = calculate_currency_strength_score(api, symbol, 'SELL')
                
                total_score = rsi_sell['score'] + hma_sell['score'] + mtf_sell['score'] + cs_sell['score']
                
                if total_score >= min_score:
                    if total_score >= 9 and mtf_sell['quality'] in ['A+', 'A'] and cs_sell['score'] == 2:
                        quality = "🔥 PREMIUM"
                        quality_color = "#ff6b00"
                    elif total_score >= 8 and mtf_sell['quality'] in ['A+', 'A']:
                        quality = "💎 EXCELLENT"
                        quality_color = "#28a745"
                    elif total_score >= 6:
                        quality = "⭐ FORT"
                        quality_color = "#ffc107"
                    elif total_score >= 5:
                        quality = "✓ BON"
                        quality_color = "#17a2b8"
                    else:
                        quality = "⚠️ MOYEN"
                        quality_color = "#6c757d"
                    
                    warning = ""
                    if cs_sell['score'] == 0 and symbol in FOREX_PAIRS:
                        if quality in ["🔥 PREMIUM", "💎 EXCELLENT"]:
                            quality = "⭐ FORT"
                            quality_color = "#ffc107"
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
                        "currency_strength": cs_sell
                    })
        
        except Exception:
            skipped += 1
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if skipped > 0:
        st.caption(f"ℹ️ {skipped} actif(s) ignoré(s) (marché fermé ou données insuffisantes)")
    
    return signals

# ==========================================
# FONCTION D'AFFICHAGE DES SIGNAUX
# ==========================================

def display_signal(sig: Dict):
    """Affiche un signal formaté"""
    if sig['type'] == 'BUY':
        icon = "🚀"
        bg_color = "#d4edda"
        border_color = "#28a745"
        text_color = "#155724"
    else:
        icon = "📉"
        bg_color = "#f8d7da"
        border_color = "#dc3545"
        text_color = "#721c24"
    
    # Analyse MTF détaillée
    mtf_details = []
    for tf in ['D1', 'H4', 'H1']:
        tf_data = sig['mtf']['analysis'][tf]
        trend_icon = ""
        if tf_data['trend'] == 'Bullish':
            trend_icon = "🟢"
        elif tf_data['trend'] == 'Bearish':
            trend_icon = "🔴"
        elif tf_data['trend'] == 'Retracement':
            trend_icon = "🟠"
        else:
            trend_icon = "⚪"
        
        atr_display = f"{tf_data['atr']:.5f}" if tf_data['atr'] < 1 else f"{tf_data['atr']:.2f}"
        mtf_details.append(f"{trend_icon} <strong>{tf}:</strong> {tf_data['trend']} (ATR: {atr_display})")
    
    mtf_html = "<br>".join(mtf_details)
    
    # Construction du HTML
    atr_display = f"{sig['atr_m15']:.5f}" if sig['atr_m15'] < 1 else f"{sig['atr_m15']:.2f}"
    
    # Warning badge si divergence
    warning_html = ""
    if sig.get('warning'):
        warning_html = f'<div class="score-badge" style="background-color: #ff9800; color: white;">{sig["warning"]}</div>'
    
    html_content = f"""
    <div style="background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 20px; border-radius: 10px; margin-bottom: 15px; color: {text_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div>
                <h2 style="margin: 0; color: {text_color};">{icon} {sig['symbol']}</h2>
                <span style="font-weight: bold; font-size: 1.3em;">{sig['type']} SIGNAL</span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2.5em; font-weight: bold; color: {border_color};">{sig['total_score']}<span style="font-size: 0.5em;">/10</span></div>
                <div class="score-badge" style="background-color: {sig['quality_color']}; color: white;">{sig['quality']}</div>
                <div class="score-badge" style="background-color: #6c757d; color: white;">GPS: {sig['mtf']['quality']}</div>
                {warning_html}
            </div>
        </div>
        <div style="font-size: 1.5em; font-weight: bold; margin-bottom: 10px;">Prix : {sig['price']:.5f}</div>
        <div style="font-size: 1.1em; color: {text_color}; margin-bottom: 15px;">ATR M15 : {atr_display}</div>
        <div style="margin-top: 15px; padding-top: 15px; border-top: 2px solid {border_color};">
            <div style="margin-bottom: 10px;">
                <strong>📊 RSI(7) [{sig['rsi']['score']}/3]:</strong> {sig['rsi']['value']:.1f}<br>
                <span style="font-size: 0.9em;">{sig['rsi']['details']}</span>
            </div>
            <div style="margin-bottom: 10px;">
                <strong>📈 HMA(20) [{sig['hma']['score']}/2]:</strong> {sig['hma']['color']}<br>
                <span style="font-size: 0.9em;">{sig['hma']['details']}</span>
            </div>
            <div style="margin-bottom: 10px;">
                <strong>🌍 MTF GPS [{sig['mtf']['score']}/3] - Qualité: {sig['mtf']['quality']}</strong><br>
                <span style="font-size: 0.9em;">Alignement: {sig['mtf']['alignment']} | {sig['mtf']['details']}</span>
            </div>
            <div style="margin-bottom: 10px;">
                <strong>💪 Currency Strength [{sig['currency_strength']['score']}/2]:</strong><br>
                <span style="font-size: 0.9em;">{sig['currency_strength']['details']}</span><br>
                <span style="font-size: 0.85em; color: #666;">Scores: {sig['currency_strength']['base_score']:.2f}% vs {sig['currency_strength']['quote_score']:.2f}% | {sig['currency_strength']['rank_info']}</span>
            </div>
            <div style="margin-top: 10px; padding: 10px; background-color: rgba(255,255,255,0.3); border-radius: 5px; font-size: 0.9em;">
                {mtf_html}
            </div>
        </div>
    </div>
    """
    
    st.markdown(html_content, unsafe_allow_html=True)

# ==========================================
# INTERFACE UTILISATEUR
# ==========================================

st.title("⚡ Bluestar M15 Sniper Pro + Currency Strength")
st.caption(f"Scanner GPS + Currency Strength | {len(ASSETS)} actifs | Score max: 10/10")

# Réglage du score minimum
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
    
    # Métriques
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
        st.info(f"😴 **Aucun signal ≥ {min_score}/10 points**")
        st.caption("💡 Essaye de baisser le score minimum ou attends la prochaine opportunité")
    
    else:
        # Tri par score décroissant
        results_sorted = sorted(results, key=lambda x: (x['total_score'], x['mtf']['quality'], x['currency_strength']['score']), reverse=True)
        
        st.success(f"🎯 **{len(results)} signal(aux) détecté(s) !**")
        
        for sig in results_sorted:
            display_signal(sig)
    
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Score: RSI(3) + HMA(2) + MTF GPS(3) + Currency Strength(2) | 🔥 PREMIUM = 9-10/10")
