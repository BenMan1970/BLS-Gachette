
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

# Cache réduit pour alertes rapides
if 'cache' not in st.session_state:
    st.session_state.cache = {}
    st.session_state.cache_time = {}

CACHE_DURATION = 30  # Réduit à 30s pour alertes rapides

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

def get_bluestar_trend(df: pd.DataFrame) -> int:
    """Tendance ZLEMA"""
    if len(df) < 50:
        return 0
    
    length = min(70, len(df) - 10)
    src = df['close']
    lag = int((length - 1) / 2)
    src_lagged = src.shift(lag)
    zlema_input = src + (src - src_lagged)
    zlema = calculate_ema(zlema_input, length)
    
    if pd.isna(zlema.iloc[-1]):
        return 0
    
    current_close = src.iloc[-1]
    current_zlema = zlema.iloc[-1]
    return 1 if current_close > current_zlema else -1

# ==========================================
# SYSTÈME DE SCORING
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
        # Croisement confirmé = 3 points
        if prev_rsi < 50 and curr_rsi > 50:
            score = 3
            details.append("✅ Croisement haussier confirmé")
        # Proche du seuil avec momentum = 2 points
        elif 45 < curr_rsi < 50 and curr_rsi > prev_rsi:
            score = 2
            details.append("⚠️ Approche haussière (momentum +)")
        # Zone basse mais momentum positif = 1 point
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
        # Changement de couleur = 2 points
        if prev == -1 and curr == 1:
            score = 2
            details.append("✅ Changement VERT")
        # Déjà vert = 1 point
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

def calculate_mtf_score(api: OandaClient, symbol: str, direction: str) -> Dict:
    """Score MTF (2 TF seulement) : 0-2 points"""
    # On prend H1 et H4 seulement (pas D1 pour plus de réactivité)
    timeframes = {'H1': 'H1', 'H4': 'H4'}
    trends = {}
    
    for tf_name, tf_code in timeframes.items():
        df = api.get_candles(symbol, tf_code, count=100)
        if df.empty or len(df) < 50:
            trends[tf_name] = 0
        else:
            trends[tf_name] = get_bluestar_trend(df)
    
    score = 0
    details = []
    
    expected = 1 if direction == 'BUY' else -1
    
    # 2 TF alignés = 2 points
    if trends['H1'] == expected and trends['H4'] == expected:
        score = 2
        details.append("✅ H1+H4 alignés")
    # 1 TF aligné = 1 point
    elif trends['H1'] == expected or trends['H4'] == expected:
        score = 1
        aligned = 'H1' if trends['H1'] == expected else 'H4'
        details.append(f"⚠️ {aligned} aligné seulement")
    
    return {
        'score': score,
        'trends': trends,
        'details': ' | '.join(details) if details else 'Pas d\'alignement'
    }

# ==========================================
# SCANNER AVEC SCORING
# ==========================================

def run_sniper_scan(api: OandaClient, min_score: int = 3) -> List[Dict]:
    """Scanner avec système de scoring
    Score total : 0-7 points
    - RSI : 0-3 points
    - HMA : 0-2 points
    - MTF : 0-2 points
    """
    signals = []
    skipped = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(ASSETS)
    
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
            
            # 3. Test BUY
            rsi_buy = calculate_rsi_score(rsi_series, 'BUY')
            if rsi_buy['score'] > 0:  # Au moins un signal RSI
                hma_buy = calculate_hma_score(hma_trend, 'BUY')
                mtf_buy = calculate_mtf_score(api, symbol, 'BUY')
                
                total_score = rsi_buy['score'] + hma_buy['score'] + mtf_buy['score']
                
                if total_score >= min_score:
                    # Déterminer la qualité
                    if total_score >= 6:
                        quality = "🔥 EXCELLENT"
                        quality_color = "#28a745"
                    elif total_score >= 5:
                        quality = "⭐ FORT"
                        quality_color = "#ffc107"
                    elif total_score >= 4:
                        quality = "✓ BON"
                        quality_color = "#17a2b8"
                    else:
                        quality = "⚠️ MOYEN"
                        quality_color = "#6c757d"
                    
                    signals.append({
                        "symbol": symbol,
                        "type": "BUY",
                        "price": current_price,
                        "total_score": total_score,
                        "quality": quality,
                        "quality_color": quality_color,
                        "rsi": rsi_buy,
                        "hma": hma_buy,
                        "mtf": mtf_buy
                    })
            
            # 4. Test SELL
            rsi_sell = calculate_rsi_score(rsi_series, 'SELL')
            if rsi_sell['score'] > 0:
                hma_sell = calculate_hma_score(hma_trend, 'SELL')
                mtf_sell = calculate_mtf_score(api, symbol, 'SELL')
                
                total_score = rsi_sell['score'] + hma_sell['score'] + mtf_sell['score']
                
                if total_score >= min_score:
                    if total_score >= 6:
                        quality = "🔥 EXCELLENT"
                        quality_color = "#28a745"
                    elif total_score >= 5:
                        quality = "⭐ FORT"
                        quality_color = "#ffc107"
                    elif total_score >= 4:
                        quality = "✓ BON"
                        quality_color = "#17a2b8"
                    else:
                        quality = "⚠️ MOYEN"
                        quality_color = "#6c757d"
                    
                    signals.append({
                        "symbol": symbol,
                        "type": "SELL",
                        "price": current_price,
                        "total_score": total_score,
                        "quality": quality,
                        "quality_color": quality_color,
                        "rsi": rsi_sell,
                        "hma": hma_sell,
                        "mtf": mtf_sell
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
# INTERFACE UTILISATEUR
# ==========================================

st.title("⚡ Bluestar M15 Sniper Pro")
st.caption(f"Scanner Scoring System | {len(ASSETS)} actifs | Cache 30s")

# Réglage du score minimum
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    min_score = st.slider("Score minimum", 2, 7, 3, help="Score total sur 7 (RSI:3 + HMA:2 + MTF:2)")
with col2:
    scan_button = st.button("🎯 LANCER LE SCAN", type="primary", use_container_width=True)
with col3:
    if st.button("🗑️", use_container_width=True):
        st.session_state.cache = {}
        st.session_state.cache_time = {}
        st.success("✓")

if scan_button:
    api = OandaClient()
    
    start_time = time.time()
    
    with st.spinner("🔎 Analyse en cours..."):
        results = run_sniper_scan(api, min_score=min_score)
    
    scan_duration = time.time() - start_time
    
    st.divider()
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Signaux", len(results))
    with col2:
        st.metric("Score min", f"{min_score}/7")
    with col3:
        st.metric("Scan", f"{scan_duration:.1f}s")
    with col4:
        st.metric("Requêtes", api.request_count)
    
    st.divider()
    
    if not results:
        st.info(f"😴 **Aucun signal ≥ {min_score}/7 points**")
        st.caption("💡 Essaye de baisser le score minimum ou attends la prochaine opportunité")
    
    else:
        # Tri par score décroissant
        results_sorted = sorted(results, key=lambda x: x['total_score'], reverse=True)
        
        st.success(f"🎯 **{len(results)} signal(aux) détecté(s) !**")
        
        for sig in results_sorted:
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
            
            # Préparation des icônes MTF
            mtf_h1 = '✅' if sig['mtf']['trends']['H1'] == (1 if sig['type'] == 'BUY' else -1) else ('❌' if sig['mtf']['trends']['H1'] == (-1 if sig['type'] == 'BUY' else 1) else '⚪')
            mtf_h4 = '✅' if sig['mtf']['trends']['H4'] == (1 if sig['type'] == 'BUY' else -1) else ('❌' if sig['mtf']['trends']['H4'] == (-1 if sig['type'] == 'BUY' else 1) else '⚪')
            
            # Affichage avec colonnes Streamlit
            with st.container():
                st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    border-left: 5px solid {border_color};
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                    color: {text_color};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div>
                            <h2 style="margin: 0; color: {text_color};">{icon} {sig['symbol']}</h2>
                            <span style="font-weight: bold; font-size: 1.3em;">{sig['type']} SIGNAL</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 2.5em; font-weight: bold; color: {border_color};">
                                {sig['total_score']}<span style="font-size: 0.5em;">/7</span>
                            </div>
                            <div class="score-badge" style="background-color: {sig['quality_color']}; color: white;">
                                {sig['quality']}
                            </div>
                        </div>
                    </div>
                    
                    <div style="font-size: 1.5em; font-weight: bold; margin-bottom: 15px;">
                        Prix : {sig['price']:.5f}
                    </div>
                    
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 2px solid {border_color};">
                        <div style="margin-bottom: 10px;">
                            <strong>📊 RSI(7) [{sig['rsi']['score']}/3]:</strong> {sig['rsi']['value']:.1f}<br>
                            <span style="font-size: 0.9em;">{sig['rsi']['details']}</span>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <strong>📈 HMA(20) [{sig['hma']['score']}/2]:</strong> {sig['hma']['color']}<br>
                            <span style="font-size: 0.9em;">{sig['hma']['details']}</span>
                        </div>
                        <div>
                            <strong>🌍 MTF [{sig['mtf']['score']}/2]:</strong> H1: {mtf_h1} | H4: {mtf_h4}<br>
                            <span style="font-size: 0.9em;">{sig['mtf']['details']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Cache: {len(st.session_state.cache)} items | Score system: RSI(3) + HMA(2) + MTF(2)")
