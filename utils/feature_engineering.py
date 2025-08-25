#!/usr/bin/env python3
"""
Advanced Feature Engineering for Gold Trading
Comprehensive feature creation and engineering for AI models

Author: AI Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Advanced feature engineering for gold trading analysis
    Creates comprehensive features from market, fundamental, and sentiment data
    """
    
    def __init__(self):
        self.scalers = {}
        print("🔧 Feature Engineer initialized")

    def create_features_from_single_dataframe(self, data):
        """
        Create comprehensive features from a single OHLCV dataframe
        Designed for training pipeline compatibility

        Args:
            data (pd.DataFrame): OHLCV data with columns: open, high, low, close, volume

        Returns:
            pd.DataFrame: Enhanced dataset with engineered features
        """
        try:
            print("🔧 Creating features from single dataframe...")

            # Start with original data
            features = data.copy()

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in features.columns:
                    print(f"⚠️  Missing column {col}, using close price")
                    features[col] = features.get('close', 2000.0)

            # Add volume if missing
            if 'volume' not in features.columns:
                features['volume'] = np.random.lognormal(10, 1, len(features))

            # Technical indicators
            features = self._add_single_df_technical_indicators(features)

            # Price patterns and relationships
            features = self._add_single_df_price_patterns(features)

            # Volume analysis
            features = self._add_single_df_volume_features(features)

            # Statistical features
            features = self._add_single_df_statistical_features(features)

            # Time-based features
            features = self._add_single_df_time_features(features)

            # Market regime features
            features = self._add_single_df_market_regime_features(features)

            # Advanced mathematical features
            features = self._add_single_df_mathematical_features(features)

            # Clean features
            features = self._clean_single_df_features(features)

            print(f"✅ Single dataframe feature engineering complete: {features.shape[1]} features")
            return features

        except Exception as e:
            print(f"❌ Single dataframe feature engineering error: {e}")
            return data

    def _add_single_df_technical_indicators(self, features):
        """Add comprehensive technical indicators"""
        try:
            # Convert to float arrays for TA-Lib
            close = pd.to_numeric(features['close'], errors='coerce').values.astype(np.float64)
            high = pd.to_numeric(features['high'], errors='coerce').values.astype(np.float64)
            low = pd.to_numeric(features['low'], errors='coerce').values.astype(np.float64)
            open_prices = pd.to_numeric(features['open'], errors='coerce').values.astype(np.float64)

            if 'volume' in features.columns:
                volume = pd.to_numeric(features['volume'], errors='coerce').values.astype(np.float64)
            else:
                volume = None

            # Moving averages
            for period in [5, 10, 20, 50, 100]:
                features[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                features[f'ema_{period}'] = talib.EMA(close, timeperiod=period)

            # Price ratios
            features['price_sma20_ratio'] = close / features['sma_20']
            features['sma20_sma50_ratio'] = features['sma_20'] / features['sma_50']

            # Momentum indicators
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            features['rsi_21'] = talib.RSI(close, timeperiod=21)
            features['roc_10'] = talib.ROC(close, timeperiod=10)
            features['roc_20'] = talib.ROC(close, timeperiod=20)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d

            # ATR and volatility
            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features['atr_20'] = talib.ATR(high, low, close, timeperiod=20)

            # Williams %R
            features['williams_r'] = talib.WILLR(high, low, close)

            # CCI
            features['cci'] = talib.CCI(high, low, close)

            # ADX
            features['adx'] = talib.ADX(high, low, close)
            features['plus_di'] = talib.PLUS_DI(high, low, close)
            features['minus_di'] = talib.MINUS_DI(high, low, close)

            # Volume indicators (if volume available)
            if volume is not None:
                features['obv'] = talib.OBV(close, volume)
                features['ad'] = talib.AD(high, low, close, volume)
                features['adosc'] = talib.ADOSC(high, low, close, volume)

            return features

        except Exception as e:
            print(f"⚠️  Technical indicators error: {e}")
            return features

    def _add_single_df_price_patterns(self, features):
        """Add price pattern features"""
        try:
            # Convert to float arrays
            close = pd.to_numeric(features['close'], errors='coerce').values.astype(np.float64)
            high = pd.to_numeric(features['high'], errors='coerce').values.astype(np.float64)
            low = pd.to_numeric(features['low'], errors='coerce').values.astype(np.float64)
            open_prices = pd.to_numeric(features['open'], errors='coerce').values.astype(np.float64)

            # Price changes
            features['price_change'] = close - open_prices
            features['price_change_pct'] = (close - open_prices) / open_prices
            features['high_low_range'] = high - low
            features['high_low_range_pct'] = (high - low) / close

            # Candlestick patterns
            features['doji'] = talib.CDLDOJI(open_prices, high, low, close)
            features['hammer'] = talib.CDLHAMMER(open_prices, high, low, close)
            features['hanging_man'] = talib.CDLHANGINGMAN(open_prices, high, low, close)
            features['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high, low, close)
            features['engulfing'] = talib.CDLENGULFING(open_prices, high, low, close)

            # Price momentum
            for period in [3, 5, 10, 20]:
                features[f'momentum_{period}'] = talib.MOM(close, timeperiod=period)
                features[f'price_change_{period}'] = np.log(close / np.roll(close, period))

            # Support and resistance levels
            features['resistance_level'] = pd.Series(high).rolling(20).max()
            features['support_level'] = pd.Series(low).rolling(20).min()
            features['price_to_resistance'] = close / features['resistance_level']
            features['price_to_support'] = close / features['support_level']

            # Trend features
            features['higher_highs'] = (pd.Series(high).rolling(5).max() >
                                      pd.Series(high).shift(5).rolling(5).max()).astype(int)
            features['lower_lows'] = (pd.Series(low).rolling(5).min() <
                                    pd.Series(low).shift(5).rolling(5).min()).astype(int)

            return features

        except Exception as e:
            print(f"⚠️  Price patterns error: {e}")
            return features

    def _add_single_df_volume_features(self, features):
        """Add volume-based features"""
        try:
            if 'volume' not in features.columns:
                return features

            # Convert to float arrays
            close = pd.to_numeric(features['close'], errors='coerce').values.astype(np.float64)
            volume = pd.to_numeric(features['volume'], errors='coerce').values.astype(np.float64)

            # Volume moving averages
            features['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
            features['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            features['volume_ratio'] = volume / features['volume_sma_20']

            # Price-volume features
            features['vwap'] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
            features['price_vwap_ratio'] = close / features['vwap']

            # Volume oscillator
            features['volume_oscillator'] = ((features['volume_sma_10'] - features['volume_sma_20']) /
                                           features['volume_sma_20'] * 100)

            # Money flow
            typical_price = (features['high'] + features['low'] + features['close']) / 3
            money_flow = typical_price * volume
            features['money_flow_20'] = money_flow.rolling(20).sum()

            return features

        except Exception as e:
            print(f"⚠️  Volume features error: {e}")
            return features

    def _add_single_df_statistical_features(self, features):
        """Add statistical features"""
        try:
            close = features['close'].values

            # Rolling statistics
            for window in [10, 20, 50]:
                close_series = pd.Series(close)
                features[f'volatility_{window}'] = close_series.pct_change().rolling(window).std()
                features[f'skewness_{window}'] = close_series.rolling(window).skew()
                features[f'kurtosis_{window}'] = close_series.rolling(window).kurt()

            # Z-scores
            features['zscore_20'] = ((close - pd.Series(close).rolling(20).mean()) /
                                   pd.Series(close).rolling(20).std())

            # Percentile ranks
            features['percentile_rank_50'] = pd.Series(close).rolling(50).rank(pct=True)

            # Autocorrelation
            close_series = pd.Series(close)
            features['autocorr_1'] = close_series.rolling(20).apply(lambda x: x.autocorr(lag=1))
            features['autocorr_5'] = close_series.rolling(20).apply(lambda x: x.autocorr(lag=5))

            return features

        except Exception as e:
            print(f"⚠️  Statistical features error: {e}")
            return features

    def _add_single_df_time_features(self, features):
        """Add time-based features"""
        try:
            if 'datetime' in features.columns:
                dt = pd.to_datetime(features['datetime'])

                # Basic time features
                features['hour'] = dt.dt.hour
                features['day_of_week'] = dt.dt.dayofweek
                features['month'] = dt.dt.month
                features['quarter'] = dt.dt.quarter

                # Cyclical encoding
                features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
                features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
                features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
                features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
                features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
                features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

                # Market session features
                features['is_market_open'] = ((features['hour'] >= 9) & (features['hour'] <= 16)).astype(int)
                features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)

            else:
                # Create synthetic time features if datetime not available
                features['synthetic_time'] = np.arange(len(features))
                features['hour_sin'] = np.sin(2 * np.pi * features['synthetic_time'] / 24)
                features['hour_cos'] = np.cos(2 * np.pi * features['synthetic_time'] / 24)

            return features

        except Exception as e:
            print(f"⚠️  Time features error: {e}")
            return features

    def _add_single_df_market_regime_features(self, features):
        """Add market regime identification features"""
        try:
            close = features['close'].values
            close_series = pd.Series(close)

            # Trend identification
            sma_20 = close_series.rolling(20).mean()
            sma_50 = close_series.rolling(50).mean()

            features['trend_short'] = (close > sma_20).astype(int)
            features['trend_medium'] = (sma_20 > sma_50).astype(int)
            features['trend_strength'] = np.abs(close - sma_20) / sma_20

            # Volatility regime
            volatility = close_series.pct_change().rolling(20).std()
            vol_percentile = volatility.rolling(100).rank(pct=True)

            features['vol_regime_low'] = (vol_percentile < 0.33).astype(int)
            features['vol_regime_medium'] = ((vol_percentile >= 0.33) & (vol_percentile < 0.67)).astype(int)
            features['vol_regime_high'] = (vol_percentile >= 0.67).astype(int)

            # Market state
            returns = close_series.pct_change()
            features['bull_market'] = (returns.rolling(20).mean() > 0).astype(int)
            features['bear_market'] = (returns.rolling(20).mean() < 0).astype(int)

            return features

        except Exception as e:
            print(f"⚠️  Market regime features error: {e}")
            return features

    def _add_single_df_mathematical_features(self, features):
        """Add advanced mathematical features"""
        try:
            close = features['close'].values
            high = features['high'].values
            low = features['low'].values

            # Fibonacci retracements
            recent_high = pd.Series(high).rolling(50).max()
            recent_low = pd.Series(low).rolling(50).min()
            fib_range = recent_high - recent_low

            features['fib_23.6'] = recent_low + 0.236 * fib_range
            features['fib_38.2'] = recent_low + 0.382 * fib_range
            features['fib_50.0'] = recent_low + 0.500 * fib_range
            features['fib_61.8'] = recent_low + 0.618 * fib_range

            # Distance to Fibonacci levels
            features['dist_to_fib_50'] = np.abs(close - features['fib_50.0']) / close

            # Fractal dimension (simplified)
            def hurst_exponent(ts, max_lag=20):
                """Calculate Hurst exponent"""
                try:
                    lags = range(2, max_lag)
                    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return poly[0] * 2.0
                except:
                    return 0.5

            # Calculate Hurst exponent for rolling windows
            close_series = pd.Series(close)
            features['hurst_50'] = close_series.rolling(50).apply(lambda x: hurst_exponent(x.values))

            # Entropy (simplified)
            def shannon_entropy(ts, bins=10):
                """Calculate Shannon entropy"""
                try:
                    hist, _ = np.histogram(ts, bins=bins)
                    hist = hist[hist > 0]
                    prob = hist / hist.sum()
                    return -np.sum(prob * np.log2(prob))
                except:
                    return 0

            features['entropy_20'] = close_series.rolling(20).apply(lambda x: shannon_entropy(x.values))

            return features

        except Exception as e:
            print(f"⚠️  Mathematical features error: {e}")
            return features

    def _clean_single_df_features(self, features):
        """Clean and validate features"""
        try:
            # Handle infinite values
            features = features.replace([np.inf, -np.inf], np.nan)

            # Forward fill then backward fill NaN values
            features = features.fillna(method='ffill').fillna(method='bfill')

            # Fill remaining NaN with 0
            features = features.fillna(0)

            # Remove columns with all zeros or constant values
            for col in features.columns:
                if features[col].nunique() <= 1:
                    features = features.drop(columns=[col])

            return features

        except Exception as e:
            print(f"⚠️  Feature cleaning error: {e}")
            return features
        
    def create_comprehensive_features(self, market_data, fundamental_data=None, sentiment_data=None):
        """
        Create comprehensive features from all data sources
        
        Args:
            market_data (dict): Market data for different timeframes
            fundamental_data (dict): Fundamental economic data
            sentiment_data (dict): Market sentiment data
            
        Returns:
            pd.DataFrame: Engineered features ready for ML models
        """
        try:
            print("🔧 Creating comprehensive features...")
            
            # Use the highest resolution data as base
            base_timeframe = self._select_base_timeframe(market_data)
            base_data = market_data[base_timeframe].copy()
            
            print(f"   Using {base_timeframe} as base timeframe ({len(base_data)} records)")
            
            # Create technical features
            technical_features = self._create_technical_features(base_data)
            
            # Create multi-timeframe features
            mtf_features = self._create_multi_timeframe_features(market_data, base_data)
            
            # Create fundamental features
            fundamental_features = self._create_fundamental_features(
                fundamental_data, len(base_data)
            )
            
            # Create sentiment features
            sentiment_features = self._create_sentiment_features(
                sentiment_data, len(base_data)
            )
            
            # Create advanced pattern features
            pattern_features = self._create_pattern_features(base_data)
            
            # Create volatility and risk features
            volatility_features = self._create_volatility_features(base_data)
            
            # Create time-based features
            time_features = self._create_time_features(base_data)
            
            # Combine all features
            all_features = pd.concat([
                technical_features,
                mtf_features,
                fundamental_features,
                sentiment_features,
                pattern_features,
                volatility_features,
                time_features
            ], axis=1)
            
            # Clean features
            all_features = self._clean_features(all_features)
            
            print(f"✅ Feature engineering complete: {all_features.shape[1]} features created")
            return all_features
            
        except Exception as e:
            print(f"❌ Feature engineering error: {e}")
            return self._create_basic_features(market_data)
            
    def _select_base_timeframe(self, market_data):
        """Select the best timeframe as base for feature engineering"""
        # Prefer 1h data, fallback to available timeframes
        preference_order = ['1h', '4h', '1d', '15m', '5m', '1m']
        
        for timeframe in preference_order:
            if timeframe in market_data and len(market_data[timeframe]) > 50:
                return timeframe
                
        # Return any available timeframe
        return list(market_data.keys())[0]
        
    def _create_technical_features(self, data):
        """Create comprehensive technical analysis features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price features
            features['price_change'] = data['close'].pct_change()
            features['price_change_abs'] = np.abs(features['price_change'])
            features['high_low_ratio'] = data['high'] / data['low']
            features['close_open_ratio'] = data['close'] / data['open']
            
            # Moving averages
            ma_periods = [5, 10, 20, 50, 100, 200]
            for period in ma_periods:
                if len(data) > period:
                    features[f'sma_{period}'] = data['close'].rolling(period).mean()
                    features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                    features[f'price_sma_ratio_{period}'] = data['close'] / features[f'sma_{period}']
                    features[f'sma_slope_{period}'] = features[f'sma_{period}'].diff(5)
                    
            # RSI indicators
            rsi_periods = [14, 21, 30]
            for period in rsi_periods:
                if len(data) > period:
                    try:
                        features[f'rsi_{period}'] = talib.RSI(data['close'].values, timeperiod=period)
                    except:
                        features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
                        
            # MACD
            try:
                macd, macd_signal, macd_hist = talib.MACD(data['close'].values)
                features['macd'] = macd
                features['macd_signal'] = macd_signal
                features['macd_histogram'] = macd_hist
                features['macd_crossover'] = (features['macd'] > features['macd_signal']).astype(int)
            except:
                macd_line, macd_signal, macd_hist = self._calculate_macd(data['close'])
                features['macd'] = macd_line
                features['macd_signal'] = macd_signal
                features['macd_histogram'] = macd_hist
                
            # Bollinger Bands
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values)
                features['bb_upper'] = bb_upper
                features['bb_middle'] = bb_middle
                features['bb_lower'] = bb_lower
                features['bb_width'] = (bb_upper - bb_lower) / bb_middle
                features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            except:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'])
                features['bb_upper'] = bb_upper
                features['bb_middle'] = bb_middle
                features['bb_lower'] = bb_lower
                
            # Stochastic Oscillator
            try:
                slowk, slowd = talib.STOCH(data['high'].values, data['low'].values, data['close'].values)
                features['stoch_k'] = slowk
                features['stoch_d'] = slowd
            except:
                features['stoch_k'] = self._calculate_stochastic(data)
                
            # Williams %R
            try:
                features['williams_r'] = talib.WILLR(data['high'].values, data['low'].values, data['close'].values)
            except:
                features['williams_r'] = self._calculate_williams_r(data)
                
            # Average True Range (ATR)
            try:
                features['atr'] = talib.ATR(data['high'].values, data['low'].values, data['close'].values)
                features['atr_ratio'] = features['atr'] / data['close']
            except:
                features['atr'] = self._calculate_atr(data)
                
            # Commodity Channel Index (CCI)
            try:
                features['cci'] = talib.CCI(data['high'].values, data['low'].values, data['close'].values)
            except:
                features['cci'] = self._calculate_cci(data)
                
            # Volume features
            if 'volume' in data.columns:
                features['volume_sma_10'] = data['volume'].rolling(10).mean()
                features['volume_ratio'] = data['volume'] / features['volume_sma_10']
                features['volume_price_trend'] = data['volume'] * features['price_change']
                
                # On-Balance Volume
                try:
                    features['obv'] = talib.OBV(data['close'].values, data['volume'].values)
                except:
                    features['obv'] = self._calculate_obv(data)
                    
            return features
            
        except Exception as e:
            print(f"❌ Technical features error: {e}")
            return pd.DataFrame(index=data.index)
            
    def _create_multi_timeframe_features(self, market_data, base_data):
        """Create features from multiple timeframes"""
        try:
            features = pd.DataFrame(index=base_data.index)
            
            # Get features from different timeframes
            for timeframe, data in market_data.items():
                if timeframe != self._select_base_timeframe(market_data) and len(data) > 20:
                    # Calculate key indicators for this timeframe
                    tf_rsi = self._calculate_rsi(data['close'], 14).iloc[-1] if len(data) > 14 else 50
                    tf_ma_20 = data['close'].rolling(20).mean().iloc[-1] if len(data) > 20 else data['close'].iloc[-1]
                    tf_price = data['close'].iloc[-1]
                    
                    # Add as constant features (broadcast to all rows)
                    features[f'{timeframe}_rsi'] = tf_rsi
                    features[f'{timeframe}_ma_ratio'] = tf_price / tf_ma_20
                    features[f'{timeframe}_trend'] = 1 if tf_price > tf_ma_20 else 0
                    
            return features
            
        except Exception as e:
            print(f"❌ Multi-timeframe features error: {e}")
            return pd.DataFrame(index=base_data.index)
            
    def _create_fundamental_features(self, fundamental_data, length):
        """Create features from fundamental economic data"""
        try:
            features = pd.DataFrame(index=range(length))
            
            if fundamental_data:
                # DXY features
                if 'dxy' in fundamental_data:
                    dxy = fundamental_data['dxy']
                    features['dxy_current'] = dxy.get('current', 103.0)
                    features['dxy_change'] = dxy.get('change_pct', 0.0)
                    features['dxy_strength'] = 1 if dxy.get('change_pct', 0) > 0.2 else 0
                    
                # Federal Reserve features
                if 'fed_rate' in fundamental_data:
                    fed = fundamental_data['fed_rate']
                    features['fed_rate'] = fed.get('rate', 5.0)
                    features['fed_dovish'] = 1 if fed.get('trend') == 'dovish' else 0
                    features['fed_hawkish'] = 1 if fed.get('trend') == 'hawkish' else 0
                    
                # Inflation features
                if 'inflation' in fundamental_data:
                    inflation = fundamental_data['inflation']
                    features['inflation_rate'] = inflation.get('rate', 3.0)
                    features['high_inflation'] = 1 if inflation.get('rate', 3.0) > 3.5 else 0
                    
                # Oil correlation features
                if 'oil' in fundamental_data:
                    oil = fundamental_data['oil']
                    features['oil_price'] = oil.get('current', 75.0)
                    features['oil_change'] = oil.get('change_pct', 0.0)
                    
                # Silver correlation features
                if 'silver' in fundamental_data:
                    silver = fundamental_data['silver']
                    features['silver_price'] = silver.get('current', 24.0)
                    features['silver_change'] = silver.get('change_pct', 0.0)
                    features['gold_silver_ratio'] = 2000.0 / silver.get('current', 24.0)  # Approximate
                    
            else:
                # Default values when no fundamental data available
                features['dxy_current'] = 103.0
                features['dxy_change'] = 0.0
                features['fed_rate'] = 5.0
                features['inflation_rate'] = 3.0
                features['oil_price'] = 75.0
                features['silver_price'] = 24.0
                
            return features
            
        except Exception as e:
            print(f"❌ Fundamental features error: {e}")
            return pd.DataFrame(index=range(length))
            
    def _create_sentiment_features(self, sentiment_data, length):
        """Create features from sentiment data"""
        try:
            features = pd.DataFrame(index=range(length))
            
            if sentiment_data:
                # Fear & Greed Index
                features['fear_greed'] = sentiment_data.get('fear_greed_index', 50)
                features['extreme_fear'] = 1 if sentiment_data.get('fear_greed_index', 50) < 25 else 0
                features['extreme_greed'] = 1 if sentiment_data.get('fear_greed_index', 50) > 75 else 0
                
                # News sentiment
                features['news_sentiment'] = sentiment_data.get('news_sentiment', 50)
                features['positive_news'] = 1 if sentiment_data.get('news_sentiment', 50) > 60 else 0
                
                # Social sentiment
                features['social_sentiment'] = sentiment_data.get('social_sentiment', 50)
                
                # VIX (volatility index)
                features['vix'] = sentiment_data.get('vix', 20.0)
                features['high_volatility'] = 1 if sentiment_data.get('vix', 20.0) > 25 else 0
                
            else:
                # Default neutral sentiment values
                features['fear_greed'] = 50
                features['news_sentiment'] = 50
                features['social_sentiment'] = 50
                features['vix'] = 20.0
                
            return features
            
        except Exception as e:
            print(f"❌ Sentiment features error: {e}")
            return pd.DataFrame(index=range(length))
            
    def _create_pattern_features(self, data):
        """Create pattern recognition features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Candlestick patterns (simplified)
            features['doji'] = self._detect_doji(data)
            features['hammer'] = self._detect_hammer(data)
            features['shooting_star'] = self._detect_shooting_star(data)
            features['engulfing_bullish'] = self._detect_bullish_engulfing(data)
            features['engulfing_bearish'] = self._detect_bearish_engulfing(data)
            
            # Price patterns
            features['higher_highs'] = self._detect_higher_highs(data)
            features['lower_lows'] = self._detect_lower_lows(data)
            features['support_level'] = self._detect_support_level(data)
            features['resistance_level'] = self._detect_resistance_level(data)
            
            # Gap analysis
            features['gap_up'] = self._detect_gap_up(data)
            features['gap_down'] = self._detect_gap_down(data)
            
            return features
            
        except Exception as e:
            print(f"❌ Pattern features error: {e}")
            return pd.DataFrame(index=data.index)
            
    def _create_volatility_features(self, data):
        """Create volatility and risk features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Historical volatility
            returns = data['close'].pct_change()
            
            for period in [5, 10, 20, 30]:
                if len(data) > period:
                    features[f'volatility_{period}'] = returns.rolling(period).std()
                    features[f'volatility_rank_{period}'] = features[f'volatility_{period}'].rolling(100).rank(pct=True)
                    
            # Parkinson volatility (using high-low)
            features['parkinson_vol'] = np.sqrt(
                0.361 * np.log(data['high'] / data['low']) ** 2
            )
            
            # Garman-Klass volatility
            features['gk_vol'] = np.sqrt(
                0.5 * np.log(data['high'] / data['low']) ** 2 - 
                (2 * np.log(2) - 1) * np.log(data['close'] / data['open']) ** 2
            )
            
            # Range features
            features['true_range'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    np.abs(data['high'] - data['close'].shift(1)),
                    np.abs(data['low'] - data['close'].shift(1))
                )
            )
            
            return features
            
        except Exception as e:
            print(f"❌ Volatility features error: {e}")
            return pd.DataFrame(index=data.index)
            
    def _create_time_features(self, data):
        """Create time-based features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            if 'datetime' in data.columns:
                dt = pd.to_datetime(data['datetime'])
                
                # Time components
                features['hour'] = dt.dt.hour
                features['day_of_week'] = dt.dt.dayofweek
                features['month'] = dt.dt.month
                features['quarter'] = dt.dt.quarter
                
                # Market session features
                features['london_session'] = ((dt.dt.hour >= 8) & (dt.dt.hour <= 16)).astype(int)
                features['ny_session'] = ((dt.dt.hour >= 13) & (dt.dt.hour <= 21)).astype(int)
                features['asian_session'] = ((dt.dt.hour >= 22) | (dt.dt.hour <= 6)).astype(int)
                features['session_overlap'] = (features['london_session'] & features['ny_session']).astype(int)
                
                # Weekend/holiday effects
                features['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
                features['is_month_end'] = (dt.dt.day >= 28).astype(int)
                
            else:
                # Default time features if datetime not available
                features['hour'] = 12
                features['day_of_week'] = 2
                features['month'] = 6
                features['london_session'] = 1
                
            return features
            
        except Exception as e:
            print(f"❌ Time features error: {e}")
            return pd.DataFrame(index=data.index)
            
    def _clean_features(self, features):
        """Clean and prepare features for ML models"""
        try:
            # Handle infinite values
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill then backward fill NaN values
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            # Fill any remaining NaN with 0
            features = features.fillna(0)
            
            # Remove constant columns
            constant_columns = features.columns[features.nunique() <= 1]
            if len(constant_columns) > 0:
                features = features.drop(columns=constant_columns)
                print(f"   Removed {len(constant_columns)} constant columns")
                
            return features
            
        except Exception as e:
            print(f"❌ Feature cleaning error: {e}")
            return features
            
    def _create_basic_features(self, market_data):
        """Create basic features as fallback"""
        try:
            base_timeframe = list(market_data.keys())[0]
            data = market_data[base_timeframe]
            
            features = pd.DataFrame(index=data.index)
            features['price_change'] = data['close'].pct_change()
            features['sma_20'] = data['close'].rolling(20).mean()
            features['rsi_14'] = self._calculate_rsi(data['close'], 14)
            
            return features.fillna(0)
            
        except:
            return pd.DataFrame()
            
    # Helper methods for technical indicators
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD manually"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
        
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands manually"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, ma, lower
        
    def _calculate_atr(self, data, period=14):
        """Calculate ATR manually"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
        
    def _calculate_stochastic(self, data, period=14):
        """Calculate Stochastic Oscillator manually"""
        lowest_low = data['low'].rolling(period).min()
        highest_high = data['high'].rolling(period).max()
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        return k_percent
        
    def _calculate_williams_r(self, data, period=14):
        """Calculate Williams %R manually"""
        highest_high = data['high'].rolling(period).max()
        lowest_low = data['low'].rolling(period).min()
        williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
        return williams_r
        
    def _calculate_cci(self, data, period=20):
        """Calculate CCI manually"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(period).mean()
        mean_deviation = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci
        
    def _calculate_obv(self, data):
        """Calculate On-Balance Volume manually"""
        obv = [0]
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=data.index)
        
    # Pattern detection methods (simplified)
    def _detect_doji(self, data):
        """Detect Doji candlestick pattern"""
        body_size = np.abs(data['close'] - data['open'])
        range_size = data['high'] - data['low']
        return (body_size / range_size < 0.1).astype(int)
        
    def _detect_hammer(self, data):
        """Detect Hammer candlestick pattern"""
        body_size = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        return ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
        
    def _detect_shooting_star(self, data):
        """Detect Shooting Star candlestick pattern"""
        body_size = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        return ((upper_shadow > 2 * body_size) & (lower_shadow < body_size)).astype(int)
        
    def _detect_bullish_engulfing(self, data):
        """Detect Bullish Engulfing pattern"""
        prev_bearish = (data['close'].shift(1) < data['open'].shift(1))
        curr_bullish = (data['close'] > data['open'])
        engulfing = (data['open'] < data['close'].shift(1)) & (data['close'] > data['open'].shift(1))
        return (prev_bearish & curr_bullish & engulfing).astype(int)
        
    def _detect_bearish_engulfing(self, data):
        """Detect Bearish Engulfing pattern"""
        prev_bullish = (data['close'].shift(1) > data['open'].shift(1))
        curr_bearish = (data['close'] < data['open'])
        engulfing = (data['open'] > data['close'].shift(1)) & (data['close'] < data['open'].shift(1))
        return (prev_bullish & curr_bearish & engulfing).astype(int)
        
    def _detect_higher_highs(self, data, period=5):
        """Detect higher highs pattern"""
        rolling_max = data['high'].rolling(period).max()
        return (data['high'] > rolling_max.shift(1)).astype(int)
        
    def _detect_lower_lows(self, data, period=5):
        """Detect lower lows pattern"""
        rolling_min = data['low'].rolling(period).min()
        return (data['low'] < rolling_min.shift(1)).astype(int)
        
    def _detect_support_level(self, data, period=20):
        """Detect support level"""
        rolling_min = data['low'].rolling(period).min()
        return (np.abs(data['low'] - rolling_min) / data['close'] < 0.01).astype(int)
        
    def _detect_resistance_level(self, data, period=20):
        """Detect resistance level"""
        rolling_max = data['high'].rolling(period).max()
        return (np.abs(data['high'] - rolling_max) / data['close'] < 0.01).astype(int)
        
    def _detect_gap_up(self, data):
        """Detect gap up"""
        return (data['open'] > data['high'].shift(1)).astype(int)
        
    def _detect_gap_down(self, data):
        """Detect gap down"""
        return (data['open'] < data['low'].shift(1)).astype(int)
