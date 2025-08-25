#!/usr/bin/env python3
"""
Technical Analysis Module for Gold Trading
Comprehensive technical analysis with 50+ indicators

Author: AI Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import talib
import warnings
warnings.filterwarnings('ignore')


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis for gold trading
    Implements 50+ technical indicators and pattern recognition
    """
    
    def __init__(self):
        self.indicators = {}
        self.signals = {}
        print("📊 Technical Analyzer initialized")
        
    def analyze_comprehensive(self, data):
        """
        Perform comprehensive technical analysis
        
        Args:
            data (pd.DataFrame): OHLCV price data
            
        Returns:
            dict: Complete technical analysis results
        """
        try:
            print("📊 Performing comprehensive technical analysis...")
            
            # Trend indicators
            trend_analysis = self._analyze_trend_indicators(data)
            
            # Momentum indicators
            momentum_analysis = self._analyze_momentum_indicators(data)
            
            # Volatility indicators
            volatility_analysis = self._analyze_volatility_indicators(data)
            
            # Volume indicators
            volume_analysis = self._analyze_volume_indicators(data)
            
            # Support/Resistance levels
            sr_analysis = self._analyze_support_resistance(data)
            
            # Candlestick patterns
            pattern_analysis = self._analyze_candlestick_patterns(data)
            
            # Fibonacci levels
            fibonacci_analysis = self._analyze_fibonacci_levels(data)
            
            # Market structure
            structure_analysis = self._analyze_market_structure(data)
            
            # Generate overall technical score
            technical_score = self._calculate_technical_score(
                trend_analysis, momentum_analysis, volatility_analysis,
                volume_analysis, sr_analysis, pattern_analysis
            )
            
            # Compile comprehensive results
            analysis_results = {
                'technical_score': technical_score,
                'trend_analysis': trend_analysis,
                'momentum_analysis': momentum_analysis,
                'volatility_analysis': volatility_analysis,
                'volume_analysis': volume_analysis,
                'support_resistance': sr_analysis,
                'patterns': pattern_analysis,
                'fibonacci': fibonacci_analysis,
                'market_structure': structure_analysis,
                'signals': self._generate_trading_signals(data),
                'summary': self._generate_analysis_summary(technical_score, trend_analysis, momentum_analysis)
            }
            
            print(f"✅ Technical analysis complete - Score: {technical_score}/100")
            return analysis_results
            
        except Exception as e:
            print(f"❌ Technical analysis error: {e}")
            return self._get_default_analysis()
            
    def _analyze_trend_indicators(self, data):
        """Analyze trend-following indicators"""
        try:
            trend_indicators = {}
            
            # Moving Averages
            ma_periods = [5, 10, 20, 50, 100, 200]
            for period in ma_periods:
                if len(data) > period:
                    trend_indicators[f'sma_{period}'] = data['close'].rolling(period).mean().iloc[-1]
                    trend_indicators[f'ema_{period}'] = data['close'].ewm(span=period).mean().iloc[-1]
                    
            current_price = data['close'].iloc[-1]
            
            # Moving Average signals
            ma_signals = []
            if len(data) > 50:
                sma_20 = trend_indicators.get('sma_20', current_price)
                sma_50 = trend_indicators.get('sma_50', current_price)
                
                if current_price > sma_20 > sma_50:
                    ma_signals.append('STRONG_BULLISH')
                elif current_price > sma_20:
                    ma_signals.append('BULLISH')
                elif current_price < sma_20 < sma_50:
                    ma_signals.append('STRONG_BEARISH')
                elif current_price < sma_20:
                    ma_signals.append('BEARISH')
                else:
                    ma_signals.append('NEUTRAL')
                    
            # Parabolic SAR
            try:
                sar = talib.SAR(data['high'].values, data['low'].values)
                trend_indicators['sar'] = sar[-1]
                trend_indicators['sar_signal'] = 'BULLISH' if current_price > sar[-1] else 'BEARISH'
            except:
                trend_indicators['sar_signal'] = 'NEUTRAL'
                
            # Average Directional Index (ADX)
            try:
                adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values)
                plus_di = talib.PLUS_DI(data['high'].values, data['low'].values, data['close'].values)
                minus_di = talib.MINUS_DI(data['high'].values, data['low'].values, data['close'].values)
                
                trend_indicators['adx'] = adx[-1]
                trend_indicators['plus_di'] = plus_di[-1]
                trend_indicators['minus_di'] = minus_di[-1]
                
                if adx[-1] > 25:
                    if plus_di[-1] > minus_di[-1]:
                        trend_indicators['adx_signal'] = 'STRONG_BULLISH'
                    else:
                        trend_indicators['adx_signal'] = 'STRONG_BEARISH'
                else:
                    trend_indicators['adx_signal'] = 'WEAK_TREND'
            except:
                trend_indicators['adx_signal'] = 'NEUTRAL'
                
            # Ichimoku Cloud
            try:
                ichimoku = self._calculate_ichimoku(data)
                trend_indicators.update(ichimoku)
            except:
                trend_indicators['ichimoku_signal'] = 'NEUTRAL'
                
            trend_indicators['ma_signals'] = ma_signals
            return trend_indicators
            
        except Exception as e:
            print(f"❌ Trend analysis error: {e}")
            return {'trend_signal': 'NEUTRAL'}
            
    def _analyze_momentum_indicators(self, data):
        """Analyze momentum oscillators"""
        try:
            momentum_indicators = {}
            
            # RSI (multiple periods)
            rsi_periods = [14, 21, 30]
            for period in rsi_periods:
                if len(data) > period:
                    try:
                        rsi = talib.RSI(data['close'].values, timeperiod=period)
                        momentum_indicators[f'rsi_{period}'] = rsi[-1]
                        
                        # RSI signals
                        if rsi[-1] > 70:
                            momentum_indicators[f'rsi_{period}_signal'] = 'OVERBOUGHT'
                        elif rsi[-1] < 30:
                            momentum_indicators[f'rsi_{period}_signal'] = 'OVERSOLD'
                        else:
                            momentum_indicators[f'rsi_{period}_signal'] = 'NEUTRAL'
                    except:
                        momentum_indicators[f'rsi_{period}'] = 50
                        
            # MACD
            try:
                macd, macd_signal, macd_hist = talib.MACD(data['close'].values)
                momentum_indicators['macd'] = macd[-1]
                momentum_indicators['macd_signal'] = macd_signal[-1]
                momentum_indicators['macd_histogram'] = macd_hist[-1]
                
                if macd[-1] > macd_signal[-1] and macd_hist[-1] > 0:
                    momentum_indicators['macd_trend'] = 'BULLISH'
                elif macd[-1] < macd_signal[-1] and macd_hist[-1] < 0:
                    momentum_indicators['macd_trend'] = 'BEARISH'
                else:
                    momentum_indicators['macd_trend'] = 'NEUTRAL'
            except:
                momentum_indicators['macd_trend'] = 'NEUTRAL'
                
            # Stochastic Oscillator
            try:
                slowk, slowd = talib.STOCH(data['high'].values, data['low'].values, data['close'].values)
                momentum_indicators['stoch_k'] = slowk[-1]
                momentum_indicators['stoch_d'] = slowd[-1]
                
                if slowk[-1] > 80:
                    momentum_indicators['stoch_signal'] = 'OVERBOUGHT'
                elif slowk[-1] < 20:
                    momentum_indicators['stoch_signal'] = 'OVERSOLD'
                else:
                    momentum_indicators['stoch_signal'] = 'NEUTRAL'
            except:
                momentum_indicators['stoch_signal'] = 'NEUTRAL'
                
            # Williams %R
            try:
                williams_r = talib.WILLR(data['high'].values, data['low'].values, data['close'].values)
                momentum_indicators['williams_r'] = williams_r[-1]
                
                if williams_r[-1] > -20:
                    momentum_indicators['williams_signal'] = 'OVERBOUGHT'
                elif williams_r[-1] < -80:
                    momentum_indicators['williams_signal'] = 'OVERSOLD'
                else:
                    momentum_indicators['williams_signal'] = 'NEUTRAL'
            except:
                momentum_indicators['williams_signal'] = 'NEUTRAL'
                
            # Commodity Channel Index (CCI)
            try:
                cci = talib.CCI(data['high'].values, data['low'].values, data['close'].values)
                momentum_indicators['cci'] = cci[-1]
                
                if cci[-1] > 100:
                    momentum_indicators['cci_signal'] = 'OVERBOUGHT'
                elif cci[-1] < -100:
                    momentum_indicators['cci_signal'] = 'OVERSOLD'
                else:
                    momentum_indicators['cci_signal'] = 'NEUTRAL'
            except:
                momentum_indicators['cci_signal'] = 'NEUTRAL'
                
            # Rate of Change (ROC)
            try:
                roc = talib.ROC(data['close'].values, timeperiod=10)
                momentum_indicators['roc'] = roc[-1]
                momentum_indicators['roc_signal'] = 'BULLISH' if roc[-1] > 0 else 'BEARISH'
            except:
                momentum_indicators['roc_signal'] = 'NEUTRAL'
                
            return momentum_indicators
            
        except Exception as e:
            print(f"❌ Momentum analysis error: {e}")
            return {'momentum_signal': 'NEUTRAL'}
            
    def _analyze_volatility_indicators(self, data):
        """Analyze volatility indicators"""
        try:
            volatility_indicators = {}
            
            # Bollinger Bands
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values)
                current_price = data['close'].iloc[-1]
                
                volatility_indicators['bb_upper'] = bb_upper[-1]
                volatility_indicators['bb_middle'] = bb_middle[-1]
                volatility_indicators['bb_lower'] = bb_lower[-1]
                volatility_indicators['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                volatility_indicators['bb_position'] = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                
                if current_price > bb_upper[-1]:
                    volatility_indicators['bb_signal'] = 'OVERBOUGHT'
                elif current_price < bb_lower[-1]:
                    volatility_indicators['bb_signal'] = 'OVERSOLD'
                else:
                    volatility_indicators['bb_signal'] = 'NEUTRAL'
            except:
                volatility_indicators['bb_signal'] = 'NEUTRAL'
                
            # Average True Range (ATR)
            try:
                atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values)
                volatility_indicators['atr'] = atr[-1]
                volatility_indicators['atr_ratio'] = atr[-1] / data['close'].iloc[-1]
                
                # ATR percentile ranking
                atr_percentile = (atr[-1] - np.min(atr[-100:])) / (np.max(atr[-100:]) - np.min(atr[-100:]))
                volatility_indicators['atr_percentile'] = atr_percentile
                
                if atr_percentile > 0.8:
                    volatility_indicators['volatility_level'] = 'HIGH'
                elif atr_percentile > 0.5:
                    volatility_indicators['volatility_level'] = 'MODERATE'
                else:
                    volatility_indicators['volatility_level'] = 'LOW'
            except:
                volatility_indicators['volatility_level'] = 'MODERATE'
                
            # Keltner Channels
            try:
                kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(data)
                current_price = data['close'].iloc[-1]
                
                volatility_indicators['kc_upper'] = kc_upper
                volatility_indicators['kc_middle'] = kc_middle
                volatility_indicators['kc_lower'] = kc_lower
                
                if current_price > kc_upper:
                    volatility_indicators['kc_signal'] = 'BREAKOUT_UP'
                elif current_price < kc_lower:
                    volatility_indicators['kc_signal'] = 'BREAKOUT_DOWN'
                else:
                    volatility_indicators['kc_signal'] = 'RANGE_BOUND'
            except:
                volatility_indicators['kc_signal'] = 'NEUTRAL'
                
            return volatility_indicators
            
        except Exception as e:
            print(f"❌ Volatility analysis error: {e}")
            return {'volatility_signal': 'NEUTRAL'}
            
    def _analyze_volume_indicators(self, data):
        """Analyze volume-based indicators"""
        try:
            volume_indicators = {}
            
            if 'volume' not in data.columns:
                return {'volume_signal': 'NO_DATA'}
                
            # On-Balance Volume (OBV)
            try:
                obv = talib.OBV(data['close'].values, data['volume'].values)
                volume_indicators['obv'] = obv[-1]
                
                # OBV trend
                obv_ma = pd.Series(obv).rolling(20).mean()
                if obv[-1] > obv_ma.iloc[-1]:
                    volume_indicators['obv_trend'] = 'BULLISH'
                else:
                    volume_indicators['obv_trend'] = 'BEARISH'
            except:
                volume_indicators['obv_trend'] = 'NEUTRAL'
                
            # Volume Rate of Change
            try:
                volume_roc = talib.ROC(data['volume'].values, timeperiod=10)
                volume_indicators['volume_roc'] = volume_roc[-1]
            except:
                volume_indicators['volume_roc'] = 0
                
            # Accumulation/Distribution Line
            try:
                ad = talib.AD(data['high'].values, data['low'].values, data['close'].values, data['volume'].values)
                volume_indicators['ad_line'] = ad[-1]
                
                # A/D Line trend
                ad_ma = pd.Series(ad).rolling(20).mean()
                if ad[-1] > ad_ma.iloc[-1]:
                    volume_indicators['ad_trend'] = 'ACCUMULATION'
                else:
                    volume_indicators['ad_trend'] = 'DISTRIBUTION'
            except:
                volume_indicators['ad_trend'] = 'NEUTRAL'
                
            # Chaikin Money Flow
            try:
                cmf = self._calculate_chaikin_money_flow(data)
                volume_indicators['cmf'] = cmf
                
                if cmf > 0.1:
                    volume_indicators['cmf_signal'] = 'BULLISH'
                elif cmf < -0.1:
                    volume_indicators['cmf_signal'] = 'BEARISH'
                else:
                    volume_indicators['cmf_signal'] = 'NEUTRAL'
            except:
                volume_indicators['cmf_signal'] = 'NEUTRAL'
                
            # Volume analysis
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            volume_indicators['volume_ratio'] = current_volume / avg_volume
            
            if current_volume > avg_volume * 1.5:
                volume_indicators['volume_signal'] = 'HIGH_VOLUME'
            elif current_volume < avg_volume * 0.5:
                volume_indicators['volume_signal'] = 'LOW_VOLUME'
            else:
                volume_indicators['volume_signal'] = 'NORMAL_VOLUME'
                
            return volume_indicators
            
        except Exception as e:
            print(f"❌ Volume analysis error: {e}")
            return {'volume_signal': 'NEUTRAL'}
            
    def _analyze_support_resistance(self, data):
        """Analyze support and resistance levels"""
        try:
            sr_analysis = {}
            
            # Calculate pivot points
            pivot_points = self._calculate_pivot_points(data)
            sr_analysis.update(pivot_points)
            
            # Dynamic support/resistance
            current_price = data['close'].iloc[-1]
            
            # Recent highs and lows
            recent_highs = data['high'].rolling(20).max()
            recent_lows = data['low'].rolling(20).min()
            
            sr_analysis['resistance_level'] = recent_highs.iloc[-1]
            sr_analysis['support_level'] = recent_lows.iloc[-1]
            
            # Distance to support/resistance
            sr_analysis['distance_to_resistance'] = (recent_highs.iloc[-1] - current_price) / current_price
            sr_analysis['distance_to_support'] = (current_price - recent_lows.iloc[-1]) / current_price
            
            # Support/Resistance strength
            sr_analysis['resistance_strength'] = self._calculate_level_strength(data, recent_highs.iloc[-1], 'resistance')
            sr_analysis['support_strength'] = self._calculate_level_strength(data, recent_lows.iloc[-1], 'support')
            
            return sr_analysis
            
        except Exception as e:
            print(f"❌ Support/Resistance analysis error: {e}")
            return {'sr_signal': 'NEUTRAL'}
            
    def _analyze_candlestick_patterns(self, data):
        """Analyze candlestick patterns"""
        try:
            patterns = {}
            
            # Major reversal patterns
            try:
                patterns['doji'] = talib.CDLDOJI(data['open'].values, data['high'].values, data['low'].values, data['close'].values)[-1]
                patterns['hammer'] = talib.CDLHAMMER(data['open'].values, data['high'].values, data['low'].values, data['close'].values)[-1]
                patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(data['open'].values, data['high'].values, data['low'].values, data['close'].values)[-1]
                patterns['engulfing'] = talib.CDLENGULFING(data['open'].values, data['high'].values, data['low'].values, data['close'].values)[-1]
                patterns['harami'] = talib.CDLHARAMI(data['open'].values, data['high'].values, data['low'].values, data['close'].values)[-1]
            except:
                patterns['pattern_signal'] = 'NO_PATTERN'
                
            # Pattern interpretation
            bullish_patterns = ['hammer', 'engulfing'] if patterns.get('engulfing', 0) > 0 else []
            bearish_patterns = ['shooting_star', 'engulfing'] if patterns.get('engulfing', 0) < 0 else []
            
            if bullish_patterns:
                patterns['pattern_signal'] = 'BULLISH_REVERSAL'
            elif bearish_patterns:
                patterns['pattern_signal'] = 'BEARISH_REVERSAL'
            elif patterns.get('doji', 0) != 0:
                patterns['pattern_signal'] = 'INDECISION'
            else:
                patterns['pattern_signal'] = 'NO_SIGNIFICANT_PATTERN'
                
            return patterns
            
        except Exception as e:
            print(f"❌ Pattern analysis error: {e}")
            return {'pattern_signal': 'NEUTRAL'}
            
    def _analyze_fibonacci_levels(self, data):
        """Calculate Fibonacci retracement levels"""
        try:
            # Find recent swing high and low
            lookback = min(100, len(data))
            recent_data = data.tail(lookback)
            
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            # Calculate Fibonacci levels
            diff = swing_high - swing_low
            
            fib_levels = {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'fib_23.6': swing_high - 0.236 * diff,
                'fib_38.2': swing_high - 0.382 * diff,
                'fib_50.0': swing_high - 0.500 * diff,
                'fib_61.8': swing_high - 0.618 * diff,
                'fib_78.6': swing_high - 0.786 * diff
            }
            
            # Current price position relative to Fibonacci levels
            current_price = data['close'].iloc[-1]
            
            for level_name, level_value in fib_levels.items():
                if level_name.startswith('fib_'):
                    distance = abs(current_price - level_value) / current_price
                    if distance < 0.005:  # Within 0.5%
                        fib_levels['near_fib_level'] = level_name
                        break
            else:
                fib_levels['near_fib_level'] = 'NONE'
                
            return fib_levels
            
        except Exception as e:
            print(f"❌ Fibonacci analysis error: {e}")
            return {'fib_signal': 'NEUTRAL'}
            
    def _analyze_market_structure(self, data):
        """Analyze market structure and trends"""
        try:
            structure = {}
            
            # Higher highs and higher lows (uptrend)
            highs = data['high'].rolling(5).max()
            lows = data['low'].rolling(5).min()
            
            recent_highs = highs.tail(10)
            recent_lows = lows.tail(10)
            
            # Trend analysis
            if recent_highs.is_monotonic_increasing and recent_lows.is_monotonic_increasing:
                structure['trend'] = 'STRONG_UPTREND'
            elif recent_highs.is_monotonic_decreasing and recent_lows.is_monotonic_decreasing:
                structure['trend'] = 'STRONG_DOWNTREND'
            elif recent_highs.iloc[-1] > recent_highs.iloc[0]:
                structure['trend'] = 'UPTREND'
            elif recent_highs.iloc[-1] < recent_highs.iloc[0]:
                structure['trend'] = 'DOWNTREND'
            else:
                structure['trend'] = 'SIDEWAYS'
                
            # Market phase
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            volume_trend = data['volume'].rolling(10).mean().iloc[-1] / data['volume'].rolling(50).mean().iloc[-1] if 'volume' in data.columns else 1.0
            
            if volatility > 0.02 and volume_trend > 1.2:
                structure['phase'] = 'TRENDING'
            elif volatility < 0.01:
                structure['phase'] = 'CONSOLIDATION'
            else:
                structure['phase'] = 'TRANSITION'
                
            return structure
            
        except Exception as e:
            print(f"❌ Market structure analysis error: {e}")
            return {'structure_signal': 'NEUTRAL'}
            
    def _calculate_technical_score(self, trend, momentum, volatility, volume, sr, patterns):
        """Calculate overall technical score (0-100)"""
        try:
            score = 50  # Start neutral
            
            # Trend analysis contribution (30%)
            trend_signals = trend.get('ma_signals', ['NEUTRAL'])
            if 'STRONG_BULLISH' in trend_signals:
                score += 15
            elif 'BULLISH' in trend_signals:
                score += 10
            elif 'STRONG_BEARISH' in trend_signals:
                score -= 15
            elif 'BEARISH' in trend_signals:
                score -= 10
                
            # Momentum contribution (25%)
            rsi_14 = momentum.get('rsi_14', 50)
            if 30 <= rsi_14 <= 70:
                score += 10  # Good RSI range
            elif rsi_14 < 30:
                score += 5   # Oversold - potential bullish
            elif rsi_14 > 70:
                score -= 5   # Overbought - potential bearish
                
            macd_trend = momentum.get('macd_trend', 'NEUTRAL')
            if macd_trend == 'BULLISH':
                score += 8
            elif macd_trend == 'BEARISH':
                score -= 8
                
            # Volatility contribution (20%)
            bb_signal = volatility.get('bb_signal', 'NEUTRAL')
            if bb_signal == 'OVERSOLD':
                score += 8
            elif bb_signal == 'OVERBOUGHT':
                score -= 8
                
            # Volume contribution (15%)
            volume_signal = volume.get('volume_signal', 'NEUTRAL')
            if volume_signal == 'HIGH_VOLUME':
                score += 5  # High volume confirms moves
                
            # Pattern contribution (10%)
            pattern_signal = patterns.get('pattern_signal', 'NEUTRAL')
            if 'BULLISH' in pattern_signal:
                score += 5
            elif 'BEARISH' in pattern_signal:
                score -= 5
                
            # Ensure score is within bounds
            score = max(0, min(100, score))
            
            return round(score)
            
        except Exception as e:
            print(f"❌ Technical score calculation error: {e}")
            return 50
            
    def _generate_trading_signals(self, data):
        """Generate specific trading signals"""
        try:
            signals = []
            
            # Simple signal generation based on multiple confirmations
            current_price = data['close'].iloc[-1]
            
            # Moving average signal
            if len(data) > 20:
                ma_20 = data['close'].rolling(20).mean().iloc[-1]
                if current_price > ma_20:
                    signals.append('MA_BULLISH')
                else:
                    signals.append('MA_BEARISH')
                    
            # RSI signal
            try:
                rsi = talib.RSI(data['close'].values)[-1]
                if rsi < 30:
                    signals.append('RSI_OVERSOLD')
                elif rsi > 70:
                    signals.append('RSI_OVERBOUGHT')
            except:
                pass
                
            # Volume confirmation
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                if current_volume > avg_volume * 1.5:
                    signals.append('HIGH_VOLUME_CONFIRMATION')
                    
            return signals
            
        except Exception as e:
            print(f"❌ Signal generation error: {e}")
            return ['NEUTRAL']
            
    def _generate_analysis_summary(self, technical_score, trend_analysis, momentum_analysis):
        """Generate human-readable analysis summary"""
        try:
            summary = []
            
            # Overall assessment
            if technical_score >= 70:
                summary.append("Strong bullish technical setup")
            elif technical_score >= 60:
                summary.append("Moderately bullish technical outlook")
            elif technical_score >= 40:
                summary.append("Neutral technical conditions")
            elif technical_score >= 30:
                summary.append("Moderately bearish technical outlook")
            else:
                summary.append("Strong bearish technical setup")
                
            # Trend assessment
            ma_signals = trend_analysis.get('ma_signals', ['NEUTRAL'])
            if 'STRONG_BULLISH' in ma_signals:
                summary.append("Strong uptrend confirmed by multiple moving averages")
            elif 'BULLISH' in ma_signals:
                summary.append("Bullish trend indicated by moving averages")
                
            # Momentum assessment
            rsi_14 = momentum_analysis.get('rsi_14', 50)
            if rsi_14 > 70:
                summary.append("Momentum indicators showing overbought conditions")
            elif rsi_14 < 30:
                summary.append("Momentum indicators showing oversold conditions")
                
            return ". ".join(summary) + "."
            
        except Exception as e:
            print(f"❌ Summary generation error: {e}")
            return "Technical analysis summary unavailable."
            
    def _get_default_analysis(self):
        """Return default analysis in case of errors"""
        return {
            'technical_score': 50,
            'trend_analysis': {'trend_signal': 'NEUTRAL'},
            'momentum_analysis': {'momentum_signal': 'NEUTRAL'},
            'volatility_analysis': {'volatility_signal': 'NEUTRAL'},
            'volume_analysis': {'volume_signal': 'NEUTRAL'},
            'support_resistance': {'sr_signal': 'NEUTRAL'},
            'patterns': {'pattern_signal': 'NEUTRAL'},
            'fibonacci': {'fib_signal': 'NEUTRAL'},
            'market_structure': {'structure_signal': 'NEUTRAL'},
            'signals': ['NEUTRAL'],
            'summary': 'Technical analysis unavailable due to data limitations.'
        }
        
    # Helper methods
    def _calculate_ichimoku(self, data):
        """Calculate Ichimoku Cloud components"""
        try:
            # Tenkan-sen (Conversion Line)
            tenkan_sen = (data['high'].rolling(9).max() + data['low'].rolling(9).min()) / 2
            
            # Kijun-sen (Base Line)
            kijun_sen = (data['high'].rolling(26).max() + data['low'].rolling(26).min()) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            senkou_span_b = ((data['high'].rolling(52).max() + data['low'].rolling(52).min()) / 2).shift(26)
            
            current_price = data['close'].iloc[-1]
            
            # Determine cloud signal
            if current_price > senkou_span_a.iloc[-1] and current_price > senkou_span_b.iloc[-1]:
                ichimoku_signal = 'BULLISH'
            elif current_price < senkou_span_a.iloc[-1] and current_price < senkou_span_b.iloc[-1]:
                ichimoku_signal = 'BEARISH'
            else:
                ichimoku_signal = 'NEUTRAL'
                
            return {
                'tenkan_sen': tenkan_sen.iloc[-1],
                'kijun_sen': kijun_sen.iloc[-1],
                'ichimoku_signal': ichimoku_signal
            }
            
        except:
            return {'ichimoku_signal': 'NEUTRAL'}
            
    def _calculate_keltner_channels(self, data, period=20, multiplier=2):
        """Calculate Keltner Channels"""
        try:
            ema = data['close'].ewm(span=period).mean()
            atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
            
            upper = ema.iloc[-1] + multiplier * atr[-1]
            middle = ema.iloc[-1]
            lower = ema.iloc[-1] - multiplier * atr[-1]
            
            return upper, middle, lower
            
        except:
            current_price = data['close'].iloc[-1]
            return current_price * 1.02, current_price, current_price * 0.98
            
    def _calculate_chaikin_money_flow(self, data, period=20):
        """Calculate Chaikin Money Flow"""
        try:
            mfv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['volume']
            cmf = mfv.rolling(period).sum() / data['volume'].rolling(period).sum()
            return cmf.iloc[-1]
        except:
            return 0
            
    def _calculate_pivot_points(self, data):
        """Calculate pivot points"""
        try:
            high = data['high'].iloc[-1]
            low = data['low'].iloc[-1]
            close = data['close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            return {
                'pivot': pivot,
                'resistance_1': r1,
                'support_1': s1,
                'resistance_2': r2,
                'support_2': s2
            }
        except:
            current_price = data['close'].iloc[-1]
            return {
                'pivot': current_price,
                'resistance_1': current_price * 1.01,
                'support_1': current_price * 0.99
            }
            
    def _calculate_level_strength(self, data, level, level_type):
        """Calculate support/resistance level strength"""
        try:
            # Count how many times price has tested this level
            tolerance = level * 0.002  # 0.2% tolerance
            
            if level_type == 'resistance':
                tests = ((data['high'] >= level - tolerance) & (data['high'] <= level + tolerance)).sum()
            else:
                tests = ((data['low'] >= level - tolerance) & (data['low'] <= level + tolerance)).sum()
                
            # Strength based on number of tests
            if tests >= 3:
                return 'STRONG'
            elif tests >= 2:
                return 'MODERATE'
            else:
                return 'WEAK'
                
        except:
            return 'UNKNOWN'
