#!/usr/bin/env python3
"""
Multiple Timeframes Testing for Gold Trading AI
Tests M1, M5, H1 timeframes and signal consistency
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.gold_trading_analyzer import GoldTradingAnalyzer
from modules.technical_analysis import TechnicalAnalyzer
from utils.advanced_data_fetcher import AdvancedDataFetcher


def test_timeframe_configuration():
    """Test configuration and data fetching for multiple timeframes"""
    print("⏰ TESTING MULTIPLE TIMEFRAMES CONFIGURATION")
    print("=" * 60)
    
    # Initialize components
    data_fetcher = AdvancedDataFetcher()
    technical_analyzer = TechnicalAnalyzer()
    
    # Define timeframes to test
    timeframes = {
        'M1': '1m',   # 1-minute
        'M5': '5m',   # 5-minute  
        'H1': '1h'    # 1-hour
    }
    
    timeframe_results = {}
    
    print("📊 Testing timeframe data availability...")
    
    for name, tf in timeframes.items():
        print(f"\n🔍 Testing {name} ({tf}) timeframe:")
        
        try:
            # Fetch data for this timeframe
            market_data = data_fetcher.fetch_current_data('GC=F', [tf])
            
            if market_data and tf in market_data:
                data = market_data[tf]
                
                # Analyze data quality
                data_quality = {
                    'records_count': len(data),
                    'date_range': f"{data.index[0]} to {data.index[-1]}" if len(data) > 0 else "No data",
                    'price_range': f"${data['close'].min():.2f} - ${data['close'].max():.2f}" if len(data) > 0 else "N/A",
                    'avg_volume': data['volume'].mean() if 'volume' in data.columns and len(data) > 0 else 0,
                    'data_completeness': (len(data.dropna()) / len(data) * 100) if len(data) > 0 else 0
                }
                
                # Run technical analysis
                analysis = technical_analyzer.analyze_comprehensive(data)
                
                timeframe_results[name] = {
                    'timeframe': tf,
                    'data_available': True,
                    'data_quality': data_quality,
                    'technical_score': analysis['technical_score'],
                    'trend_signal': analysis.get('trend_analysis', {}).get('adx_signal', 'NEUTRAL'),
                    'signals': analysis.get('signals', []),
                    'volatility': analysis.get('volatility_analysis', {}).get('volatility_level', 'MODERATE')
                }
                
                print(f"   ✅ Data Available: {data_quality['records_count']} records")
                print(f"   📊 Technical Score: {analysis['technical_score']}/100")
                print(f"   📈 Trend: {timeframe_results[name]['trend_signal']}")
                print(f"   📊 Data Completeness: {data_quality['data_completeness']:.1f}%")
                
            else:
                timeframe_results[name] = {
                    'timeframe': tf,
                    'data_available': False,
                    'error': 'No data returned'
                }
                print(f"   ❌ No data available for {tf}")
                
        except Exception as e:
            timeframe_results[name] = {
                'timeframe': tf,
                'data_available': False,
                'error': str(e)
            }
            print(f"   ❌ Error: {e}")
    
    return timeframe_results


def test_signal_generation_across_timeframes():
    """Test signal generation across different timeframes"""
    print(f"\n{'='*60}")
    print("🚦 TESTING SIGNAL GENERATION ACROSS TIMEFRAMES")
    print(f"{'='*60}")
    
    analyzer = GoldTradingAnalyzer()
    data_fetcher = AdvancedDataFetcher()
    
    timeframes = ['1m', '5m', '1h']
    signal_results = {}
    
    for tf in timeframes:
        print(f"\n📊 Generating signals for {tf} timeframe...")
        
        try:
            # Fetch data for this timeframe
            market_data = data_fetcher.fetch_current_data('GC=F', [tf])
            
            if market_data and tf in market_data:
                # Generate signal using the analyzer
                start_time = time.time()
                signal_result = analyzer.analyze_gold_market(real_time=False)
                execution_time = time.time() - start_time
                
                signal_results[tf] = {
                    'signal': signal_result['signal'],
                    'confidence': signal_result['confidence'],
                    'entry_price': signal_result['entry_price'],
                    'stop_loss': signal_result['stop_loss'],
                    'take_profit': signal_result['take_profit'],
                    'execution_time': execution_time,
                    'technical_score': signal_result.get('technical_score', 0),
                    'fundamental_score': signal_result.get('fundamental_score', 0),
                    'success': True
                }
                
                print(f"   ✅ Signal: {signal_result['signal']}")
                print(f"   🎯 Confidence: {signal_result['confidence']:.1f}%")
                print(f"   💰 Entry: ${signal_result['entry_price']:.2f}")
                print(f"   ⏱️  Time: {execution_time:.2f}s")
                
            else:
                signal_results[tf] = {
                    'success': False,
                    'error': 'No data available'
                }
                print(f"   ❌ No data available for {tf}")
                
        except Exception as e:
            signal_results[tf] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Error: {e}")
    
    return signal_results


def analyze_timeframe_consistency(signal_results):
    """Analyze consistency across timeframes"""
    print(f"\n{'='*60}")
    print("📊 ANALYZING TIMEFRAME CONSISTENCY")
    print(f"{'='*60}")
    
    successful_signals = {tf: result for tf, result in signal_results.items() if result.get('success', False)}
    
    if len(successful_signals) < 2:
        print("❌ Insufficient successful signals for consistency analysis")
        return None
    
    # Analyze signal consistency
    signals = [result['signal'] for result in successful_signals.values()]
    confidences = [result['confidence'] for result in successful_signals.values()]
    entry_prices = [result['entry_price'] for result in successful_signals.values()]
    
    print(f"📊 CONSISTENCY ANALYSIS:")
    print(f"   🚦 Signals: {', '.join(signals)}")
    print(f"   🎯 Confidence Range: {min(confidences):.1f}% - {max(confidences):.1f}%")
    print(f"   💰 Entry Price Range: ${min(entry_prices):.2f} - ${max(entry_prices):.2f}")
    
    # Calculate consistency scores
    signal_consistency = len(set(signals)) == 1  # All signals are the same
    confidence_variance = max(confidences) - min(confidences)
    price_variance = max(entry_prices) - min(entry_prices)
    
    consistency_score = 0
    if signal_consistency:
        consistency_score += 40
        print(f"   ✅ Signal Consistency: EXCELLENT (all {signals[0]})")
    else:
        print(f"   ⚠️  Signal Consistency: MIXED")
    
    if confidence_variance <= 10:
        consistency_score += 30
        print(f"   ✅ Confidence Consistency: GOOD (variance: {confidence_variance:.1f}%)")
    else:
        print(f"   ⚠️  Confidence Consistency: HIGH VARIANCE ({confidence_variance:.1f}%)")
    
    if price_variance <= 50:
        consistency_score += 30
        print(f"   ✅ Price Consistency: GOOD (variance: ${price_variance:.2f})")
    else:
        print(f"   ⚠️  Price Consistency: HIGH VARIANCE (${price_variance:.2f})")
    
    print(f"\n🏆 OVERALL CONSISTENCY SCORE: {consistency_score}/100")
    
    if consistency_score >= 80:
        print("✅ EXCELLENT - Very consistent across timeframes")
    elif consistency_score >= 60:
        print("⚠️  GOOD - Reasonably consistent")
    else:
        print("❌ POOR - Inconsistent across timeframes")
    
    return {
        'signal_consistency': signal_consistency,
        'confidence_variance': confidence_variance,
        'price_variance': price_variance,
        'consistency_score': consistency_score
    }


def test_timeframe_performance():
    """Test performance metrics for each timeframe"""
    print(f"\n{'='*60}")
    print("⚡ TESTING TIMEFRAME PERFORMANCE")
    print(f"{'='*60}")
    
    data_fetcher = AdvancedDataFetcher()
    technical_analyzer = TechnicalAnalyzer()
    
    timeframes = ['1m', '5m', '1h']
    performance_results = {}
    
    for tf in timeframes:
        print(f"\n📊 Performance testing for {tf}...")
        
        try:
            # Multiple performance tests
            execution_times = []
            technical_scores = []
            
            for test_run in range(3):
                start_time = time.time()
                
                # Fetch data
                market_data = data_fetcher.fetch_current_data('GC=F', [tf])
                
                if market_data and tf in market_data:
                    data = market_data[tf]
                    
                    # Run technical analysis
                    analysis = technical_analyzer.analyze_comprehensive(data)
                    
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                    technical_scores.append(analysis['technical_score'])
                    
                else:
                    break
            
            if execution_times:
                performance_results[tf] = {
                    'avg_execution_time': sum(execution_times) / len(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times),
                    'avg_technical_score': sum(technical_scores) / len(technical_scores),
                    'score_variance': max(technical_scores) - min(technical_scores),
                    'test_runs': len(execution_times),
                    'success': True
                }
                
                print(f"   ⚡ Avg Execution Time: {performance_results[tf]['avg_execution_time']:.2f}s")
                print(f"   📊 Avg Technical Score: {performance_results[tf]['avg_technical_score']:.1f}/100")
                print(f"   📈 Score Variance: {performance_results[tf]['score_variance']:.1f}")
                
            else:
                performance_results[tf] = {
                    'success': False,
                    'error': 'No data available'
                }
                print(f"   ❌ No data available")
                
        except Exception as e:
            performance_results[tf] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Error: {e}")
    
    return performance_results


if __name__ == "__main__":
    try:
        print("🧪 COMPREHENSIVE MULTIPLE TIMEFRAMES TESTING")
        print("=" * 70)
        
        # Test 1: Configuration and data availability
        timeframe_config = test_timeframe_configuration()
        
        # Test 2: Signal generation across timeframes
        signal_results = test_signal_generation_across_timeframes()
        
        # Test 3: Consistency analysis
        consistency_analysis = analyze_timeframe_consistency(signal_results)
        
        # Test 4: Performance testing
        performance_results = test_timeframe_performance()
        
        # Final summary
        print(f"\n{'='*70}")
        print("📊 FINAL TIMEFRAMES TESTING SUMMARY")
        print(f"{'='*70}")
        
        # Count successful configurations
        successful_configs = sum(1 for result in timeframe_config.values() if result.get('data_available', False))
        total_configs = len(timeframe_config)
        
        successful_signals = sum(1 for result in signal_results.values() if result.get('success', False))
        total_signals = len(signal_results)
        
        successful_performance = sum(1 for result in performance_results.values() if result.get('success', False))
        total_performance = len(performance_results)
        
        print(f"📊 Configuration Success: {successful_configs}/{total_configs}")
        print(f"🚦 Signal Generation Success: {successful_signals}/{total_signals}")
        print(f"⚡ Performance Testing Success: {successful_performance}/{total_performance}")
        
        if consistency_analysis:
            print(f"🔄 Consistency Score: {consistency_analysis['consistency_score']}/100")
        
        # Overall assessment
        overall_success_rate = (successful_configs + successful_signals + successful_performance) / (total_configs + total_signals + total_performance)
        
        print(f"\n🎯 OVERALL TIMEFRAMES ASSESSMENT:")
        print(f"📊 Success Rate: {overall_success_rate*100:.1f}%")
        
        if overall_success_rate >= 0.8:
            print("✅ EXCELLENT - All timeframes working well")
        elif overall_success_rate >= 0.6:
            print("⚠️  GOOD - Most timeframes working")
        else:
            print("❌ NEEDS ATTENTION - Multiple timeframe issues")
            
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing error: {e}")
