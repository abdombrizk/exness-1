#!/usr/bin/env python3
"""
Test Script for Gold Trading AI Signal Generation
Tests the signal generation functionality comprehensively
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.gold_trading_analyzer import GoldTradingAnalyzer
from modules.technical_analysis import TechnicalAnalyzer
from modules.fundamental_analysis import FundamentalAnalyzer
from modules.risk_management import RiskManager
from utils.advanced_data_fetcher import AdvancedDataFetcher


def test_signal_generation():
    """Test comprehensive signal generation"""
    print("🧪 TESTING GOLD TRADING AI SIGNAL GENERATION")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing AI components...")
    analyzer = GoldTradingAnalyzer()
    technical_analyzer = TechnicalAnalyzer()
    fundamental_analyzer = FundamentalAnalyzer()
    risk_manager = RiskManager()
    data_fetcher = AdvancedDataFetcher()
    
    print("✅ All components initialized")
    
    # Test multiple signal generations
    test_results = []
    
    for test_run in range(3):
        print(f"\n{'='*20} TEST RUN {test_run + 1} {'='*20}")
        
        try:
            # Record start time
            start_time = time.time()
            
            print("🚀 Running AI analysis...")
            
            # Run comprehensive analysis
            result = analyzer.analyze_gold_market(real_time=False)
            
            # Record execution time
            execution_time = time.time() - start_time
            
            # Display results
            print(f"\n📊 ANALYSIS RESULTS (Run {test_run + 1}):")
            print(f"   🚀 Signal: {result['signal']}")
            print(f"   🎯 Confidence: {result['confidence']:.1f}%")
            print(f"   📊 Model Accuracy: {result['accuracy_estimate']:.1f}%")
            print(f"   💰 Entry Price: ${result['entry_price']:.2f}")
            print(f"   🛑 Stop Loss: ${result['stop_loss']:.2f}")
            print(f"   🎯 Take Profit: ${result['take_profit']:.2f}")
            print(f"   📏 Position Size: {result['position_size']} lots")
            print(f"   ⚖️  Risk/Reward: {result['risk_reward_ratio']:.1f}:1")
            print(f"   🎲 Win Probability: {result['win_probability']}%")
            print(f"   ⏱️  Execution Time: {execution_time:.2f}s")
            
            # Store result for analysis
            test_result = {
                'run': test_run + 1,
                'timestamp': datetime.now(),
                'signal': result['signal'],
                'confidence': result['confidence'],
                'accuracy_estimate': result['accuracy_estimate'],
                'entry_price': result['entry_price'],
                'stop_loss': result['stop_loss'],
                'take_profit': result['take_profit'],
                'position_size': result['position_size'],
                'risk_reward_ratio': result['risk_reward_ratio'],
                'win_probability': result['win_probability'],
                'execution_time': execution_time,
                'technical_score': result.get('technical_score', 0),
                'fundamental_score': result.get('fundamental_score', 0),
                'risk_score': result.get('risk_score', 0),
                'market_regime': result.get('market_regime', 'UNKNOWN'),
                'volatility_level': result.get('volatility_level', 'MODERATE')
            }
            
            test_results.append(test_result)
            
            print(f"✅ Test run {test_run + 1} completed successfully")
            
        except Exception as e:
            print(f"❌ Test run {test_run + 1} failed: {e}")
            test_results.append({
                'run': test_run + 1,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
        # Wait between tests
        if test_run < 2:
            print("⏳ Waiting 2 seconds before next test...")
            time.sleep(2)
    
    # Analyze test results
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    successful_tests = [r for r in test_results if 'error' not in r]
    failed_tests = [r for r in test_results if 'error' in r]
    
    print(f"✅ Successful tests: {len(successful_tests)}/3")
    print(f"❌ Failed tests: {len(failed_tests)}/3")
    
    if successful_tests:
        print(f"\n📈 SIGNAL ANALYSIS:")
        
        # Analyze signals
        signals = [r['signal'] for r in successful_tests]
        confidences = [r['confidence'] for r in successful_tests]
        execution_times = [r['execution_time'] for r in successful_tests]
        
        print(f"   🚦 Signals Generated: {', '.join(signals)}")
        print(f"   🎯 Average Confidence: {sum(confidences)/len(confidences):.1f}%")
        print(f"   ⏱️  Average Execution Time: {sum(execution_times)/len(execution_times):.2f}s")
        
        # Check consistency
        unique_signals = set(signals)
        if len(unique_signals) == 1:
            print(f"   ✅ Signal Consistency: EXCELLENT (all {list(unique_signals)[0]})")
        elif len(unique_signals) == 2:
            print(f"   ⚠️  Signal Consistency: MODERATE (mixed signals)")
        else:
            print(f"   ❌ Signal Consistency: POOR (inconsistent)")
            
        # Display detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in successful_tests:
            print(f"   Run {result['run']}: {result['signal']} "
                  f"({result['confidence']:.1f}% confidence, "
                  f"${result['entry_price']:.2f} entry, "
                  f"{result['execution_time']:.2f}s)")
    
    if failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for result in failed_tests:
            print(f"   Run {result['run']}: {result['error']}")
    
    return test_results


def test_timeframe_analysis():
    """Test analysis across different timeframes"""
    print(f"\n{'='*60}")
    print("⏰ TESTING MULTIPLE TIMEFRAMES")
    print(f"{'='*60}")
    
    data_fetcher = AdvancedDataFetcher()
    technical_analyzer = TechnicalAnalyzer()
    
    timeframes = ['1h', '4h', '1d']
    timeframe_results = {}
    
    for timeframe in timeframes:
        print(f"\n📊 Testing {timeframe} timeframe...")
        
        try:
            # Fetch data for timeframe
            market_data = data_fetcher.fetch_current_data('GC=F', [timeframe])
            
            if market_data and timeframe in market_data:
                data = market_data[timeframe]
                
                # Run technical analysis
                analysis = technical_analyzer.analyze_comprehensive(data)
                
                timeframe_results[timeframe] = {
                    'technical_score': analysis['technical_score'],
                    'signals': analysis.get('signals', []),
                    'trend': analysis.get('trend_analysis', {}).get('adx_signal', 'NEUTRAL'),
                    'data_points': len(data)
                }
                
                print(f"   ✅ {timeframe}: Score {analysis['technical_score']}/100, "
                      f"Trend: {timeframe_results[timeframe]['trend']}, "
                      f"Data: {len(data)} points")
            else:
                print(f"   ❌ {timeframe}: No data available")
                
        except Exception as e:
            print(f"   ❌ {timeframe}: Error - {e}")
    
    # Compare timeframes
    if len(timeframe_results) > 1:
        print(f"\n📊 TIMEFRAME COMPARISON:")
        scores = [r['technical_score'] for r in timeframe_results.values()]
        avg_score = sum(scores) / len(scores)
        print(f"   📈 Average Technical Score: {avg_score:.1f}/100")
        
        trends = [r['trend'] for r in timeframe_results.values()]
        unique_trends = set(trends)
        if len(unique_trends) == 1:
            print(f"   ✅ Trend Consistency: EXCELLENT (all {list(unique_trends)[0]})")
        else:
            print(f"   ⚠️  Trend Consistency: MIXED ({', '.join(unique_trends)})")
    
    return timeframe_results


if __name__ == "__main__":
    try:
        # Run signal generation tests
        signal_results = test_signal_generation()
        
        # Run timeframe tests
        timeframe_results = test_timeframe_analysis()
        
        print(f"\n{'='*60}")
        print("🎉 TESTING COMPLETE")
        print(f"{'='*60}")
        
        # Overall assessment
        successful_signals = len([r for r in signal_results if 'error' not in r])
        total_signals = len(signal_results)
        
        if successful_signals == total_signals:
            print("✅ OVERALL RESULT: ALL TESTS PASSED")
        elif successful_signals >= total_signals * 0.7:
            print("⚠️  OVERALL RESULT: MOSTLY SUCCESSFUL")
        else:
            print("❌ OVERALL RESULT: NEEDS ATTENTION")
            
        print(f"📊 Signal Generation Success Rate: {successful_signals}/{total_signals}")
        print(f"⏰ Timeframes Tested: {len(timeframe_results)}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing error: {e}")
