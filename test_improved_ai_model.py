#!/usr/bin/env python3
"""
Test Improved AI Model - Comprehensive testing of the fixed AI system
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.gold_trading_analyzer import GoldTradingAnalyzer
from utils.advanced_data_fetcher import AdvancedDataFetcher


def test_signal_diversity():
    """Test if the AI model generates diverse signals"""
    print("🎯 TESTING SIGNAL DIVERSITY")
    print("=" * 50)
    
    analyzer = GoldTradingAnalyzer(target_accuracy=0.90)
    
    # Run multiple analyses to see signal diversity
    signals = []
    confidences = []
    entry_prices = []
    analysis_methods = []
    
    print("Running 10 analysis cycles...")
    
    for i in range(10):
        try:
            result = analyzer.analyze_gold_market(real_time=False)
            
            signals.append(result['signal'])
            confidences.append(result['confidence'])
            entry_prices.append(result['entry_price'])
            analysis_methods.append(result.get('analysis_method', 'Standard'))
            
            print(f"   Cycle {i+1:2d}: {result['signal']:12} | {result['confidence']:5.1f}% | ${result['entry_price']:7.2f} | {result.get('analysis_method', 'Standard')}")
            
            # Small delay to potentially get different market conditions
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   Cycle {i+1:2d}: ERROR - {e}")
            signals.append('ERROR')
            confidences.append(0)
            entry_prices.append(0)
            analysis_methods.append('ERROR')
    
    # Analyze results
    print(f"\n📊 SIGNAL DIVERSITY ANALYSIS:")
    
    successful_signals = [s for s in signals if s != 'ERROR']
    unique_signals = set(successful_signals)
    
    print(f"   ✅ Successful analyses: {len(successful_signals)}/10")
    print(f"   🚦 Unique signals generated: {len(unique_signals)}")
    print(f"   📊 Signal distribution:")
    
    for signal in unique_signals:
        count = successful_signals.count(signal)
        percentage = (count / len(successful_signals)) * 100
        print(f"      {signal:12}: {count:2d} times ({percentage:5.1f}%)")
    
    # Confidence analysis
    valid_confidences = [c for c in confidences if c > 0]
    if valid_confidences:
        avg_confidence = sum(valid_confidences) / len(valid_confidences)
        min_confidence = min(valid_confidences)
        max_confidence = max(valid_confidences)
        
        print(f"\n   🎯 Confidence Analysis:")
        print(f"      Average: {avg_confidence:.1f}%")
        print(f"      Range: {min_confidence:.1f}% - {max_confidence:.1f}%")
    
    # Entry price analysis
    valid_prices = [p for p in entry_prices if p > 0]
    if valid_prices:
        avg_price = sum(valid_prices) / len(valid_prices)
        min_price = min(valid_prices)
        max_price = max(valid_prices)
        
        print(f"\n   💰 Entry Price Analysis:")
        print(f"      Average: ${avg_price:.2f}")
        print(f"      Range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Analysis method distribution
    unique_methods = set(analysis_methods)
    print(f"\n   🔧 Analysis Methods Used:")
    for method in unique_methods:
        count = analysis_methods.count(method)
        percentage = (count / len(analysis_methods)) * 100
        print(f"      {method:20}: {count:2d} times ({percentage:5.1f}%)")
    
    # Overall assessment
    diversity_score = len(unique_signals)
    confidence_score = avg_confidence if valid_confidences else 0
    success_rate = len(successful_signals) / 10
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"   📊 Diversity Score: {diversity_score}/5 signals")
    print(f"   🎯 Average Confidence: {confidence_score:.1f}%")
    print(f"   ✅ Success Rate: {success_rate*100:.0f}%")
    
    if diversity_score >= 3 and confidence_score >= 60 and success_rate >= 0.8:
        print("   🎉 EXCELLENT: AI model performing very well!")
        return True
    elif diversity_score >= 2 and confidence_score >= 50 and success_rate >= 0.7:
        print("   ✅ GOOD: AI model performing adequately")
        return True
    else:
        print("   ⚠️  NEEDS IMPROVEMENT: AI model needs further work")
        return False


def test_real_time_mode():
    """Test real-time analysis mode"""
    print("\n⚡ TESTING REAL-TIME MODE")
    print("=" * 50)
    
    analyzer = GoldTradingAnalyzer(target_accuracy=0.90)
    
    print("Testing real-time analysis (3 cycles)...")
    
    results = []
    
    for i in range(3):
        try:
            print(f"\n   Real-time cycle {i+1}:")
            start_time = time.time()
            
            result = analyzer.analyze_gold_market(real_time=True)
            
            execution_time = time.time() - start_time
            
            print(f"      Signal: {result['signal']}")
            print(f"      Confidence: {result['confidence']:.1f}%")
            print(f"      Entry Price: ${result['entry_price']:.2f}")
            print(f"      Execution Time: {execution_time:.2f}s")
            print(f"      Market Regime: {result.get('market_regime', 'UNKNOWN')}")
            print(f"      Volatility: {result.get('volatility_level', 'MODERATE')}")
            
            results.append({
                'signal': result['signal'],
                'confidence': result['confidence'],
                'execution_time': execution_time,
                'success': True
            })
            
            # Wait between real-time cycles
            if i < 2:
                time.sleep(2)
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results.append({
                'signal': 'ERROR',
                'confidence': 0,
                'execution_time': 0,
                'success': False
            })
    
    # Analyze real-time performance
    successful_results = [r for r in results if r['success']]
    
    print(f"\n📊 REAL-TIME PERFORMANCE:")
    print(f"   ✅ Successful cycles: {len(successful_results)}/3")
    
    if successful_results:
        avg_execution_time = sum(r['execution_time'] for r in successful_results) / len(successful_results)
        avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
        
        print(f"   ⚡ Average execution time: {avg_execution_time:.2f}s")
        print(f"   🎯 Average confidence: {avg_confidence:.1f}%")
        
        if avg_execution_time <= 10 and avg_confidence >= 50:
            print("   ✅ Real-time performance is good!")
            return True
        else:
            print("   ⚠️  Real-time performance needs improvement")
            return False
    else:
        print("   ❌ Real-time mode failed")
        return False


def test_accuracy_improvements():
    """Test accuracy improvements"""
    print("\n🎯 TESTING ACCURACY IMPROVEMENTS")
    print("=" * 50)
    
    data_fetcher = AdvancedDataFetcher()
    
    # Get recent market data for validation
    print("1. Fetching recent market data...")
    try:
        market_data = data_fetcher.fetch_current_data('GC=F', ['1h'], {'1h': 50})
        
        if market_data and '1h' in market_data:
            data = market_data['1h']
            print(f"   ✅ Retrieved {len(data)} hourly records")
            
            current_price = data['close'].iloc[-1]
            price_change_24h = ((current_price - data['close'].iloc[-24]) / data['close'].iloc[-24]) * 100 if len(data) >= 24 else 0
            
            print(f"   💰 Current price: ${current_price:.2f}")
            print(f"   📈 24h change: {price_change_24h:+.2f}%")
            
        else:
            print("   ❌ No market data available")
            return False
            
    except Exception as e:
        print(f"   ❌ Data fetching error: {e}")
        return False
    
    # Test prediction accuracy on recent data
    print("\n2. Testing prediction accuracy...")
    analyzer = GoldTradingAnalyzer(target_accuracy=0.90)
    
    try:
        result = analyzer.analyze_gold_market(real_time=False)
        
        print(f"   🚀 Generated signal: {result['signal']}")
        print(f"   🎯 Confidence: {result['confidence']:.1f}%")
        print(f"   📊 Accuracy estimate: {result['accuracy_estimate']:.1f}%")
        print(f"   💰 Predicted entry: ${result['entry_price']:.2f}")
        print(f"   📏 Risk/Reward: {result['risk_reward_ratio']:.1f}:1")
        
        # Check if prediction is reasonable
        price_diff = abs(result['entry_price'] - current_price) / current_price
        
        if price_diff <= 0.05:  # Within 5% of current price
            print("   ✅ Entry price prediction is reasonable")
            price_reasonable = True
        else:
            print(f"   ⚠️  Entry price seems off by {price_diff*100:.1f}%")
            price_reasonable = False
        
        # Check confidence levels
        confidence_good = result['confidence'] >= 40  # At least 40% confidence
        accuracy_good = result['accuracy_estimate'] >= 70  # At least 70% accuracy estimate
        
        print(f"\n   📊 ACCURACY ASSESSMENT:")
        print(f"      Price Prediction: {'✅' if price_reasonable else '❌'}")
        print(f"      Confidence Level: {'✅' if confidence_good else '❌'} ({result['confidence']:.1f}%)")
        print(f"      Accuracy Estimate: {'✅' if accuracy_good else '❌'} ({result['accuracy_estimate']:.1f}%)")
        
        return price_reasonable and confidence_good and accuracy_good
        
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return False


def main():
    """Main testing function"""
    print("🧪 COMPREHENSIVE AI MODEL TESTING")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    test_results = {
        'signal_diversity': False,
        'real_time_mode': False,
        'accuracy_improvements': False
    }
    
    try:
        # Test 1: Signal Diversity
        test_results['signal_diversity'] = test_signal_diversity()
        
        # Test 2: Real-time Mode
        test_results['real_time_mode'] = test_real_time_mode()
        
        # Test 3: Accuracy Improvements
        test_results['accuracy_improvements'] = test_accuracy_improvements()
        
        # Final Assessment
        print(f"\n{'='*60}")
        print("🏆 FINAL TEST RESULTS")
        print(f"{'='*60}")
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name.replace('_', ' ').title():20}: {status}")
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"\n📊 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.0f}%)")
        
        if success_rate >= 80:
            print("\n🎉 AI MODEL IMPROVEMENTS SUCCESSFUL!")
            print("   The AI model is now working significantly better!")
        elif success_rate >= 60:
            print("\n✅ AI MODEL IMPROVEMENTS GOOD!")
            print("   The AI model has been improved but could use more work.")
        else:
            print("\n⚠️  AI MODEL IMPROVEMENTS PARTIAL!")
            print("   Some improvements made but significant issues remain.")
        
        return success_rate >= 60
        
    except Exception as e:
        print(f"\n❌ TESTING ERROR: {e}")
        return False
    
    finally:
        print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
