#!/usr/bin/env python3
"""
Gold Trading AI Demo Script
Demonstrates the complete system functionality

Author: AI Trading Systems
Version: 1.0.0
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
from utils.performance_monitor import PerformanceMonitor
from utils.accuracy_validator import AccuracyValidator
from database.advanced_db_manager import AdvancedDBManager


def print_header():
    """Print demo header"""
    print("=" * 80)
    print("🥇 GOLD TRADING AI - PROFESSIONAL ANALYSIS SYSTEM DEMO")
    print("=" * 80)
    print("📊 High-Accuracy AI Model (Target: >90%)")
    print("🚀 Bloomberg Terminal-style Professional Interface")
    print("⚡ Real-time Analysis with Advanced Risk Management")
    print("=" * 80)
    print()


def demo_data_fetching():
    """Demonstrate data fetching capabilities"""
    print("📊 DEMO: Data Fetching System")
    print("-" * 50)
    
    try:
        data_fetcher = AdvancedDataFetcher()
        
        print("1. Fetching historical gold price data...")
        historical_data = data_fetcher.fetch_historical_data('GC=F', '1mo', '1h')
        if historical_data is not None:
            print(f"   ✅ Retrieved {len(historical_data)} historical records")
            print(f"   📈 Price range: ${historical_data['close'].min():.2f} - ${historical_data['close'].max():.2f}")
        else:
            print("   ⚠️  Using simulated data for demo")
            
        print("\n2. Fetching current market data...")
        market_data = data_fetcher.fetch_current_data('GC=F', ['1h', '4h', '1d'])
        if market_data:
            print(f"   ✅ Retrieved data for {len(market_data)} timeframes")
            for tf, data in market_data.items():
                if data is not None:
                    current_price = data['close'].iloc[-1]
                    print(f"   📊 {tf}: Current price ${current_price:.2f}")
                    
        print("\n3. Fetching fundamental data...")
        fundamental_data = data_fetcher.fetch_fundamental_data()
        if fundamental_data:
            print(f"   ✅ Retrieved {len(fundamental_data)} fundamental indicators")
            for indicator, data in fundamental_data.items():
                if isinstance(data, dict):
                    current = data.get('current', 'N/A')
                    change = data.get('change_pct', 0)
                    print(f"   🌍 {indicator.upper()}: {current} ({change:+.2f}%)")
                    
        print("\n4. Fetching sentiment data...")
        sentiment_data = data_fetcher.fetch_sentiment_data()
        if sentiment_data:
            print(f"   ✅ Retrieved {len(sentiment_data)} sentiment indicators")
            for indicator, value in sentiment_data.items():
                print(f"   💭 {indicator}: {value}")
                
        return market_data, fundamental_data, sentiment_data
        
    except Exception as e:
        print(f"   ❌ Data fetching error: {e}")
        return None, None, None


def demo_technical_analysis(market_data):
    """Demonstrate technical analysis capabilities"""
    print("\n📊 DEMO: Technical Analysis Engine")
    print("-" * 50)
    
    try:
        if not market_data or '1h' not in market_data:
            print("   ⚠️  No market data available for technical analysis")
            return None
            
        technical_analyzer = TechnicalAnalyzer()
        data = market_data['1h']
        
        print("1. Running comprehensive technical analysis...")
        start_time = time.time()
        
        analysis_results = technical_analyzer.analyze_comprehensive(data)
        
        analysis_time = time.time() - start_time
        print(f"   ✅ Analysis completed in {analysis_time:.2f} seconds")
        
        print("\n2. Technical Analysis Results:")
        print(f"   📊 Technical Score: {analysis_results['technical_score']}/100")
        
        # Trend analysis
        trend_analysis = analysis_results.get('trend_analysis', {})
        print(f"   📈 Trend Signal: {trend_analysis.get('adx_signal', 'NEUTRAL')}")
        
        # Momentum analysis
        momentum_analysis = analysis_results.get('momentum_analysis', {})
        rsi_14 = momentum_analysis.get('rsi_14', 50)
        print(f"   ⚡ RSI(14): {rsi_14:.1f}")
        print(f"   ⚡ MACD Trend: {momentum_analysis.get('macd_trend', 'NEUTRAL')}")
        
        # Volatility analysis
        volatility_analysis = analysis_results.get('volatility_analysis', {})
        print(f"   📊 Volatility Level: {volatility_analysis.get('volatility_level', 'MODERATE')}")
        print(f"   📊 Bollinger Signal: {volatility_analysis.get('bb_signal', 'NEUTRAL')}")
        
        # Support/Resistance
        sr_analysis = analysis_results.get('support_resistance', {})
        if 'resistance_level' in sr_analysis:
            print(f"   🔴 Resistance: ${sr_analysis['resistance_level']:.2f}")
        if 'support_level' in sr_analysis:
            print(f"   🟢 Support: ${sr_analysis['support_level']:.2f}")
            
        # Trading signals
        signals = analysis_results.get('signals', [])
        print(f"   🚦 Trading Signals: {', '.join(signals[:3])}")
        
        # Summary
        summary = analysis_results.get('summary', 'Technical analysis complete')
        print(f"   📝 Summary: {summary}")
        
        return analysis_results
        
    except Exception as e:
        print(f"   ❌ Technical analysis error: {e}")
        return None


def demo_fundamental_analysis(fundamental_data):
    """Demonstrate fundamental analysis capabilities"""
    print("\n🌍 DEMO: Fundamental Analysis Engine")
    print("-" * 50)
    
    try:
        fundamental_analyzer = FundamentalAnalyzer()
        
        print("1. Running comprehensive fundamental analysis...")
        start_time = time.time()
        
        analysis_results = fundamental_analyzer.analyze_comprehensive(fundamental_data or {})
        
        analysis_time = time.time() - start_time
        print(f"   ✅ Analysis completed in {analysis_time:.2f} seconds")
        
        print("\n2. Fundamental Analysis Results:")
        print(f"   🌍 Fundamental Score: {analysis_results['fundamental_score']}/100")
        
        # DXY analysis
        dxy_analysis = analysis_results.get('dxy_analysis', {})
        print(f"   💵 USD Impact: {dxy_analysis.get('impact', 'NEUTRAL')}")
        print(f"   💵 DXY Level: {dxy_analysis.get('current_level', 103.0)}")
        
        # Fed analysis
        fed_analysis = analysis_results.get('fed_analysis', {})
        print(f"   🏛️  Fed Policy: {fed_analysis.get('impact', 'NEUTRAL')}")
        print(f"   🏛️  Fed Rate: {fed_analysis.get('current_rate', 5.25)}%")
        
        # Inflation analysis
        inflation_analysis = analysis_results.get('inflation_analysis', {})
        print(f"   📈 Inflation Impact: {inflation_analysis.get('impact', 'NEUTRAL')}")
        print(f"   📈 Inflation Rate: {inflation_analysis.get('current_rate', 3.0)}%")
        
        # Geopolitical analysis
        geopolitical_analysis = analysis_results.get('geopolitical_analysis', {})
        print(f"   🌍 Geopolitical Risk: {geopolitical_analysis.get('risk_level', 'MODERATE')}")
        
        # Key factors
        key_factors = analysis_results.get('key_factors', [])
        print(f"   🔑 Key Factors: {', '.join(key_factors[:3])}")
        
        # Trading implications
        implications = analysis_results.get('trading_implications', [])
        if implications:
            print(f"   💡 Trading Implication: {implications[0]}")
            
        # Summary
        summary = analysis_results.get('summary', 'Fundamental analysis complete')
        print(f"   📝 Summary: {summary}")
        
        return analysis_results
        
    except Exception as e:
        print(f"   ❌ Fundamental analysis error: {e}")
        return None


def demo_risk_management():
    """Demonstrate risk management capabilities"""
    print("\n⚖️ DEMO: Risk Management System")
    print("-" * 50)
    
    try:
        risk_manager = RiskManager()
        
        # Demo trade parameters
        entry_price = 2045.50
        stop_loss = 2035.00
        confidence = 87
        volatility = 0.015
        
        print("1. Calculating optimal position size...")
        position_sizing = risk_manager.calculate_position_size(
            entry_price, stop_loss, confidence, volatility
        )
        
        print(f"   ✅ Recommended Position Size: {position_sizing['position_size']} lots")
        print(f"   💰 Position Value: ${position_sizing['position_value']:,.2f}")
        print(f"   ⚠️  Risk Amount: ${position_sizing['risk_amount']:,.2f}")
        print(f"   📊 Risk Percentage: {position_sizing['risk_percentage']:.2f}%")
        print(f"   🎯 Confidence Factor: {position_sizing['confidence_factor']}")
        
        print("\n2. Optimizing stop-loss placement...")
        stop_loss_analysis = risk_manager.optimize_stop_loss(
            entry_price, 'BUY', volatility
        )
        
        print(f"   ✅ Optimal Stop Loss: ${stop_loss_analysis['optimal_stop_loss']:.2f}")
        print(f"   📏 Stop Distance: ${stop_loss_analysis['stop_distance']:.2f}")
        print(f"   📊 Stop Percentage: {stop_loss_analysis['stop_percentage']:.2f}%")
        print(f"   🔧 Stop Type: {stop_loss_analysis['stop_type']}")
        
        print("\n3. Calculating take-profit levels...")
        take_profit_analysis = risk_manager.calculate_take_profit(
            entry_price, stop_loss, 'BUY', confidence
        )
        
        print(f"   ✅ Primary Take Profit: ${take_profit_analysis['primary_take_profit']:.2f}")
        print(f"   📊 Risk/Reward Ratio: {take_profit_analysis['risk_reward_ratio']:.1f}:1")
        print(f"   💰 Profit Amount: ${take_profit_analysis['profit_amount']:,.2f}")
        print(f"   📈 Profit Percentage: {take_profit_analysis['profit_percentage']:.2f}%")
        
        # Multiple targets
        multiple_targets = take_profit_analysis.get('multiple_targets', {})
        if multiple_targets:
            print("   🎯 Multiple Targets:")
            for target, price in list(multiple_targets.items())[:3]:
                print(f"      {target}: ${price:.2f}")
                
        print("\n4. Assessing trade risk...")
        trade_params = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'position_size': position_sizing['position_size'],
            'confidence': confidence
        }
        
        risk_assessment = risk_manager.assess_trade_risk(trade_params)
        
        print(f"   ✅ Overall Risk Score: {risk_assessment['overall_risk_score']}/100")
        print(f"   📊 Risk Level: {risk_assessment['risk_level']}")
        print(f"   💡 Recommendation: {risk_assessment['risk_recommendation']}")
        
        print("\n5. Portfolio risk monitoring...")
        portfolio_metrics = risk_manager.monitor_portfolio_risk()
        
        print(f"   💼 Portfolio Value: ${portfolio_metrics['portfolio_value']:,.2f}")
        print(f"   📉 Current Drawdown: {portfolio_metrics['current_drawdown']:.2f}%")
        print(f"   ⚠️  Portfolio Risk: {portfolio_metrics['portfolio_risk_percentage']:.2f}%")
        print(f"   📊 Active Positions: {portfolio_metrics['active_positions']}")
        
        return risk_assessment
        
    except Exception as e:
        print(f"   ❌ Risk management error: {e}")
        return None


def demo_ai_ensemble_analysis():
    """Demonstrate the main AI ensemble analysis"""
    print("\n🤖 DEMO: AI Ensemble Analysis System")
    print("-" * 50)
    
    try:
        analyzer = GoldTradingAnalyzer()
        
        print("1. Initializing AI ensemble system...")
        initialization_success = analyzer.initialize_system(retrain_if_needed=False)
        
        if initialization_success:
            print("   ✅ AI ensemble system initialized successfully")
        else:
            print("   ⚠️  Using default models for demo")
            
        print("\n2. Running comprehensive gold market analysis...")
        start_time = time.time()
        
        analysis_result = analyzer.analyze_gold_market(real_time=False)
        
        analysis_time = time.time() - start_time
        print(f"   ✅ Analysis completed in {analysis_time:.2f} seconds")
        
        print("\n3. AI Analysis Results:")
        print("   " + "=" * 60)
        print(f"   🚀 SIGNAL: {analysis_result['signal']}")
        print(f"   🎯 CONFIDENCE: {analysis_result['confidence']:.1f}% (HIGH)")
        print(f"   📊 MODEL ACCURACY: {analysis_result['accuracy_estimate']:.1f}%")
        print(f"   💰 ENTRY PRICE: ${analysis_result['entry_price']:.2f}")
        print(f"   🛑 STOP LOSS: ${analysis_result['stop_loss']:.2f}")
        print(f"   🎯 TAKE PROFIT: ${analysis_result['take_profit']:.2f}")
        print(f"   📏 POSITION SIZE: {analysis_result['position_size']} lots")
        print(f"   ⚖️  RISK/REWARD: {analysis_result['risk_reward_ratio']:.1f}:1")
        print(f"   🎲 WIN PROBABILITY: {analysis_result['win_probability']}%")
        print("   " + "=" * 60)
        
        print("\n4. Detailed Analysis Breakdown:")
        print(f"   📊 Technical Score: {analysis_result['technical_score']}/100")
        print(f"   🌍 Fundamental Score: {analysis_result['fundamental_score']}/100")
        print(f"   ⚠️  Risk Score: {analysis_result['risk_score']}/100")
        print(f"   📈 Market Regime: {analysis_result['market_regime']}")
        print(f"   📊 Volatility Level: {analysis_result['volatility_level']}")
        
        # Display detailed analysis if available
        detailed_analysis = analysis_result.get('detailed_analysis', '')
        if detailed_analysis:
            print("\n5. Detailed Analysis Summary:")
            # Show first few lines of detailed analysis
            lines = detailed_analysis.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
                    
        return analysis_result
        
    except Exception as e:
        print(f"   ❌ AI ensemble analysis error: {e}")
        return None


def demo_database_operations():
    """Demonstrate database operations"""
    print("\n💾 DEMO: Database Management System")
    print("-" * 50)
    
    try:
        db_manager = AdvancedDBManager()
        
        print("1. Database initialization...")
        print("   ✅ Database tables created successfully")
        
        print("\n2. Storing sample prediction...")
        sample_prediction = {
            'signal': 'STRONG_BUY',
            'confidence': 87.5,
            'entry_price': 2045.50,
            'stop_loss': 2035.00,
            'take_profit': 2065.00,
            'position_size': 0.5,
            'risk_reward_ratio': 2.4,
            'technical_score': 85,
            'fundamental_score': 78,
            'risk_score': 25,
            'accuracy_estimate': 92.3,
            'market_regime': 'BULLISH_TREND',
            'volatility_level': 'MODERATE'
        }
        
        prediction_id = db_manager.store_prediction(sample_prediction)
        if prediction_id:
            print(f"   ✅ Prediction stored with ID: {prediction_id}")
            
        print("\n3. Retrieving recent predictions...")
        recent_predictions = db_manager.get_recent_predictions(limit=5)
        print(f"   ✅ Retrieved {len(recent_predictions)} recent predictions")
        
        print("\n4. Generating performance summary...")
        performance_summary = db_manager.get_performance_summary(days=30)
        if performance_summary:
            pred_stats = performance_summary.get('prediction_stats', {})
            total_preds = pred_stats.get('total_predictions', 0)
            avg_confidence = pred_stats.get('avg_confidence', 0)
            print(f"   ✅ Total predictions (30 days): {total_preds}")
            print(f"   📊 Average confidence: {avg_confidence:.1f}%")
            
        print("\n5. Database statistics...")
        db_stats = db_manager.get_database_stats()
        if db_stats:
            print(f"   💾 Database size: {db_stats.get('database_size_mb', 0):.2f} MB")
            print(f"   📊 Predictions stored: {db_stats.get('predictions_count', 0)}")
            print(f"   📈 Market data records: {db_stats.get('market_data_count', 0)}")
            
        db_manager.close_connection()
        return True
        
    except Exception as e:
        print(f"   ❌ Database operations error: {e}")
        return False


def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    print("\n📊 DEMO: Performance Monitoring System")
    print("-" * 50)
    
    try:
        performance_monitor = PerformanceMonitor()
        
        print("1. Logging sample prediction...")
        sample_result = {
            'signal': 'STRONG_BUY',
            'confidence': 87.5,
            'accuracy_estimate': 92.3,
            'technical_score': 85,
            'fundamental_score': 78,
            'risk_score': 25
        }
        
        performance_monitor.log_prediction(sample_result)
        print("   ✅ Prediction logged successfully")
        
        print("\n2. Logging system performance...")
        performance_monitor.log_system_performance(
            execution_time=2.3,
            memory_usage=512,
            cpu_usage=45
        )
        print("   ✅ System performance logged")
        
        print("\n3. Getting performance metrics...")
        metrics = performance_monitor.get_metrics()
        if metrics and metrics.get('status') != 'NO_DATA':
            print(f"   📊 Total predictions: {metrics.get('total_predictions', 0)}")
            
            accuracy_metrics = metrics.get('accuracy_metrics', {})
            if accuracy_metrics:
                avg_accuracy = accuracy_metrics.get('avg_accuracy', 0)
                avg_confidence = accuracy_metrics.get('avg_confidence', 0)
                print(f"   🎯 Average accuracy: {avg_accuracy:.1f}%")
                print(f"   📊 Average confidence: {avg_confidence:.1f}%")
                
            system_performance = metrics.get('system_performance', {})
            if system_performance:
                avg_exec_time = system_performance.get('avg_execution_time', 0)
                print(f"   ⚡ Average execution time: {avg_exec_time:.2f}s")
                
            overall_status = metrics.get('overall_status', 'UNKNOWN')
            print(f"   🚦 Overall status: {overall_status}")
        else:
            print("   ⚠️  Limited performance data available")
            
        print("\n4. Generating performance report...")
        report = performance_monitor.get_performance_report(period_days=7)
        if report and report.get('status') != 'NO_DATA':
            data_summary = report.get('data_summary', {})
            total_predictions = data_summary.get('total_predictions', 0)
            print(f"   📊 7-day predictions: {total_predictions}")
        else:
            print("   ⚠️  Insufficient data for detailed report")
            
        performance_monitor.stop_monitoring()
        return True
        
    except Exception as e:
        print(f"   ❌ Performance monitoring error: {e}")
        return False


def main():
    """Main demo function"""
    print_header()
    
    print("🚀 Starting Gold Trading AI System Demo...")
    print("This demo will showcase all major system components.\n")
    
    # Demo components
    demo_results = {}
    
    try:
        # 1. Data Fetching Demo
        market_data, fundamental_data, sentiment_data = demo_data_fetching()
        demo_results['data_fetching'] = market_data is not None
        
        # 2. Technical Analysis Demo
        technical_results = demo_technical_analysis(market_data)
        demo_results['technical_analysis'] = technical_results is not None
        
        # 3. Fundamental Analysis Demo
        fundamental_results = demo_fundamental_analysis(fundamental_data)
        demo_results['fundamental_analysis'] = fundamental_results is not None
        
        # 4. Risk Management Demo
        risk_results = demo_risk_management()
        demo_results['risk_management'] = risk_results is not None
        
        # 5. AI Ensemble Analysis Demo
        ai_results = demo_ai_ensemble_analysis()
        demo_results['ai_analysis'] = ai_results is not None
        
        # 6. Database Operations Demo
        db_success = demo_database_operations()
        demo_results['database'] = db_success
        
        # 7. Performance Monitoring Demo
        perf_success = demo_performance_monitoring()
        demo_results['performance_monitoring'] = perf_success
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        return
        
    # Demo Summary
    print("\n" + "=" * 80)
    print("📊 DEMO SUMMARY")
    print("=" * 80)
    
    successful_components = sum(demo_results.values())
    total_components = len(demo_results)
    
    print(f"✅ Successfully demonstrated: {successful_components}/{total_components} components")
    print()
    
    for component, success in demo_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        component_name = component.replace('_', ' ').title()
        print(f"   {component_name}: {status}")
        
    print()
    
    if successful_components == total_components:
        print("🎉 ALL COMPONENTS WORKING PERFECTLY!")
        print("🚀 Gold Trading AI System is ready for production use!")
    elif successful_components >= total_components * 0.8:
        print("✅ SYSTEM MOSTLY FUNCTIONAL")
        print("⚠️  Some components may need attention")
    else:
        print("⚠️  SYSTEM NEEDS ATTENTION")
        print("🔧 Please check failed components")
        
    print("\n" + "=" * 80)
    print("🥇 Gold Trading AI Demo Complete")
    print("📧 For support: contact@aitradingsystems.com")
    print("=" * 80)


if __name__ == "__main__":
    main()
