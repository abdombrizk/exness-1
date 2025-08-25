#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Gold Trading AI
Tests all components, integration, functionality, performance, and error handling
"""

import sys
import os
import time
import traceback
import psutil
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all modules for testing
try:
    from modules.gold_trading_analyzer import GoldTradingAnalyzer
    from modules.technical_analysis import TechnicalAnalyzer
    from modules.fundamental_analysis import FundamentalAnalyzer
    from modules.risk_management import RiskManager
    from utils.advanced_data_fetcher import AdvancedDataFetcher
    from utils.accuracy_validator import AccuracyValidator
    from utils.performance_monitor import PerformanceMonitor
    # Skip database manager if not available
    try:
        from utils.database_manager import DatabaseManager
        DATABASE_AVAILABLE = True
    except ImportError:
        DATABASE_AVAILABLE = False
        print("⚠️  Database manager not available - skipping database tests")
except ImportError as e:
    print(f"❌ Critical import error: {e}")
    sys.exit(1)


class ComprehensiveTestSuite:
    """Comprehensive testing suite for Gold Trading AI"""
    
    def __init__(self):
        self.test_results = {
            'component_tests': {},
            'integration_tests': {},
            'functional_tests': {},
            'performance_tests': {},
            'error_handling_tests': {},
            'ui_tests': {}
        }
        self.start_time = datetime.now()
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_test_result(self, category, test_name, status, details=None, metrics=None):
        """Log test result"""
        self.test_count += 1
        if status:
            self.passed_tests += 1
            status_str = "✅ PASS"
        else:
            self.failed_tests += 1
            status_str = "❌ FAIL"
            
        result = {
            'status': status,
            'status_str': status_str,
            'details': details or {},
            'metrics': metrics or {},
            'timestamp': datetime.now()
        }
        
        if category not in self.test_results:
            self.test_results[category] = {}
        self.test_results[category][test_name] = result
        
        print(f"   {status_str} {test_name}")
        if details and not status:
            print(f"      Error: {details.get('error', 'Unknown error')}")
    
    def measure_performance(self, func, *args, **kwargs):
        """Measure function performance"""
        process = psutil.Process()
        
        # Initial measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = process.cpu_percent()
        
        metrics = {
            'execution_time': round(end_time - start_time, 3),
            'memory_usage': round(end_memory - start_memory, 2),
            'cpu_usage': round(end_cpu - start_cpu, 2),
            'success': success,
            'error': error
        }
        
        return result, metrics

    def test_component_gold_trading_analyzer(self):
        """Test Gold Trading Analyzer component"""
        print("\n🥇 Testing Gold Trading Analyzer...")
        
        try:
            # Test initialization
            analyzer = GoldTradingAnalyzer(target_accuracy=0.90)
            self.log_test_result('component_tests', 'GoldTradingAnalyzer_Init', True, 
                               {'target_accuracy': analyzer.target_accuracy})
            
            # Test system initialization
            result, metrics = self.measure_performance(analyzer.initialize_system, retrain_if_needed=False)
            self.log_test_result('component_tests', 'GoldTradingAnalyzer_SystemInit', 
                               metrics['success'], {'error': metrics.get('error')}, metrics)
            
            # Test analysis
            result, metrics = self.measure_performance(analyzer.analyze_gold_market, real_time=False)
            analysis_success = metrics['success'] and result is not None
            
            analysis_details = {}
            if result:
                analysis_details = {
                    'signal': result.get('signal', 'N/A'),
                    'confidence': result.get('confidence', 0),
                    'entry_price': result.get('entry_price', 0),
                    'analysis_method': result.get('analysis_method', 'Unknown')
                }
            
            self.log_test_result('component_tests', 'GoldTradingAnalyzer_Analysis', 
                               analysis_success, analysis_details, metrics)
            
        except Exception as e:
            self.log_test_result('component_tests', 'GoldTradingAnalyzer_Exception', False, 
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def test_component_technical_analyzer(self):
        """Test Technical Analyzer component"""
        print("\n📊 Testing Technical Analyzer...")
        
        try:
            # Test initialization
            tech_analyzer = TechnicalAnalyzer()
            self.log_test_result('component_tests', 'TechnicalAnalyzer_Init', True)
            
            # Create sample data for testing
            dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
            sample_data = pd.DataFrame({
                'open': np.random.uniform(2000, 2100, 100),
                'high': np.random.uniform(2050, 2150, 100),
                'low': np.random.uniform(1950, 2050, 100),
                'close': np.random.uniform(2000, 2100, 100),
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
            
            # Test comprehensive analysis
            result, metrics = self.measure_performance(tech_analyzer.analyze_comprehensive, sample_data)
            analysis_success = metrics['success'] and result is not None
            
            analysis_details = {}
            if result:
                analysis_details = {
                    'technical_score': result.get('technical_score', 0),
                    'indicators_count': len(result.get('indicators', {})),
                    'signals_count': len(result.get('signals', []))
                }
            
            self.log_test_result('component_tests', 'TechnicalAnalyzer_Analysis', 
                               analysis_success, analysis_details, metrics)
            
            # Test individual indicators
            indicators_to_test = ['calculate_rsi', 'calculate_macd', 'calculate_bollinger_bands']
            for indicator in indicators_to_test:
                if hasattr(tech_analyzer, indicator):
                    try:
                        result, metrics = self.measure_performance(getattr(tech_analyzer, indicator), sample_data['close'])
                        self.log_test_result('component_tests', f'TechnicalAnalyzer_{indicator}', 
                                           metrics['success'], {'error': metrics.get('error')}, metrics)
                    except Exception as e:
                        self.log_test_result('component_tests', f'TechnicalAnalyzer_{indicator}', False, 
                                           {'error': str(e)})
            
        except Exception as e:
            self.log_test_result('component_tests', 'TechnicalAnalyzer_Exception', False, 
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def test_component_fundamental_analyzer(self):
        """Test Fundamental Analyzer component"""
        print("\n🌍 Testing Fundamental Analyzer...")
        
        try:
            # Test initialization
            fund_analyzer = FundamentalAnalyzer()
            self.log_test_result('component_tests', 'FundamentalAnalyzer_Init', True)
            
            # Test economic data fetching
            result, metrics = self.measure_performance(fund_analyzer.fetch_economic_data)
            self.log_test_result('component_tests', 'FundamentalAnalyzer_EconomicData', 
                               metrics['success'], {'error': metrics.get('error')}, metrics)
            
            # Test sentiment analysis
            result, metrics = self.measure_performance(fund_analyzer.analyze_market_sentiment)
            self.log_test_result('component_tests', 'FundamentalAnalyzer_Sentiment', 
                               metrics['success'], {'error': metrics.get('error')}, metrics)
            
            # Test comprehensive analysis
            result, metrics = self.measure_performance(fund_analyzer.analyze_comprehensive)
            analysis_success = metrics['success'] and result is not None
            
            analysis_details = {}
            if result:
                analysis_details = {
                    'fundamental_score': result.get('fundamental_score', 0),
                    'economic_factors': len(result.get('economic_factors', {})),
                    'sentiment_score': result.get('sentiment_score', 0)
                }
            
            self.log_test_result('component_tests', 'FundamentalAnalyzer_Analysis', 
                               analysis_success, analysis_details, metrics)
            
        except Exception as e:
            self.log_test_result('component_tests', 'FundamentalAnalyzer_Exception', False, 
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def test_component_risk_manager(self):
        """Test Risk Manager component"""
        print("\n⚖️ Testing Risk Manager...")
        
        try:
            # Test initialization
            risk_manager = RiskManager()
            self.log_test_result('component_tests', 'RiskManager_Init', True, 
                               {'max_risk_per_trade': risk_manager.max_risk_per_trade,
                                'max_portfolio_risk': risk_manager.max_portfolio_risk})
            
            # Test position sizing
            test_params = {
                'account_balance': 10000,
                'entry_price': 2000,
                'stop_loss': 1950,
                'confidence': 0.8
            }
            
            result, metrics = self.measure_performance(risk_manager.calculate_position_size, **test_params)
            position_success = metrics['success'] and result is not None and result > 0
            
            position_details = {'position_size': result} if result else {}
            self.log_test_result('component_tests', 'RiskManager_PositionSize', 
                               position_success, position_details, metrics)
            
            # Test risk assessment
            result, metrics = self.measure_performance(risk_manager.assess_trade_risk, **test_params)
            risk_success = metrics['success'] and result is not None
            
            risk_details = {}
            if result:
                risk_details = {
                    'risk_score': result.get('risk_score', 0),
                    'risk_level': result.get('risk_level', 'Unknown')
                }
            
            self.log_test_result('component_tests', 'RiskManager_RiskAssessment', 
                               risk_success, risk_details, metrics)
            
        except Exception as e:
            self.log_test_result('component_tests', 'RiskManager_Exception', False, 
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def test_component_data_fetcher(self):
        """Test Data Fetcher component"""
        print("\n📈 Testing Data Fetcher...")
        
        try:
            # Test initialization
            data_fetcher = AdvancedDataFetcher()
            self.log_test_result('component_tests', 'DataFetcher_Init', True)
            
            # Test current data fetching
            result, metrics = self.measure_performance(
                data_fetcher.fetch_current_data, 'GC=F', ['1h'], {'1h': 50}
            )
            current_data_success = metrics['success'] and result is not None
            
            current_data_details = {}
            if result:
                current_data_details = {
                    'timeframes': len(result),
                    'total_records': sum(len(data) for data in result.values()) if result else 0
                }
            
            self.log_test_result('component_tests', 'DataFetcher_CurrentData', 
                               current_data_success, current_data_details, metrics)
            
            # Test historical data fetching
            result, metrics = self.measure_performance(
                data_fetcher.fetch_historical_data, 'GC=F', '1mo', '1h'
            )
            historical_success = metrics['success'] and result is not None and len(result) > 0
            
            historical_details = {}
            if result is not None:
                historical_details = {
                    'records_count': len(result),
                    'date_range': f"{result.index[0]} to {result.index[-1]}" if len(result) > 0 else "No data"
                }
            
            self.log_test_result('component_tests', 'DataFetcher_HistoricalData', 
                               historical_success, historical_details, metrics)
            
        except Exception as e:
            self.log_test_result('component_tests', 'DataFetcher_Exception', False, 
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def test_component_database_manager(self):
        """Test Database Manager component"""
        print("\n💾 Testing Database Manager...")

        if not DATABASE_AVAILABLE:
            self.log_test_result('component_tests', 'DatabaseManager_NotAvailable', False,
                               {'error': 'Database manager module not available'})
            return

        try:
            # Test initialization
            db_manager = DatabaseManager()
            self.log_test_result('component_tests', 'DatabaseManager_Init', True)

            # Test database connection
            result, metrics = self.measure_performance(db_manager.connect)
            self.log_test_result('component_tests', 'DatabaseManager_Connect',
                               metrics['success'], {'error': metrics.get('error')}, metrics)

            # Test table creation
            result, metrics = self.measure_performance(db_manager.create_tables)
            self.log_test_result('component_tests', 'DatabaseManager_CreateTables',
                               metrics['success'], {'error': metrics.get('error')}, metrics)

            # Test data insertion
            test_prediction = {
                'signal': 'BUY',
                'confidence': 75.0,
                'entry_price': 2000.0,
                'stop_loss': 1950.0,
                'take_profit': 2100.0,
                'position_size': 0.5,
                'risk_reward_ratio': 2.0
            }

            result, metrics = self.measure_performance(db_manager.save_prediction, test_prediction)
            self.log_test_result('component_tests', 'DatabaseManager_SavePrediction',
                               metrics['success'], {'error': metrics.get('error')}, metrics)

        except Exception as e:
            self.log_test_result('component_tests', 'DatabaseManager_Exception', False,
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def run_component_tests(self):
        """Run all component tests"""
        print("🔧 COMPONENT TESTING")
        print("=" * 60)
        
        self.test_component_gold_trading_analyzer()
        self.test_component_technical_analyzer()
        self.test_component_fundamental_analyzer()
        self.test_component_risk_manager()
        self.test_component_data_fetcher()
        self.test_component_database_manager()
        
        # Component test summary
        component_results = self.test_results['component_tests']
        passed = sum(1 for result in component_results.values() if result['status'])
        total = len(component_results)
        
        print(f"\n📊 Component Tests Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
        return passed, total

    def test_integration_data_flow(self):
        """Test data flow between modules"""
        print("\n🔄 Testing Integration - Data Flow...")

        try:
            # Test complete data pipeline
            data_fetcher = AdvancedDataFetcher()
            tech_analyzer = TechnicalAnalyzer()
            fund_analyzer = FundamentalAnalyzer()
            risk_manager = RiskManager()

            # Step 1: Fetch data
            market_data, metrics1 = self.measure_performance(
                data_fetcher.fetch_current_data, 'GC=F', ['1h'], {'1h': 100}
            )

            if not metrics1['success'] or not market_data:
                self.log_test_result('integration_tests', 'DataFlow_DataFetch', False,
                                   {'error': 'Failed to fetch market data'})
                return

            # Step 2: Technical analysis
            tech_result, metrics2 = self.measure_performance(
                tech_analyzer.analyze_comprehensive, market_data['1h']
            )

            # Step 3: Fundamental analysis
            fund_result, metrics3 = self.measure_performance(
                fund_analyzer.analyze_comprehensive
            )

            # Step 4: Risk assessment
            if tech_result and 'entry_price' in tech_result:
                risk_params = {
                    'account_balance': 10000,
                    'entry_price': tech_result.get('entry_price', 2000),
                    'stop_loss': tech_result.get('entry_price', 2000) * 0.98,
                    'confidence': 0.75
                }
                risk_result, metrics4 = self.measure_performance(
                    risk_manager.calculate_position_size, **risk_params
                )
            else:
                risk_result, metrics4 = None, {'success': False, 'error': 'No entry price from technical analysis'}

            # Evaluate integration success
            integration_success = all([
                metrics1['success'], metrics2['success'],
                metrics3['success'], metrics4['success']
            ])

            integration_details = {
                'data_fetch_time': metrics1['execution_time'],
                'tech_analysis_time': metrics2['execution_time'],
                'fund_analysis_time': metrics3['execution_time'],
                'risk_calc_time': metrics4['execution_time'],
                'total_pipeline_time': sum([
                    metrics1['execution_time'], metrics2['execution_time'],
                    metrics3['execution_time'], metrics4['execution_time']
                ])
            }

            self.log_test_result('integration_tests', 'DataFlow_Pipeline',
                               integration_success, integration_details)

        except Exception as e:
            self.log_test_result('integration_tests', 'DataFlow_Exception', False,
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def test_integration_api_connections(self):
        """Test API integrations"""
        print("\n🌐 Testing Integration - API Connections...")

        apis_to_test = [
            ('Yahoo Finance', 'GC=F'),
            ('Alternative Symbols', 'GLD'),
            ('Economic Data', 'DXY')
        ]

        data_fetcher = AdvancedDataFetcher()

        for api_name, symbol in apis_to_test:
            try:
                result, metrics = self.measure_performance(
                    data_fetcher.fetch_historical_data, symbol, '5d', '1h'
                )

                api_success = metrics['success'] and result is not None and len(result) > 0
                api_details = {
                    'symbol': symbol,
                    'records_fetched': len(result) if result is not None else 0,
                    'response_time': metrics['execution_time']
                }

                self.log_test_result('integration_tests', f'API_{api_name.replace(" ", "_")}',
                                   api_success, api_details, metrics)

            except Exception as e:
                self.log_test_result('integration_tests', f'API_{api_name.replace(" ", "_")}', False,
                                   {'error': str(e)})

    def test_integration_model_pipeline(self):
        """Test model training and prediction pipeline"""
        print("\n🤖 Testing Integration - Model Pipeline...")

        try:
            analyzer = GoldTradingAnalyzer()

            # Test model initialization
            init_result, metrics1 = self.measure_performance(analyzer.initialize_system, retrain_if_needed=False)

            # Test prediction pipeline
            prediction_result, metrics2 = self.measure_performance(analyzer.analyze_gold_market, real_time=False)

            pipeline_success = metrics2['success'] and prediction_result is not None
            pipeline_details = {
                'initialization_time': metrics1['execution_time'],
                'prediction_time': metrics2['execution_time'],
                'total_pipeline_time': metrics1['execution_time'] + metrics2['execution_time']
            }

            if prediction_result:
                pipeline_details.update({
                    'signal_generated': prediction_result.get('signal', 'None'),
                    'confidence_level': prediction_result.get('confidence', 0),
                    'analysis_method': prediction_result.get('analysis_method', 'Unknown')
                })

            self.log_test_result('integration_tests', 'ModelPipeline_Complete',
                               pipeline_success, pipeline_details)

        except Exception as e:
            self.log_test_result('integration_tests', 'ModelPipeline_Exception', False,
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def run_integration_tests(self):
        """Run all integration tests"""
        print("\n🔗 INTEGRATION TESTING")
        print("=" * 60)

        self.test_integration_data_flow()
        self.test_integration_api_connections()
        self.test_integration_model_pipeline()

        # Integration test summary
        integration_results = self.test_results['integration_tests']
        passed = sum(1 for result in integration_results.values() if result['status'])
        total = len(integration_results)

        print(f"\n📊 Integration Tests Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
        return passed, total

    def test_functional_signal_generation(self):
        """Test signal generation functionality"""
        print("\n🚦 Testing Functional - Signal Generation...")

        try:
            analyzer = GoldTradingAnalyzer()

            # Test multiple signal generations
            signals = []
            confidences = []
            execution_times = []

            for i in range(5):
                result, metrics = self.measure_performance(analyzer.analyze_gold_market, real_time=False)

                if metrics['success'] and result:
                    signals.append(result.get('signal', 'UNKNOWN'))
                    confidences.append(result.get('confidence', 0))
                    execution_times.append(metrics['execution_time'])

                time.sleep(0.1)  # Small delay between tests

            # Analyze signal generation
            unique_signals = set(signals)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

            signal_success = len(signals) >= 3 and avg_confidence > 0
            signal_details = {
                'signals_generated': len(signals),
                'unique_signals': len(unique_signals),
                'signal_types': list(unique_signals),
                'avg_confidence': round(avg_confidence, 1),
                'avg_execution_time': round(avg_execution_time, 3)
            }

            self.log_test_result('functional_tests', 'SignalGeneration_Multiple',
                               signal_success, signal_details)

        except Exception as e:
            self.log_test_result('functional_tests', 'SignalGeneration_Exception', False,
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def test_functional_timeframe_analysis(self):
        """Test multiple timeframe analysis"""
        print("\n⏰ Testing Functional - Timeframe Analysis...")

        try:
            data_fetcher = AdvancedDataFetcher()
            tech_analyzer = TechnicalAnalyzer()

            timeframes = ['1h', '4h', '1d']
            timeframe_results = {}

            for tf in timeframes:
                # Fetch data for timeframe
                data, metrics = self.measure_performance(
                    data_fetcher.fetch_current_data, 'GC=F', [tf], {tf: 50}
                )

                if metrics['success'] and data and tf in data:
                    # Analyze timeframe
                    analysis, analysis_metrics = self.measure_performance(
                        tech_analyzer.analyze_comprehensive, data[tf]
                    )

                    timeframe_results[tf] = {
                        'data_fetch_success': True,
                        'analysis_success': analysis_metrics['success'],
                        'data_points': len(data[tf]),
                        'technical_score': analysis.get('technical_score', 0) if analysis else 0,
                        'execution_time': metrics['execution_time'] + analysis_metrics['execution_time']
                    }
                else:
                    timeframe_results[tf] = {
                        'data_fetch_success': False,
                        'analysis_success': False,
                        'error': metrics.get('error', 'Unknown error')
                    }

            # Evaluate timeframe analysis
            successful_timeframes = sum(1 for result in timeframe_results.values()
                                      if result['data_fetch_success'] and result['analysis_success'])

            timeframe_success = successful_timeframes >= 2
            timeframe_details = {
                'successful_timeframes': successful_timeframes,
                'total_timeframes': len(timeframes),
                'timeframe_results': timeframe_results
            }

            self.log_test_result('functional_tests', 'TimeframeAnalysis_Multiple',
                               timeframe_success, timeframe_details)

        except Exception as e:
            self.log_test_result('functional_tests', 'TimeframeAnalysis_Exception', False,
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def run_functional_tests(self):
        """Run all functional tests"""
        print("\n⚙️ FUNCTIONAL TESTING")
        print("=" * 60)

        self.test_functional_signal_generation()
        self.test_functional_timeframe_analysis()

        # Functional test summary
        functional_results = self.test_results['functional_tests']
        passed = sum(1 for result in functional_results.values() if result['status'])
        total = len(functional_results)

        print(f"\n📊 Functional Tests Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
        return passed, total

    def test_performance_execution_times(self):
        """Test system execution performance"""
        print("\n⚡ Testing Performance - Execution Times...")

        try:
            analyzer = GoldTradingAnalyzer()

            # Test multiple analysis cycles for performance consistency
            execution_times = []
            memory_usage = []

            for i in range(3):
                result, metrics = self.measure_performance(analyzer.analyze_gold_market, real_time=False)

                if metrics['success']:
                    execution_times.append(metrics['execution_time'])
                    memory_usage.append(abs(metrics['memory_usage']))

                time.sleep(0.5)  # Brief pause between tests

            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)
                max_execution_time = max(execution_times)
                min_execution_time = min(execution_times)
                avg_memory_usage = sum(memory_usage) / len(memory_usage)

                # Performance criteria
                performance_good = avg_execution_time <= 10.0  # Under 10 seconds average
                memory_reasonable = avg_memory_usage <= 100.0  # Under 100MB memory increase

                performance_success = performance_good and memory_reasonable
                performance_details = {
                    'avg_execution_time': round(avg_execution_time, 3),
                    'min_execution_time': round(min_execution_time, 3),
                    'max_execution_time': round(max_execution_time, 3),
                    'avg_memory_usage': round(avg_memory_usage, 2),
                    'test_cycles': len(execution_times),
                    'performance_rating': 'GOOD' if performance_good else 'NEEDS_IMPROVEMENT',
                    'memory_rating': 'GOOD' if memory_reasonable else 'HIGH'
                }

                self.log_test_result('performance_tests', 'ExecutionTimes_Analysis',
                                   performance_success, performance_details)
            else:
                self.log_test_result('performance_tests', 'ExecutionTimes_Analysis', False,
                                   {'error': 'No successful executions to measure'})

        except Exception as e:
            self.log_test_result('performance_tests', 'ExecutionTimes_Exception', False,
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def test_performance_api_response_times(self):
        """Test API response performance"""
        print("\n🌐 Testing Performance - API Response Times...")

        try:
            data_fetcher = AdvancedDataFetcher()

            # Test different API calls
            api_tests = [
                ('Current_Data_1h', lambda: data_fetcher.fetch_current_data('GC=F', ['1h'], {'1h': 50})),
                ('Historical_Data_1mo', lambda: data_fetcher.fetch_historical_data('GC=F', '1mo', '1h')),
                ('Historical_Data_5d', lambda: data_fetcher.fetch_historical_data('GC=F', '5d', '1h'))
            ]

            api_performance = {}

            for test_name, api_call in api_tests:
                result, metrics = self.measure_performance(api_call)

                api_performance[test_name] = {
                    'success': metrics['success'],
                    'response_time': metrics['execution_time'],
                    'data_size': len(result) if result and hasattr(result, '__len__') else 0,
                    'error': metrics.get('error')
                }

            # Evaluate API performance
            successful_apis = sum(1 for perf in api_performance.values() if perf['success'])
            avg_response_time = sum(perf['response_time'] for perf in api_performance.values()
                                  if perf['success']) / max(successful_apis, 1)

            api_success = successful_apis >= 2 and avg_response_time <= 15.0  # Under 15 seconds average
            api_details = {
                'successful_apis': successful_apis,
                'total_apis': len(api_tests),
                'avg_response_time': round(avg_response_time, 3),
                'api_performance': api_performance
            }

            self.log_test_result('performance_tests', 'APIResponseTimes_Multiple',
                               api_success, api_details)

        except Exception as e:
            self.log_test_result('performance_tests', 'APIResponseTimes_Exception', False,
                               {'error': str(e), 'traceback': traceback.format_exc()})

    def run_performance_tests(self):
        """Run all performance tests"""
        print("\n⚡ PERFORMANCE TESTING")
        print("=" * 60)

        self.test_performance_execution_times()
        self.test_performance_api_response_times()

        # Performance test summary
        performance_results = self.test_results['performance_tests']
        passed = sum(1 for result in performance_results.values() if result['status'])
        total = len(performance_results)

        print(f"\n📊 Performance Tests Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
        return passed, total

    def test_error_handling_network_issues(self):
        """Test error handling for network issues"""
        print("\n🌐 Testing Error Handling - Network Issues...")

        try:
            data_fetcher = AdvancedDataFetcher()

            # Test with invalid symbol
            result, metrics = self.measure_performance(
                data_fetcher.fetch_historical_data, 'INVALID_SYMBOL_12345', '1mo', '1h'
            )

            # Should handle gracefully (not crash)
            graceful_handling = not metrics['success']  # Should fail but not crash
            error_details = {
                'handled_gracefully': graceful_handling,
                'error_message': metrics.get('error', 'No error message'),
                'execution_completed': True  # If we reach here, it didn't crash
            }

            self.log_test_result('error_handling_tests', 'NetworkIssues_InvalidSymbol',
                               graceful_handling, error_details)

        except Exception as e:
            # If we get here, error handling failed (system crashed)
            self.log_test_result('error_handling_tests', 'NetworkIssues_InvalidSymbol', False,
                               {'error': 'System crashed instead of handling error gracefully',
                                'exception': str(e)})

    def test_error_handling_invalid_data(self):
        """Test error handling for invalid data scenarios"""
        print("\n📊 Testing Error Handling - Invalid Data...")

        try:
            tech_analyzer = TechnicalAnalyzer()

            # Test with empty data
            empty_data = pd.DataFrame()
            result, metrics = self.measure_performance(tech_analyzer.analyze_comprehensive, empty_data)

            empty_data_handled = not metrics['success']  # Should fail gracefully

            # Test with invalid data structure
            invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
            result2, metrics2 = self.measure_performance(tech_analyzer.analyze_comprehensive, invalid_data)

            invalid_data_handled = not metrics2['success']  # Should fail gracefully

            error_handling_success = empty_data_handled and invalid_data_handled
            error_details = {
                'empty_data_handled': empty_data_handled,
                'invalid_data_handled': invalid_data_handled,
                'empty_data_error': metrics.get('error', 'No error'),
                'invalid_data_error': metrics2.get('error', 'No error')
            }

            self.log_test_result('error_handling_tests', 'InvalidData_Scenarios',
                               error_handling_success, error_details)

        except Exception as e:
            self.log_test_result('error_handling_tests', 'InvalidData_Scenarios', False,
                               {'error': 'System crashed instead of handling invalid data',
                                'exception': str(e)})

    def run_error_handling_tests(self):
        """Run all error handling tests"""
        print("\n🛡️ ERROR HANDLING TESTING")
        print("=" * 60)

        self.test_error_handling_network_issues()
        self.test_error_handling_invalid_data()

        # Error handling test summary
        error_results = self.test_results['error_handling_tests']
        passed = sum(1 for result in error_results.values() if result['status'])
        total = len(error_results)

        print(f"\n📊 Error Handling Tests Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
        return passed, total

    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 80)

        total_time = datetime.now() - self.start_time

        print(f"🕒 Test Duration: {total_time}")
        print(f"📊 Total Tests: {self.test_count}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📈 Success Rate: {self.passed_tests/self.test_count*100:.1f}%")

        # Category summaries
        categories = [
            ('Component Tests', 'component_tests'),
            ('Integration Tests', 'integration_tests'),
            ('Functional Tests', 'functional_tests'),
            ('Performance Tests', 'performance_tests'),
            ('Error Handling Tests', 'error_handling_tests')
        ]

        print(f"\n📊 DETAILED RESULTS BY CATEGORY:")
        for category_name, category_key in categories:
            if category_key in self.test_results:
                results = self.test_results[category_key]
                passed = sum(1 for result in results.values() if result['status'])
                total = len(results)
                percentage = passed/total*100 if total > 0 else 0

                print(f"   {category_name:20}: {passed:2d}/{total:2d} ({percentage:5.1f}%)")

        # Overall system health assessment
        overall_percentage = self.passed_tests/self.test_count*100

        print(f"\n🏥 SYSTEM HEALTH ASSESSMENT:")
        if overall_percentage >= 90:
            health_status = "🟢 EXCELLENT"
            recommendation = "System is performing excellently and ready for production use."
        elif overall_percentage >= 75:
            health_status = "🟡 GOOD"
            recommendation = "System is performing well with minor issues that should be addressed."
        elif overall_percentage >= 60:
            health_status = "🟠 FAIR"
            recommendation = "System has moderate issues that need attention before production use."
        else:
            health_status = "🔴 POOR"
            recommendation = "System has significant issues that must be resolved before use."

        print(f"   Status: {health_status}")
        print(f"   Overall Score: {overall_percentage:.1f}%")
        print(f"   Recommendation: {recommendation}")

        return {
            'total_tests': self.test_count,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'success_rate': overall_percentage,
            'health_status': health_status,
            'recommendation': recommendation,
            'test_duration': str(total_time),
            'detailed_results': self.test_results
        }

    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 COMPREHENSIVE GOLD TRADING AI TEST SUITE")
        print("=" * 80)
        print(f"Started at: {self.start_time}")

        try:
            # Run all test categories
            self.run_component_tests()
            self.run_integration_tests()
            self.run_functional_tests()
            self.run_performance_tests()
            self.run_error_handling_tests()

            # Generate final report
            report = self.generate_test_report()

            return report

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR IN TEST SUITE: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None


def main():
    """Main function to run comprehensive tests"""
    test_suite = ComprehensiveTestSuite()
    report = test_suite.run_all_tests()

    if report:
        print(f"\n✅ Test suite completed successfully!")
        print(f"📊 Final Score: {report['success_rate']:.1f}%")
        return report['success_rate'] >= 60
    else:
        print(f"\n❌ Test suite failed to complete!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
