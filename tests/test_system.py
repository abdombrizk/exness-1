#!/usr/bin/env python3
"""
Basic System Tests for Gold Trading AI
Simple tests to verify system functionality

Author: AI Trading Systems
Version: 1.0.0
"""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.advanced_data_fetcher import AdvancedDataFetcher
from modules.technical_analysis import TechnicalAnalyzer
from modules.fundamental_analysis import FundamentalAnalyzer
from modules.risk_management import RiskManager
from database.advanced_db_manager import AdvancedDBManager


class TestSystemComponents(unittest.TestCase):
    """Test basic functionality of system components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_fetcher = AdvancedDataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.risk_manager = RiskManager()
        
    def test_data_fetcher_initialization(self):
        """Test data fetcher initialization"""
        self.assertIsNotNone(self.data_fetcher)
        self.assertIsInstance(self.data_fetcher.cache, dict)
        
    def test_technical_analyzer_initialization(self):
        """Test technical analyzer initialization"""
        self.assertIsNotNone(self.technical_analyzer)
        self.assertIsInstance(self.technical_analyzer.indicators, dict)
        
    def test_fundamental_analyzer_initialization(self):
        """Test fundamental analyzer initialization"""
        self.assertIsNotNone(self.fundamental_analyzer)
        self.assertIsInstance(self.fundamental_analyzer.weights, dict)
        
    def test_risk_manager_initialization(self):
        """Test risk manager initialization"""
        self.assertIsNotNone(self.risk_manager)
        self.assertEqual(self.risk_manager.max_risk_per_trade, 0.02)
        self.assertEqual(self.risk_manager.max_portfolio_risk, 0.10)
        
    def test_position_sizing_calculation(self):
        """Test position sizing calculation"""
        position_sizing = self.risk_manager.calculate_position_size(
            entry_price=2000.0,
            stop_loss=1990.0,
            confidence=80
        )
        
        self.assertIsInstance(position_sizing, dict)
        self.assertIn('position_size', position_sizing)
        self.assertIn('risk_percentage', position_sizing)
        self.assertGreater(position_sizing['position_size'], 0)
        
    def test_stop_loss_optimization(self):
        """Test stop loss optimization"""
        stop_loss_analysis = self.risk_manager.optimize_stop_loss(
            entry_price=2000.0,
            signal_direction='BUY',
            volatility=0.02
        )
        
        self.assertIsInstance(stop_loss_analysis, dict)
        self.assertIn('optimal_stop_loss', stop_loss_analysis)
        self.assertLess(stop_loss_analysis['optimal_stop_loss'], 2000.0)
        
    def test_database_initialization(self):
        """Test database initialization"""
        db_manager = AdvancedDBManager(':memory:')  # Use in-memory database for testing
        self.assertIsNotNone(db_manager.connection)
        
        # Test storing a sample prediction
        sample_prediction = {
            'signal': 'BUY',
            'confidence': 85.0,
            'entry_price': 2000.0,
            'stop_loss': 1990.0,
            'take_profit': 2020.0,
            'position_size': 0.5,
            'risk_reward_ratio': 2.0,
            'technical_score': 80,
            'fundamental_score': 75,
            'risk_score': 30,
            'accuracy_estimate': 90.0,
            'market_regime': 'BULLISH',
            'volatility_level': 'MODERATE'
        }
        
        prediction_id = db_manager.store_prediction(sample_prediction)
        self.assertIsNotNone(prediction_id)
        
        db_manager.close_connection()


class TestDataGeneration(unittest.TestCase):
    """Test data generation and processing"""
    
    def test_sample_data_generation(self):
        """Test sample data generation"""
        data_fetcher = AdvancedDataFetcher()
        sample_data = data_fetcher._generate_sample_data('1mo', '1h')
        
        self.assertIsNotNone(sample_data)
        self.assertGreater(len(sample_data), 0)
        self.assertIn('close', sample_data.columns)
        self.assertIn('open', sample_data.columns)
        self.assertIn('high', sample_data.columns)
        self.assertIn('low', sample_data.columns)
        
    def test_technical_analysis_with_sample_data(self):
        """Test technical analysis with sample data"""
        data_fetcher = AdvancedDataFetcher()
        sample_data = data_fetcher._generate_sample_data('1mo', '1h')
        
        technical_analyzer = TechnicalAnalyzer()
        analysis_result = technical_analyzer.analyze_comprehensive(sample_data)
        
        self.assertIsInstance(analysis_result, dict)
        self.assertIn('technical_score', analysis_result)
        self.assertGreaterEqual(analysis_result['technical_score'], 0)
        self.assertLessEqual(analysis_result['technical_score'], 100)


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading"""
    
    def test_model_config_import(self):
        """Test model configuration import"""
        try:
            from config.model_config import MODEL_CONFIG, RISK_CONFIG
            
            self.assertIsInstance(MODEL_CONFIG, dict)
            self.assertIsInstance(RISK_CONFIG, dict)
            self.assertIn('target_accuracy', MODEL_CONFIG)
            self.assertIn('max_risk_per_trade', RISK_CONFIG)
            
        except ImportError:
            self.fail("Could not import configuration")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSystemComponents))
    test_suite.addTest(unittest.makeSuite(TestDataGeneration))
    test_suite.addTest(unittest.makeSuite(TestConfigurationLoading))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
