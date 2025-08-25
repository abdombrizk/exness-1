#!/usr/bin/env python3
"""
GUI Interface Testing for Gold Trading AI
Tests the graphical user interface components and functionality
"""

import sys
import os
import time
import threading
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_gui_launch():
    """Test GUI application launch"""
    print("🖥️ TESTING GUI INTERFACE")
    print("=" * 50)
    
    test_results = {
        'main_gui_launch': False,
        'alternative_gui_launch': False,
        'gui_components': False,
        'gui_responsiveness': False
    }
    
    # Test 1: Main GUI Launch
    print("\n1. Testing main GUI launch...")
    try:
        import subprocess
        import signal
        
        # Launch main GUI in background
        main_gui_process = subprocess.Popen(
            [sys.executable, 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Wait a few seconds for initialization
        time.sleep(5)
        
        # Check if process is still running (not crashed)
        if main_gui_process.poll() is None:
            print("   ✅ Main GUI launched successfully")
            test_results['main_gui_launch'] = True
            
            # Terminate the process
            main_gui_process.terminate()
            try:
                main_gui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                main_gui_process.kill()
        else:
            stdout, stderr = main_gui_process.communicate()
            print("   ❌ Main GUI failed to launch or crashed")
            print(f"      Error: {stderr.decode()[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Main GUI launch error: {e}")
    
    # Test 2: Alternative GUI Launch
    print("\n2. Testing alternative GUI launch...")
    try:
        alt_gui_process = subprocess.Popen(
            [sys.executable, 'gui_app/main_application.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        time.sleep(5)
        
        if alt_gui_process.poll() is None:
            print("   ✅ Alternative GUI launched successfully")
            test_results['alternative_gui_launch'] = True
            
            alt_gui_process.terminate()
            try:
                alt_gui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                alt_gui_process.kill()
        else:
            stdout, stderr = alt_gui_process.communicate()
            print("   ❌ Alternative GUI failed to launch or crashed")
            print(f"      Error: {stderr.decode()[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Alternative GUI launch error: {e}")
    
    # Test 3: GUI Components (simulated)
    print("\n3. Testing GUI components...")
    try:
        # Since we can't directly interact with GUI, we'll test the underlying components
        from modules.gold_trading_analyzer import GoldTradingAnalyzer
        
        # Test if the analyzer can be initialized (used by GUI)
        analyzer = GoldTradingAnalyzer()
        
        # Test if analysis can be performed (GUI button functionality)
        result = analyzer.analyze_gold_market(real_time=False)
        
        if result and 'signal' in result:
            print("   ✅ GUI backend components working")
            test_results['gui_components'] = True
        else:
            print("   ❌ GUI backend components failed")
            
    except Exception as e:
        print(f"   ❌ GUI components test error: {e}")
    
    # Test 4: GUI Responsiveness (simulated)
    print("\n4. Testing GUI responsiveness...")
    try:
        # Test multiple rapid analysis calls (simulating GUI button clicks)
        analyzer = GoldTradingAnalyzer()
        
        response_times = []
        for i in range(3):
            start_time = time.time()
            result = analyzer.analyze_gold_market(real_time=False)
            end_time = time.time()
            
            if result:
                response_times.append(end_time - start_time)
            
            time.sleep(0.1)  # Brief pause
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            if avg_response_time <= 10.0:  # Under 10 seconds is responsive
                print(f"   ✅ GUI responsiveness good (avg: {avg_response_time:.2f}s)")
                test_results['gui_responsiveness'] = True
            else:
                print(f"   ⚠️  GUI responsiveness slow (avg: {avg_response_time:.2f}s)")
        else:
            print("   ❌ GUI responsiveness test failed")
            
    except Exception as e:
        print(f"   ❌ GUI responsiveness test error: {e}")
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    success_rate = passed_tests / total_tests * 100
    
    print(f"\n📊 GUI TESTING SUMMARY:")
    print(f"   ✅ Passed: {passed_tests}/{total_tests}")
    print(f"   📈 Success Rate: {success_rate:.1f}%")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    if success_rate >= 75:
        print("   🎉 GUI TESTING: EXCELLENT")
    elif success_rate >= 50:
        print("   ✅ GUI TESTING: GOOD")
    else:
        print("   ⚠️  GUI TESTING: NEEDS IMPROVEMENT")
    
    return test_results, success_rate


def test_gui_tabs_functionality():
    """Test individual GUI tabs functionality"""
    print("\n📑 TESTING GUI TABS FUNCTIONALITY")
    print("=" * 50)
    
    # Test the modules that power each GUI tab
    tab_tests = {
        'Gold_AI_Trading_Tab': False,
        'Technical_Analysis_Tab': False,
        'Fundamental_Analysis_Tab': False,
        'Risk_Management_Tab': False,
        'Database_History_Tab': False
    }
    
    # Test Gold AI Trading Tab (main analyzer)
    print("\n1. Testing Gold AI Trading tab backend...")
    try:
        from modules.gold_trading_analyzer import GoldTradingAnalyzer
        analyzer = GoldTradingAnalyzer()
        result = analyzer.analyze_gold_market(real_time=False)
        
        if result and 'signal' in result:
            print("   ✅ Gold AI Trading tab backend working")
            tab_tests['Gold_AI_Trading_Tab'] = True
        else:
            print("   ❌ Gold AI Trading tab backend failed")
    except Exception as e:
        print(f"   ❌ Gold AI Trading tab error: {e}")
    
    # Test Technical Analysis Tab
    print("\n2. Testing Technical Analysis tab backend...")
    try:
        from modules.technical_analysis import TechnicalAnalyzer
        import pandas as pd
        import numpy as np
        
        tech_analyzer = TechnicalAnalyzer()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'open': np.random.uniform(2000, 2100, 50),
            'high': np.random.uniform(2050, 2150, 50),
            'low': np.random.uniform(1950, 2050, 50),
            'close': np.random.uniform(2000, 2100, 50),
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        result = tech_analyzer.analyze_comprehensive(sample_data)
        
        if result and 'technical_score' in result:
            print("   ✅ Technical Analysis tab backend working")
            tab_tests['Technical_Analysis_Tab'] = True
        else:
            print("   ❌ Technical Analysis tab backend failed")
    except Exception as e:
        print(f"   ❌ Technical Analysis tab error: {e}")
    
    # Test Fundamental Analysis Tab
    print("\n3. Testing Fundamental Analysis tab backend...")
    try:
        from modules.fundamental_analysis import FundamentalAnalyzer
        
        fund_analyzer = FundamentalAnalyzer()
        result = fund_analyzer.analyze_comprehensive()
        
        if result and 'fundamental_score' in result:
            print("   ✅ Fundamental Analysis tab backend working")
            tab_tests['Fundamental_Analysis_Tab'] = True
        else:
            print("   ❌ Fundamental Analysis tab backend failed")
    except Exception as e:
        print(f"   ❌ Fundamental Analysis tab error: {e}")
    
    # Test Risk Management Tab
    print("\n4. Testing Risk Management tab backend...")
    try:
        from modules.risk_management import RiskManager
        
        risk_manager = RiskManager()
        position_size = risk_manager.calculate_position_size(
            account_balance=10000,
            entry_price=2000,
            stop_loss=1950,
            confidence=0.8
        )
        
        if position_size and position_size > 0:
            print("   ✅ Risk Management tab backend working")
            tab_tests['Risk_Management_Tab'] = True
        else:
            print("   ❌ Risk Management tab backend failed")
    except Exception as e:
        print(f"   ❌ Risk Management tab error: {e}")
    
    # Test Database & History Tab
    print("\n5. Testing Database & History tab backend...")
    try:
        from utils.database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        db_manager.connect()
        db_manager.create_tables()
        
        print("   ✅ Database & History tab backend working")
        tab_tests['Database_History_Tab'] = True
    except Exception as e:
        print(f"   ❌ Database & History tab error: {e}")
    
    # Tab functionality summary
    passed_tabs = sum(tab_tests.values())
    total_tabs = len(tab_tests)
    tab_success_rate = passed_tabs / total_tabs * 100
    
    print(f"\n📊 GUI TABS TESTING SUMMARY:")
    print(f"   ✅ Working Tabs: {passed_tabs}/{total_tabs}")
    print(f"   📈 Success Rate: {tab_success_rate:.1f}%")
    
    for tab_name, result in tab_tests.items():
        status = "✅ WORKING" if result else "❌ FAILED"
        print(f"   {tab_name.replace('_', ' ')}: {status}")
    
    return tab_tests, tab_success_rate


def main():
    """Main GUI testing function"""
    print("🖥️ COMPREHENSIVE GUI TESTING SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Test GUI launch
        gui_results, gui_success_rate = test_gui_launch()
        
        # Test GUI tabs functionality
        tab_results, tab_success_rate = test_gui_tabs_functionality()
        
        # Overall GUI assessment
        overall_success_rate = (gui_success_rate + tab_success_rate) / 2
        
        print(f"\n" + "=" * 70)
        print("🏆 FINAL GUI TESTING RESULTS")
        print("=" * 70)
        
        print(f"🖥️  GUI Launch Tests: {gui_success_rate:.1f}%")
        print(f"📑 GUI Tabs Tests: {tab_success_rate:.1f}%")
        print(f"📊 Overall GUI Score: {overall_success_rate:.1f}%")
        
        if overall_success_rate >= 80:
            status = "🟢 EXCELLENT"
            recommendation = "GUI is fully functional and ready for use."
        elif overall_success_rate >= 60:
            status = "🟡 GOOD"
            recommendation = "GUI is mostly functional with minor issues."
        else:
            status = "🔴 NEEDS WORK"
            recommendation = "GUI has significant issues that need attention."
        
        print(f"🏥 GUI Health Status: {status}")
        print(f"💡 Recommendation: {recommendation}")
        
        return overall_success_rate >= 60
        
    except Exception as e:
        print(f"\n❌ GUI TESTING FAILED: {e}")
        return False
    
    finally:
        print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
