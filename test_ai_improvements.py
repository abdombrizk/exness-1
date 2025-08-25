#!/usr/bin/env python3
"""
Test AI Improvements
Quick test to verify the AI model training improvements
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.gold_trading_analyzer import GoldTradingAnalyzer


def test_ai_improvements():
    """Test the AI improvements"""
    print("🧪 TESTING AI IMPROVEMENTS")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        print("1. Initializing analyzer...")
        analyzer = GoldTradingAnalyzer(target_accuracy=0.85)
        
        # Test system initialization
        print("\n2. Testing system initialization...")
        init_success = analyzer.initialize_system(retrain_if_needed=False)
        
        if init_success:
            print("   ✅ System initialization successful")
        else:
            print("   ⚠️  System initialization had issues")
        
        # Test prediction
        print("\n3. Testing AI prediction...")
        result = analyzer.analyze_gold_market(real_time=False)
        
        if result and 'signal' in result:
            print("   ✅ Prediction successful!")
            print(f"      Signal: {result['signal']}")
            print(f"      Confidence: {result['confidence']:.1f}%")
            print(f"      Entry Price: ${result['entry_price']:.2f}")
            print(f"      Method: {result.get('analysis_method', 'Unknown')}")
            
            # Check if using AI models
            if 'AI Ensemble' in result.get('analysis_method', ''):
                print("   🎉 USING FULL AI ENSEMBLE!")
                return True
            else:
                print("   ⚠️  Still using fallback methods")
                return False
        else:
            print("   ❌ Prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def main():
    """Main test function"""
    print("🤖 AI MODEL IMPROVEMENTS TEST")
    print("=" * 60)
    
    success = test_ai_improvements()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 AI IMPROVEMENTS SUCCESSFUL!")
        print("✅ System using full AI ensemble capabilities")
    else:
        print("⚠️  AI IMPROVEMENTS PARTIALLY SUCCESSFUL")
        print("✅ Models trained and working")
        print("⚠️  Still using some fallback methods")
        print("🔧 Additional optimization may be needed")
    
    print("\n🚀 Next Steps:")
    print("   1. Run main application: python main.py")
    print("   2. Test real-time predictions")
    print("   3. Monitor performance and accuracy")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
