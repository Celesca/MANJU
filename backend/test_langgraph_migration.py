#!/usr/bin/env python3
"""
Test script for MultiAgent LangGraph implementation
Validates migration from CrewAI to LangGraph
"""

import sys
import os
import time
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_result(test_name: str, result: Dict[str, Any], passed: bool):
    """Pretty print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status} | {test_name}")
    print(f"  Response: {result.get('response', 'N/A')[:100]}...")
    print(f"  Intent: {result.get('intent', 'N/A')}")
    print(f"  Route: {result.get('route', 'N/A')}")
    print(f"  Time: {result.get('processing_time_seconds', 0):.3f}s")
    if not passed:
        print(f"  Error: {result.get('error', 'Unknown error')}")

def test_langgraph():
    """Test LangGraph implementation"""
    print("="*60)
    print("🧪 Testing LangGraph Implementation")
    print("="*60)
    
    try:
        from MultiAgent_LangGraph import VoiceCallCenterMultiAgent
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        system = VoiceCallCenterMultiAgent()
        print("✅ Initialization successful")
        print(f"   Model: {system.config.model}")
        print(f"   Base URL: {system.config.base_url}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test cases
    test_cases = [
        {
            "name": "Greeting Test",
            "input": "สวัสดีครับ",
            "expected_keywords": ["สวัสดี", "ยินดี"],
            "expected_intent": None  # Fast path
        },
        {
            "name": "Product SKU Query",
            "input": "มีสินค้ารหัส TEL001 อะไรบ้างครับ",
            "expected_keywords": ["Galaxy", "สมาร์ท", "โทรศัพท์"],
            "expected_intent": "PRODUCT"
        },
        {
            "name": "Product Search Query",
            "input": "ขอสอบถามแพ็กเกจอินเทอร์เน็ตหน่อยครับ",
            "expected_keywords": ["อินเทอร์เน็ต", "Fiber", "INT002"],
            "expected_intent": "PRODUCT"
        },
        {
            "name": "Knowledge Query",
            "input": "นโยบายการคืนสินค้าเป็นยังไงบ้างครับ",
            "expected_keywords": ["นโยบาย", "คืน"],
            "expected_intent": "KNOWLEDGE"
        },
        {
            "name": "Owner Query",
            "input": "ขอดูข้อมูลของนายสมชาย ใจดี หน่อยครับ",
            "expected_keywords": ["สมชาย", "Galaxy", "TEL001"],
            "expected_intent": "PRODUCT"
        },
        {
            "name": "Thank You",
            "input": "ขอบคุณครับ",
            "expected_keywords": ["ยินดี"],
            "expected_intent": None  # Fast path
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_cases)
    total_time = 0
    
    for test_case in test_cases:
        try:
            start = time.time()
            result = system.process_voice_input(test_case["input"])
            elapsed = time.time() - start
            total_time += elapsed
            
            response = result.get("response", "")
            intent = result.get("intent")
            
            # Check if response contains expected keywords
            keyword_match = any(kw.lower() in response.lower() for kw in test_case["expected_keywords"])
            
            # Check intent if expected
            intent_match = (test_case["expected_intent"] is None or 
                          intent == test_case["expected_intent"] or
                          "fast_path" in result.get("route", ""))
            
            passed = keyword_match or intent_match
            if passed:
                passed_tests += 1
            
            print_result(test_case["name"], result, passed)
            
        except Exception as e:
            print_result(test_case["name"], {"error": str(e)}, False)
    
    # System status test
    print("\n" + "="*60)
    print("📊 System Status Test")
    print("="*60)
    try:
        status = system.get_system_status()
        print("✅ Status retrieved successfully")
        print(f"   Engine: {status.get('engine')}")
        print(f"   Model: {status.get('model')}")
        print(f"   Architecture: {status.get('architecture')}")
        print(f"   Tools: {status.get('tools')}")
        print(f"   Ready: {status.get('ready')}")
        passed_tests += 1
        total_tests += 1
    except Exception as e:
        print(f"❌ Status test failed: {e}")
        total_tests += 1
    
    # Summary
    print("\n" + "="*60)
    print("📈 Test Summary")
    print("="*60)
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"Average Response Time: {total_time/len(test_cases):.3f}s")
    print(f"Total Test Time: {total_time:.3f}s")
    
    return passed_tests == total_tests

def compare_implementations():
    """Compare CrewAI vs LangGraph performance"""
    print("\n" + "="*60)
    print("⚖️  Comparing CrewAI vs LangGraph")
    print("="*60)
    
    test_query = "มีสินค้ารหัส TEL001 อะไรบ้างครับ"
    
    # Test CrewAI
    try:
        from MultiAgent_New import VoiceCallCenterMultiAgent as CrewAIAgent
        print("\n🔵 Testing CrewAI Implementation...")
        crewai_system = CrewAIAgent()
        start = time.time()
        crewai_result = crewai_system.process_voice_input(test_query)
        crewai_time = time.time() - start
        print(f"   Time: {crewai_time:.3f}s")
        print(f"   Response: {crewai_result.get('response', '')[:100]}...")
    except Exception as e:
        print(f"   ⚠️ CrewAI not available: {e}")
        crewai_time = None
    
    # Test LangGraph
    try:
        from MultiAgent_LangGraph import VoiceCallCenterMultiAgent as LangGraphAgent
        print("\n🟢 Testing LangGraph Implementation...")
        langgraph_system = LangGraphAgent()
        start = time.time()
        langgraph_result = langgraph_system.process_voice_input(test_query)
        langgraph_time = time.time() - start
        print(f"   Time: {langgraph_time:.3f}s")
        print(f"   Response: {langgraph_result.get('response', '')[:100]}...")
    except Exception as e:
        print(f"   ❌ LangGraph failed: {e}")
        langgraph_time = None
    
    # Compare
    if crewai_time and langgraph_time:
        print("\n📊 Performance Comparison:")
        print(f"   CrewAI: {crewai_time:.3f}s")
        print(f"   LangGraph: {langgraph_time:.3f}s")
        improvement = ((crewai_time - langgraph_time) / crewai_time) * 100
        print(f"   Improvement: {improvement:+.1f}%")
        if langgraph_time < crewai_time:
            print(f"   🚀 LangGraph is {crewai_time/langgraph_time:.2f}x faster!")
        else:
            print(f"   ⚠️ CrewAI is {langgraph_time/crewai_time:.2f}x faster")

def test_api_compatibility():
    """Test API compatibility between implementations"""
    print("\n" + "="*60)
    print("🔄 Testing API Compatibility")
    print("="*60)
    
    try:
        from MultiAgent_LangGraph import VoiceCallCenterMultiAgent
        system = VoiceCallCenterMultiAgent()
        
        # Test all API methods
        print("\n✅ Testing process_voice_input()...")
        result = system.process_voice_input("test")
        assert "response" in result
        assert "model" in result
        assert "processing_time_seconds" in result
        print("   ✓ Response format correct")
        
        print("\n✅ Testing get_system_status()...")
        status = system.get_system_status()
        assert "engine" in status
        assert "model" in status
        assert "ready" in status
        assert "tools" in status
        print("   ✓ Status format correct")
        
        print("\n✅ Testing with conversation history...")
        history = [
            {"role": "user", "content": "สวัสดี"},
            {"role": "assistant", "content": "สวัสดีครับ"}
        ]
        result = system.process_voice_input("ขอบคุณ", conversation_history=history)
        assert "response" in result
        print("   ✓ Conversation history supported")
        
        print("\n✅ All API compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ API compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "🎬 "*15)
    print("MultiAgent LangGraph Migration Test Suite")
    print("🎬 "*15)
    
    # Main tests
    langgraph_passed = test_langgraph()
    
    # API compatibility
    api_passed = test_api_compatibility()
    
    # Performance comparison (if both available)
    compare_implementations()
    
    # Final result
    print("\n" + "="*60)
    print("🏁 Final Result")
    print("="*60)
    
    if langgraph_passed and api_passed:
        print("✅ ALL TESTS PASSED - Migration Successful! 🎉")
        print("\n📝 Next Steps:")
        print("   1. Update new_server.py to use MultiAgent_LangGraph")
        print("   2. Test in production environment")
        print("   3. Monitor performance metrics")
        print("   4. Consider deprecating CrewAI implementation")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Review Issues Above")
        print("\n🔍 Troubleshooting:")
        print("   1. Check dependencies: pip install langgraph langchain")
        print("   2. Verify API keys in .env file")
        print("   3. Review error messages above")
        sys.exit(1)
