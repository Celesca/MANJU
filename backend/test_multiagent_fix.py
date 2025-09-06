"""Test script for MultiAgent with Together AI API"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

from MultiAgent import MultiAgent

def test_multiagent():
    """Test MultiAgent initialization and run"""
    
    # Make sure API key is set
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("❌ TOGETHER_API_KEY not found in environment!")
        print("Please set it with: export TOGETHER_API_KEY=your_key_here")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...")
    
    try:
        # Initialize MultiAgent
        ma = MultiAgent()
        print("✅ MultiAgent initialized successfully!")
        print(f"Model: {ma.config.model}")
        print(f"Base URL: {ma.config.base_url}")
        
        # Test run method
        print("\n🧪 Testing run method...")
        result = ma.run("Hello, who are you?")
        print("✅ run() method successful!")
        print(f"Response: {result['response']}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_multiagent()
