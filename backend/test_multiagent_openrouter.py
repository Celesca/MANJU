"""Test updated MultiAgent with OpenRouter support"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

def test_multiagent_with_openrouter():
    """Test MultiAgent with OpenRouter API"""
    
    try:
        from MultiAgent import MultiAgent
        
        print("Testing MultiAgent initialization...")
        ma = MultiAgent()
        print("✅ MultiAgent initialized successfully!")
        print(f"Provider: {ma.provider}")
        print(f"Model: {ma.config.model}")
        print(f"Base URL: {ma.config.base_url}")
        print(f"API Key prefix: {ma.config.api_key[:8]}...")
        
        print("\n🧪 Testing run method...")
        result = ma.run("Hello, who are you?")
        print("✅ run() method successful!")
        print(f"Response: {result['response']}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_multiagent_with_openrouter()
