"""Test MultiAgent with Together AI configuration"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

def test_together_ai():
    """Test MultiAgent with Together AI"""
    
    print("Testing MultiAgent with Together AI...")
    
    try:
        from MultiAgent import MultiAgent
        
        # Initialize with default settings (should use Together AI)
        ma = MultiAgent()
        print("✅ MultiAgent initialized successfully!")
        print(f"Provider: {ma.provider}")
        print(f"Model: {ma.config.model}")
        print(f"Base URL: {ma.config.base_url}")
        print(f"API Key prefix: {ma.config.api_key[:8]}...")
        
        # Test with a simple message
        print("\n🧪 Testing run method...")
        result = ma.run("Hello, please introduce yourself briefly.")
        print("✅ run() method successful!")
        print(f"Response: {result['response']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_together_ai()
