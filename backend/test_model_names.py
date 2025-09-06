"""Test MultiAgent with provider-prefixed model names"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

# Load environment from both locations
def load_env():
    # Load from backend/.env
    backend_env = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(backend_env):
        with open(backend_env, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    # Load from root/.env (don't override existing)
    root_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(root_env):
        with open(root_env, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = value.strip()

def test_model_names():
    """Test MultiAgent with provider-prefixed model names"""
    
    load_env()
    
    print("Environment:")
    print(f"TOGETHER_API_KEY: {bool(os.getenv('TOGETHER_API_KEY'))}")
    print(f"OPENROUTER_API_KEY: {bool(os.getenv('OPENROUTER_API_KEY'))}")
    
    try:
        from MultiAgent import MultiAgent
        
        print("\n=== Testing with default configuration ===")
        ma = MultiAgent()
        print("✅ MultiAgent initialized successfully!")
        print(f"Provider: {ma.provider}")
        print(f"Model: {ma.config.model}")
        print(f"Base URL: {ma.config.base_url}")
        
        # Test with a simple English message
        print("\n🧪 Testing run method...")
        result = ma.run("Hello, who are you? Please respond briefly.")
        print("✅ run() method successful!")
        print(f"Response: {result['response']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_names()
