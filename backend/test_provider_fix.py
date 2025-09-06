"""Test MultiAgent with explicit provider configuration"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

def test_with_provider():
    """Test MultiAgent with explicit provider setting"""
    
    print("Testing MultiAgent with explicit provider configuration...")
    
    # Set environment variables manually to test both providers
    print("=== Testing with Together AI ===")
    os.environ.pop("OPENROUTER_API_KEY", None)  # Remove OpenRouter key if present
    
    try:
        from MultiAgent import MultiAgent
        
        # Test with Together AI
        ma = MultiAgent()
        print("✅ MultiAgent initialized with Together AI!")
        print(f"Provider: {ma.provider}")
        print(f"Model: {ma.config.model}")
        print(f"Base URL: {ma.config.base_url}")
        
        # Test a simple call
        result = ma.run("Hello, respond briefly.")
        print("✅ Together AI call successful!")
        print(f"Response: {result['response'][:100]}...")
        
    except Exception as e:
        print(f"❌ Together AI failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Now test with OpenRouter if available
    print("\n=== Testing with OpenRouter ===")
    backend_env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(backend_env_path):
        with open(backend_env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('OPENROUTER_API_KEY='):
                    openrouter_key = line.split('=', 1)[1]
                    if openrouter_key and openrouter_key != 'your_openrouter_api_key_here':
                        os.environ["OPENROUTER_API_KEY"] = openrouter_key
                        break
    
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            # Force reload of the module to pick up new env vars
            import importlib
            import MultiAgent as MA
            importlib.reload(MA)
            
            ma_or = MA.MultiAgent()
            print("✅ MultiAgent initialized with OpenRouter!")
            print(f"Provider: {ma_or.provider}")
            print(f"Model: {ma_or.config.model}")
            print(f"Base URL: {ma_or.config.base_url}")
            
            # Test a simple call
            result_or = ma_or.run("Hello, respond briefly.")
            print("✅ OpenRouter call successful!")
            print(f"Response: {result_or['response'][:100]}...")
            
        except Exception as e:
            print(f"❌ OpenRouter failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No OpenRouter API key available for testing")

if __name__ == "__main__":
    test_with_provider()
