"""Test environment loading for MultiAgent"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

def test_env_loading():
    """Test if .env file is loaded correctly"""
    
    print("Testing environment loading...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(__file__)}")
    
    # Check if .env file exists
    env_paths = [
        os.path.join(os.path.dirname(__file__), '.env'),  # backend/.env
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),  # root/.env
    ]
    
    for env_path in env_paths:
        print(f"Checking: {env_path}")
        if os.path.exists(env_path):
            print(f"✅ Found .env at: {env_path}")
            with open(env_path, 'r') as f:
                content = f.read()
                print(f"Content preview:\n{content[:200]}...")
        else:
            print(f"❌ Not found: {env_path}")
    
    # Test environment variable
    api_key = os.getenv("TOGETHER_API_KEY")
    print(f"\nTOGETHER_API_KEY in environment: {bool(api_key)}")
    if api_key:
        print(f"Key prefix: {api_key[:8]}...")
    
    # Test the _late_env_hydrate function
    print("\nTesting _late_env_hydrate function...")
    from MultiAgent import _late_env_hydrate
    _late_env_hydrate()
    
    api_key_after = os.getenv("TOGETHER_API_KEY")
    print(f"TOGETHER_API_KEY after hydrate: {bool(api_key_after)}")
    if api_key_after:
        print(f"Key prefix after hydrate: {api_key_after[:8]}...")

if __name__ == "__main__":
    test_env_loading()
