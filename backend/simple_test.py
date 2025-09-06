"""Simple test for MultiAgent"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

# Manually load .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ Loaded .env from {env_path}")
    else:
        print(f"❌ .env not found at {env_path}")

def simple_test():
    # Load environment first
    load_env()
    
    print("Current environment variables:")
    print(f"OPENROUTER_API_KEY: {bool(os.getenv('OPENROUTER_API_KEY'))}")
    print(f"TOGETHER_API_KEY: {bool(os.getenv('TOGETHER_API_KEY'))}")
    
    try:
        from MultiAgent import MultiAgent, MultiAgentConfig
        
        # Try with a simpler model
        config = MultiAgentConfig(
            model="gpt-3.5-turbo",  # Use a reliable model
            temperature=0.7,
            max_tokens=100
        )
        
        print("\nInitializing MultiAgent...")
        ma = MultiAgent(config)
        print(f"✅ Success! Using {ma.provider} with model {ma.config.model}")
        
        print("\nTesting run method...")
        result = ma.run("Say hello")
        print(f"✅ Response: {result['response']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
