"""Minimal MultiAgent test with basic LiteLLM"""
import os
import sys

# Load environment
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

print("Environment loaded")
print(f"OPENROUTER_API_KEY: {bool(os.getenv('OPENROUTER_API_KEY'))}")

# Test LiteLLM directly
try:
    import litellm
    print("✅ LiteLLM imported")
    
    # Test direct call
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_tokens=50
    )
    print(f"✅ LiteLLM works: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ LiteLLM failed: {e}")
    import traceback
    traceback.print_exc()
