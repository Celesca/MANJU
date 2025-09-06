"""Simple test to verify model configuration"""
import os

# Load environment
backend_env = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(backend_env):
    with open(backend_env, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

root_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(root_env):
    with open(root_env, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key.strip() not in os.environ:
                    os.environ[key.strip()] = value.strip()

print("Testing configuration resolution...")
print(f"TOGETHER_API_KEY: {bool(os.getenv('TOGETHER_API_KEY'))}")
print(f"OPENROUTER_API_KEY: {bool(os.getenv('OPENROUTER_API_KEY'))}")

# Import and test config
import sys
sys.path.append(os.path.dirname(__file__))

from MultiAgent import MultiAgentConfig

config = MultiAgentConfig()
print(f"\nInitial model: {config.model}")
print(f"Initial base_url: {config.base_url}")

resolved = config.resolve()
print(f"\nAfter resolve:")
print(f"Model: {resolved.model}")
print(f"Base URL: {resolved.base_url}")
print(f"API Key: {bool(resolved.api_key)}")

# Determine provider
provider = "openrouter" if "openrouter.ai" in resolved.base_url else "together"
print(f"Provider: {provider}")

print("\n✅ Configuration test completed!")
