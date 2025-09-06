"""Debug LLM initialization parameters"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

# Load environment
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Also check parent .env
parent_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(parent_env):
    with open(parent_env, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key.strip() not in os.environ:  # Don't override
                    os.environ[key.strip()] = value.strip()

print("Environment loaded:")
print(f"TOGETHER_API_KEY: {bool(os.getenv('TOGETHER_API_KEY'))}")
print(f"OPENROUTER_API_KEY: {bool(os.getenv('OPENROUTER_API_KEY'))}")

try:
    from MultiAgent import MultiAgentConfig
    config = MultiAgentConfig()
    resolved_config = config.resolve()
    
    print(f"\nResolved config:")
    print(f"Model: {resolved_config.model}")
    print(f"Base URL: {resolved_config.base_url}")
    print(f"API Key: {bool(resolved_config.api_key)}")
    if resolved_config.api_key:
        print(f"API Key prefix: {resolved_config.api_key[:8]}...")
    
    # Determine provider
    provider = "openrouter" if "openrouter.ai" in resolved_config.base_url else "together"
    print(f"Provider: {provider}")
    
    # Show what LLM parameters would be used
    llm_kwargs = {
        "model": resolved_config.model,
        "api_key": resolved_config.api_key,
        "base_url": resolved_config.base_url,
        "temperature": resolved_config.temperature,
        "max_tokens": resolved_config.max_tokens,
        "timeout": resolved_config.request_timeout,
    }
    
    # Add provider specification
    if provider == "openrouter":
        llm_kwargs["provider"] = "openrouter"
    elif provider == "together":
        llm_kwargs["provider"] = "together_ai"
    
    print(f"\nLLM kwargs that would be used:")
    for k, v in llm_kwargs.items():
        if k == "api_key" and v:
            print(f"  {k}: {v[:8]}...")
        else:
            print(f"  {k}: {v}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
