"""Basic import test"""
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

# Load .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

print("Testing imports...")
try:
    import crewai
    print("✅ crewai imported")
except Exception as e:
    print(f"❌ crewai import failed: {e}")

try:
    from crewai import LLM
    print("✅ LLM imported")
except Exception as e:
    print(f"❌ LLM import failed: {e}")

try:
    from MultiAgent import MultiAgentConfig
    config = MultiAgentConfig()
    print(f"✅ MultiAgentConfig created: {config}")
except Exception as e:
    print(f"❌ MultiAgentConfig failed: {e}")
    import traceback
    traceback.print_exc()
