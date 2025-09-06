"""Debug script to test LiteLLM with Together AI API"""
import os
import litellm

# Enable debug mode
litellm.set_verbose = True

def test_litellm_together():
    """Test LiteLLM directly with Together AI"""
    
    # Get the API key
    api_key = os.getenv("TOGETHER_API_KEY")
    print(f"API Key found: {bool(api_key)}")
    if api_key:
        print(f"API Key prefix: {api_key[:8]}...")
    
    try:
        # Test direct LiteLLM call
        response = litellm.completion(
            model="together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Hello, who are you?"}],
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            max_tokens=50
        )
        print("✅ LiteLLM direct call successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ LiteLLM direct call failed: {e}")
        return False

if __name__ == "__main__":
    test_litellm_together()
