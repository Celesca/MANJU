# LangGraph Multi-Agent System - Quick Start Guide

Get started with the LangGraph implementation in 5 minutes!

## Prerequisites

- Python 3.9+
- pip installed
- API key for OpenRouter, Together AI, or OpenAI

## Installation

### 1. Install Dependencies

```bash
# Navigate to backend directory
cd backend

# Install LangGraph and dependencies
pip install langgraph langchain langchain-openai langchain-community

# Optional: Install additional features
pip install gspread oauth2client  # Google Sheets
pip install PyPDF2 faiss-cpu sentence-transformers  # RAG
```

### 2. Set Up Environment Variables

Create or update `.env` file:

```bash
# Choose ONE of these:

# Option 1: OpenRouter (Recommended - cheapest)
OPENROUTER_API_KEY=your_openrouter_key_here

# Option 2: Together AI
TOGETHER_API_KEY=your_together_key_here

# Option 3: OpenAI
OPENAI_API_KEY=your_openai_key_here

# Option 4: Local Ollama (Free)
# No API key needed, just run: ollama serve
```

### 3. Test the Installation

```bash
# Run test script
python test_langgraph_migration.py
```

Expected output:
```
✅ ALL TESTS PASSED - Migration Successful! 🎉
```

## Usage

### Basic Usage

```python
from MultiAgent_LangGraph import VoiceCallCenterMultiAgent

# Initialize system
system = VoiceCallCenterMultiAgent()

# Process a query
result = system.process_voice_input("สวัสดีครับ ขอสอบถามแพ็กเกจอินเทอร์เน็ต")

print(result["response"])
# Output: "สวัสดีครับ พบแพ็กเกจอินเทอร์เน็ต Fiber 100/30 ..."
```

### With Conversation History

```python
history = [
    {"role": "user", "content": "สวัสดีครับ"},
    {"role": "assistant", "content": "สวัสดีครับ มีอะไรให้ช่วยไหม"}
]

result = system.process_voice_input(
    "ขอบคุณครับ",
    conversation_history=history
)
```

### Get System Status

```python
status = system.get_system_status()
print(status)
# {
#   "engine": "langgraph",
#   "model": "openrouter/qwen/qwen3-4b:free",
#   "ready": true,
#   ...
# }
```

## Server Integration

### Update Server

Edit `new_server.py`:

```python
# Change this line:
from MultiAgent_New import VoiceCallCenterMultiAgent

# To:
from MultiAgent_LangGraph import VoiceCallCenterMultiAgent
```

The server is already configured to auto-detect and prefer LangGraph!

### Start Server

```bash
python new_server.py
```

### Test Server

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test LLM endpoint
curl -X POST http://localhost:8000/llm \
  -H "Content-Type: application/json" \
  -d '{"text": "สวัสดีครับ"}'
```

## Configuration

### Custom Model

```python
from MultiAgent_LangGraph import VoiceCallCenterMultiAgent, VoiceCallCenterConfig

config = VoiceCallCenterConfig(
    model="gpt-4",  # or any compatible model
    temperature=0.3,
    max_tokens=256,
    speed_mode=False  # Disable for better quality
)

system = VoiceCallCenterMultiAgent(config=config)
```

### Environment Variables

```bash
# Override model
LLM_MODEL=gpt-4

# OpenRouter settings
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Together AI settings
TOGETHER_API_KEY=...
TOGETHER_BASE_URL=https://api.together.xyz/v1

# Speed mode (default: true)
SPEED_MODE=true
```

## Testing

### Run All Tests

```bash
python test_langgraph_migration.py
```

### Test Specific Functionality

```python
from MultiAgent_LangGraph import VoiceCallCenterMultiAgent

system = VoiceCallCenterMultiAgent()

# Test greeting
assert "สวัสดี" in system.process_voice_input("สวัสดี")["response"]

# Test product query
result = system.process_voice_input("มีสินค้ารหัส TEL001 อะไรบ้าง")
assert result["intent"] == "PRODUCT"

# Test knowledge query
result = system.process_voice_input("นโยบายการคืนสินค้า")
assert result["intent"] == "KNOWLEDGE"

print("✅ All tests passed!")
```

## Examples

### Example 1: Product Query

```python
result = system.process_voice_input("มีสินค้ารหัส TEL001 อะไรบ้างครับ")

print(f"Response: {result['response']}")
# "สินค้า สมาร์ทโฟน Galaxy A54 รหัส TEL001 ..."

print(f"Route: {result['route']}")
# "supervisor -> PRODUCT -> product -> response"

print(f"Time: {result['processing_time_seconds']:.2f}s")
# 0.52s
```

### Example 2: Knowledge Query

```python
result = system.process_voice_input("นโยบายการคืนสินค้าเป็นยังไงบ้างครับ")

print(f"Response: {result['response']}")
# "นโยบายการคืนสินค้า: สามารถคืนได้ภายใน 7 วัน..."

print(f"Intent: {result['intent']}")
# "KNOWLEDGE"
```

### Example 3: Fast Path

```python
result = system.process_voice_input("สวัสดีครับ")

print(f"Response: {result['response']}")
# "สวัสดีครับ ยินดีต้อนรับ..."

print(f"Route: {result['route']}")
# "fast_path_greeting"

print(f"Time: {result['processing_time_seconds']:.3f}s")
# 0.031s (super fast!)
```

## Troubleshooting

### Issue: Module not found

```
ModuleNotFoundError: No module named 'langgraph'
```

**Solution:**
```bash
pip install langgraph langchain langchain-openai langchain-community
```

### Issue: API key error

```
RuntimeError: Missing API keys
```

**Solution:**
1. Check `.env` file exists
2. Verify API key is set: `OPENROUTER_API_KEY=sk-...`
3. Restart Python/server

### Issue: Ollama connection

```
Error: Connection refused to localhost:11434
```

**Solution:**
```bash
# Start Ollama server
ollama serve

# Pull model (if needed)
ollama pull qwen2.5:7b
```

### Issue: Slow responses

**Solution:**
```python
# Enable speed mode
config = VoiceCallCenterConfig(
    speed_mode=True,
    max_tokens=64,  # Shorter responses
    request_timeout=15  # Faster timeout
)
```

### Issue: Empty responses

**Solution:**
1. Check LLM model is working:
   ```python
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="...", api_key="...")
   response = llm.invoke([{"role": "user", "content": "test"}])
   print(response)
   ```

2. Check logs for errors:
   ```bash
   # Enable debug logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## Performance Tips

### 1. Use Fast Path
- Handles greetings, thanks, and direct SKU queries instantly
- Enabled by default with `speed_mode=True`

### 2. Enable Caching
- Product queries are cached automatically
- RAG results are cached
- Sheets queries are cached

### 3. Optimize Token Limits
```python
config = VoiceCallCenterConfig(
    max_tokens=64,  # Shorter = faster
    temperature=0.0  # More deterministic
)
```

### 4. Use Faster Models
```python
# OpenRouter free models (fast)
model = "openrouter/qwen/qwen3-4b:free"

# Together AI (fast + good quality)
model = "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo"

# Local Ollama (fastest, no API calls)
model = "ollama/qwen2.5:7b"
```

## Next Steps

1. ✅ Read `MIGRATION_GUIDE.md` for detailed architecture
2. ✅ Read `COMPARISON.md` for CrewAI vs LangGraph comparison
3. ✅ Test in your environment
4. ✅ Integrate with your application
5. ✅ Monitor performance
6. ✅ Customize nodes for your use case

## Resources

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangChain Docs**: https://python.langchain.com/
- **Code**: `MultiAgent_LangGraph.py`
- **Tests**: `test_langgraph_migration.py`
- **Server**: `new_server.py`

## Support

For issues or questions:
1. Check logs: `logger.info()` statements throughout code
2. Run test suite: `python test_langgraph_migration.py`
3. Review documentation: `MIGRATION_GUIDE.md`, `COMPARISON.md`
4. Check examples in this guide

---

**Happy Coding!** 🚀

*Last Updated: November 24, 2025*
