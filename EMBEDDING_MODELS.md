# Alternative Embedding Models for RAG

If you encounter issues with the default embedding model, you can try these alternatives by modifying the code:

## Recommended Models (in order of preference)

### 1. Current Default (Best Balance)
```python
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```
- **Size**: ~420MB
- **Performance**: Fast and accurate
- **Languages**: 50+ languages including Thai
- **Dimensions**: 384

### 2. Higher Quality Option
```python
embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```
- **Size**: ~1.1GB
- **Performance**: Better accuracy, slower
- **Languages**: 50+ languages including Thai
- **Dimensions**: 768

### 3. Lightweight Option
```python
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"
```
- **Size**: ~90MB
- **Performance**: Fastest but less accurate
- **Languages**: 50+ languages including Thai
- **Dimensions**: 384

### 4. Thai-Specific Option (Experimental)
```python
embedding_model = "sentence-transformers/distiluse-base-multilingual-cased"
```
- **Size**: ~500MB
- **Performance**: Good for Thai
- **Languages**: 15 languages including Thai
- **Dimensions**: 512

## How to Change the Model

1. Open `voice_chatbot.py`
2. Find the `RAGEnabledOpenRouterLLM` class initialization
3. Change the `embedding_model` parameter:

```python
def __init__(self, model_name: str = "tencent/hunyuan-a13b-instruct:free", api_key: str = None, 
             vector_db_path: str = "./vector_db", embedding_model: str = "YOUR_CHOSEN_MODEL"):
```

4. Clear your vector database to rebuild with the new model:
   - Delete the `vector_db` folder, or
   - Use "🗑️ Clear Knowledge Base" in the sidebar

## Testing Different Models

You can test model performance with Thai text by:

1. Loading sample data
2. Asking questions in Thai
3. Checking relevance scores in the retrieved documents

The higher the relevance scores, the better the model understands your queries.

## Troubleshooting

If you get model download errors:
- Check internet connection
- Try a smaller model first
- Clear pip cache: `pip cache purge`
- Update transformers: `pip install --upgrade transformers`

## Performance Comparison

| Model | Size | Speed | Thai Quality | Memory Usage |
|-------|------|-------|--------------|--------------|
| MiniLM-L12-v2 | 420MB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low |
| mpnet-base-v2 | 1.1GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium |
| MiniLM-L6-v2 | 90MB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Very Low |
| distiluse-multilingual | 500MB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low |
