# RAG (Retrieval-Augmented Generation) Setup Guide

## What is RAG?

RAG (Retrieval-Augmented Generation) enhances the chatbot by allowing it to search through a knowledge base of documents before generating responses. This makes the AI more accurate and able to provide specific information from your uploaded documents.

## Features

- **Vector Search**: Uses Qwen3-Embedding-0.6B for multilingual embeddings
- **Document Storage**: ChromaDB for persistent vector database
- **Thai Language Support**: Optimized for Thai language with multilingual embeddings
- **Relevance Scoring**: Filters results by similarity score
- **Document Management**: Upload, search, and manage your knowledge base

## Installation

1. **Install RAG Dependencies**:
   ```bash
   pip install chromadb>=0.4.0 sentence-transformers>=2.2.0
   ```

   Or run the installation script:
   ```bash
   install_rag.bat
   ```

2. **Verify Installation**:
   Start the chatbot and check the sidebar for RAG options.

## How to Use

### 1. Enable RAG
- In the sidebar, check "Enable RAG"
- The system will initialize Qwen3-Embedding-0.6B and ChromaDB
- Wait for "✅ RAG system ready!" message

### 2. Add Documents
- **Load Sample Data**: Click "📝 Load Sample Thai Data" for testing
- **Upload Files**: Use the file uploader to add your .txt or .md files
- **Monitor Status**: Check document count in the sidebar

### 3. Configure Settings
- **Top K results**: Number of relevant documents to retrieve (1-10)
- **Min relevance**: Minimum similarity score (0.0-1.0)
- Higher values = more selective results

### 4. Chat with RAG
- Ask questions normally through voice or text
- The system will search your knowledge base
- See retrieved documents in the expandable section
- Responses will reference your documents

## Tips for Better Results

### Document Preparation
- Use clear, concise text
- Include relevant keywords
- Organize information logically
- Use Thai language for Thai queries

### Query Optimization
- Ask specific questions
- Include relevant keywords
- Use natural language
- Be clear about what you're looking for

### Settings Tuning
- **High relevance needs**: Increase min relevance score (0.6-0.8)
- **More context**: Increase Top K results (5-10)
- **Faster responses**: Decrease Top K results (1-3)

## Technical Details

### Embedding Model
- **Model**: Qwen/Qwen3-Embedding-0.6B
- **Language Support**: Multilingual (Thai, English, etc.)
- **Dimensions**: 768-dimensional embeddings
- **Similarity**: Cosine similarity scoring

### Vector Database
- **Engine**: ChromaDB
- **Storage**: Persistent local storage in `./vector_db`
- **Collection**: "knowledge_base"
- **Metadata**: Source, length, timestamp, custom metadata

### Integration
- **Fallback**: Automatically falls back to standard LLM if RAG fails
- **Optional**: Can be enabled/disabled without affecting basic functionality
- **Context**: Injects retrieved context into LLM prompts

## Troubleshooting

### Common Issues

1. **"RAG dependencies not available"**
   - Install: `pip install chromadb sentence-transformers`
   - Restart the application

2. **"RAG initialization failed"**
   - Check disk space for vector database
   - Ensure internet access for downloading models
   - Try clearing the knowledge base

3. **No relevant documents found**
   - Check document content relevance
   - Lower the min relevance score
   - Add more diverse documents

4. **Slow performance**
   - Reduce Top K results
   - Use fewer documents
   - Consider GPU acceleration for embeddings

### Performance Optimization

- **CPU**: Works on CPU but slower
- **GPU**: CUDA acceleration for embeddings (recommended)
- **Memory**: ~2GB for Qwen3-Embedding model
- **Storage**: Vector database grows with documents

## File Structure

```
MANJU/
├── voice_chatbot.py          # Main application with RAG
├── requirements.txt          # Dependencies including RAG
├── install_rag.bat          # RAG installation script
├── vector_db/               # ChromaDB storage (auto-created)
│   ├── chroma.sqlite3       # Vector database
│   └── ...                  # ChromaDB files
└── ...
```

## Advanced Usage

### Custom Metadata
When uploading documents programmatically:
```python
documents = [{
    'content': 'Your document content',
    'source': 'filename.txt',
    'metadata': {
        'category': 'technical',
        'language': 'thai',
        'priority': 'high'
    }
}]
```

### API Access
The RAG system provides methods for:
- `add_documents()`: Add multiple documents
- `search_knowledge_base()`: Search for relevant content
- `get_knowledge_base_stats()`: Get database statistics
- `clear_knowledge_base()`: Clear all documents

### Model Customization
You can modify the embedding model in the code:
```python
# In RAGEnabledOpenRouterLLM class
embedding_model = "Qwen/Qwen3-Embedding-0.6B"  # Default
# Or try: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## Support

For issues or questions:
1. Check this documentation
2. Review error messages in the sidebar
3. Try the troubleshooting steps
4. Ensure all dependencies are installed correctly
