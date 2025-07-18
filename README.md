# 🎤 Voice Chatbot with ASR & LLM

## Multi-agent AI for Natural Just-in-Time Understanding - Voice Chatbot

A complete voice chatbot solution that combines:
- **ASR (Automatic Speech Recognition)** using Whisper with overlapping chunks
- **LLM (Large Language Model)** using OpenRouter API
- **TTS (Text-to-Speech)** with F5-TTS-THAI and pyttsx3 support
- **Streamlit** web interface

## 🚀 Features

- **Voice Recording**: Record your voice directly in the web interface
- **File Upload**: Upload audio files for transcription
- **Real-time ASR**: Advanced overlapping chunk processing for better accuracy
- **AI Chat**: Conversation with various models via OpenRouter API
- **Advanced TTS**: High-quality Thai text-to-speech with F5-TTS-THAI
- **Multiple TTS Engines**: F5-TTS-THAI for Thai, pyttsx3 for fallback
- **Conversation History**: Track your entire conversation
- **Multiple Input Methods**: Voice, file upload, or text input

## 📋 Prerequisites

1. **Python 3.8+**
2. **OpenRouter API Key** - Get from [https://openrouter.ai](https://openrouter.ai)
3. **Audio drivers** for recording (usually built-in on Windows)
4. **Optional: CUDA** for GPU-accelerated TTS

## 🛠️ Installation

### Option 1: Quick Setup

1. **Install basic dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install F5-TTS-THAI (for better Thai TTS):**
```bash
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
```
*Or run `install_f5_tts.bat` on Windows*

3. **Set up OpenRouter API key:**
```bash
# Option 1: Environment variable (recommended)
set OPENROUTER_API_KEY=your_api_key_here

# Option 2: Enter in the app sidebar when running
```

4. **Run the chatbot:**
```bash
streamlit run voice_chatbot.py
```

### Option 2: Manual Setup

1. **Install Python dependencies:**
```bash
pip install streamlit transformers torch torchaudio openai
pip install sounddevice soundfile pyttsx3 requests pandas numpy
```

2. **Install F5-TTS-THAI:**
```bash
# For CUDA support (recommended)
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install F5-TTS-THAI
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
```

## 🎯 Usage

1. **Start the application:**
   ```bash
   streamlit run voice_chatbot.py
   ```

2. **Configure API Key:**
   - Set `OPENROUTER_API_KEY` environment variable, OR
   - Enter API key in the sidebar when app starts

3. **Choose your input method:**
   - 🎤 **Voice Recording**: Click "Start Recording" and speak
   - 📁 **File Upload**: Upload an audio file (WAV, MP3, MP4, M4A, FLAC)
   - 💬 **Text Input**: Type your message directly

4. **Select TTS Engine:**
   - **F5-TTS-THAI**: High-quality neural TTS for Thai (recommended)
   - **pyttsx3**: Traditional TTS engine (fallback)
   - **Auto**: Automatically chooses the best available engine

## 🔧 Configuration

### Available Models (OpenRouter)
- `tencent/hunyuan-a13b-instruct:free` (Default)
- `moonshotai/kimi-k2:free`
- `meta-llama/llama-3.2-11b-vision-instruct:free`
- `microsoft/phi-3-mini-128k-instruct:free`
- And many more free models!

### ASR Settings
Modify in `voice_chatbot.py`:
```python
audio_config = AudioConfig(
    chunk_length_ms=27000,  # 27 seconds per chunk
    overlap_ms=2000,        # 2 seconds overlap
    sample_rate=16000       # 16kHz sample rate
)
```

### TTS Settings
- **F5-TTS-THAI**: Automatic setup with Thai optimization
- **pyttsx3**: Configurable voice, speed, and volume
- **Engine Selection**: Choose in sidebar

## 📁 Project Structure

```
MANJU/
├── voice_chatbot.py        # Main Streamlit application
├── whisper.py             # ASR pipeline with overlapping chunks
├── requirements.txt       # Python dependencies
├── install_f5_tts.bat    # F5-TTS-THAI installation script
└── README.md             # This file
```

## 🎤 ASR Technology

Advanced **overlapping chunk ASR system**:
- **Chunk Length**: 27 seconds per chunk
- **Overlap**: 2 seconds between chunks
- **Benefits**: Better accuracy for long audio, handles word boundaries
- **Model**: Whisper Thai Large V3 (nectec/Pathumma-whisper-th-large-v3)

## 🧠 LLM Integration

- **OpenRouter API**: Access to multiple state-of-the-art models
- **Free Models**: Many free options available
- **Conversation Memory**: Maintains chat history
- **Flexible**: Easy model switching via dropdown

## 🔊 Text-to-Speech

### F5-TTS-THAI (Recommended)
- **High Quality**: Neural speech synthesis for Thai
- **Natural Voice**: More human-like than traditional TTS
- **GPU Accelerated**: CUDA support for faster generation
- **Thai Optimized**: Specifically trained on Thai language data

### pyttsx3 (Fallback)
- **Cross-platform**: Works on Windows, Mac, Linux
- **Voice Selection**: Automatic female voice preference
- **Configurable**: Speed and volume adjustable
- **Reliable**: Always available fallback option

## 🐛 Troubleshooting

### Common Issues:

1. **"OpenRouter not connected"**
   - Set your API key: `set OPENROUTER_API_KEY=your_api_key_here`
   - Or enter it manually in the sidebar
   - Get free API key from [https://openrouter.ai](https://openrouter.ai)

2. **"F5-TTS-THAI initialization failed"**
   - Install dependencies: `pip install torch torchaudio`
   - Install F5-TTS-THAI: `pip install git+https://github.com/VYNCX/F5-TTS-THAI.git`
   - Use pyttsx3 as fallback if F5-TTS fails

3. **"Recording not available"**
   - Install audio dependencies: `pip install sounddevice soundfile`
   - Check microphone permissions

4. **"TTS not available"**
   - Install TTS engine: `pip install pyttsx3`
   - On Linux: `sudo apt-get install espeak espeak-data libespeak1`
   - For F5-TTS: Install CUDA for GPU acceleration

5. **Slow transcription**
   - GPU acceleration is enabled by default
   - For CPU-only: Set `use_gpu=False` in ProcessingConfig

### Performance Tips:

- **GPU Usage**: Ensure CUDA is available for faster F5-TTS generation
- **Model Loading**: First F5-TTS generation may be slower (model loading)
- **API Limits**: Check OpenRouter rate limits if requests fail
- **TTS Engine**: Use "auto" to automatically select best available engine

### F5-TTS-THAI Specific:

- **CUDA Required**: GPU acceleration highly recommended
- **Model Size**: First run downloads ~2GB model
- **Memory**: Requires 4GB+ GPU memory for optimal performance
- **Fallback**: System automatically uses pyttsx3 if F5-TTS fails

## 🔒 Privacy

- **Local ASR**: Speech recognition happens locally with Whisper
- **API Calls**: Only LLM requests sent to OpenRouter (encrypted)
- **No Audio Upload**: Your voice stays on your machine
- **Temporary Files**: Audio chunks automatically cleaned up

## 📊 System Requirements

### Minimum:
- **RAM**: 4GB (for basic operation)
- **Storage**: 3GB+ for models
- **CPU**: Dual-core
- **Network**: Internet for OpenRouter API

### Recommended:
- **RAM**: 8GB+ (for smooth operation)
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (for F5-TTS)
- **Storage**: 5GB+ for models and cache
- **CPU**: Quad-core or better

## 🤝 Contributing

Feel free to contribute by:
- Adding new TTS voices or engines
- Supporting more OpenRouter models
- Improving the UI/UX
- Adding new audio formats
- Optimizing F5-TTS integration
- Adding multi-language support

## 📝 License

This project is open source. Feel free to use and modify as needed.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify OpenRouter API key is set
4. Check the Streamlit logs for detailed error messages
5. Try using pyttsx3 TTS if F5-TTS fails

## 🔗 Useful Links

- **OpenRouter**: [https://openrouter.ai](https://openrouter.ai) - Get API keys
- **F5-TTS-THAI**: [https://github.com/VYNCX/F5-TTS-THAI](https://github.com/VYNCX/F5-TTS-THAI) - Thai TTS model
- **Whisper Model**: [https://huggingface.co/nectec/Pathumma-whisper-th-large-v3](https://huggingface.co/nectec/Pathumma-whisper-th-large-v3) - Thai ASR model

---

🎉 **Enjoy your enhanced voice chatbot!** Now with OpenRouter LLM and F5-TTS-THAI for premium Thai text-to-speech!

# RAG (Retrieval-Augmented Generation) Setup Guide

## What is RAG?

RAG (Retrieval-Augmented Generation) enhances the chatbot by allowing it to search through a knowledge base of documents before generating responses. This makes the AI more accurate and able to provide specific information from your uploaded documents.

## Features

- **Vector Search**: Uses sentence-transformers multilingual model for embeddings
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
- The system will initialize the multilingual embedding model and ChromaDB
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
- **Model**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Language Support**: Multilingual (Thai, English, etc.)
- **Dimensions**: 384-dimensional embeddings
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
- **Memory**: ~500MB for multilingual embedding model
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
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Default
# Or try: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

## Support

For issues or questions:
1. Check this documentation
2. Review error messages in the sidebar
3. Try the troubleshooting steps
4. Ensure all dependencies are installed correctly

---

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
