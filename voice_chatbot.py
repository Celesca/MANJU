import streamlit as st
import tempfile
import os
import requests
import json
from pathlib import Path
import time
import base64
from io import BytesIO
import pandas as pd
from openai import OpenAI


# Import your ASR pipeline
from whisper import OverlappingASRPipeline, AudioConfig, ProcessingConfig

# RAG imports
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from typing import List, Dict, Optional
    import re
    import threading
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Text-to-Speech imports
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# F5-TTS-THAI imports
try:
    import torch
    import torchaudio
    import soundfile as sf
    import tempfile
    import subprocess
    import threading
    import time
    
    # Try to import the inference function directly
    F5TTS_INFER = None
    F5TTS_CLASS = None
    F5TTS_LOAD_MODEL = None
    F5TTS_DIT_MODEL = None
    
    try:
        from f5_tts.infer.utils_infer import infer_process, load_model
        F5TTS_INFER = infer_process
        F5TTS_LOAD_MODEL = load_model
    except ImportError:
        pass
    
    try:
        from f5_tts.model import DiT
        F5TTS_DIT_MODEL = DiT
    except ImportError:
        pass
    
    # Try alternative API imports
    try:
        from f5_tts.api import F5TTS
        F5TTS_CLASS = F5TTS
    except ImportError:
        pass
    
    # Check if CLI is available
    F5TTS_CLI_AVAILABLE = False
    try:
        result = subprocess.run(['python', '-c', 'import f5_tts.infer.infer_cli'], 
                              capture_output=True, text=True, timeout=5)
        F5TTS_CLI_AVAILABLE = (result.returncode == 0)
    except:
        pass
    
    # Legacy variable for backward compatibility
    F5TTS = F5TTS_CLASS
    
    F5_TTS_AVAILABLE = (F5TTS_INFER is not None or F5TTS_CLASS is not None or F5TTS_CLI_AVAILABLE or 
                       (F5TTS_LOAD_MODEL is not None and F5TTS_DIT_MODEL is not None))
    
except ImportError:
    F5_TTS_AVAILABLE = False
    F5TTS = None
    F5TTS_INFER = None
    F5TTS_CLASS = None
    F5TTS_LOAD_MODEL = None
    F5TTS_DIT_MODEL = None
    F5TTS_CLI_AVAILABLE = False
    F5TTS_CLASS = None
    F5TTS_CLI_AVAILABLE = False

# Audio recording imports
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    st.warning("Audio recording not available. Install with: pip install sounddevice soundfile")

# Add dotenv import to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    st.warning("python-dotenv not installed. Install with: pip install python-dotenv")


class OpenRouterLLM:
    """Interface for OpenRouter API"""
    
    def __init__(self, model_name: str = "tencent/hunyuan-a13b-instruct:free", api_key: str = None):
        self.model_name = model_name
        
        # Try to get API key from multiple sources
        self.api_key = (
            api_key or 
            os.getenv("OPENROUTER_API_KEY") or
            self._load_api_key_from_env_file()
        )
        
        if self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        else:
            self.client = None
    
    def _load_api_key_from_env_file(self) -> str:
        """Manually load API key from .env file if dotenv is not available"""
        try:
            env_file_path = Path(".env")
            if env_file_path.exists():
                with open(env_file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENROUTER_API_KEY="):
                            return line.split("=", 1)[1].strip()
        except Exception as e:
            st.warning(f"Could not read .env file: {str(e)}")
        return None
        
    def is_available(self) -> bool:
        """Check if OpenRouter is available"""
        if not self.client or not self.api_key:
            return False
        
        try:
            # Test with a simple request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            st.error(f"OpenRouter connection failed: {str(e)}")
            return False
    
    def get_available_models(self) -> list:
        """Get list of commonly available models on OpenRouter"""
        # Common free/popular models on OpenRouter
        return [
            "tencent/hunyuan-a13b-instruct:free",
            "moonshotai/kimi-k2:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "meta-llama/llama-3.2-1b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "microsoft/phi-3-medium-128k-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "huggingfaceh4/zephyr-7b-beta:free"
        ]
    
    def chat(self, message: str, conversation_history: list = None) -> str:
        """Send message to OpenRouter and get response"""
        if not self.client:
            return "Error: OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable."
        
        try:
            # Prepare messages
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": message})
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Get the response content
            content = response.choices[0].message.content
            
            # Clean up the response by removing <answer> and </answer> tags
            content = self._clean_response(content)
            
            return content
                
        except Exception as e:
            return f"Error communicating with OpenRouter: {str(e)}"
    
    def _clean_response(self, text: str) -> str:
        """Remove <answer> and </answer> tags from the response"""
        if not text:
            return text
        
        import re
        
        # Remove <answer> and </answer> tags (case insensitive)
        # This handles various formats:
        # <answer>content</answer>
        # <Answer>content</Answer>
        # <ANSWER>content</ANSWER>
        cleaned_text = re.sub(r'<\s*answer\s*>', '', text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'</\s*answer\s*>', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove any extra whitespace that might be left
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text


class RAGEnabledOpenRouterLLM(OpenRouterLLM):
    """OpenRouter LLM with RAG (Retrieval-Augmented Generation) capabilities using Qwen3-Embedding"""
    
    def __init__(self, model_name: str = "tencent/hunyuan-a13b-instruct:free", api_key: str = None, 
                 vector_db_path: str = "./vector_db", embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        super().__init__(model_name, api_key)
        
        # Initialize vector database and embedding model
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.rag_enabled = False
        
        # Try to initialize RAG components
        if RAG_AVAILABLE:
            try:
                self._initialize_rag()
            except Exception as e:
                st.warning(f"RAG initialization failed: {str(e)}. Running without RAG.")
                self.rag_enabled = False
        else:
            st.info("RAG dependencies not available. Install with: pip install chromadb sentence-transformers")
    
    def _initialize_rag(self):
        """Initialize RAG components (vector DB and Qwen3 embedding model)"""
        try:
            with st.spinner("🔧 Initializing RAG system with multilingual embeddings..."):
                # Initialize multilingual embedding model that supports Thai
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                
                # Initialize ChromaDB with persistent storage
                self.chroma_client = chromadb.PersistentClient(path=self.vector_db_path)
                
                # Get or create collection
                try:
                    self.collection = self.chroma_client.get_collection("knowledge_base")
                    st.success(f"✅ Loaded existing knowledge base with {self.collection.count()} documents")
                except Exception:
                    self.collection = self.chroma_client.create_collection(
                        name="knowledge_base",
                        metadata={"description": "Knowledge base for RAG with multilingual embeddings"}
                    )
                    st.info("📚 Created new knowledge base")
                
                self.rag_enabled = True
                st.success(f"✅ RAG system initialized with {self.embedding_model_name}")
                
        except ImportError as e:
            st.error("Missing RAG dependencies. Install with: pip install chromadb sentence-transformers")
            raise e
        except Exception as e:
            st.error(f"RAG initialization error: {str(e)}")
            raise e
    
    def is_rag_available(self) -> bool:
        """Check if RAG system is available"""
        return self.rag_enabled and self.embedding_model is not None and self.collection is not None
    
    def add_documents(self, documents: List[Dict[str, str]]) -> bool:
        """Add documents to vector database
        
        Args:
            documents: List of dicts with 'content', 'source', and optional 'metadata'
        """
        if not self.is_rag_available():
            return False
        
        try:
            with st.spinner("📝 Adding documents to knowledge base..."):
                # Prepare data for ChromaDB
                contents = []
                metadatas = []
                ids = []
                
                for i, doc in enumerate(documents):
                    content = doc.get('content', '')
                    source = doc.get('source', 'unknown')
                    metadata = doc.get('metadata', {})
                    
                    # Skip empty content
                    if not content.strip():
                        continue
                    
                    contents.append(content)
                    metadatas.append({
                        'source': source,
                        'length': len(content),
                        'added_time': str(time.time()),
                        **metadata
                    })
                    ids.append(f"doc_{int(time.time())}_{i}_{hash(content) % 100000}")
                
                if not contents:
                    st.warning("No valid content to add")
                    return False
                
                # Generate embeddings using multilingual model
                embeddings = self.embedding_model.encode(contents, 
                                                       convert_to_tensor=False,
                                                       normalize_embeddings=True).tolist()
                
                # Add to ChromaDB
                self.collection.add(
                    documents=contents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
                
                st.success(f"✅ Added {len(contents)} documents to knowledge base")
                return True
                
        except Exception as e:
            st.error(f"Failed to add documents: {str(e)}")
            return False
    
    def search_knowledge_base(self, query: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """Search vector database for relevant documents"""
        if not self.is_rag_available():
            return []
        
        try:
            # Generate query embedding using multilingual model
            query_embedding = self.embedding_model.encode([query], 
                                                        convert_to_tensor=False,
                                                        normalize_embeddings=True).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results with relevance filtering
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    # Convert distance to similarity score (cosine similarity)
                    similarity_score = 1 - results['distances'][0][i]
                    
                    # Only include documents above minimum score threshold
                    if similarity_score >= min_score:
                        retrieved_docs.append({
                            'content': doc,
                            'source': results['metadatas'][0][i].get('source', 'unknown'),
                            'score': similarity_score,
                            'metadata': results['metadatas'][0][i]
                        })
            
            # Sort by relevance score (highest first)
            retrieved_docs.sort(key=lambda x: x['score'], reverse=True)
            return retrieved_docs
            
        except Exception as e:
            st.error(f"Knowledge base search failed: {str(e)}")
            return []
    
    def chat_with_rag(self, message: str, conversation_history: list = None, 
                      use_rag: bool = True, top_k: int = 3, min_score: float = 0.4) -> str:
        """Enhanced chat with RAG capabilities using multilingual embeddings"""
        
        # If RAG is disabled or not available, fall back to normal chat
        if not use_rag or not self.is_rag_available():
            return self.chat(message, conversation_history)
        
        try:
            # Search knowledge base for relevant information
            retrieved_docs = self.search_knowledge_base(message, top_k=top_k, min_score=min_score)
            
            # Prepare enhanced prompt with retrieved context
            if retrieved_docs:
                # Format context from retrieved documents
                context_sections = []
                for i, doc in enumerate(retrieved_docs[:top_k], 1):
                    context_sections.append(
                        f"**ข้อมูลที่ {i}** (ความเกี่ยวข้อง: {doc['score']:.2f})\n"
                        f"แหล่งที่มา: {doc['source']}\n"
                        f"เนื้อหา: {doc['content']}\n"
                    )
                
                context_text = "\n".join(context_sections)
                
                # Create enhanced prompt in Thai
                enhanced_message = f"""โปรดตอบคำถามต่อไปนี้โดยใช้ข้อมูลจากฐานความรู้ที่กำหนดให้:

🔍 **ข้อมูลจากฐานความรู้:**
{context_text}

❓ **คำถามจากผู้ใช้:** {message}

📋 **คำแนะนำในการตอบ:**
- ตอบเป็นภาषาไทยที่เป็นธรรมชาติและเข้าใจง่าย
- อ้างอิงข้อมูลจากฐานความรู้ที่ให้มาข้างต้น
- หากข้อมูลไม่เพียงพอหรือไม่เกี่ยวข้อง ให้บอกอย่างชัดเจน
- ระบุแหล่งที่มาของข้อมูลที่ใช้ในการตอบ
- ให้คำตอบที่มีประโยชน์และตรงประเด็น"""

                # Show retrieved context in UI
                with st.expander(f"🔍 ข้อมูลที่ค้นพบ ({len(retrieved_docs)} รายการ)"):
                    for doc in retrieved_docs:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**แหล่งที่มา:** {doc['source']}")
                            st.write(f"**เนื้อหา:** {doc['content'][:200]}...")
                        with col2:
                            st.metric("ความเกี่ยวข้อง", f"{doc['score']:.2f}")
                        st.divider()
            else:
                enhanced_message = f"""คำถาม: {message}

ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้ โปรดตอบจากความรู้ทั่วไปและระบุว่าไม่มีข้อมูลเฉพาะจากฐานความรู้"""
                
                st.info("🔍 ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้ - ใช้ความรู้ทั่วไป")
            
            # Get response from LLM
            response = self.chat(enhanced_message, conversation_history)
            
            return response
            
        except Exception as e:
            st.error(f"RAG chat failed: {str(e)}. Falling back to normal chat.")
            return self.chat(message, conversation_history)
    
    def get_knowledge_base_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        if not self.is_rag_available():
            return {"status": "RAG not available"}
        
        try:
            count = self.collection.count()
            return {
                "status": "Available",
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "database_path": self.vector_db_path,
                "collection_name": "knowledge_base"
            }
        except Exception as e:
            return {"status": f"Error: {str(e)}"}
    
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from knowledge base"""
        if not self.is_rag_available():
            return False
        
        try:
            # Delete and recreate collection
            self.chroma_client.delete_collection("knowledge_base")
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base",
                metadata={"description": "Knowledge base for RAG with multilingual embeddings"}
            )
            st.success("🗑️ Knowledge base cleared successfully")
            return True
        except Exception as e:
            st.error(f"Failed to clear knowledge base: {str(e)}")
            return False
    
    def load_sample_data(self) -> bool:
        """Load sample Thai data for testing"""
        if not self.is_rag_available():
            return False
        
        sample_docs = [
            {
                'content': 'ประเทศไทยมีเมืองหลวงคือกรุงเทพมหานคร มีประชากรประมาณ 70 ล้านคน และมีภาษาราชการคือภาษาไทย',
                'source': 'ข้อมูลทั่วไปเกี่ยวกับประเทศไทย',
                'metadata': {'category': 'geography', 'language': 'thai'}
            },
            {
                'content': 'อาหารไทยที่มีชื่อเสียงระดับโลก ได้แก่ ต้มยำกุ้ง ผัดไทย ส้มตำ และมะม่วงข้าวเหนียว',
                'source': 'วัฒนธรรมอาหารไทย',
                'metadata': {'category': 'culture', 'language': 'thai'}
            },
            {
                'content': 'ประวัติศาสตร์ไทยมีความยาวนานกว่า 700 ปี เริ่มจากสมัยสุโขทัย อยุธยา ธนบุรี และปัจจุบันคือสมัยรัตนโกสินทร์',
                'source': 'ประวัติศาสตร์ไทย',
                'metadata': {'category': 'history', 'language': 'thai'}
            }
        ]
        
        return self.add_documents(sample_docs)


class F5TTSThai:
    """F5-TTS-THAI handler for Thai text-to-speech using VIZINTZOR pre-trained model"""
    
    def __init__(self):
        # Always initialize all attributes first to prevent AttributeError
        self.model = None
        self.infer_function = None
        self.available = False
        self.thai_ref_audio = None
        self.thai_ref_text = "สวัสดีครับ ผมเป็นผู้ช่วยเสียงภาษาไทย ยินดีให้บริการครับ"
        self.model_path = None
        self.vocab_path = None
        
        # Only try initialization if F5-TTS is available
        if F5_TTS_AVAILABLE:
            try:
                if self._check_thai_model_files():
                    self._initialize_model()
                    self._prepare_thai_reference()
                else:
                    self.available = False
            except Exception as e:
                # Ensure we remain in a safe state even if initialization fails
                self.available = False
                st.warning(f"F5-TTS-THAI initialization failed: {str(e)}")
        else:
            # F5-TTS not available - leave attributes as None but initialized
            self.available = False
    
    def _check_thai_model_files(self):
        """Check if Thai model files are available"""
        possible_paths = [
            # Check in current directory
            "model_1000000.pt",
            "VIZINTZOR_model_1000000.pt",
            # Check in models directory
            "models/model_1000000.pt",
            "models/VIZINTZOR_model_1000000.pt",
            # Check in F5-TTS-THAI directory
            "F5-TTS-THAI/model_1000000.pt",
            # Check in ckpts directory (common F5-TTS convention)
            "ckpts/model_1000000.pt",
            "ckpts/VIZINTZOR/model_1000000.pt",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.model_path = path
                st.success(f"✅ Found Thai model: {path}")
                break
        
        # Check for vocab file
        vocab_paths = [
            "vocab.txt",
            "models/vocab.txt",
            "F5-TTS-THAI/vocab.txt",
            "ckpts/vocab.txt"
        ]
        
        for path in vocab_paths:
            if os.path.exists(path):
                self.vocab_path = path
                break
        
        if not self.model_path:
            st.error("❌ Thai model checkpoint not found! Please download model_1000000.pt from https://huggingface.co/VIZINTZOR/F5-TTS-THAI")
            return False
        
        return True
    
    def _prepare_thai_reference(self):
        """Create proper Thai reference audio for authentic Thai voice"""
        try:
            import tempfile
            import os
            
            # Create temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            
            ref_audio_path = os.path.join("temp", "thai_reference.wav")
            
            # Only create if it doesn't exist
            if not os.path.exists(ref_audio_path):
                try:
                    # Create Thai reference audio using gTTS
                    from gtts import gTTS
                    
                    # Use a longer Thai reference text for better voice quality
                    tts = gTTS(text=self.thai_ref_text, lang='th', slow=False)
                    temp_mp3 = os.path.join("temp", "thai_ref_temp.mp3")
                    tts.save(temp_mp3)
                    
                    # Convert MP3 to WAV if soundfile is available
                    try:
                        import soundfile as sf
                        data, samplerate = sf.read(temp_mp3)
                        sf.write(ref_audio_path, data, samplerate)
                        os.remove(temp_mp3)  # Clean up temp MP3
                        self.thai_ref_audio = ref_audio_path
                        print(f"✅ Thai reference audio created: {ref_audio_path}")
                    except ImportError:
                        # Fallback: use MP3 file directly (rename it)
                        mp3_path = ref_audio_path.replace('.wav', '.mp3')
                        os.rename(temp_mp3, mp3_path)
                        self.thai_ref_audio = mp3_path
                        print(f"✅ Thai reference audio created: {mp3_path}")
                        
                except Exception as e:
                    print(f"Warning: Could not create Thai reference audio: {e}")
                    self.thai_ref_audio = None
            else:
                self.thai_ref_audio = ref_audio_path
                print(f"✅ Using existing Thai reference audio: {ref_audio_path}")
                
        except Exception as e:
            print(f"Warning: Thai reference preparation failed: {e}")
            self.thai_ref_audio = None
    
    def _initialize_model(self):
        """Initialize the F5-TTS model with Thai checkpoint using proper loading approach"""
        try:
            # Method 1: Try loading Thai model with proper F5-TTS configuration (following VIZINTZOR approach)
            if self.model_path and os.path.exists(self.model_path) and F5TTS_LOAD_MODEL is not None and F5TTS_DIT_MODEL is not None:
                try:
                    st.info(f"🔄 Loading Thai model from: {self.model_path}")
                    
                    # Import necessary F5-TTS components locally
                    import torch
                    from huggingface_hub import cached_path
                    
                    # F5-TTS model configuration for Thai (following VIZINTZOR config)
                    F5TTS_model_cfg = dict(
                        dim=1024, 
                        depth=22, 
                        heads=16, 
                        ff_mult=2, 
                        text_dim=512, 
                        conv_layers=4
                    )
                    
                    # Determine vocab file path
                    vocab_file = self.vocab_path
                    if not vocab_file or not os.path.exists(vocab_file):
                        # Try to download from HuggingFace
                        try:
                            vocab_file = str(cached_path("hf://VIZINTZOR/F5-TTS-THAI/vocab.txt"))
                            st.info(f"📁 Downloaded vocab from HuggingFace: {vocab_file}")
                        except Exception as e:
                            st.warning(f"Could not download vocab file: {e}")
                            vocab_file = None
                    
                    # Load the Thai model using proper F5-TTS approach
                    self.model = F5TTS_LOAD_MODEL(
                        F5TTS_DIT_MODEL, 
                        F5TTS_model_cfg, 
                        self.model_path, 
                        vocab_file=vocab_file,
                        use_ema=True
                    )
                    
                    self.available = True
                    st.success("✅ F5-TTS-THAI (VIZINTZOR) loaded with Thai checkpoint!")
                    print(f"Loaded Thai model from {self.model_path}")
                    return
                    
                except ImportError as e:
                    st.warning(f"⚠️ F5-TTS imports failed: {str(e)}. Trying alternative approach...")
                except Exception as e:
                    st.warning(f"⚠️ Failed to load Thai checkpoint with proper method: {str(e)}")
            
            # Method 2: Fallback to original F5TTS_CLASS approach if available
            if F5TTS_CLASS is not None:
                try:
                    self.model = F5TTS_CLASS()
                    self.available = True
                    st.warning("⚠️ Using default F5-TTS model (not Thai-optimized)")
                    return
                except Exception as e:
                    st.warning(f"⚠️ F5-TTS API initialization failed: {str(e)}")
            
            # Method 2: Try function-based approach with CLI pointing to Thai model
            if F5TTS_INFER is not None:
                try:
                    self.infer_function = F5TTS_INFER
                    self.available = True
                    st.success("✅ F5-TTS-THAI initialized via inference function")
                    return
                except Exception as e:
                    st.warning(f"⚠️ F5-TTS inference function failed: {str(e)}")
                
            # Method 3: Check if command line tool is available with Thai model path
            if F5TTS_CLI_AVAILABLE:
                try:
                    self.available = True
                    st.info("✅ F5-TTS-THAI available via command line")
                    return
                except Exception as e:
                    st.warning(f"⚠️ F5-TTS CLI check failed: {str(e)}")
            
            # If we get here, F5-TTS is not available
            self.available = False
                
        except Exception as e:
            # Don't show error in UI, just mark as unavailable
            self.available = False
            st.warning(f"F5-TTS-THAI initialization failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if F5-TTS-THAI is available"""
        return self.available
    
    def speak(self, text: str, ref_audio: str = None, ref_text: str = None, save_file: str = None) -> bool:
        """Convert text to speech using F5-TTS-THAI with VIZINTZOR checkpoint"""
        if not self.is_available():
            return False
        
        # Ensure we have necessary imports
        try:
            import tempfile
            import os
            import time
            import threading
            import soundfile as sf
            import torch
            import torchaudio
        except ImportError as e:
            st.error(f"Missing required dependencies for F5-TTS: {str(e)}")
            return False
        
        try:
            st.info("🔊 F5-TTS-THAI (VIZINTZOR): Generating authentic Thai speech...")
            
            # Use Thai reference audio and text for authentic Thai voice
            use_ref_audio = ref_audio or self.thai_ref_audio
            use_ref_text = ref_text or self.thai_ref_text
            
            # If we still don't have reference audio, create it on-the-fly
            if use_ref_audio is None and GTTS_AVAILABLE:
                try:
                    temp_ref_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    temp_ref_file = temp_ref_obj.name
                    temp_ref_obj.close()
                    
                    from gtts import gTTS
                    ref_tts = gTTS(text=use_ref_text, lang="th")
                    ref_tts.save(temp_ref_file)
                    use_ref_audio = temp_ref_file
                    
                    st.info("📁 Created temporary Thai reference audio")
                except Exception as e:
                    st.warning(f"Could not create temporary Thai reference: {e}")
            
            # Determine output file
            if save_file is None:
                # Create a unique temporary file
                temp_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                output_file = temp_file_obj.name
                temp_file_obj.close()  # Close the file handle
            else:
                output_file = save_file
            
            # Try different F5-TTS approaches with Thai model
            success = False
            
            # Method 1: Try API-based approach with Thai model
            if self.model is not None:
                try:
                    st.info(f"🎤 Using Thai model with reference: {use_ref_text[:50]}...")
                    
                    # Use Thai-optimized parameters
                    audio_data = self.model.infer(
                        gen_text=text,  # Text to synthesize
                        ref_text=use_ref_text,  # Thai reference text
                        ref_file=use_ref_audio,  # Thai reference audio
                        remove_silence=True,
                        speed=0.9,  # Slightly slower for clearer Thai
                        nfe_step=32,  # Good balance of quality and speed
                        cfg_strength=2.0,  # Classifier-free guidance
                        sway_sampling_coef=-1.0  # Sway sampling for better quality
                    )
                    
                    # Process and save audio
                    if audio_data is not None:
                        # Convert to numpy array if needed
                        if hasattr(audio_data, 'cpu'):
                            audio_data = audio_data.cpu().numpy()
                        elif hasattr(audio_data, 'numpy'):
                            audio_data = audio_data.numpy()
                        
                        # Handle tuple/list returns (audio, sample_rate)
                        if isinstance(audio_data, (list, tuple)):
                            if len(audio_data) >= 2:
                                audio_array, sample_rate = audio_data[0], audio_data[1]
                            else:
                                audio_array = audio_data[0]
                                sample_rate = 24000  # Default F5-TTS sample rate
                        else:
                            audio_array = audio_data
                            sample_rate = 24000  # Default F5-TTS sample rate
                        
                        # Ensure proper format
                        if hasattr(audio_array, 'cpu'):
                            audio_array = audio_array.cpu().numpy()
                        elif hasattr(audio_array, 'numpy'):
                            audio_array = audio_array.numpy()
                        
                        audio_array = np.array(audio_array).flatten().astype(np.float32)
                        audio_array = np.clip(audio_array, -1.0, 1.0)
                        
                        # Save with detected or default sample rate
                        sf.write(output_file, audio_array, int(sample_rate))
                        success = True
                        st.success("✅ F5-TTS-THAI (VIZINTZOR): Audio generated with Thai model!")
                    else:
                        st.warning("API returned no audio data")
                    
                except Exception as e:
                    st.warning(f"⚠️ Thai model generation failed: {str(e)}")
            
            # Method 2: Try CLI approach with Thai model path
            if not success and F5TTS_CLI_AVAILABLE and self.model_path:
                try:
                    st.info("🔄 Using F5-TTS CLI with Thai model...")
                    
                    # Create temporary text file
                    temp_text_obj = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
                    temp_text_obj.write(text)
                    text_file = temp_text_obj.name
                    temp_text_obj.close()
                    
                    # Prepare CLI command with Thai model
                    cmd = [
                        'python', '-m', 'f5_tts.infer.infer_cli',
                        '--gen_text', text,
                        '--output_file', output_file,
                        '--model', 'F5-TTS',
                        '--ckpt_file', self.model_path  # Use Thai checkpoint
                    ]
                    
                    if use_ref_text:
                        cmd.extend(['--ref_text', use_ref_text])
                    if use_ref_audio:
                        cmd.extend(['--ref_audio', use_ref_audio])
                    if self.vocab_path:
                        cmd.extend(['--vocab_file', self.vocab_path])
                    
                    # Add Thai-optimized parameters
                    cmd.extend([
                        '--remove_silence',
                        '--nfe_step', '32',
                        '--cfg_strength', '2.0',
                        '--speed', '0.9'
                    ])
                    
                    # Run command
                    env = os.environ.copy()
                    env['PYTHONHASHSEED'] = '0'
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
                    
                    # Clean up text file
                    try:
                        if os.path.exists(text_file):
                            os.unlink(text_file)
                    except:
                        pass
                    
                    if result.returncode == 0 and os.path.exists(output_file):
                        if os.path.getsize(output_file) > 0:
                            success = True
                            st.success("✅ F5-TTS-THAI (VIZINTZOR): Audio generated via CLI with Thai model!")
                        else:
                            st.warning("CLI generated empty file")
                    else:
                        error_msg = result.stderr.strip() if result.stderr else "Unknown CLI error"
                        st.warning(f"⚠️ CLI with Thai model failed: {error_msg}")
                        
                except Exception as e:
                    st.warning(f"⚠️ CLI method with Thai model failed: {str(e)}")
            
            # Method 3: Fallback to standard CLI without model path
            if not success and F5TTS_CLI_AVAILABLE:
                try:
                    st.info("🔄 Fallback: Using standard F5-TTS CLI...")
                    # [Include standard CLI fallback similar to original code but simplified]
                    # This is a fallback if the Thai model path doesn't work
                    
                except Exception as e:
                    st.warning(f"⚠️ Standard CLI fallback failed: {str(e)}")
            
            # Clean up temporary reference file
            if 'temp_ref_file' in locals() and temp_ref_file and os.path.exists(temp_ref_file):
                try:
                    os.unlink(temp_ref_file)
                except:
                    pass
            
            # If successful, play the audio
            if success and os.path.exists(output_file):
                if save_file is None:
                    # Play using streamlit audio player
                    with open(output_file, 'rb') as audio_file:
                        st.audio(audio_file.read(), format='audio/wav')
                    
                    # Schedule cleanup
                    def cleanup_file():
                        time.sleep(10)  # Wait longer for F5-TTS audio
                        try:
                            if os.path.exists(output_file):
                                os.unlink(output_file)
                        except:
                            pass
                    
                    cleanup_thread = threading.Thread(target=cleanup_file, daemon=True)
                    cleanup_thread.start()
                
                return True
            else:
                st.warning("🔄 F5-TTS-THAI: All methods failed, falling back to gTTS for Thai TTS")
                return False
            
        except Exception as e:
            st.error(f"F5-TTS-THAI generation failed: {str(e)}")
            return False


class GTTSThai:
    """Google Text-to-Speech handler for Thai language"""
    
    def __init__(self):
        self.available = GTTS_AVAILABLE
        # No need to initialize pygame since we're using streamlit audio
    
    def is_available(self) -> bool:
        """Check if gTTS is available"""
        return self.available
    
    def speak(self, text: str, language: str = "th", save_file: str = None) -> bool:
        """Convert text to speech using Google TTS"""
        if not self.available:
            return False
        
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            if save_file:
                # Save to specified file
                tts.save(save_file)
                return True
            else:
                # Create temporary file and play
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.close()  # Close the file handle so gTTS can write to it
                
                # Save audio to temp file
                tts.save(temp_file.name)
                
                # Check if file was created successfully
                if not os.path.exists(temp_file.name):
                    st.error("Failed to create audio file")
                    return False
                
                # Play using streamlit audio player instead of pygame
                with open(temp_file.name, 'rb') as audio_file:
                    st.audio(audio_file.read(), format='audio/mp3')
                
                # Schedule cleanup after a delay
                import threading
                def cleanup_file():
                    time.sleep(5)  # Wait 5 seconds for streamlit to load the audio
                    try:
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
                    except:
                        pass
                
                cleanup_thread = threading.Thread(target=cleanup_file, daemon=True)
                cleanup_thread.start()
            
            return True
            
        except Exception as e:
            st.error(f"gTTS speech generation failed: {str(e)}")
            return False


class PyttsxTTS:
    """pyttsx3 TTS handler (fallback for non-Thai)"""
    
    def __init__(self):
        self.engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self._setup_voice()
            except:
                self.engine = None
    
    def _setup_voice(self):
        """Setup pyttsx3 voice properties"""
        if self.engine:
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice or use first available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    self.engine.setProperty('voice', voices[0].id)
            
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.8)  # Volume
    
    def is_available(self) -> bool:
        """Check if pyttsx3 is available"""
        return self.engine is not None
    
    def speak(self, text: str, save_file: str = None) -> bool:
        """Convert text to speech using pyttsx3"""
        if not self.engine:
            return False
        
        try:
            if save_file:
                self.engine.save_to_file(text, save_file)
                self.engine.runAndWait()
            else:
                self.engine.say(text)
                self.engine.runAndWait()
            return True
        except:
            return False


class TextToSpeech:
    """Text-to-Speech handler with multiple engine support"""
    
    def __init__(self, prefer_f5_tts: bool = True):
        self.prefer_f5_tts = prefer_f5_tts
        self.gtts_engine = None
        self.pyttsx3_engine = None
        self.f5_tts_engine = None
        
        # Initialize F5-TTS-THAI if preferred and available
        if prefer_f5_tts and F5_TTS_AVAILABLE:
            self.f5_tts_engine = F5TTSThai()
        
        # Initialize gTTS for Thai (primary choice)
        if GTTS_AVAILABLE:
            self.gtts_engine = GTTSThai()
        
        # Initialize pyttsx3 as last fallback
        if PYTTSX3_AVAILABLE:
            self.pyttsx3_engine = PyttsxTTS()
    
    def get_available_engines(self) -> list:
        """Get list of available TTS engines"""
        engines = []
        if self.f5_tts_engine and self.f5_tts_engine.is_available():
            engines.append("F5-TTS-THAI")
        if self.gtts_engine and self.gtts_engine.is_available():
            engines.append("gTTS-Thai")
        if self.pyttsx3_engine and self.pyttsx3_engine.is_available():
            engines.append("pyttsx3")
        return engines
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is Thai or English"""
        # Simple Thai detection - check for Thai Unicode characters
        thai_chars = sum(1 for char in text if '\u0e00' <= char <= '\u0e7f')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return "th"  # Default to Thai
        
        thai_ratio = thai_chars / total_chars
        return "th" if thai_ratio > 0.3 else "en"
    
    def speak(self, text: str, engine: str = "auto", save_file: str = None) -> bool:
        """Convert text to speech using specified engine"""
        # Determine which engine to use
        if engine == "auto":
            # Auto-select based on preference and availability
            if self.prefer_f5_tts and self.f5_tts_engine and self.f5_tts_engine.is_available():
                engine = "F5-TTS-THAI"
            elif self.gtts_engine and self.gtts_engine.is_available():
                engine = "gTTS-Thai"
            elif self.pyttsx3_engine and self.pyttsx3_engine.is_available():
                engine = "pyttsx3"
            else:
                return False
        
        # Use F5-TTS-THAI (best for Thai)
        if engine == "F5-TTS-THAI" and self.f5_tts_engine and self.f5_tts_engine.is_available():
            success = self.f5_tts_engine.speak(text, save_file=save_file)
            if success:
                return True
            # Fall back to gTTS if F5-TTS fails
            engine = "gTTS-Thai"
        
        # Use gTTS (good for Thai and other languages)
        if engine == "gTTS-Thai" and self.gtts_engine and self.gtts_engine.is_available():
            language = self._detect_language(text)
            return self.gtts_engine.speak(text, language=language, save_file=save_file)
        
        # Use pyttsx3 as last fallback (not ideal for Thai)
        elif engine == "pyttsx3" and self.pyttsx3_engine and self.pyttsx3_engine.is_available():
            return self.pyttsx3_engine.speak(text, save_file=save_file)
        
        return False
    
    def is_available(self) -> bool:
        """Check if any TTS engine is available"""
        f5_available = self.f5_tts_engine and self.f5_tts_engine.is_available()
        gtts_available = self.gtts_engine and self.gtts_engine.is_available()
        pyttsx3_available = self.pyttsx3_engine and self.pyttsx3_engine.is_available()
        return f5_available or gtts_available or pyttsx3_available


class AudioRecorder:
    """Audio recording handler"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_data = None
    
    def record_audio(self, duration: int = 5) -> str:
        """Record audio and save to temporary file"""
        if not RECORDING_AVAILABLE:
            return None
        
        try:
            st.info(f"Recording for {duration} seconds... Speak now!")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait until recording is finished
            
            # Save to temporary file as WAV (no ffmpeg needed)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio_data, self.sample_rate)
            
            st.success("Recording completed!")
            return temp_file.name
            
        except Exception as e:
            st.error(f"Recording failed: {str(e)}")
            return None


def check_ffmpeg_available():
    """Check if FFmpeg is available"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'asr_pipeline' not in st.session_state:
        # Initialize ASR pipeline
        audio_config = AudioConfig(
            chunk_length_ms=27000,
            overlap_ms=2000,
            sample_rate=16000
        )
        processing_config = ProcessingConfig(
            model_name="nectec/Pathumma-whisper-th-large-v3",
            batch_size=2,
            use_gpu=True
        )
        st.session_state.asr_pipeline = OverlappingASRPipeline(
            input_path="",  # Will be set dynamically
            audio_config=audio_config,
            processing_config=processing_config
        )
    
    # Initialize standard LLM
    if 'llm' not in st.session_state:
        st.session_state.llm = OpenRouterLLM(model_name="moonshotai/kimi-k2:free")
    
    # Initialize RAG-enabled LLM
    if 'rag_llm' not in st.session_state:
        st.session_state.rag_llm = None
    
    if 'use_rag' not in st.session_state:
        st.session_state.use_rag = False

    if 'tts' not in st.session_state:
        # Initialize TTS with F5-TTS preference for Thai
        st.session_state.tts = TextToSpeech(prefer_f5_tts=True)
    
    if 'tts_engine' not in st.session_state:
        st.session_state.tts_engine = "auto"
    
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()


def display_conversation():
    """Display conversation history"""
    st.subheader("💬 Conversation History")
    
    if not st.session_state.conversation_history:
        st.info("No conversation yet. Start by recording your voice or typing a message!")
        return
    
    for i, msg in enumerate(st.session_state.conversation_history):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if role == 'user':
            with st.chat_message("user"):
                st.write(f"🎤 **You:** {content}")
        elif role == 'assistant':
            with st.chat_message("assistant"):
                st.write(f"🤖 **Assistant:** {content}")
                
                # Add TTS button for assistant responses
                if st.session_state.tts.is_available():
                    if st.button(f"🔊 Play Response {i}", key=f"tts_{i}"):
                        st.session_state.tts.speak(content, engine=st.session_state.tts_engine)

def main():
    st.set_page_config(
        page_title="Voice Chatbot",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 MANJU Voice Chatbot: Multi-agent AI for Natural Just-in-Time Understanding")
    st.markdown("*Speak, transcribe, chat, and listen to responses!*")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check system status
        st.subheader("System Status")
        
        # FFmpeg Status
        ffmpeg_available = check_ffmpeg_available()
        if ffmpeg_available:
            st.write("🎵 **FFmpeg:** ✅ Available")
        else:
            st.write("🎵 **FFmpeg:** ❌ Missing")
            st.error("FFmpeg required for audio processing")
            if st.button("📥 Install FFmpeg Guide"):
                st.markdown("""
                **Install FFmpeg:**
                1. Run: `install_ffmpeg.bat`
                2. Or download: [FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases)
                3. Add to PATH or restart
                """)
        
        # ASR Status
        if ffmpeg_available:
            st.write("🎯 **ASR (Whisper):** ✅ Ready")
        else:
            st.write("🎯 **ASR (Whisper):** ❌ Needs FFmpeg")
        
        # OpenRouter Status
        if st.session_state.llm.is_available():
            available_models = st.session_state.llm.get_available_models()
            st.write("🧠 **OpenRouter:** ✅ Connected")
            
            # Model selection
            if available_models:
                current_model = st.session_state.llm.model_name
                selected_model = st.selectbox(
                    "Select Model:",
                    available_models,
                    index=available_models.index(current_model) if current_model in available_models else 0
                )
                st.session_state.llm.model_name = selected_model
            else:
                st.write("No models available")
        else:
            st.write("🧠 **OpenRouter:** ❌ Not connected")
            st.error("Please set OPENROUTER_API_KEY environment variable")
            
            # Add API key input option
            api_key_input = st.text_input("Or enter API key here:", type="password")
            if api_key_input and st.button("Connect with API Key"):
                st.session_state.llm = OpenRouterLLM(api_key=api_key_input)
                st.rerun()
        
        # RAG (Vector Search) Settings
        st.divider()
        st.subheader("🔍 RAG (Vector Search)")
        
        # RAG toggle
        use_rag = st.checkbox("Enable RAG", value=st.session_state.use_rag, 
                             help="Use vector search for enhanced knowledge retrieval")
        st.session_state.use_rag = use_rag
        
        if use_rag:
            # Initialize RAG LLM if needed
            if st.session_state.rag_llm is None:
                with st.spinner("🔧 Initializing RAG system..."):
                    try:
                        st.session_state.rag_llm = RAGEnabledOpenRouterLLM(
                            model_name=st.session_state.llm.model_name,
                            api_key=st.session_state.llm.api_key
                        )
                        st.success("✅ RAG system ready!")
                    except Exception as e:
                        st.error(f"❌ RAG initialization failed: {str(e)}")
                        st.session_state.use_rag = False
                        st.session_state.rag_llm = None
            
            # RAG status and controls
            if st.session_state.rag_llm and st.session_state.rag_llm.is_rag_available():
                stats = st.session_state.rag_llm.get_knowledge_base_stats()
                st.write(f"📊 **Status:** {stats['status']}")
                st.write(f"📄 **Documents:** {stats.get('document_count', 0)}")
                st.write(f"🤖 **Model:** paraphrase-multilingual-MiniLM-L12-v2")
                
                # RAG settings
                col1, col2 = st.columns(2)
                with col1:
                    top_k = st.number_input("Top K results", min_value=1, max_value=10, value=3)
                with col2:
                    min_score = st.slider("Min relevance", 0.0, 1.0, 0.4, 0.1)
                
                # Document management
                with st.expander("📚 Manage Knowledge Base"):
                    # Add sample data
                    if st.button("📝 Load Sample Thai Data"):
                        if st.session_state.rag_llm.load_sample_data():
                            st.success("✅ Sample data loaded!")
                            st.rerun()
                    
                    # Upload documents
                    uploaded_files = st.file_uploader(
                        "Upload text files (.txt, .md):",
                        type=['txt', 'md'],
                        accept_multiple_files=True
                    )
                    
                    if uploaded_files and st.button("📤 Upload Documents"):
                        documents = []
                        for file in uploaded_files:
                            content = file.read().decode('utf-8')
                            documents.append({
                                'content': content,
                                'source': file.name,
                                'metadata': {'file_type': file.type}
                            })
                        
                        if st.session_state.rag_llm.add_documents(documents):
                            st.success(f"✅ Uploaded {len(documents)} documents!")
                            st.rerun()
                    
                    # Clear knowledge base
                    if st.button("🗑️ Clear Knowledge Base", type="secondary"):
                        if st.session_state.rag_llm.clear_knowledge_base():
                            st.rerun()
            else:
                st.error("❌ RAG system not available")
                if RAG_AVAILABLE:
                    st.info("💡 Try reinstalling: pip install chromadb sentence-transformers")
                else:
                    st.info("💡 Install RAG: pip install chromadb sentence-transformers")
        
        else:
            # RAG disabled
            if st.session_state.rag_llm:
                st.info("🔍 RAG disabled - using standard LLM")
        
        # TTS Status
        if st.session_state.tts.is_available():
            available_engines = st.session_state.tts.get_available_engines()
            st.write("🔊 **TTS:** ✅ Available")
            
            # Show available engines
            if available_engines:
                st.write(f"**Engines:** {', '.join(available_engines)}")
                
                # Engine selection
                engine_options = ["auto"] + available_engines
                selected_engine = st.selectbox(
                    "TTS Engine:",
                    engine_options,
                    index=engine_options.index(st.session_state.tts_engine) if st.session_state.tts_engine in engine_options else 0
                )
                st.session_state.tts_engine = selected_engine
                
                # F5-TTS Debug Info
                if st.session_state.tts.f5_tts_engine:
                    with st.expander("🔧 F5-TTS-THAI (VIZINTZOR) Debug Info"):
                        st.write(f"**Available:** {'✅' if st.session_state.tts.f5_tts_engine.is_available() else '❌'}")
                        st.write(f"**F5_TTS_AVAILABLE:** {'✅' if F5_TTS_AVAILABLE else '❌'}")
                        st.write(f"**Model initialized:** {'✅' if st.session_state.tts.f5_tts_engine.model else '❌'}")
                        
                        # Safe attribute access with getattr
                        model_path = getattr(st.session_state.tts.f5_tts_engine, 'model_path', None)
                        vocab_path = getattr(st.session_state.tts.f5_tts_engine, 'vocab_path', None)
                        infer_function = getattr(st.session_state.tts.f5_tts_engine, 'infer_function', None)
                        
                        st.write(f"**Thai model path:** {model_path or '❌ Not found'}")
                        st.write(f"**Vocab path:** {vocab_path or '❌ Not found'}")
                        st.write(f"**Infer function:** {'✅' if infer_function else '❌'}")
                        
                        if st.button("🧪 Test F5-TTS-THAI (VIZINTZOR)"):
                            test_text = "สวัสดีครับ นี่คือการทดสอบเสียงไทยจากโมเดล VIZINTZOR"
                            success = st.session_state.tts.f5_tts_engine.speak(test_text)
                            if success:
                                st.success("✅ F5-TTS-THAI (VIZINTZOR) test successful!")
                            else:
                                st.error("❌ F5-TTS-THAI (VIZINTZOR) test failed")
                
                # Show F5-TTS installation guide if not available
                if "F5-TTS-THAI" not in available_engines:
                    with st.expander("📥 Install F5-TTS-THAI (VIZINTZOR) for authentic Thai TTS"):
                        st.markdown("""
                        **Step 1: Install F5-TTS-THAI:**
                        ```bash
                        pip install torch torchaudio soundfile
                        pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
                        ```
                        
                        **Step 2: Download Thai Model:**
                        Run the setup script to download the VIZINTZOR Thai model (1.35GB):
                        ```bash
                        python setup_thai_model.py
                        ```
                        
                        **Or download manually:**
                        1. Go to: [VIZINTZOR/F5-TTS-THAI](https://huggingface.co/VIZINTZOR/F5-TTS-THAI)
                        2. Download `model_1000000.pt` to `models/` folder
                        3. Optionally download `vocab.txt` and `config.json`
                        
                        **Note:** Requires CUDA GPU for best performance
                        """)
                        
                        if st.button("🚀 Run Setup Script"):
                            st.code("python setup_thai_model.py", language="bash")
                            st.info("Run this command in your terminal to download the Thai model")
                
                # Show gTTS info if not available
                if "gTTS-Thai" not in available_engines:
                    with st.expander("📥 Install gTTS for Thai TTS"):
                        st.markdown("""
                        **Install gTTS (Google Text-to-Speech):**
                        ```bash
                        pip install gtts
                        ```
                        **Features:** Excellent Thai support, natural voice, requires internet
                        """)
        else:
            st.write("🔊 **TTS:** ❌ Not available")
            st.error("Install TTS engines for Thai support:")
            
            # Installation guides
            with st.expander("📥 Recommended: Install gTTS (Best for Thai)"):
                st.markdown("""
                **Install gTTS:**
                ```bash
                pip install gtts
                ```
                **Why gTTS?**
                - ✅ Excellent Thai language support
                - ✅ Natural Google voices
                - ✅ Easy to install
                - ✅ Plays through Streamlit interface
                - ❌ Requires internet connection
                """)
            
            with st.expander("📥 Premium: Install F5-TTS-THAI"):
                st.markdown("""
                **Install F5-TTS-THAI:**
                ```bash
                pip install torch torchaudio
                pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
                ```
                **Why F5-TTS?**
                - ✅ Highest quality Thai TTS
                - ✅ Works offline
                - ✅ Customizable voice
                - ❌ Requires GPU for best performance
                """)
            
            with st.expander("📥 Fallback: Install pyttsx3"):
                st.markdown("""
                **Install pyttsx3:**
                ```bash
                pip install pyttsx3
                ```
                **Note:** Limited Thai support, better for English
                """)
        
        # Recording Status
        if RECORDING_AVAILABLE:
            st.write("🎙️ **Recording:** ✅ Available")
            
            # Recording settings
            st.subheader("Recording Settings")
            recording_duration = st.slider("Recording Duration (seconds)", 3, 15, 5)
        else:
            st.write("🎙️ **Recording:** ❌ Not available")
            recording_duration = 5
        
        st.divider()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display conversation
        display_conversation()
    
    with col2:
        st.subheader("🎙️ Voice Input")
        
        # Voice recording
        if RECORDING_AVAILABLE and st.button("🎤 Start Recording", type="primary"):
            audio_file = st.session_state.recorder.record_audio(recording_duration)
            
            if audio_file:
                try:
                    # Display audio player
                    st.audio(audio_file)
                    
                    # Transcribe audio
                    with st.spinner("Transcribing audio..."):
                        # Create temporary pipeline for this file
                        temp_pipeline = OverlappingASRPipeline(
                            input_path=audio_file,
                            audio_config=st.session_state.asr_pipeline.audio_config,
                            processing_config=st.session_state.asr_pipeline.processing_config
                        )
                        transcription = temp_pipeline()
                    
                    if transcription and not transcription.startswith("[ERROR"):
                        st.success(f"Transcribed: {transcription}")
                        
                        # Add to conversation
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": transcription
                        })
                        
                        # Get LLM response
                        if st.session_state.llm.is_available():
                            with st.spinner("Getting AI response..."):
                                # Use RAG-enabled LLM if available and enabled
                                if (st.session_state.use_rag and 
                                    st.session_state.rag_llm and 
                                    st.session_state.rag_llm.is_rag_available()):
                                    response = st.session_state.rag_llm.chat_with_rag(
                                        transcription, 
                                        st.session_state.conversation_history[:-1],
                                        use_rag=True,
                                        top_k=3,
                                        min_score=0.4
                                    )
                                else:
                                    response = st.session_state.llm.chat(
                                        transcription, 
                                        st.session_state.conversation_history[:-1]
                                    )
                            
                            # Add response to conversation
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                            st.success("Response generated!")
                            
                            # Display the response immediately
                            st.info(f"🤖 **Assistant Response:** {response}")
                            
                            # Auto-play response if TTS is available
                            if st.session_state.tts.is_available():
                                with st.spinner("Playing response..."):
                                    st.session_state.tts.speak(response, engine=st.session_state.tts_engine)
                        else:
                            st.error("OpenRouter not available for response generation")
                        
                        st.rerun()
                    else:
                        st.error("Transcription failed or empty")
                
                finally:
                    # Cleanup temporary file
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
        
        st.divider()
        
        # File upload option
        st.subheader("📁 Upload Audio")
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=['wav', 'mp3', 'mp4', 'm4a', 'flac'],
            help="Upload an audio file to transcribe"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}')
            temp_file.write(uploaded_file.read())
            temp_file.close()
            
            try:
                # Display audio player
                st.audio(uploaded_file)
                
                if st.button("🎯 Transcribe Uploaded File"):
                    with st.spinner("Transcribing uploaded audio..."):
                        temp_pipeline = OverlappingASRPipeline(
                            input_path=temp_file.name,
                            audio_config=st.session_state.asr_pipeline.audio_config,
                            processing_config=st.session_state.asr_pipeline.processing_config
                        )
                        transcription = temp_pipeline()
                    
                    if transcription and not transcription.startswith("[ERROR"):
                        st.success(f"Transcribed: {transcription}")
                        
                        # Add to conversation and get response (same as recording)
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": transcription
                        })
                        
                        if st.session_state.llm.is_available():
                            with st.spinner("Getting AI response..."):
                                # Use RAG-enabled LLM if available and enabled
                                if (st.session_state.use_rag and 
                                    st.session_state.rag_llm and 
                                    st.session_state.rag_llm.is_rag_available()):
                                    response = st.session_state.rag_llm.chat_with_rag(
                                        transcription, 
                                        st.session_state.conversation_history[:-1],
                                        use_rag=True,
                                        top_k=3,
                                        min_score=0.4
                                    )
                                else:
                                    response = st.session_state.llm.chat(
                                        transcription, 
                                        st.session_state.conversation_history[:-1]
                                    )
                            
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                            # Display the response immediately
                            st.success("Response generated!")
                            st.info(f"🤖 **Assistant Response:** {response}")
                            
                            if st.session_state.tts.is_available():
                                with st.spinner("Playing response..."):
                                    st.session_state.tts.speak(response, engine=st.session_state.tts_engine)
                        
                        st.rerun()
                    else:
                        st.error("Transcription failed")
            
            finally:
                # Cleanup
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
        
        st.divider()
        
        # Text input fallback
        st.subheader("💬 Text Input")
        text_input = st.text_area("Type your message:", height=100)
        
        if st.button("💬 Send Text Message"):
            if text_input.strip():
                # Add to conversation
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": text_input
                })
                
                # Get LLM response
                if st.session_state.llm.is_available():
                    with st.spinner("Getting AI response..."):
                        # Use RAG-enabled LLM if available and enabled
                        if (st.session_state.use_rag and 
                            st.session_state.rag_llm and 
                            st.session_state.rag_llm.is_rag_available()):
                            response = st.session_state.rag_llm.chat_with_rag(
                                text_input, 
                                st.session_state.conversation_history[:-1],
                                use_rag=True,
                                top_k=3,
                                min_score=0.4
                            )
                        else:
                            response = st.session_state.llm.chat(
                                text_input, 
                                st.session_state.conversation_history[:-1]
                            )
                    
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Display the response immediately
                    st.success("Response generated!")
                    st.info(f"🤖 **Assistant Response:** {response}")
                    
                    if st.session_state.tts.is_available():
                        with st.spinner("Playing response..."):
                            st.session_state.tts.speak(response, engine=st.session_state.tts_engine)
                
                st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🎤 Voice Chatbot powered by Whisper ASR + OpenRouter LLM + gTTS/F5-TTS Thai
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
