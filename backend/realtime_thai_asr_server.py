#!/usr/bin/env python3
"""
Real-time Thai ASR WebSocket Server
Serves faster-whisper Thai models with WebSocket interface for real-time transcription

Features:
- WebSocket-based real-time audio streaming
- GPU-optimized faster-whisper Thai models (95% GPU utilization)
- Voice Activity Detection (VAD)
- Real-time and final transcription results
- Multiple client support
- Audio chunking and buffering
- Thai language optimized settings
- Maximum GPU memory utilization for speed

Usage:
    python realtime_thai_asr_server.py [OPTIONS]

WebSocket Endpoints:
    - ws://localhost:8765/control - Control messages and configuration
    - ws://localhost:8766/audio - Audio data streaming and transcription results
"""

import os
import sys
import json
import asyncio
import logging
import time
import base64
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Set
from pathlib import Path
import tempfile
from collections import deque
import numpy as np

# Audio processing imports
import pyaudio
import wave
from scipy.signal import butter, filtfilt, resample_poly
from pydub import AudioSegment

# WebSocket imports
import websockets
from websockets.server import WebSocketServerProtocol

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import our optimized Thai ASR components
try:
    from whisper.model_manager import get_model_manager, ModelManager
    from whisper.faster_whisper_thai import WhisperConfig, FasterWhisperThai, create_thai_asr
except ImportError as e:
    print(f"❌ Failed to import Thai ASR components: {e}")
    print("Make sure you're running from the backend directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Audio configuration constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
BUFFER_SIZE = 8192

# WebSocket ports
CONTROL_PORT = 8765
AUDIO_PORT = 8766

class AudioProcessor:
    """Handles audio processing, buffering, and VAD"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_buffer = deque()
        self.vad_threshold = 0.02  # Adjusted for Thai speech
        self.silence_threshold = 30  # chunks of silence before stop
        self.min_speech_chunks = 20  # minimum chunks for valid speech
        
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Simple VAD based on energy level"""
        try:
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            return rms > self.vad_threshold
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False
    
    def add_chunk(self, audio_chunk: bytes) -> bool:
        """Add audio chunk to buffer and return if speech is detected"""
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Normalize to [-1, 1] range
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Add to buffer
            self.audio_buffer.append(audio_float)
            
            # Check for speech
            return self.is_speech(audio_float)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return False
    
    def get_audio_for_transcription(self) -> Optional[np.ndarray]:
        """Get accumulated audio for transcription"""
        if len(self.audio_buffer) < self.min_speech_chunks:
            return None
        
        try:
            # Concatenate all chunks
            audio_data = np.concatenate(list(self.audio_buffer))
            return audio_data
        except Exception as e:
            logger.error(f"Error getting audio for transcription: {e}")
            return None
    
    def clear_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer.clear()
    
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply simple noise reduction filter"""
        try:
            # Simple high-pass filter to remove low-frequency noise
            from scipy.signal import butter, filtfilt
            nyquist = self.sample_rate / 2
            low_cutoff = 300  # 300 Hz high-pass for speech
            b, a = butter(4, low_cutoff / nyquist, btype='high')
            filtered_audio = filtfilt(b, a, audio_data)
            return filtered_audio
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data

# Global variables
model_manager: Optional[ModelManager] = None
thai_asr: Optional[FasterWhisperThai] = None
audio_processor: Optional[AudioProcessor] = None
control_connections: Set[WebSocketServerProtocol] = set()
audio_connections: Set[WebSocketServerProtocol] = set()
audio_queue = asyncio.Queue()
is_recording = False
current_model_id = "biodatlab-large-faster"  # Use the working large model

# Audio processing state
audio_buffer = deque(maxlen=100)  # Store last 100 chunks
silence_counter = 0
speech_detected = False
last_transcription_time = 0

# WebSocket message handlers

async def handle_control_message(websocket: WebSocketServerProtocol, message: Dict[str, Any]):
    """Handle control WebSocket messages"""
    global model_manager, thai_asr, is_recording, current_model_id
    
    try:
        msg_type = message.get('type')
        
        if msg_type == 'load_model':
            model_id = message.get('model_id', 'biodatlab-medium-faster')
            await load_model(model_id)
            await websocket.send(json.dumps({
                'type': 'model_loaded',
                'model_id': model_id,
                'status': 'success'
            }))
        
        elif msg_type == 'get_models':
            try:
                if model_manager is None:
                    # Initialize model manager if not already done
                    from whisper.model_manager import get_model_manager
                    model_manager = get_model_manager()
                
                models = model_manager.get_available_models() if model_manager else []
                await websocket.send(json.dumps({
                    'type': 'available_models',
                    'models': models
                }))
            except Exception as e:
                logger.error(f"Error getting models: {e}")
                await websocket.send(json.dumps({
                    'type': 'available_models',
                    'models': [],
                    'error': str(e)
                }))
        
        elif msg_type == 'start_recording':
            is_recording = True
            await broadcast_to_audio_clients({
                'type': 'recording_started',
                'timestamp': datetime.now().isoformat()
            })
        
        elif msg_type == 'stop_recording':
            is_recording = False
            await broadcast_to_audio_clients({
                'type': 'recording_stopped',
                'timestamp': datetime.now().isoformat()
            })
        
        elif msg_type == 'get_status':
            await websocket.send(json.dumps({
                'type': 'status',
                'is_recording': is_recording,
                'current_model': current_model_id,
                'model_initialized': thai_asr is not None and thai_asr.model is not None,
                'connected_audio_clients': len(audio_connections)
            }))
        
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {msg_type}'
            }))
    
    except Exception as e:
        logger.error(f"Error handling control message: {e}")
        await websocket.send(json.dumps({
            'type': 'error',
            'message': str(e)
        }))

async def handle_audio_message(websocket: WebSocketServerProtocol, message: Dict[str, Any]):
    """Handle audio WebSocket messages"""
    global thai_asr, audio_processor, is_recording
    
    try:
        msg_type = message.get('type')
        
        if msg_type == 'audio_chunk':
            if not is_recording:
                return
            
            # Decode base64 audio data
            audio_b64 = message.get('data', '')
            if not audio_b64:
                return
            
            audio_bytes = base64.b64decode(audio_b64)
            
            # Process the audio chunk
            if thai_asr and thai_asr.model is not None and audio_processor:
                has_speech = audio_processor.add_chunk(audio_bytes)
                if has_speech:
                    # Get accumulated audio for partial transcription
                    audio_data = audio_processor.get_audio_for_transcription()
                    if audio_data is not None:
                        try:
                            # Create a simple transcription for real-time feedback
                            transcription = await transcribe_audio_data(audio_data)
                            if transcription:
                                # Broadcast realtime transcription to all audio clients
                                await broadcast_to_audio_clients({
                                    'type': 'realtime_transcription',
                                    'text': transcription,
                                    'timestamp': datetime.now().isoformat(),
                                    'is_final': False
                                })
                        except Exception as e:
                            logger.warning(f"Realtime transcription error: {e}")
        
        elif msg_type == 'audio_end':
            # Process final transcription with accumulated audio
            if thai_asr and thai_asr.model is not None and audio_processor:
                audio_data = audio_processor.get_audio_for_transcription()
                if audio_data is not None:
                    try:
                        # Apply noise reduction
                        audio_data = audio_processor.apply_noise_reduction(audio_data)
                        
                        # Get final transcription
                        result = await transcribe_audio_data_full(audio_data)
                        
                        # Broadcast final transcription
                        await broadcast_to_audio_clients({
                            'type': 'final_transcription',
                            'text': result.get('text', ''),
                            'duration': result.get('duration', 0),
                            'processing_time': result.get('processing_time', 0),
                            'model': current_model_id,
                            'timestamp': datetime.now().isoformat(),
                            'is_final': True
                        })
                        
                        # Clear the buffer
                        audio_processor.clear_buffer()
                    except Exception as e:
                        logger.error(f"Final transcription error: {e}")
    
    except Exception as e:
        logger.error(f"Error handling audio message: {e}")

async def transcribe_audio_data(audio_data: np.ndarray) -> Optional[str]:
    """Simple transcription for real-time feedback"""
    global thai_asr
    
    try:
        if not thai_asr or not thai_asr.model:
            return None
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            temp_path = tmp_file.name
        
        # Quick transcription
        result = thai_asr.transcribe(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        return result.get('text', '').strip() if result else None
        
    except Exception as e:
        logger.warning(f"Transcription error: {e}")
        return None

async def transcribe_audio_data_full(audio_data: np.ndarray) -> Dict[str, Any]:
    """Full transcription with detailed results"""
    global thai_asr
    
    try:
        if not thai_asr or not thai_asr.model:
            return {'text': '', 'duration': 0, 'processing_time': 0}
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            temp_path = tmp_file.name
        
        # Full transcription with timing
        start_time = time.time()
        result = thai_asr.transcribe(temp_path)
        processing_time = time.time() - start_time
        
        # Clean up
        os.unlink(temp_path)
        
        # Add timing information
        if result:
            result['processing_time'] = processing_time
            result['duration'] = len(audio_data) / SAMPLE_RATE
        
        return result or {'text': '', 'duration': 0, 'processing_time': processing_time}
        
    except Exception as e:
        logger.error(f"Full transcription error: {e}")
        return {'text': '', 'duration': 0, 'processing_time': 0}

# Utility functions

async def broadcast_to_control_clients(message: Dict[str, Any]):
    """Broadcast message to all control clients"""
    global control_connections
    if control_connections:
        message_json = json.dumps(message)
        disconnected = set()
        
        for websocket in control_connections:
            try:
                await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        control_connections -= disconnected

async def broadcast_to_audio_clients(message: Dict[str, Any]):
    """Broadcast message to all audio clients"""
    global audio_connections
    if audio_connections:
        message_json = json.dumps(message)
        disconnected = set()
        
        for websocket in audio_connections:
            try:
                await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        audio_connections -= disconnected

async def load_model(model_id: str):
    """
    Load a specific Thai ASR model
    
    Available models:
    - biodatlab-medium-faster: Vinxscribe/biodatlab-whisper-th-medium-faster
    - biodatlab-faster: Vinxscribe/biodatlab-whisper-th-large-v3-faster (legacy alias)
    - biodatlab-large-faster: Vinxscribe/biodatlab-whisper-th-large-v3-faster
    - pathumma-large: PathummaApiwat/Pathumma-whisper-large-v3-th
    """
    global thai_asr, current_model_id
    
    try:
        logger.info(f"🔄 Loading model: {model_id}")
        
        # Log GPU optimization settings
        logger.info("🚀 GPU Optimization Settings:")
        logger.info("   - GPU Memory Fraction: 95% (maximum utilization)")
        logger.info("   - Compute Type: float16 (GPU optimized)")
        logger.info("   - High Batch Sizes: 16-24 (depending on model)")
        logger.info("   - Parallel Workers: 8")
        logger.info("   - Beam Size: 3 (quality + speed balance)")
        
        # Create appropriate config based on model_id
        if model_id == "biodatlab-medium-faster":
            config = WhisperConfig(
                model_name="Vinxscribe/biodatlab-whisper-th-medium-faster",
                language="th",
                device="auto",
                compute_type="float16",
                gpu_memory_fraction=0.95,  # Increased from 0.8 to 0.95
                batch_size=16,  # Increased from 8 to 16
                num_workers=8,  # Increased from 4 to 8
                beam_size=3,    # Increased for better accuracy
                chunk_length_ms=15000,  # Smaller chunks for faster processing
                overlap_ms=300
            )
        elif model_id == "biodatlab-faster":
            # Fix: Use the correct large model path with max GPU optimization
            config = WhisperConfig(
                model_name="Vinxscribe/biodatlab-whisper-th-large-v3-faster",
                language="th", 
                device="auto",
                compute_type="float16",
                gpu_memory_fraction=0.95,  # Max GPU usage
                batch_size=20,  # Higher batch size for large model
                num_workers=8,
                beam_size=3,
                chunk_length_ms=15000,
                overlap_ms=300
            )
        elif model_id == "biodatlab-large-faster":
            # Optimized config for large model with maximum GPU utilization
            config = WhisperConfig(
                model_name="Vinxscribe/biodatlab-whisper-th-large-v3-faster",
                language="th",
                device="auto", 
                compute_type="float16",
                gpu_memory_fraction=0.95,  # Use 95% of GPU RAM
                batch_size=24,  # Higher batch size for large model
                num_workers=8,  # Maximize parallel processing
                beam_size=3,    # Better quality with more GPU
                chunk_length_ms=15000,  # Optimized chunk size
                overlap_ms=300,
                cpu_threads=16  # Increased CPU threads
            )
        elif model_id == "pathumma-large":
            config = WhisperConfig(
                model_name="PathummaApiwat/Pathumma-whisper-large-v3-th",
                language="th",
                device="auto", 
                compute_type="float16",
                gpu_memory_fraction=0.95,  # Max GPU usage
                batch_size=20,  # Higher batch size
                num_workers=8,
                beam_size=3,
                chunk_length_ms=15000,
                overlap_ms=300
            )
        else:
            # Default config with maximum GPU optimization
            logger.warning(f"Unknown model_id '{model_id}', using default large model")
            config = WhisperConfig(
                model_name="Vinxscribe/biodatlab-whisper-th-large-v3-faster",
                language="th",
                device="auto",
                compute_type="float16", 
                gpu_memory_fraction=0.95,  # Max GPU utilization
                batch_size=24,  # Large batch size
                num_workers=8,
                beam_size=3,
                chunk_length_ms=15000,
                overlap_ms=300,
                cpu_threads=16
            )
        
        # Initialize new ASR instance with the specific config
        thai_asr = FasterWhisperThai(config=config)
        
        # Check if model was loaded successfully
        if thai_asr.model is not None:
            current_model_id = model_id
            logger.info(f"✅ Model {model_id} loaded successfully")
        else:
            logger.error(f"❌ Failed to load model {model_id}")
            raise Exception(f"Model {model_id} failed to load")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_id}: {e}")
        raise

# WebSocket server handlers

async def control_handler(websocket: WebSocketServerProtocol):
    """Handle control WebSocket connections"""
    logger.info(f"🔌 Control client connected: {websocket.remote_address}")
    control_connections.add(websocket)
    
    try:
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'connected',
            'server': 'Thai ASR Real-time Server',
            'version': '1.0.0',
            'current_model': current_model_id
        }))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_control_message(websocket, data)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"🔌 Control client disconnected: {websocket.remote_address}")
    finally:
        control_connections.discard(websocket)

async def audio_handler(websocket: WebSocketServerProtocol):
    """Handle audio WebSocket connections"""
    logger.info(f"🎵 Audio client connected: {websocket.remote_address}")
    audio_connections.add(websocket)
    
    try:
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'audio_connected',
            'server': 'Thai ASR Audio Stream',
            'sample_rate': SAMPLE_RATE,
            'channels': CHANNELS,
            'chunk_size': CHUNK_SIZE
        }))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_audio_message(websocket, data)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"🎵 Audio client disconnected: {websocket.remote_address}")
    finally:
        audio_connections.discard(websocket)

async def start_servers():
    """Start both WebSocket servers"""
    global model_manager, audio_processor
    
    logger.info("🚀 Starting Thai ASR WebSocket servers...")
    
    # Initialize model manager if not already done
    if model_manager is None:
        logger.info("📦 Initializing model manager...")
        model_manager = ModelManager()
        models = model_manager.get_available_models()
        model_names = [model.get('name', 'Unknown') for model in models]
        logger.info(f"📦 Loaded {len(models)} models: {', '.join(model_names)}")
    
    # Initialize audio processor
    if audio_processor is None:
        logger.info("🎤 Initializing audio processor...")
        audio_processor = AudioProcessor()
    
    # Initialize the default model
    await load_model(current_model_id)
    
    # Start control server
    control_server = await websockets.serve(
        control_handler,
        "0.0.0.0",  # Listen on all interfaces for ngrok
        CONTROL_PORT,
        ping_interval=20,
        ping_timeout=10
    )
    logger.info(f"🔧 Control server started on ws://0.0.0.0:{CONTROL_PORT}")
    
    # Start audio server
    audio_server = await websockets.serve(
        audio_handler,
        "0.0.0.0",  # Listen on all interfaces for ngrok
        AUDIO_PORT,
        ping_interval=20,
        ping_timeout=10
    )
    logger.info(f"🎵 Audio server started on ws://0.0.0.0:{AUDIO_PORT}")
    
    logger.info("✅ Thai ASR WebSocket servers are ready!")
    logger.info("📖 Usage:")
    logger.info(f"   - Control: ws://0.0.0.0:{CONTROL_PORT}")
    logger.info(f"   - Audio: ws://0.0.0.0:{AUDIO_PORT}")
    logger.info("🌐 Use ngrok to expose these ports for external access")
    
    # Keep servers running
    await asyncio.gather(
        control_server.wait_closed(),
        audio_server.wait_closed()
    )

def main():
    """Main function"""
    try:
        # Set up Windows event loop policy if needed
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        print("🎤 Thai ASR Real-time WebSocket Server")
        print("=" * 50)
        print(f"🚀 Starting servers...")
        print(f"🔧 Control WebSocket: ws://0.0.0.0:{CONTROL_PORT}")
        print(f"🎵 Audio WebSocket: ws://0.0.0.0:{AUDIO_PORT}")
        print(f"📦 Default Model: {current_model_id}")
        print("🌐 Use ngrok to expose ports for external access")
        print("📝 Available models:")
        print("   - biodatlab-large-faster: Vinxscribe/biodatlab-whisper-th-large-v3-faster")
        print("   - biodatlab-medium-faster: Vinxscribe/biodatlab-whisper-th-medium-faster") 
        print("   - pathumma-large: PathummaApiwat/Pathumma-whisper-large-v3-th")
        print("=" * 50)
        
        # Run the servers
        asyncio.run(start_servers())
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise

if __name__ == "__main__":
    main()
