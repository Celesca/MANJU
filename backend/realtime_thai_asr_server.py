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
    """Enhanced audio processing with RealtimeSTT techniques"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Enhanced buffering with circular buffers
        self.audio_buffer = deque(maxlen=100)  # Circular buffer for efficiency
        self.pre_recording_buffer = deque(
            maxlen=int((sample_rate // chunk_size) * 1.0)  # 1 second pre-recording buffer
        )
        
        # Enhanced VAD parameters (from RealtimeSTT)
        self.vad_threshold = 0.02  # Base threshold
        self.silence_threshold = 30  # chunks of silence before stop
        self.min_speech_chunks = 20  # minimum chunks for valid speech
        self.post_speech_silence_duration = 0.6  # seconds to wait after speech ends
        self.min_length_of_recording = 0.5  # minimum recording length
        self.pre_recording_buffer_duration = 1.0  # pre-recording buffer duration
        
        # Advanced VAD state tracking
        self.is_speech_active = False
        self.speech_start_time = 0
        self.speech_end_time = 0
        self.silence_start_time = 0
        self.consecutive_silence_chunks = 0
        self.consecutive_speech_chunks = 0
        
        # Buffer overflow handling (from RealtimeSTT)
        self.allowed_latency_limit = 100  # max unprocessed chunks
        self.handle_buffer_overflow = True
        
        # Audio quality enhancement
        self.normalize_audio = True
        self.noise_reduction_enabled = True
        
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Enhanced VAD with multiple detection methods"""
        try:
            # Primary RMS-based detection
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            rms_speech = rms > self.vad_threshold
            
            # Secondary zero-crossing rate detection (for speech characteristics)
            zero_crossings = np.sum(np.diff(np.signbit(audio_chunk)))
            zcr_speech = 20 < zero_crossings < 200  # Typical speech range
            
            # Combine multiple indicators
            is_speech_detected = rms_speech and zcr_speech
            
            # Update speech state tracking
            if is_speech_detected:
                self.consecutive_speech_chunks += 1
                self.consecutive_silence_chunks = 0
                if not self.is_speech_active:
                    self.speech_start_time = time.time()
                    self.is_speech_active = True
                    logger.debug("🎤 Speech activity started")
            else:
                self.consecutive_silence_chunks += 1
                self.consecutive_speech_chunks = 0
                if self.is_speech_active and self.consecutive_silence_chunks > self.silence_threshold:
                    self.speech_end_time = time.time()
                    self.is_speech_active = False
                    logger.debug("🔇 Speech activity ended")
            
            return is_speech_detected
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False
    
    def add_chunk(self, audio_chunk: bytes) -> tuple[bool, bool]:
        """
        Add audio chunk with enhanced buffering and return (has_speech, recording_should_stop)
        Returns tuple: (speech_detected, should_stop_recording)
        """
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Normalize to [-1, 1] range
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Apply noise reduction if enabled
            if self.noise_reduction_enabled:
                audio_float = self.apply_noise_reduction(audio_float)
            
            # Add to pre-recording buffer (always)
            self.pre_recording_buffer.append(audio_float)
            
            # Check for speech
            has_speech = self.is_speech(audio_float)
            
            # Handle buffer overflow (RealtimeSTT technique)
            if self.handle_buffer_overflow and len(self.audio_buffer) > self.allowed_latency_limit:
                logger.warning(f"Audio buffer overflow: {len(self.audio_buffer)} chunks, discarding old data")
                # Remove oldest chunks
                for _ in range(len(self.audio_buffer) - self.allowed_latency_limit):
                    self.audio_buffer.popleft()
            
            # Add to main buffer if recording
            if self.is_speech_active or has_speech:
                # Include pre-recording buffer when speech starts
                if has_speech and not self.is_speech_active:
                    # Add pre-recording buffer to capture speech start
                    for buffered_chunk in list(self.pre_recording_buffer):
                        self.audio_buffer.append(buffered_chunk)
                
                self.audio_buffer.append(audio_float)
            
            # Determine if recording should stop
            should_stop = self._should_stop_recording()
            
            return has_speech, should_stop
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return False, False
    
    def _should_stop_recording(self) -> bool:
        """
        Enhanced logic to determine when to stop recording (from RealtimeSTT)
        """
        if not self.is_speech_active:
            return False
        
        # Check if we've had enough silence after speech
        if self.consecutive_silence_chunks > 0:
            silence_duration = self.consecutive_silence_chunks * (self.chunk_size / self.sample_rate)
            if silence_duration >= self.post_speech_silence_duration:
                # Check minimum recording length
                recording_duration = time.time() - self.speech_start_time
                if recording_duration >= self.min_length_of_recording:
                    return True
        
        return False
    
    def get_audio_for_transcription(self, clear_buffer: bool = False) -> Optional[np.ndarray]:
        """Get accumulated audio with enhanced processing"""
        if len(self.audio_buffer) < self.min_speech_chunks:
            return None
        
        try:
            # Concatenate all chunks
            audio_data = np.concatenate(list(self.audio_buffer))
            
            # Apply final processing
            if self.normalize_audio:
                audio_data = self._normalize_audio(audio_data)
            
            # Clear buffer if requested
            if clear_buffer:
                self.clear_buffer()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error getting audio for transcription: {e}")
            return None
    
    def clear_buffer(self):
        """Clear the audio buffer and reset state"""
        self.audio_buffer.clear()
        self.is_speech_active = False
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        logger.debug("Audio buffer cleared")
    
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhanced noise reduction with multiple filters"""
        try:
            # High-pass filter to remove low-frequency noise
            from scipy.signal import butter, filtfilt
            nyquist = self.sample_rate / 2
            
            # Remove very low frequencies (< 80Hz) - electrical hum, rumble
            low_cutoff = 80 / nyquist
            b_high, a_high = butter(2, low_cutoff, btype='high')
            filtered_audio = filtfilt(b_high, a_high, audio_data)
            
            # Optional: Band-pass filter for speech frequencies (300-3400 Hz)
            # Uncomment for more aggressive noise reduction
            # low_freq = 300 / nyquist
            # high_freq = 3400 / nyquist
            # b_band, a_band = butter(2, [low_freq, high_freq], btype='band')
            # filtered_audio = filtfilt(b_band, a_band, filtered_audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to optimal range for transcription"""
        try:
            # Find peak amplitude
            peak = np.abs(audio_data).max()
            
            if peak > 0:
                # Normalize to 95% of maximum to prevent clipping
                normalized = (audio_data / peak) * 0.95
                return normalized
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
            return audio_data
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get current recording statistics"""
        return {
            'buffer_size': len(self.audio_buffer),
            'pre_buffer_size': len(self.pre_recording_buffer),
            'is_speech_active': self.is_speech_active,
            'consecutive_speech_chunks': self.consecutive_speech_chunks,
            'consecutive_silence_chunks': self.consecutive_silence_chunks,
            'recording_duration': time.time() - self.speech_start_time if self.is_speech_active else 0
        }

# Global variables for server state
control_connections: Set[WebSocketServerProtocol] = set()
audio_connections: Set[WebSocketServerProtocol] = set()
connected_clients: Dict[str, Dict] = {}
thai_asr = None
model_manager = None
audio_processor = None
audio_queue = asyncio.Queue()
is_recording = False
current_model_id = "biodatlab-large-faster"  # Use the working large model

server_config = {
    'sample_rate': 16000,
    'channels': 1,
    'chunk_size': 1024,
    'vad_threshold': 0.5,
    'min_speech_duration': 0.5,
    'max_silence_duration': 1.0,
    # Enhanced configuration from RealtimeSTT
    'early_transcription_on_silence': 200,  # milliseconds
    'realtime_processing_pause': 0.2,  # seconds between real-time updates
    'post_speech_silence_duration': 0.6,  # seconds of silence to auto-stop
    'min_length_of_recording': 0.5,  # minimum recording duration
    'pre_recording_buffer_duration': 1.0,  # pre-recording buffer
    'allowed_latency_limit': 100,  # max buffered chunks
    'normalize_audio': True,
    'noise_reduction': True
}

# Parallel processing globals
transcription_semaphore = asyncio.Semaphore(4)  # Allow 4 concurrent transcriptions
active_transcriptions = set()  # Track active transcription tasks
max_concurrent_transcriptions = 4  # Adjust based on GPU capacity

# Performance monitoring (inspired by RealtimeSTT)
performance_stats = {
    'total_requests': 0,
    'concurrent_requests': 0,
    'avg_processing_time': 0.0,
    'peak_concurrent': 0,
    'start_time': time.time(),
    'total_audio_processed_seconds': 0.0,
    'total_transcriptions': 0,
    'real_time_factor': 0.0,  # Processing time vs audio duration ratio
    'error_count': 0
}

# Enhanced state tracking (RealtimeSTT inspired)
recording_state = {
    'is_recording': False,
    'recording_start_time': 0,
    'last_activity_time': 0,
    'current_session_id': None,
    'auto_stop_enabled': True,
    'speech_detected': False,
    'continuous_silence_duration': 0
}

# Audio processing state
audio_buffer = deque(maxlen=100)  # Store last 100 chunks
silence_counter = 0
speech_detected = False
last_transcription_time = 0

# WebSocket message handlers

async def handle_control_message(websocket: WebSocketServerProtocol, message: Dict[str, Any]):
    """Enhanced control WebSocket message handler"""
    global model_manager, thai_asr, is_recording, current_model_id, recording_state
    
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
            recording_state['is_recording'] = True
            recording_state['recording_start_time'] = time.time()
            recording_state['current_session_id'] = f"session_{int(time.time())}"
            
            # Reset audio processor if needed
            if audio_processor:
                audio_processor.clear_buffer()
            
            await broadcast_to_audio_clients({
                'type': 'recording_started',
                'session_id': recording_state['current_session_id'],
                'timestamp': datetime.now().isoformat()
            })
        
        elif msg_type == 'stop_recording':
            is_recording = False
            recording_state['is_recording'] = False
            
            await broadcast_to_audio_clients({
                'type': 'recording_stopped',
                'session_id': recording_state.get('current_session_id'),
                'timestamp': datetime.now().isoformat()
            })
        
        elif msg_type == 'get_status':
            await websocket.send(json.dumps({
                'type': 'status',
                'is_recording': is_recording,
                'current_model': current_model_id,
                'model_initialized': thai_asr is not None and thai_asr.model is not None,
                'connected_audio_clients': len(audio_connections),
                'recording_state': recording_state,
                'active_transcriptions': len(active_transcriptions)
            }))
        
        elif msg_type == 'get_statistics':
            # New enhanced statistics endpoint
            stats = await get_server_statistics()
            await websocket.send(json.dumps({
                'type': 'server_statistics',
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            }))
        
        elif msg_type == 'update_config':
            # Allow dynamic configuration updates
            config_updates = message.get('config', {})
            for key, value in config_updates.items():
                if key in server_config:
                    server_config[key] = value
                    logger.info(f"Updated config: {key} = {value}")
            
            await websocket.send(json.dumps({
                'type': 'config_updated',
                'updated_config': server_config,
                'timestamp': datetime.now().isoformat()
            }))
        
        elif msg_type == 'toggle_auto_stop':
            # Toggle automatic recording stop feature
            recording_state['auto_stop_enabled'] = not recording_state['auto_stop_enabled']
            
            await websocket.send(json.dumps({
                'type': 'auto_stop_toggled',
                'auto_stop_enabled': recording_state['auto_stop_enabled'],
                'timestamp': datetime.now().isoformat()
            }))
        
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {msg_type}'
            }))
    
    except Exception as e:
        logger.error(f"Error handling control message: {e}")
        performance_stats['error_count'] += 1
        await websocket.send(json.dumps({
            'type': 'error',
            'message': str(e)
        }))

async def handle_audio_message(websocket: WebSocketServerProtocol, message: Dict[str, Any]):
    """Enhanced audio message handler with RealtimeSTT techniques"""
    global thai_asr, audio_processor, is_recording, active_transcriptions
    
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
            
            # Process the audio chunk with enhanced VAD
            if thai_asr and thai_asr.model is not None and audio_processor:
                has_speech, should_stop = audio_processor.add_chunk(audio_bytes)
                
                # Handle real-time transcription for active speech
                if has_speech and audio_processor.is_speech_active:
                    # Get current audio for partial transcription
                    audio_data = audio_processor.get_audio_for_transcription(clear_buffer=False)
                    if audio_data is not None and len(audio_data) > audio_processor.min_speech_chunks * audio_processor.chunk_size:
                        # Create parallel real-time transcription task
                        task = asyncio.create_task(
                            transcribe_realtime_parallel(audio_data, websocket)
                        )
                        active_transcriptions.add(task)
                        task.add_done_callback(lambda t: active_transcriptions.discard(t))
                
                # Handle automatic recording stop when speech ends
                if should_stop:
                    logger.info("🛑 Auto-stopping recording: silence detected after speech")
                    
                    # Get final audio for transcription
                    final_audio = audio_processor.get_audio_for_transcription(clear_buffer=True)
                    if final_audio is not None:
                        # Create parallel final transcription task
                        task = asyncio.create_task(
                            transcribe_final_parallel(final_audio, websocket, auto_stop=True)
                        )
                        active_transcriptions.add(task)
                        task.add_done_callback(lambda t: active_transcriptions.discard(t))
                    
                    # Notify clients that recording auto-stopped
                    await broadcast_to_audio_clients({
                        'type': 'recording_auto_stopped',
                        'reason': 'silence_detected',
                        'timestamp': datetime.now().isoformat()
                    })
        
        elif msg_type == 'audio_end':
            # Manual end of recording - process final transcription
            if thai_asr and thai_asr.model is not None and audio_processor:
                logger.info("🔚 Manual recording end")
                
                # Get final accumulated audio
                audio_data = audio_processor.get_audio_for_transcription(clear_buffer=True)
                if audio_data is not None:
                    # Create parallel final transcription task
                    task = asyncio.create_task(
                        transcribe_final_parallel(audio_data, websocket, auto_stop=False)
                    )
                    active_transcriptions.add(task)
                    task.add_done_callback(lambda t: active_transcriptions.discard(t))
        
        elif msg_type == 'get_recording_stats':
            # New message type to get recording statistics
            if audio_processor:
                stats = audio_processor.get_recording_stats()
                await websocket.send(json.dumps({
                    'type': 'recording_stats',
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                }))
    
    except Exception as e:
        logger.error(f"Error handling audio message: {e}")
        # Send error response
        try:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Audio processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }))
        except:
            pass  # Ignore if websocket is closed

async def transcribe_realtime_parallel(audio_data: np.ndarray, websocket: WebSocketServerProtocol):
    """Parallel real-time transcription task"""
    async with transcription_semaphore:  # Limit concurrent transcriptions
        try:
            transcription = await transcribe_audio_data(audio_data)
            if transcription:
                # Broadcast to all audio clients, not just the sender
                await broadcast_to_audio_clients({
                    'type': 'realtime_transcription',
                    'text': transcription,
                    'timestamp': datetime.now().isoformat(),
                    'is_final': False
                })
        except Exception as e:
            logger.warning(f"Parallel realtime transcription error: {e}")

async def transcribe_final_parallel(audio_data: np.ndarray, websocket: WebSocketServerProtocol, auto_stop: bool = False):
    """Enhanced parallel final transcription with auto-stop handling"""
    async with transcription_semaphore:  # Limit concurrent transcriptions
        try:
            # Apply noise reduction with enhanced processing
            audio_data_filtered = audio_processor.apply_noise_reduction(audio_data) if audio_processor else audio_data
            
            # Get final transcription with timing
            start_time = time.time()
            result = await transcribe_audio_data_full(audio_data_filtered)
            processing_time = time.time() - start_time
            
            # Enhanced result with additional metadata
            enhanced_result = {
                'type': 'final_transcription',
                'text': result.get('text', ''),
                'duration': result.get('duration', 0),
                'processing_time': processing_time,
                'model': current_model_id,
                'timestamp': datetime.now().isoformat(),
                'is_final': True,
                'auto_stopped': auto_stop,
                'audio_stats': {
                    'sample_rate': SAMPLE_RATE,
                    'audio_length_seconds': len(audio_data) / SAMPLE_RATE,
                    'audio_samples': len(audio_data)
                }
            }
            
            # Add quality metrics if available
            if result.get('confidence'):
                enhanced_result['confidence'] = result['confidence']
            
            # Broadcast to all audio clients
            await broadcast_to_audio_clients(enhanced_result)
            
            # Log performance metrics
            logger.info(f"🎯 Final transcription completed: "
                       f"'{result.get('text', '')[:50]}...' "
                       f"({processing_time:.2f}s, {len(audio_data)/SAMPLE_RATE:.1f}s audio, "
                       f"auto_stop={auto_stop})")
            
        except Exception as e:
            logger.error(f"Parallel final transcription error: {e}")
            # Send error notification
            await broadcast_to_audio_clients({
                'type': 'transcription_error',
                'error': str(e),
                'auto_stopped': auto_stop,
                'timestamp': datetime.now().isoformat()
            })

async def transcribe_audio_data(audio_data: np.ndarray) -> Optional[str]:
    """Simple transcription for real-time feedback (parallel)"""
    global thai_asr
    
    try:
        if not thai_asr or not thai_asr.model:
            return None
        
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _transcribe_sync, audio_data, thai_asr
        )
        
        return result.get('text', '').strip() if result else None
        
    except Exception as e:
        logger.warning(f"Transcription error: {e}")
        return None

async def transcribe_audio_data_full(audio_data: np.ndarray) -> Dict[str, Any]:
    """Enhanced full transcription with detailed results and performance tracking"""
    global thai_asr, performance_stats
    
    try:
        if not thai_asr or not thai_asr.model:
            return {'text': '', 'duration': 0, 'processing_time': 0}
        
        # Track performance
        start_time = time.time()
        audio_duration = len(audio_data) / SAMPLE_RATE
        performance_stats['concurrent_requests'] += 1
        performance_stats['peak_concurrent'] = max(
            performance_stats['peak_concurrent'], 
            performance_stats['concurrent_requests']
        )
        
        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _transcribe_sync, audio_data, thai_asr
            )
            
            processing_time = time.time() - start_time
            
            # Add timing and quality information
            if result:
                result['processing_time'] = processing_time
                result['duration'] = audio_duration
                result['real_time_factor'] = processing_time / audio_duration if audio_duration > 0 else 0
                result['audio_samples'] = len(audio_data)
                result['timestamp'] = datetime.now().isoformat()
            
            # Update performance statistics
            await update_performance_stats(processing_time, audio_duration)
            
            return result or {'text': '', 'duration': audio_duration, 'processing_time': processing_time}
            
        finally:
            performance_stats['concurrent_requests'] -= 1
        
    except Exception as e:
        logger.error(f"Full transcription error: {e}")
        performance_stats['error_count'] += 1
        return {'text': '', 'duration': 0, 'processing_time': 0, 'error': str(e)}

def _transcribe_sync(audio_data: np.ndarray, asr_model: FasterWhisperThai) -> Optional[Dict[str, Any]]:
    """Synchronous transcription function for thread pool execution"""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            temp_path = tmp_file.name
        
        # Transcribe
        result = asr_model.transcribe(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        logger.warning(f"Sync transcription error: {e}")
        return None

async def update_performance_stats(processing_time: float, audio_duration: float):
    """Update performance statistics (RealtimeSTT inspired)"""
    global performance_stats
    
    performance_stats['total_requests'] += 1
    performance_stats['total_transcriptions'] += 1
    performance_stats['total_audio_processed_seconds'] += audio_duration
    
    # Update average processing time
    total_requests = performance_stats['total_requests']
    current_avg = performance_stats['avg_processing_time']
    performance_stats['avg_processing_time'] = (current_avg * (total_requests - 1) + processing_time) / total_requests
    
    # Calculate real-time factor (processing_time / audio_duration)
    if audio_duration > 0:
        rtf = processing_time / audio_duration
        performance_stats['real_time_factor'] = rtf
        
        # Log performance warnings
        if rtf > 1.0:
            logger.warning(f"⚠️ Real-time factor > 1.0: {rtf:.2f} (processing too slow)")
        elif rtf < 0.1:
            logger.info(f"🚀 Excellent performance: RTF {rtf:.2f}")

async def get_server_statistics() -> Dict[str, Any]:
    """Get comprehensive server statistics"""
    global performance_stats, recording_state, audio_processor, active_transcriptions
    
    uptime = time.time() - performance_stats['start_time']
    
    stats = {
        'server': {
            'uptime_seconds': uptime,
            'uptime_formatted': f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m {uptime%60:.1f}s",
            'current_model': current_model_id,
            'connected_clients': len(audio_connections),
            'is_recording': is_recording
        },
        'performance': {
            **performance_stats,
            'active_transcriptions': len(active_transcriptions),
            'peak_concurrent': max(performance_stats['peak_concurrent'], len(active_transcriptions)),
            'requests_per_minute': (performance_stats['total_requests'] / uptime) * 60 if uptime > 0 else 0,
            'audio_processed_per_minute': (performance_stats['total_audio_processed_seconds'] / uptime) * 60 if uptime > 0 else 0
        },
        'recording': recording_state.copy(),
        'audio_processor': audio_processor.get_recording_stats() if audio_processor else {},
        'server_config': server_config.copy()
    }
    
    return stats

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
