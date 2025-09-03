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

# Audio configuration constants (RealtimeSTT optimized)
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Reduced from 1024 for lower latency (RealtimeSTT optimization)
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
BUFFER_SIZE = 4096  # Reduced buffer size for faster processing
INT16_MAX_ABS_VALUE = 32768.0  # For audio normalization (RealtimeSTT technique)

# WebSocket ports
CONTROL_PORT = 8765
AUDIO_PORT = 8766

# Performance constants from RealtimeSTT
TIME_SLEEP = 0.02  # Sleep time for non-blocking operations
ALLOWED_LATENCY_LIMIT = 100  # Max buffered chunks before discarding
MAX_CONCURRENT_TRANSCRIPTIONS = 6  # Increased from 4 for better GPU utilization

class AudioProcessor:
    """Enhanced audio processing with RealtimeSTT optimizations"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # RealtimeSTT-inspired sliding window approach
        self.recent_chunks = deque(maxlen=40)  # Increased from 30 for better context
        self.current_utterance = deque(maxlen=80)  # Increased from 50 for longer utterances
        
        # Enhanced VAD parameters (RealtimeSTT optimized)
        self.vad_threshold = 0.015  # More sensitive threshold
        self.silence_threshold = 12  # Reduced from 15 for faster auto-stop
        self.min_speech_chunks = 6   # Reduced from 8 for faster detection
        
        # State tracking with performance monitoring
        self.is_speech_active = False
        self.consecutive_silence_chunks = 0
        self.consecutive_speech_chunks = 0
        self.last_transcription_time = 0
        self.transcription_cooldown = 0.3  # Reduced from 0.5s for more responsive updates
        
        # RealtimeSTT audio preprocessing buffer
        self.preprocessing_buffer = bytearray()
        self.buffer_size_target = 2 * chunk_size  # Target buffer size for processing
        
        # Performance tracking
        self.total_chunks_processed = 0
        self.speech_chunks_detected = 0
        self.last_activity_time = time.time()
        
    def lowpass_filter(self, signal: np.ndarray, cutoff_freq: float, sample_rate: float) -> np.ndarray:
        """
        Apply anti-aliasing low-pass filter (RealtimeSTT technique)
        """
        try:
            from scipy.signal import butter, filtfilt
            nyquist_rate = sample_rate / 2.0
            normal_cutoff = cutoff_freq / nyquist_rate
            b, a = butter(5, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, signal)
        except Exception as e:
            logger.warning(f"Filter error: {e}")
            return signal
    
    def resample_audio(self, pcm_data: np.ndarray, target_sample_rate: int, 
                      original_sample_rate: int) -> np.ndarray:
        """
        High-quality resampling with anti-aliasing (RealtimeSTT technique)
        """
        try:
            from scipy.signal import resample_poly
            
            if target_sample_rate < original_sample_rate:
                # Downsampling with anti-aliasing filter
                pcm_filtered = self.lowpass_filter(pcm_data, target_sample_rate / 2, original_sample_rate)
                resampled = resample_poly(pcm_filtered, target_sample_rate, original_sample_rate)
            else:
                # Upsampling without filter
                resampled = resample_poly(pcm_data, target_sample_rate, original_sample_rate)
            return resampled
        except Exception as e:
            logger.warning(f"Resample error: {e}")
            return pcm_data
    
    def normalize_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Normalize audio to optimal range (RealtimeSTT technique)
        """
        try:
            # Convert to float32 for processing
            audio_float = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
            
            # Apply light normalization to improve VAD performance
            max_val = np.max(np.abs(audio_float))
            if max_val > 0.001:  # Avoid normalizing silence
                audio_float = audio_float * (0.8 / max_val)
            
            return audio_float
        except Exception as e:
            logger.warning(f"Normalization error: {e}")
            return audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
        
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Enhanced VAD with RealtimeSTT techniques"""
        try:
            # Multiple VAD approaches for better accuracy
            
            # 1. RMS energy detection (primary)
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            energy_speech = rms > self.vad_threshold
            
            # 2. Zero crossing rate (helps with fricatives and consonants)
            zero_crossings = np.sum(np.diff(np.sign(audio_chunk)) != 0)
            zcr_speech = zero_crossings > len(audio_chunk) * 0.01
            
            # 3. Peak detection (for sudden sounds)
            peak_val = np.max(np.abs(audio_chunk))
            peak_speech = peak_val > self.vad_threshold * 2
            
            # Combine all methods with weights (RealtimeSTT approach)
            speech_score = (energy_speech * 0.6 + zcr_speech * 0.2 + peak_speech * 0.2)
            
            return speech_score > 0.5
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False
    
    def preprocess_chunk(self, audio_bytes: bytes) -> np.ndarray:
        """
        Preprocess audio chunk with RealtimeSTT optimizations
        """
        try:
            # Add to preprocessing buffer (RealtimeSTT technique)
            self.preprocessing_buffer.extend(audio_bytes)
            
            # Process when we have enough data
            if len(self.preprocessing_buffer) < self.buffer_size_target:
                return None
            
            # Extract chunk for processing
            chunk_bytes = bytes(self.preprocessing_buffer[:self.buffer_size_target])
            self.preprocessing_buffer = self.preprocessing_buffer[self.buffer_size_target:]
            
            # Convert to numpy array
            audio_np = np.frombuffer(chunk_bytes, dtype=np.int16)
            
            # Normalize audio for better VAD performance
            audio_float = self.normalize_audio(audio_np)
            
            self.total_chunks_processed += 1
            return audio_float
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None
    
    def add_chunk(self, audio_chunk: bytes) -> tuple[bool, bool, Optional[np.ndarray]]:
        """
        Enhanced chunk processing with RealtimeSTT optimizations
        Returns:
        - has_speech: bool - if speech detected in this chunk
        - should_stop: bool - if silence detected after speech (auto-stop)
        - realtime_audio: np.ndarray or None - audio for real-time transcription
        """
        try:
            # Preprocess chunk with RealtimeSTT techniques
            audio_float = self.preprocess_chunk(audio_chunk)
            if audio_float is None:
                return False, False, None
            
            # Add to sliding window (RealtimeSTT approach)
            self.recent_chunks.append(audio_float)
            
            # Enhanced speech detection
            has_speech = self.is_speech(audio_float)
            
            # Update activity tracking
            current_time = time.time()
            if has_speech:
                self.last_activity_time = current_time
                self.speech_chunks_detected += 1
            
            # State management with RealtimeSTT logic
            if has_speech:
                self.consecutive_speech_chunks += 1
                self.consecutive_silence_chunks = 0
                
                # Add to current utterance with overlap handling
                self.current_utterance.append(audio_float)
                
                if not self.is_speech_active:
                    self.is_speech_active = True
                    logger.debug("🎤 Speech started (enhanced detection)")
            else:
                self.consecutive_silence_chunks += 1
                if self.is_speech_active:
                    self.consecutive_speech_chunks = 0
            
            # Real-time transcription with adaptive timing (RealtimeSTT optimization)
            realtime_audio = None
            
            if (self.is_speech_active and 
                len(self.current_utterance) >= self.min_speech_chunks and
                current_time - self.last_transcription_time > self.transcription_cooldown):
                
                # Get recent audio for real-time transcription (optimized window size)
                window_size = min(15, len(self.current_utterance))  # Adaptive window
                realtime_audio = np.concatenate(list(self.current_utterance)[-window_size:])
                self.last_transcription_time = current_time
            
            # Enhanced auto-stop logic (RealtimeSTT inspired)
            should_stop = False
            if (self.is_speech_active and 
                self.consecutive_silence_chunks > self.silence_threshold and
                len(self.current_utterance) >= self.min_speech_chunks):
                
                # Additional check: ensure minimum utterance duration
                utterance_duration = len(self.current_utterance) * self.chunk_size / self.sample_rate
                if utterance_duration >= 0.3:  # Minimum 300ms utterance
                    should_stop = True
                    self.is_speech_active = False
                    logger.debug(f"🔇 Speech ended - auto-stopping (duration: {utterance_duration:.2f}s)")
            
            return has_speech, should_stop, realtime_audio
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return False, False, None
    
    def get_final_audio(self) -> Optional[np.ndarray]:
        """Get final audio for complete transcription"""
        if len(self.current_utterance) < self.min_speech_chunks:
            return None
        
        try:
            # Return the complete current utterance
            final_audio = np.concatenate(list(self.current_utterance))
            return final_audio
        except Exception as e:
            logger.error(f"Error getting final audio: {e}")
            return None
    
    def clear_utterance(self):
        """Clear current utterance after final transcription"""
        self.current_utterance.clear()
        self.is_speech_active = False
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        logger.debug("Utterance cleared")
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get comprehensive recording statistics (RealtimeSTT inspired)"""
        current_time = time.time()
        return {
            'recent_chunks': len(self.recent_chunks),
            'current_utterance_chunks': len(self.current_utterance),
            'is_speech_active': self.is_speech_active,
            'consecutive_speech_chunks': self.consecutive_speech_chunks,
            'consecutive_silence_chunks': self.consecutive_silence_chunks,
            'utterance_duration_seconds': len(self.current_utterance) * self.chunk_size / self.sample_rate,
            'total_chunks_processed': self.total_chunks_processed,
            'speech_chunks_detected': self.speech_chunks_detected,
            'speech_detection_ratio': self.speech_chunks_detected / max(1, self.total_chunks_processed),
            'time_since_last_activity': current_time - self.last_activity_time,
            'buffer_size': len(self.preprocessing_buffer),
            'vad_threshold': self.vad_threshold,
            'transcription_cooldown': self.transcription_cooldown
        }
    
    def clear_buffer(self):
        """Clear all buffers and reset state (RealtimeSTT technique)"""
        self.recent_chunks.clear()
        self.current_utterance.clear()
        self.preprocessing_buffer.clear()
        self.is_speech_active = False
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        self.last_transcription_time = 0
        logger.debug("All audio buffers cleared")
    
    def adjust_sensitivity(self, sensitivity: float):
        """Dynamically adjust VAD sensitivity (RealtimeSTT feature)"""
        self.vad_threshold = max(0.005, min(0.1, sensitivity))
        logger.info(f"VAD sensitivity adjusted to: {self.vad_threshold}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring (RealtimeSTT inspired)"""
        return {
            'total_chunks_processed': self.total_chunks_processed,
            'speech_chunks_detected': self.speech_chunks_detected,
            'detection_accuracy': self.speech_chunks_detected / max(1, self.total_chunks_processed),
            'current_utterance_length': len(self.current_utterance),
            'buffer_utilization': len(self.preprocessing_buffer) / self.buffer_size_target,
            'is_active': self.is_speech_active,
            'avg_chunk_size': self.chunk_size,
            'sample_rate': self.sample_rate
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
    'chunk_size': 512,  # Reduced for lower latency
    'vad_threshold': 0.4,
    'min_speech_duration': 0.3,  # Reduced from 0.5s
    'max_silence_duration': 0.8,  # Reduced from 1.0s
    # Enhanced configuration from RealtimeSTT
    'early_transcription_on_silence': 150,  # Reduced from 200ms for faster response
    'realtime_processing_pause': 0.15,  # Reduced from 0.2s for more frequent updates
    'post_speech_silence_duration': 0.4,  # Reduced from 0.6s for faster auto-stop
    'min_length_of_recording': 0.3,  # Reduced from 0.5s
    'pre_recording_buffer_duration': 0.8,  # Reduced from 1.0s
    'allowed_latency_limit': 80,  # Reduced from 100 for lower latency
    'normalize_audio': True,
    'noise_reduction': True,
    # New RealtimeSTT optimizations
    'beam_size': 1,  # Fastest beam search
    'beam_size_realtime': 1,  # Even faster for real-time
    'temperature': 0.0,  # Deterministic for speed
    'compression_ratio_threshold': 2.4,
    'log_prob_threshold': -1.0,
    'no_speech_threshold': 0.6,
    'condition_on_previous_text': False,  # Faster processing
    'initial_prompt_realtime': None,  # No prompt for speed
    'suppress_tokens': [-1],  # Suppress non-speech tokens
    'faster_whisper_vad_filter': True,  # Built-in VAD
    'handle_buffer_overflow': True
}

# Parallel processing globals (RealtimeSTT optimized)
transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)  # Increased capacity
active_transcriptions = set()  # Track active transcription tasks
max_concurrent_transcriptions = MAX_CONCURRENT_TRANSCRIPTIONS
transcription_queue = asyncio.Queue(maxsize=ALLOWED_LATENCY_LIMIT)  # Bounded queue for backpressure

# Performance monitoring (RealtimeSTT inspired)
performance_stats = {
    'total_requests': 0,
    'concurrent_requests': 0,
    'avg_processing_time': 0.0,
    'peak_concurrent': 0,
    'start_time': time.time(),
    'total_audio_processed_seconds': 0.0,
    'total_transcriptions': 0,
    'real_time_factor': 0.0,  # Processing time vs audio duration ratio
    'error_count': 0,
    # New RealtimeSTT metrics
    'successful_transcriptions': 0,
    'failed_transcriptions': 0,
    'avg_queue_size': 0.0,
    'max_queue_size': 0,
    'buffer_overflows': 0,
    'vad_activations': 0,
    'auto_stops': 0,
    'last_reset_time': time.time()
}

# Enhanced state tracking (RealtimeSTT inspired)
recording_state = {
    'is_recording': False,
    'recording_start_time': 0,
    'last_activity_time': 0,
    'current_session_id': None,
    'auto_stop_enabled': True,
    'speech_detected': False,
    'continuous_silence_duration': 0,
    # New RealtimeSTT states
    'vad_active': False,
    'realtime_active': False,
    'buffer_health': 'good',  # good, warning, critical
    'processing_mode': 'normal',  # normal, fast, ultra_fast
    'adaptive_threshold': 0.015,
    'session_stats': {
        'utterances_processed': 0,
        'avg_utterance_length': 0.0,
        'total_audio_duration': 0.0
    }
}

# Audio processing state (RealtimeSTT optimized)
audio_buffer = deque(maxlen=ALLOWED_LATENCY_LIMIT)  # Configurable max length
recent_transcriptions = deque(maxlen=10)  # Store recent results for analysis
silence_counter = 0
speech_detected = False
last_transcription_time = 0

# Threading and async optimization (RealtimeSTT techniques)
transcription_executor = None  # Will be initialized in main
background_tasks = set()  # Track background tasks for cleanup

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
            stats = await get_enhanced_statistics()
            await websocket.send(json.dumps({
                'type': 'server_statistics',
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            }))
        
        elif msg_type == 'reset_statistics':
            # Reset performance statistics
            await reset_performance_stats()
            await websocket.send(json.dumps({
                'type': 'statistics_reset',
                'message': 'Performance statistics have been reset',
                'timestamp': datetime.now().isoformat()
            }))
        
        elif msg_type == 'optimize_performance':
            # Manual performance optimization trigger
            await optimize_server_performance()
            await websocket.send(json.dumps({
                'type': 'performance_optimized',
                'message': 'Performance optimization applied',
                'timestamp': datetime.now().isoformat()
            }))
        
        elif msg_type == 'adjust_vad_sensitivity':
            # Dynamic VAD sensitivity adjustment
            sensitivity = message.get('sensitivity', 0.4)
            if audio_processor:
                audio_processor.adjust_sensitivity(float(sensitivity))
                server_config['vad_threshold'] = float(sensitivity)
                await websocket.send(json.dumps({
                    'type': 'vad_sensitivity_adjusted',
                    'new_sensitivity': float(sensitivity),
                    'timestamp': datetime.now().isoformat()
                }))
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Audio processor not available'
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
    """Fixed audio handler - no more concatenation madness!"""
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
            
            # Process with new chunked approach
            if thai_asr and thai_asr.model is not None and audio_processor:
                has_speech, should_stop, realtime_audio = audio_processor.add_chunk(audio_bytes)
                
                # Real-time transcription ONLY for new audio segments
                if realtime_audio is not None:
                    # Create parallel real-time transcription for JUST this segment
                    task = asyncio.create_task(
                        transcribe_realtime_chunk(realtime_audio, websocket)
                    )
                    active_transcriptions.add(task)
                    task.add_done_callback(lambda t: active_transcriptions.discard(t))
                
                # Auto-stop when speech ends
                if should_stop:
                    logger.info("🛑 Auto-stopping: speech ended")
                    
                    # Get the complete utterance for final transcription
                    final_audio = audio_processor.get_final_audio()
                    if final_audio is not None:
                        # Final transcription of the complete utterance
                        task = asyncio.create_task(
                            transcribe_final_utterance(final_audio, websocket)
                        )
                        active_transcriptions.add(task)
                        task.add_done_callback(lambda t: active_transcriptions.discard(t))
                    
                    # Clear the utterance buffer
                    audio_processor.clear_utterance()
                    
                    # Notify auto-stop
                    await broadcast_to_audio_clients({
                        'type': 'recording_auto_stopped',
                        'reason': 'speech_ended',
                        'timestamp': datetime.now().isoformat()
                    })
        
        elif msg_type == 'audio_end':
            # Manual end - get final utterance if any
            if thai_asr and thai_asr.model is not None and audio_processor:
                logger.info("🔚 Manual recording end")
                
                final_audio = audio_processor.get_final_audio()
                if final_audio is not None:
                    # Final transcription
                    task = asyncio.create_task(
                        transcribe_final_utterance(final_audio, websocket)
                    )
                    active_transcriptions.add(task)
                    task.add_done_callback(lambda t: active_transcriptions.discard(t))
                
                # Clear everything
                audio_processor.clear_utterance()
        
        elif msg_type == 'get_recording_stats':
            if audio_processor:
                stats = audio_processor.get_recording_stats()
                await websocket.send(json.dumps({
                    'type': 'recording_stats',
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                }))
    
    except Exception as e:
        logger.error(f"Error handling audio message: {e}")
        try:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Audio processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }))
        except:
            pass

async def transcribe_realtime_chunk(audio_data: np.ndarray, websocket: WebSocketServerProtocol):
    """Enhanced real-time transcription with RealtimeSTT optimizations"""
    async with transcription_semaphore:
        transcription_start = time.time()
        try:
            # Quick quality check (RealtimeSTT technique)
            if len(audio_data) < server_config['min_length_of_recording'] * SAMPLE_RATE:
                return
            
            # Fast transcription with minimal beam search
            result = await transcribe_audio_data_fast(audio_data)
            if result and result.strip():
                processing_time = time.time() - transcription_start
                
                # Quality filtering (RealtimeSTT approach)
                confidence = estimate_transcription_confidence(result)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    await broadcast_to_audio_clients({
                        'type': 'realtime_transcription',
                        'text': result,
                        'timestamp': datetime.now().isoformat(),
                        'is_final': False,
                        'chunk_duration': len(audio_data) / SAMPLE_RATE,
                        'processing_time': processing_time,
                        'confidence': confidence,
                        'real_time_factor': processing_time / (len(audio_data) / SAMPLE_RATE)
                    })
                    
                    # Store for trend analysis
                    recent_transcriptions.append({
                        'text': result,
                        'confidence': confidence,
                        'processing_time': processing_time,
                        'timestamp': time.time()
                    })
                    
                    logger.debug(f"📝 Real-time: '{result}' (conf: {confidence:.2f}, RTF: {processing_time / (len(audio_data) / SAMPLE_RATE):.2f})")
        except Exception as e:
            logger.warning(f"Real-time transcription error: {e}")
            performance_stats['failed_transcriptions'] += 1

async def transcribe_final_utterance(audio_data: np.ndarray, websocket: WebSocketServerProtocol):
    """Enhanced final transcription with RealtimeSTT optimizations"""
    async with transcription_semaphore:
        try:
            start_time = time.time()
            
            # Use higher quality settings for final transcription
            result = await transcribe_audio_data_full_enhanced(audio_data)
            processing_time = time.time() - start_time
            
            final_text = result.get('text', '').strip()
            if final_text:
                # Enhanced result with RealtimeSTT metadata
                enhanced_result = {
                    'type': 'final_transcription',
                    'text': final_text,
                    'duration': result.get('duration', 0),
                    'processing_time': processing_time,
                    'model': current_model_id,
                    'timestamp': datetime.now().isoformat(),
                    'is_final': True,
                    'confidence': result.get('confidence', 0.0),
                    'language_probability': result.get('language_probability', 0.0),
                    'detected_language': result.get('detected_language', 'th'),
                    'real_time_factor': processing_time / result.get('duration', 1),
                    'segments_count': result.get('segments_count', 1),
                    'quality_score': result.get('quality_score', 0.0)
                }
                
                await broadcast_to_audio_clients(enhanced_result)
                
                # Update session statistics
                recording_state['session_stats']['utterances_processed'] += 1
                recording_state['session_stats']['total_audio_duration'] += result.get('duration', 0)
                
                performance_stats['successful_transcriptions'] += 1
                logger.info(f"🎯 Final: '{final_text}' ({processing_time:.2f}s, RTF: {enhanced_result['real_time_factor']:.2f})")
            else:
                logger.debug("No final text transcribed")
                
        except Exception as e:
            logger.error(f"Final transcription error: {e}")
            performance_stats['failed_transcriptions'] += 1

async def transcribe_realtime_parallel(audio_data: np.ndarray, websocket: WebSocketServerProtocol):
    """Legacy function - redirects to new chunk-based transcription"""
    await transcribe_realtime_chunk(audio_data, websocket)

async def transcribe_final_parallel(audio_data: np.ndarray, websocket: WebSocketServerProtocol, auto_stop: bool = False):
    """Legacy function - redirects to new utterance-based transcription"""
    await transcribe_final_utterance(audio_data, websocket)

async def transcribe_audio_data_fast(audio_data: np.ndarray) -> Optional[str]:
    """Fast transcription for real-time feedback (RealtimeSTT optimized)"""
    global thai_asr
    
    try:
        if not thai_asr or not thai_asr.model:
            return None
        
        # Run transcription in thread pool with minimal settings for speed
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            transcription_executor, _transcribe_sync_fast, audio_data, thai_asr
        )
        
        return result.get('text', '').strip() if result else None
        
    except Exception as e:
        logger.warning(f"Fast transcription error: {e}")
        return None

async def transcribe_audio_data_full_enhanced(audio_data: np.ndarray) -> Dict[str, Any]:
    """Enhanced full transcription with RealtimeSTT quality analysis"""
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
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                transcription_executor, _transcribe_sync_enhanced, audio_data, thai_asr
            )
            
            processing_time = time.time() - start_time
            
            # Add enhanced timing and quality information (RealtimeSTT approach)
            if result:
                result.update({
                    'processing_time': processing_time,
                    'duration': audio_duration,
                    'real_time_factor': processing_time / audio_duration if audio_duration > 0 else 0,
                    'audio_samples': len(audio_data),
                    'timestamp': datetime.now().isoformat(),
                    'quality_score': estimate_transcription_confidence(result.get('text', '')),
                    'segments_count': len(result.get('segments', [])) if 'segments' in result else 1
                })
            
            # Update performance statistics
            await update_performance_stats(processing_time, audio_duration)
            
            return result or {'text': '', 'duration': audio_duration, 'processing_time': processing_time}
            
        finally:
            performance_stats['concurrent_requests'] -= 1
        
    except Exception as e:
        logger.error(f"Enhanced transcription error: {e}")
        performance_stats['error_count'] += 1
        return {'text': '', 'duration': 0, 'processing_time': 0, 'error': str(e)}

def estimate_transcription_confidence(text: str) -> float:
    """Estimate transcription confidence based on text characteristics (RealtimeSTT technique)"""
    try:
        if not text or not text.strip():
            return 0.0
        
        # Simple heuristic-based confidence scoring
        text = text.strip()
        
        # Length-based confidence (longer texts generally more reliable)
        length_score = min(1.0, len(text) / 50.0)
        
        # Character diversity (more diverse = higher confidence)
        unique_chars = len(set(text.lower()))
        diversity_score = min(1.0, unique_chars / 20.0)
        
        # Thai language specific patterns
        thai_chars = sum(1 for c in text if '\u0e00' <= c <= '\u0e7f')
        thai_ratio = thai_chars / len(text) if text else 0
        thai_score = thai_ratio * 0.8  # Boost for Thai content
        
        # Repetition penalty (repeated patterns reduce confidence)
        words = text.split()
        if len(words) > 1:
            unique_words = len(set(words))
            repetition_score = unique_words / len(words)
        else:
            repetition_score = 1.0
        
        # Combined confidence score
        confidence = (length_score * 0.3 + diversity_score * 0.3 + 
                     thai_score * 0.3 + repetition_score * 0.1)
        
        return min(1.0, max(0.0, confidence))
        
    except Exception as e:
        logger.warning(f"Confidence estimation error: {e}")
        return 0.5  # Default moderate confidence

def _transcribe_sync_fast(audio_data: np.ndarray, asr_instance) -> Dict[str, Any]:
    """Synchronous fast transcription for real-time use (RealtimeSTT optimized)"""
    try:
        # Create temporary audio file with optimized settings
        import tempfile
        import wave
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Convert float audio to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Write WAV file with minimal overhead
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Fast transcription with minimal beam search
            if hasattr(asr_instance.model, 'transcribe'):
                segments, info = asr_instance.model.transcribe(
                    tmp_file.name,
                    language="th",
                    beam_size=1,  # Fastest setting
                    temperature=0.0,  # Deterministic
                    condition_on_previous_text=False,  # Faster
                    vad_filter=False,  # Skip VAD for speed
                    word_timestamps=False  # Skip word timing for speed
                )
                text = " ".join(segment.text for segment in segments).strip()
                result = {
                    'text': text,
                    'language': getattr(info, 'language', 'th'),
                    'language_probability': getattr(info, 'language_probability', 0.9)
                }
            else:
                # Fallback for batch model
                result = asr_instance.transcribe_audio_data(audio_data.tobytes())
            
            # Cleanup
            try:
                os.unlink(tmp_file.name)
            except:
                pass
            
            return result
            
    except Exception as e:
        logger.error(f"Sync fast transcription error: {e}")
        return {'text': '', 'error': str(e)}

def _transcribe_sync_enhanced(audio_data: np.ndarray, asr_instance) -> Dict[str, Any]:
    """Synchronous enhanced transcription for final results (RealtimeSTT optimized)"""
    try:
        # Create temporary audio file
        import tempfile
        import wave
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Convert float audio to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Enhanced transcription with better quality settings
            if hasattr(asr_instance.model, 'transcribe'):
                segments, info = asr_instance.model.transcribe(
                    tmp_file.name,
                    language="th",
                    beam_size=server_config['beam_size'],  # Use configured beam size
                    temperature=server_config.get('temperature', 0.0),
                    condition_on_previous_text=server_config.get('condition_on_previous_text', False),
                    vad_filter=server_config.get('faster_whisper_vad_filter', True),
                    word_timestamps=True,  # Get word timing for quality analysis
                    compression_ratio_threshold=server_config.get('compression_ratio_threshold', 2.4),
                    log_prob_threshold=server_config.get('log_prob_threshold', -1.0),
                    no_speech_threshold=server_config.get('no_speech_threshold', 0.6)
                )
                
                text = " ".join(segment.text for segment in segments).strip()
                segments_list = [{"text": seg.text, "start": seg.start, "end": seg.end} for seg in segments]
                
                result = {
                    'text': text,
                    'language': getattr(info, 'language', 'th'),
                    'language_probability': getattr(info, 'language_probability', 0.9),
                    'segments': segments_list,
                    'duration': getattr(info, 'duration', len(audio_data) / SAMPLE_RATE)
                }
            else:
                # Fallback for batch model
                result = asr_instance.transcribe_audio_data(audio_data.tobytes())
            
            # Cleanup
            try:
                os.unlink(tmp_file.name)
            except:
                pass
            
            return result
            
    except Exception as e:
        logger.error(f"Sync enhanced transcription error: {e}")
        return {'text': '', 'error': str(e)}

async def transcribe_audio_data(audio_data: np.ndarray) -> Optional[str]:
    """Legacy function - redirects to fast transcription"""
    return await transcribe_audio_data_fast(audio_data)

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

async def get_enhanced_statistics():
    """Get comprehensive server statistics (RealtimeSTT inspired)"""
    global performance_stats, recording_state, audio_processor
    
    current_time = time.time()
    uptime = current_time - performance_stats['start_time']
    
    # Calculate advanced metrics
    success_rate = 0
    if performance_stats['total_transcriptions'] > 0:
        success_rate = performance_stats['successful_transcriptions'] / performance_stats['total_transcriptions']
    
    # Audio processor metrics
    audio_metrics = {}
    if audio_processor:
        audio_metrics = audio_processor.get_performance_metrics()
    
    # System health assessment
    health_status = "healthy"
    if performance_stats['real_time_factor'] > 1.5:
        health_status = "warning"
    elif performance_stats['real_time_factor'] > 2.0:
        health_status = "critical"
    
    return {
        'uptime_seconds': uptime,
        'uptime_formatted': f"{uptime/3600:.1f}h",
        'total_requests': performance_stats['total_requests'],
        'concurrent_requests': performance_stats['concurrent_requests'],
        'peak_concurrent': performance_stats['peak_concurrent'],
        'avg_processing_time': performance_stats['avg_processing_time'],
        'real_time_factor': performance_stats['real_time_factor'],
        'success_rate': success_rate,
        'error_count': performance_stats['error_count'],
        'successful_transcriptions': performance_stats['successful_transcriptions'],
        'failed_transcriptions': performance_stats['failed_transcriptions'],
        'audio_processed_seconds': performance_stats['total_audio_processed_seconds'],
        'vad_activations': performance_stats['vad_activations'],
        'auto_stops': performance_stats['auto_stops'],
        'buffer_overflows': performance_stats['buffer_overflows'],
        'health_status': health_status,
        'active_transcriptions': len(active_transcriptions),
        'connected_clients': {
            'control': len(control_connections),
            'audio': len(audio_connections)
        },
        'current_model': current_model_id,
        'recording_state': recording_state,
        'audio_processor_metrics': audio_metrics,
        'server_config': {
            'chunk_size': server_config['chunk_size'],
            'sample_rate': server_config['sample_rate'],
            'vad_threshold': server_config['vad_threshold'],
            'max_concurrent': MAX_CONCURRENT_TRANSCRIPTIONS,
            'beam_size': server_config['beam_size']
        }
    }

async def reset_performance_stats():
    """Reset performance statistics (RealtimeSTT feature)"""
    global performance_stats
    
    current_time = time.time()
    performance_stats.update({
        'total_requests': 0,
        'concurrent_requests': 0,
        'avg_processing_time': 0.0,
        'peak_concurrent': 0,
        'total_audio_processed_seconds': 0.0,
        'total_transcriptions': 0,
        'successful_transcriptions': 0,
        'failed_transcriptions': 0,
        'error_count': 0,
        'vad_activations': 0,
        'auto_stops': 0,
        'buffer_overflows': 0,
        'last_reset_time': current_time
    })
    
    # Reset session stats
    recording_state['session_stats'] = {
        'utterances_processed': 0,
        'avg_utterance_length': 0.0,
        'total_audio_duration': 0.0
    }
    
    logger.info("📊 Performance statistics reset")

async def optimize_server_performance():
    """Dynamic server optimization (RealtimeSTT technique)"""
    global audio_processor, server_config
    
    try:
        # Analyze recent performance
        current_rtf = performance_stats.get('real_time_factor', 0.0)
        error_rate = performance_stats.get('error_count', 0) / max(1, performance_stats.get('total_requests', 1))
        
        # Adaptive optimization based on performance
        if current_rtf > 1.5:  # Too slow
            # Reduce quality for speed
            if server_config['beam_size'] > 1:
                server_config['beam_size'] = 1
                logger.info("🔧 Reduced beam size for speed optimization")
            
            # Increase VAD threshold (more aggressive)
            if audio_processor and server_config['vad_threshold'] < 0.03:
                new_threshold = min(0.03, server_config['vad_threshold'] * 1.2)
                audio_processor.adjust_sensitivity(new_threshold)
                server_config['vad_threshold'] = new_threshold
                logger.info(f"🔧 Increased VAD threshold to {new_threshold:.3f}")
            
            recording_state['processing_mode'] = 'fast'
            
        elif current_rtf < 0.5 and error_rate < 0.05:  # Good performance, can improve quality
            # Increase quality
            if server_config['beam_size'] < 3:
                server_config['beam_size'] = min(3, server_config['beam_size'] + 1)
                logger.info("🔧 Increased beam size for quality optimization")
            
            recording_state['processing_mode'] = 'normal'
        
        # Update buffer health status
        if performance_stats.get('buffer_overflows', 0) > 5:
            recording_state['buffer_health'] = 'critical'
        elif performance_stats.get('buffer_overflows', 0) > 2:
            recording_state['buffer_health'] = 'warning'
        else:
            recording_state['buffer_health'] = 'good'
        
        logger.debug(f"🔧 Performance optimization: RTF={current_rtf:.2f}, Mode={recording_state['processing_mode']}")
        
    except Exception as e:
        logger.warning(f"Performance optimization error: {e}")

# Background task for periodic optimization
async def performance_monitor():
    """Background task for performance monitoring (RealtimeSTT inspired)"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            await optimize_server_performance()
            
            # Log periodic statistics
            stats = await get_enhanced_statistics()
            if stats['total_requests'] > 0:
                logger.info(f"📊 Stats: {stats['total_requests']} requests, "
                           f"RTF: {stats['real_time_factor']:.2f}, "
                           f"Success: {stats['success_rate']:.1%}, "
                           f"Health: {stats['health_status']}")
        
        except Exception as e:
            logger.warning(f"Performance monitor error: {e}")
            await asyncio.sleep(5)  # Short delay before retry

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
    """Start both WebSocket servers with RealtimeSTT optimizations"""
    global model_manager, audio_processor, transcription_executor
    
    logger.info("🚀 Starting Enhanced Thai ASR WebSocket servers (RealtimeSTT optimized)...")
    
    # Initialize thread pool executor for transcriptions (RealtimeSTT technique)
    import concurrent.futures
    transcription_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_TRANSCRIPTIONS,
        thread_name_prefix="ASR-Worker"
    )
    logger.info(f"🔧 Thread pool initialized with {MAX_CONCURRENT_TRANSCRIPTIONS} workers")
    
    # Initialize model manager if not already done
    if model_manager is None:
        logger.info("📦 Initializing model manager...")
        model_manager = ModelManager()
        models = model_manager.get_available_models()
        model_names = [model.get('name', 'Unknown') for model in models]
        logger.info(f"📦 Loaded {len(models)} models: {', '.join(model_names)}")
    
    # Initialize enhanced audio processor with RealtimeSTT optimizations
    if audio_processor is None:
        logger.info("🎤 Initializing enhanced audio processor...")
        audio_processor = AudioProcessor(
            sample_rate=server_config['sample_rate'],
            chunk_size=server_config['chunk_size']
        )
        
        # Apply dynamic configuration
        audio_processor.adjust_sensitivity(server_config['vad_threshold'])
        logger.info(f"🎤 Audio processor configured with {server_config['chunk_size']} chunk size")
    
    # Initialize the default model with GPU optimization
    await load_model(current_model_id)
    
    # Start control server
    control_server = await websockets.serve(
        control_handler,
        "0.0.0.0",  # Listen on all interfaces for ngrok
        CONTROL_PORT,
        ping_interval=20,
        ping_timeout=10,
        max_size=None,  # No message size limit
        compression=None  # Disable compression for speed
    )
    logger.info(f"🔧 Control server started on ws://0.0.0.0:{CONTROL_PORT}")
    
    # Start audio server with optimized settings
    audio_server = await websockets.serve(
        audio_handler,
        "0.0.0.0",  # Listen on all interfaces for ngrok
        AUDIO_PORT,
        ping_interval=20,
        ping_timeout=10,
        max_size=None,  # No message size limit for audio data
        compression=None,  # Disable compression for speed
        max_queue=ALLOWED_LATENCY_LIMIT  # Limit queue size for low latency
    )
    logger.info(f"🎵 Audio server started on ws://0.0.0.0:{AUDIO_PORT}")
    
    logger.info("✅ Enhanced Thai ASR WebSocket servers are ready!")
    logger.info("📊 RealtimeSTT Optimizations Applied:")
    logger.info(f"   🔹 Reduced chunk size: {server_config['chunk_size']} samples")
    logger.info(f"   🔹 Concurrent transcriptions: {MAX_CONCURRENT_TRANSCRIPTIONS}")
    logger.info(f"   🔹 Enhanced VAD with multi-method detection")
    logger.info(f"   🔹 Adaptive audio processing with normalization")
    logger.info(f"   🔹 Smart buffering with backpressure control")
    logger.info(f"   🔹 Performance monitoring and auto-tuning")
    logger.info("📖 Usage:")
    logger.info(f"   - Control: ws://0.0.0.0:{CONTROL_PORT}")
    logger.info(f"   - Audio: ws://0.0.0.0:{AUDIO_PORT}")
    logger.info("🌐 Use ngrok to expose these ports for external access")
    
    # Start background performance monitor (RealtimeSTT optimization)
    monitor_task = asyncio.create_task(performance_monitor())
    background_tasks.add(monitor_task)
    monitor_task.add_done_callback(background_tasks.discard)
    logger.info("📊 Performance monitor started")
    
    # Keep servers running
    try:
        await asyncio.gather(
            control_server.wait_closed(),
            audio_server.wait_closed()
        )
    finally:
        # Cleanup tasks
        for task in background_tasks:
            task.cancel()
        
        # Cleanup executor
        if transcription_executor:
            transcription_executor.shutdown(wait=True)
            logger.info("🔧 Thread pool executor shut down")

def main():
    """Main function with RealtimeSTT optimizations"""
    try:
        # Set up Windows event loop policy if needed
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        print("🎤 Enhanced Thai ASR Real-time WebSocket Server (RealtimeSTT Optimized)")
        print("=" * 80)
        print(f"🚀 Starting optimized servers...")
        print(f"🔧 Control WebSocket: ws://0.0.0.0:{CONTROL_PORT}")
        print(f"🎵 Audio WebSocket: ws://0.0.0.0:{AUDIO_PORT}")
        print(f"📦 Default Model: {current_model_id}")
        print("🌐 Use ngrok to expose ports for external access")
        print()
        print("✨ REALTIMESTT OPTIMIZATIONS APPLIED:")
        print("   🔹 Reduced latency with 512-sample chunks")
        print("   🔹 Enhanced multi-method VAD (RMS + ZCR + Peak)")
        print("   🔹 Smart audio normalization and preprocessing")
        print("   🔹 Adaptive transcription with confidence scoring")
        print("   🔹 Dynamic performance optimization")
        print("   🔹 Advanced buffer management with backpressure")
        print("   🔹 Thread pool optimization for parallel processing")
        print("   🔹 Real-time performance monitoring")
        print("   🔹 Auto-tuning based on system performance")
        print()
        print("📊 PERFORMANCE FEATURES:")
        print("   🔹 Real-time factor monitoring")
        print("   🔹 Automatic quality/speed balancing")
        print("   🔹 Buffer overflow protection")
        print("   🔹 Comprehensive statistics tracking")
        print("   🔹 Health status monitoring")
        print()
        print("📝 Available models:")
        print("   - biodatlab-large-faster: Vinxscribe/biodatlab-whisper-th-large-v3-faster")
        print("   - biodatlab-medium-faster: Vinxscribe/biodatlab-whisper-th-medium-faster") 
        print("   - pathumma-large: PathummaApiwat/Pathumma-whisper-large-v3-th")
        print("=" * 80)
        
        # Run the servers
        asyncio.run(start_servers())
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise

if __name__ == "__main__":
    main()
