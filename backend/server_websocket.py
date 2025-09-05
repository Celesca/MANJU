#!/usr/bin/env python3
"""
WebSocket-based Thai ASR Server
Real-time Thai speech recognition with WebSocket streaming
Optimized with RealtimeSTT techniques for low-latency performance
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
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
import tempfile
import numpy as np
from collections import deque

# WebSocket imports
import websockets
from websockets.server import WebSocketServerProtocol

# Audio processing
import pyaudio
from scipy.signal import butter, filtfilt, resample_poly

# Add backend to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our optimized components
try:
    from whisper.model_manager import get_model_manager, ModelManager
    from whisper.faster_whisper_thai import FasterWhisperThai, WhisperConfig
    from realtime_thai_asr_server import AudioProcessor, MAX_CONCURRENT_TRANSCRIPTIONS
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Audio configuration (RealtimeSTT optimized)
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Reduced for lower latency
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
BUFFER_SIZE = 4096
INT16_MAX_ABS_VALUE = 32768.0

# WebSocket ports
CONTROL_PORT = 8765
AUDIO_PORT = 8766

# Global state
model_manager: Optional[ModelManager] = None
audio_processor: Optional[AudioProcessor] = None
control_connections: Set[WebSocketServerProtocol] = set()
audio_connections: Set[WebSocketServerProtocol] = set()
connected_clients: Dict[str, Dict] = {}

# Performance tracking
performance_stats = {
    'total_requests': 0,
    'concurrent_requests': 0,
    'avg_processing_time': 0.0,
    'peak_concurrent': 0,
    'start_time': time.time(),
    'total_audio_processed_seconds': 0.0,
    'total_transcriptions': 0,
    'real_time_factor': 0.0,
    'successful_transcriptions': 0,
    'failed_transcriptions': 0,
    'error_count': 0
}

# Thread pool for transcriptions
transcription_executor = None
transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)
active_transcriptions = set()
background_tasks = set()

class WSASRServer:
    """WebSocket-based ASR Server with RealtimeSTT optimizations"""

    def __init__(self):
        self.model_manager = None
        self.audio_processor = None
        self.performance_stats = performance_stats
        self.start_time = time.time()

    async def initialize(self):
        """Initialize the ASR server components"""
        logger.info("🚀 Initializing WebSocket ASR Server...")

        # Initialize model manager
        if self.model_manager is None:
            logger.info("📦 Initializing model manager...")
            self.model_manager = get_model_manager()
            models = self.model_manager.get_available_models()
            model_names = [model.get('name', 'Unknown') for model in models]
            logger.info(f"📦 Loaded {len(models)} models: {', '.join(model_names)}")

        # Initialize enhanced audio processor
        if self.audio_processor is None:
            logger.info("🎤 Initializing enhanced audio processor...")
            self.audio_processor = AudioProcessor(
                sample_rate=SAMPLE_RATE,
                chunk_size=CHUNK_SIZE
            )
            logger.info(f"🎤 Audio processor configured with {CHUNK_SIZE} chunk size")

        # Load default model
        await self.load_model("biodatlab-medium-faster")

        logger.info("✅ WebSocket ASR Server initialized successfully!")

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load ASR model with error handling"""
        try:
            logger.info(f"📦 Loading model: {model_id}")
            self.model_manager.load_model(model_id)
            model_info = self.model_manager.get_current_model_info()

            logger.info(f"✅ Model {model_id} loaded successfully")
            return {
                "status": "success",
                "model_id": model_id,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {e}")
            return {
                "status": "error",
                "model_id": model_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def transcribe_audio_data(self, audio_data: bytes, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Transcribe audio data with RealtimeSTT optimizations"""
        async with transcription_semaphore:
            start_time = time.time()

            try:
                # Default configuration
                if config is None:
                    config = {
                        "language": "th",
                        "beam_size": 1,
                        "use_vad": True
                    }

                # Track performance
                self.performance_stats['concurrent_requests'] += 1
                self.performance_stats['peak_concurrent'] = max(
                    self.performance_stats['peak_concurrent'],
                    self.performance_stats['concurrent_requests']
                )

                # Create temporary WAV file
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_duration = len(audio_np) / SAMPLE_RATE

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    # Write WAV header and data
                    import wave
                    with wave.open(tmp_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(CHANNELS)
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(SAMPLE_RATE)
                        wav_file.writeframes(audio_data)

                    try:
                        # Transcribe using model manager
                        result = self.model_manager.transcribe_with_current_model(tmp_file.name)

                        processing_time = time.time() - start_time

                        # Enhanced result with RealtimeSTT metrics
                        enhanced_result = {
                            "text": result["text"],
                            "language": result["language"],
                            "duration": result["duration"],
                            "processing_time": processing_time,
                            "speed_ratio": result["speed_ratio"],
                            "chunks_processed": result["chunks_processed"],
                            "model": result["model"],
                            "device": result["device"],
                            "real_time_factor": processing_time / audio_duration if audio_duration > 0 else 0,
                            "confidence": self._estimate_confidence(result["text"]),
                            "timestamp": datetime.now().isoformat(),
                            "status": "success"
                        }

                        # Update performance stats
                        self.performance_stats['successful_transcriptions'] += 1
                        await self._update_performance_stats(processing_time, audio_duration)

                        logger.info(f"✅ Transcription completed: {len(result['text'])} chars, RTF: {enhanced_result['real_time_factor']:.2f}")
                        return enhanced_result

                    finally:
                        # Cleanup temp file
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass

            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"❌ Transcription failed: {e}")
                self.performance_stats['failed_transcriptions'] += 1
                self.performance_stats['error_count'] += 1

                return {
                    "text": "",
                    "error": str(e),
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                }
            finally:
                self.performance_stats['concurrent_requests'] -= 1

    def _estimate_confidence(self, text: str) -> float:
        """Estimate transcription confidence (RealtimeSTT technique)"""
        try:
            if not text or not text.strip():
                return 0.0

            text = text.strip()
            length_score = min(1.0, len(text) / 50.0)
            unique_chars = len(set(text.lower()))
            diversity_score = min(1.0, unique_chars / 20.0)
            thai_chars = sum(1 for c in text if '\u0e00' <= c <= '\u0e7f')
            thai_ratio = thai_chars / len(text) if text else 0
            thai_score = thai_ratio * 0.8

            confidence = (length_score * 0.3 + diversity_score * 0.3 + thai_score * 0.4)
            return min(1.0, max(0.0, confidence))

        except Exception:
            return 0.5

    async def _update_performance_stats(self, processing_time: float, audio_duration: float):
        """Update performance statistics"""
        self.performance_stats['total_requests'] += 1
        self.performance_stats['total_transcriptions'] += 1
        self.performance_stats['total_audio_processed_seconds'] += audio_duration

        # Update average processing time
        total_requests = self.performance_stats['total_requests']
        current_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (current_avg * (total_requests - 1) + processing_time) / total_requests

        # Calculate real-time factor
        if audio_duration > 0:
            rtf = processing_time / audio_duration
            self.performance_stats['real_time_factor'] = rtf

    async def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time

        return {
            "server": {
                "uptime_seconds": uptime,
                "uptime_formatted": f"{uptime/3600:.1f}h",
                "current_model": self.model_manager.get_current_model_info() if self.model_manager else None,
                "connected_clients": {
                    "control": len(control_connections),
                    "audio": len(audio_connections),
                    "total": len(control_connections) + len(audio_connections)
                },
                "version": "1.0.0"
            },
            "performance": self.performance_stats.copy(),
            "audio_processor": self.audio_processor.get_performance_metrics() if self.audio_processor else {},
            "timestamp": datetime.now().isoformat()
        }

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        if self.model_manager is None:
            return []

        models = self.model_manager.get_available_models()
        return models

# Global server instance
asr_server = WSASRServer()

# WebSocket message handlers
async def handle_control_message(websocket: WebSocketServerProtocol, message: Dict[str, Any]):
    """Handle control WebSocket messages"""
    global asr_server

    try:
        msg_type = message.get('type')

        if msg_type == 'load_model':
            model_id = message.get('model_id', 'biodatlab-medium-faster')
            result = await asr_server.load_model(model_id)
            await websocket.send(json.dumps(result))

        elif msg_type == 'get_models':
            models = await asr_server.get_available_models()
            await websocket.send(json.dumps({
                'type': 'available_models',
                'models': models,
                'timestamp': datetime.now().isoformat()
            }))

        elif msg_type == 'get_stats':
            stats = await asr_server.get_server_stats()
            await websocket.send(json.dumps({
                'type': 'server_stats',
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }))

        elif msg_type == 'health_check':
            stats = await asr_server.get_server_stats()
            health_status = "healthy"
            if stats['performance']['real_time_factor'] > 1.5:
                health_status = "warning"
            elif stats['performance']['real_time_factor'] > 2.0:
                health_status = "critical"

            await websocket.send(json.dumps({
                'type': 'health_status',
                'status': health_status,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }))

        elif msg_type == 'ping':
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }))

        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {msg_type}',
                'timestamp': datetime.now().isoformat()
            }))

    except Exception as e:
        logger.error(f"Error handling control message: {e}")
        await websocket.send(json.dumps({
            'type': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }))

async def handle_audio_message(websocket: WebSocketServerProtocol, message: Dict[str, Any]):
    """Handle audio WebSocket messages"""
    global asr_server

    try:
        msg_type = message.get('type')

        if msg_type == 'transcribe':
            # Handle single transcription request
            audio_b64 = message.get('audio_data', '')
            config = message.get('config', {})

            if not audio_b64:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'No audio data provided',
                    'timestamp': datetime.now().isoformat()
                }))
                return

            try:
                audio_bytes = base64.b64decode(audio_b64)
                result = await asr_server.transcribe_audio_data(audio_bytes, config)

                await websocket.send(json.dumps({
                    'type': 'transcription_result',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }))

            except Exception as e:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Transcription failed: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }))

        elif msg_type == 'start_stream':
            # Handle streaming transcription start
            session_id = message.get('session_id', f"session_{int(time.time())}")
            config = message.get('config', {})

            connected_clients[session_id] = {
                'websocket': websocket,
                'config': config,
                'start_time': time.time(),
                'chunks_received': 0,
                'last_activity': time.time()
            }

            await websocket.send(json.dumps({
                'type': 'stream_started',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }))

        elif msg_type == 'audio_chunk':
            # Handle streaming audio chunk
            session_id = message.get('session_id')
            audio_b64 = message.get('audio_data')

            if not session_id or session_id not in connected_clients:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid or missing session_id',
                    'timestamp': datetime.now().isoformat()
                }))
                return

            if not audio_b64:
                return

            try:
                audio_bytes = base64.b64decode(audio_b64)
                session = connected_clients[session_id]
                session['chunks_received'] += 1
                session['last_activity'] = time.time()

                # Process with audio processor if available
                if asr_server.audio_processor:
                    has_speech, should_stop, realtime_audio = asr_server.audio_processor.add_chunk(audio_bytes)

                    if realtime_audio is not None:
                        # Real-time transcription
                        result = await asr_server.transcribe_audio_data(realtime_audio.tobytes(), session['config'])

                        if result['text'].strip():
                            await websocket.send(json.dumps({
                                'type': 'realtime_transcription',
                                'session_id': session_id,
                                'result': result,
                                'timestamp': datetime.now().isoformat()
                            }))

                    if should_stop:
                        # Final transcription
                        final_audio = asr_server.audio_processor.get_final_audio()
                        if final_audio is not None:
                            result = await asr_server.transcribe_audio_data(final_audio.tobytes(), session['config'])

                            await websocket.send(json.dumps({
                                'type': 'final_transcription',
                                'session_id': session_id,
                                'result': result,
                                'timestamp': datetime.now().isoformat()
                            }))

                        asr_server.audio_processor.clear_utterance()

            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")

        elif msg_type == 'end_stream':
            # Handle streaming end
            session_id = message.get('session_id')

            if session_id and session_id in connected_clients:
                session = connected_clients[session_id]
                duration = time.time() - session['start_time']

                await websocket.send(json.dumps({
                    'type': 'stream_ended',
                    'session_id': session_id,
                    'duration': duration,
                    'chunks_processed': session['chunks_received'],
                    'timestamp': datetime.now().isoformat()
                }))

                # Cleanup session
                del connected_clients[session_id]

        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {msg_type}',
                'timestamp': datetime.now().isoformat()
            }))

    except Exception as e:
        logger.error(f"Error handling audio message: {e}")
        try:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }))
        except:
            pass

# WebSocket handlers
async def control_handler(websocket: WebSocketServerProtocol):
    """Handle control WebSocket connections"""
    logger.info(f"🔌 Control client connected: {websocket.remote_address}")
    control_connections.add(websocket)

    try:
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'connected',
            'server': 'WebSocket Thai ASR Server',
            'version': '1.0.0',
            'features': ['realtime_transcription', 'streaming', 'model_switching', 'performance_monitoring'],
            'timestamp': datetime.now().isoformat()
        }))

        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_control_message(websocket, data)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format',
                    'timestamp': datetime.now().isoformat()
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
            'server': 'WebSocket Thai ASR Audio Stream',
            'sample_rate': SAMPLE_RATE,
            'channels': CHANNELS,
            'chunk_size': CHUNK_SIZE,
            'supported_formats': ['base64'],
            'timestamp': datetime.now().isoformat()
        }))

        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_audio_message(websocket, data)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format',
                    'timestamp': datetime.now().isoformat()
                }))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"🎵 Audio client disconnected: {websocket.remote_address}")
    finally:
        audio_connections.discard(websocket)

async def start_websocket_servers():
    """Start both WebSocket servers"""
    global transcription_executor

    logger.info("🚀 Starting WebSocket ASR Servers...")

    # Initialize thread pool executor
    import concurrent.futures
    transcription_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_TRANSCRIPTIONS,
        thread_name_prefix="ASR-Worker"
    )
    logger.info(f"🔧 Thread pool initialized with {MAX_CONCURRENT_TRANSCRIPTIONS} workers")

    # Initialize ASR server
    await asr_server.initialize()

    # Start control server
    control_server = await websockets.serve(
        control_handler,
        "0.0.0.0",
        CONTROL_PORT,
        ping_interval=20,
        ping_timeout=10,
        max_size=None,
        compression=None
    )
    logger.info(f"🔧 Control server started on ws://0.0.0.0:{CONTROL_PORT}")

    # Start audio server
    audio_server = await websockets.serve(
        audio_handler,
        "0.0.0.0",
        AUDIO_PORT,
        ping_interval=20,
        ping_timeout=10,
        max_size=None,
        compression=None
    )
    logger.info(f"🎵 Audio server started on ws://0.0.0.0:{AUDIO_PORT}")

    logger.info("✅ WebSocket ASR Servers are ready!")
    logger.info("📊 RealtimeSTT Optimizations Applied:")
    logger.info(f"   🔹 Reduced chunk size: {CHUNK_SIZE} samples")
    logger.info(f"   🔹 Concurrent transcriptions: {MAX_CONCURRENT_TRANSCRIPTIONS}")
    logger.info(f"   🔹 Enhanced VAD with multi-method detection")
    logger.info(f"   🔹 Smart buffering with backpressure control")
    logger.info(f"   🔹 Performance monitoring and auto-tuning")
    logger.info("📖 Usage:")
    logger.info(f"   - Control: ws://0.0.0.0:{CONTROL_PORT}")
    logger.info(f"   - Audio: ws://0.0.0.0:{AUDIO_PORT}")
    logger.info("🌐 Use ngrok to expose these ports for external access")

    # Start background performance monitor
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
        # Cleanup
        for task in background_tasks:
            task.cancel()

        if transcription_executor:
            transcription_executor.shutdown(wait=True)
            logger.info("🔧 Thread pool executor shut down")

async def performance_monitor():
    """Background performance monitoring task"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            stats = await asr_server.get_server_stats()

            if stats['performance']['total_requests'] > 0:
                logger.info(f"📊 Stats: {stats['performance']['total_requests']} requests, "
                           f"RTF: {stats['performance']['real_time_factor']:.2f}, "
                           f"Success: {stats['performance']['successful_transcriptions']}/{stats['performance']['total_transcriptions']}")

        except Exception as e:
            logger.warning(f"Performance monitor error: {e}")
            await asyncio.sleep(5)

def main():
    """Main function"""
    try:
        # Set up Windows event loop policy if needed
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        print("🎤 WebSocket Thai ASR Server (RealtimeSTT Optimized)")
        print("=" * 70)
        print(f"🚀 Starting servers...")
        print(f"🔧 Control WebSocket: ws://0.0.0.0:{CONTROL_PORT}")
        print(f"🎵 Audio WebSocket: ws://0.0.0.0:{AUDIO_PORT}")
        print("🌐 Use ngrok to expose ports for external access")
        print("✨ FEATURES:")
        print("   ✅ Real-time streaming transcription")
        print("   ✅ RealtimeSTT optimizations applied")
        print("   ✅ Multi-model support with hot switching")
        print("   ✅ Performance monitoring and health checks")
        print("   ✅ Enhanced VAD and audio processing")
        print("   ✅ Concurrent transcription support")
        print("=" * 70)

        # Run the servers
        asyncio.run(start_websocket_servers())

    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise

if __name__ == "__main__":
    main()
