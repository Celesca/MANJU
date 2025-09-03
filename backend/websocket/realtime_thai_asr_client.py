#!/usr/bin/env python3
"""
Real-time Thai ASR WebSocket Client Example
Demonstrates how to connect to the Thai ASR WebSocket server and stream audio
"""

import asyncio
import websockets
import json
import pyaudio
import base64
import logging
import signal
import sys
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1

class ThaiASRClient:
    """WebSocket client for Thai ASR server"""
    
    def __init__(self, control_port: int = 8765, audio_port: int = 8766):
        self.control_url = f"ws://localhost:{control_port}"
        self.audio_url = f"ws://localhost:{audio_port}"
        self.control_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.audio_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.is_recording = False
        self.running = True
        
    async def connect(self):
        """Connect to both WebSocket servers"""
        try:
            logger.info("🔌 Connecting to Thai ASR server...")
            
            # Connect to control server
            self.control_ws = await websockets.connect(self.control_url)
            logger.info("✅ Connected to control server")
            
            # Connect to audio server
            self.audio_ws = await websockets.connect(self.audio_url)
            logger.info("✅ Connected to audio server")
            
            # Initialize audio
            self.setup_audio()
            
            logger.info("🎤 Ready for audio streaming!")
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            raise
    
    def setup_audio(self):
        """Initialize PyAudio for recording"""
        try:
            self.audio_interface = pyaudio.PyAudio()
            
            # List available input devices
            logger.info("📱 Available audio input devices:")
            for i in range(self.audio_interface.get_device_count()):
                device_info = self.audio_interface.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    logger.info(f"   {i}: {device_info.get('name')}")
            
            # Open audio stream
            self.audio_stream = self.audio_interface.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            logger.info("🎵 Audio system initialized")
            
        except Exception as e:
            logger.error(f"❌ Audio setup failed: {e}")
            raise
    
    async def send_control_message(self, message: dict):
        """Send a control message"""
        if self.control_ws:
            await self.control_ws.send(json.dumps(message))
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to server"""
        if self.audio_ws:
            # Encode audio data as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            message = {
                'type': 'audio_chunk',
                'data': audio_b64
            }
            
            await self.audio_ws.send(json.dumps(message))
    
    async def send_audio_end(self):
        """Signal end of audio stream"""
        if self.audio_ws:
            message = {'type': 'audio_end'}
            await self.audio_ws.send(json.dumps(message))
    
    async def start_recording(self):
        """Start recording and streaming audio"""
        try:
            logger.info("🎙️ Starting recording...")
            
            # Send start recording command
            await self.send_control_message({'type': 'start_recording'})
            self.is_recording = True
            
            # Start streaming audio
            while self.is_recording and self.running:
                try:
                    # Read audio chunk
                    audio_data = self.audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    
                    # Send to server
                    await self.send_audio_chunk(audio_data)
                    
                    # Small delay to prevent overwhelming the server
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
                    break
            
            # Send end signal
            await self.send_audio_end()
            
        except Exception as e:
            logger.error(f"❌ Recording error: {e}")
    
    async def stop_recording(self):
        """Stop recording"""
        logger.info("🛑 Stopping recording...")
        self.is_recording = False
        await self.send_control_message({'type': 'stop_recording'})
    
    async def listen_to_responses(self):
        """Listen to server responses"""
        try:
            # Listen to control messages
            async def control_listener():
                if self.control_ws:
                    async for message in self.control_ws:
                        try:
                            data = json.loads(message)
                            await self.handle_control_response(data)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON from control server")
            
            # Listen to audio messages
            async def audio_listener():
                if self.audio_ws:
                    async for message in self.audio_ws:
                        try:
                            data = json.loads(message)
                            await self.handle_audio_response(data)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON from audio server")
            
            # Run both listeners concurrently
            await asyncio.gather(
                control_listener(),
                audio_listener(),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"❌ Response listener error: {e}")
    
    async def handle_control_response(self, data: dict):
        """Handle control server responses"""
        msg_type = data.get('type')
        
        if msg_type == 'connected':
            logger.info(f"🤝 {data.get('server', 'Server')} connected")
            logger.info(f"📦 Current model: {data.get('current_model', 'Unknown')}")
        
        elif msg_type == 'model_loaded':
            logger.info(f"✅ Model loaded: {data.get('model_id')}")
        
        elif msg_type == 'error':
            logger.error(f"❌ Control error: {data.get('message')}")
        
        else:
            logger.debug(f"Control response: {data}")
    
    async def handle_audio_response(self, data: dict):
        """Handle audio server responses"""
        msg_type = data.get('type')
        
        if msg_type == 'audio_connected':
            logger.info(f"🎵 Audio server connected")
            logger.info(f"📊 Audio config: {data.get('sample_rate')}Hz, {data.get('channels')} channel(s)")
        
        elif msg_type == 'realtime_transcription':
            text = data.get('text', '')
            if text.strip():
                print(f"\\r🎯 Realtime: {text}", end='', flush=True)
        
        elif msg_type == 'final_transcription':
            text = data.get('text', '')
            duration = data.get('duration', 0)
            processing_time = data.get('processing_time', 0)
            speed_ratio = data.get('speed_ratio', 0)
            
            print()  # New line
            print(f"✅ Final: {text}")
            print(f"⏱️ Duration: {duration:.2f}s, Processing: {processing_time:.2f}s, Speed: {speed_ratio:.1f}x")
            print()
        
        elif msg_type == 'recording_started':
            logger.info("🎙️ Server confirmed recording started")
        
        elif msg_type == 'recording_stopped':
            logger.info("🛑 Server confirmed recording stopped")
        
        else:
            logger.debug(f"Audio response: {data}")
    
    async def get_available_models(self):
        """Get list of available models"""
        await self.send_control_message({'type': 'get_models'})
    
    async def load_model(self, model_id: str):
        """Load a specific model"""
        logger.info(f"📦 Loading model: {model_id}")
        await self.send_control_message({
            'type': 'load_model',
            'model_id': model_id
        })
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            
            if self.audio_interface:
                self.audio_interface.terminate()
            
            logger.info("🧹 Resources cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def close(self):
        """Close WebSocket connections"""
        self.running = False
        
        if self.is_recording:
            await self.stop_recording()
        
        if self.control_ws:
            await self.control_ws.close()
        
        if self.audio_ws:
            await self.audio_ws.close()
        
        self.cleanup()

async def interactive_session():
    """Run an interactive session with the Thai ASR server"""
    client = ThaiASRClient()
    
    try:
        # Connect to server
        await client.connect()
        
        # Start listening to responses
        response_task = asyncio.create_task(client.listen_to_responses())
        
        print()
        print("🎤 Thai ASR Client Ready!")
        print("Commands:")
        print("  'start' - Start recording")
        print("  'stop' - Stop recording") 
        print("  'models' - List available models")
        print("  'load <model_id>' - Load specific model")
        print("  'quit' - Exit")
        print()
        
        # Interactive command loop
        while client.running:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, "👤 Command: "
                )
                
                command = command.strip().lower()
                
                if command == 'start':
                    if not client.is_recording:
                        await client.start_recording()
                    else:
                        print("Already recording!")
                
                elif command == 'stop':
                    if client.is_recording:
                        await client.stop_recording()
                    else:
                        print("Not recording!")
                
                elif command == 'models':
                    await client.get_available_models()
                
                elif command.startswith('load '):
                    model_id = command[5:].strip()
                    if model_id:
                        await client.load_model(model_id)
                    else:
                        print("Please specify a model ID")
                
                elif command in ['quit', 'exit', 'q']:
                    break
                
                else:
                    print("Unknown command!")
            
            except (EOFError, KeyboardInterrupt):
                break
        
        # Cancel response task
        response_task.cancel()
        
    except Exception as e:
        logger.error(f"❌ Session error: {e}")
    
    finally:
        await client.close()

def handle_signal(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\\n🛑 Shutting down...")
    sys.exit(0)

def main():
    """Main function"""
    try:
        # Set up signal handler
        signal.signal(signal.SIGINT, handle_signal)
        
        # Set up Windows event loop policy if needed
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        print("🎤 Thai ASR WebSocket Client")
        print("=" * 40)
        print("Make sure the Thai ASR server is running!")
        print("=" * 40)
        
        # Run interactive session
        asyncio.run(interactive_session())
        
    except KeyboardInterrupt:
        logger.info("🛑 Client stopped by user")
    except Exception as e:
        logger.error(f"❌ Client error: {e}")

if __name__ == "__main__":
    main()
