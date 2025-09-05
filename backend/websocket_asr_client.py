#!/usr/bin/env python3
"""
WebSocket ASR Client Example
Demonstrates how to connect to the WebSocket Thai ASR server
"""

import asyncio
import json
import base64
import websockets
import pyaudio
import numpy as np
import threading
import queue
import time
from datetime import datetime

class WebSocketASRClient:
    def __init__(self, control_url="ws://localhost:8765", audio_url="ws://localhost:8766"):
        self.control_url = control_url
        self.audio_url = audio_url
        self.control_ws = None
        self.audio_ws = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.session_id = f"session_{int(time.time())}"

        # Audio configuration
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 30

    async def connect_control(self):
        """Connect to control WebSocket server"""
        try:
            self.control_ws = await websockets.connect(self.control_url)
            print("✅ Connected to control server")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to control server: {e}")
            return False

    async def connect_audio(self):
        """Connect to audio WebSocket server"""
        try:
            self.audio_ws = await websockets.connect(self.audio_url)
            print("✅ Connected to audio server")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to audio server: {e}")
            return False

    async def load_model(self, model_id="biodatlab-medium-faster"):
        """Load a specific model"""
        if not self.control_ws:
            print("❌ Control connection not established")
            return False

        try:
            message = {
                "type": "load_model",
                "model_id": model_id
            }
            await self.control_ws.send(json.dumps(message))
            response = await self.control_ws.recv()
            result = json.loads(response)
            print(f"📦 Model loaded: {result}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    async def get_stats(self):
        """Get server statistics"""
        if not self.control_ws:
            print("❌ Control connection not established")
            return None

        try:
            message = {"type": "get_stats"}
            await self.control_ws.send(json.dumps(message))
            response = await self.control_ws.recv()
            stats = json.loads(response)
            print("📊 Server Stats:")
            print(json.dumps(stats, indent=2))
            return stats
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return None

    async def start_streaming_transcription(self):
        """Start streaming transcription session"""
        if not self.audio_ws:
            print("❌ Audio connection not established")
            return False

        try:
            message = {
                "type": "start_stream",
                "session_id": self.session_id,
                "config": {
                    "language": "th",
                    "beam_size": 1
                }
            }
            await self.audio_ws.send(json.dumps(message))
            print(f"🎙️ Started streaming session: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to start streaming: {e}")
            return False

    async def send_audio_chunk(self, audio_data):
        """Send audio chunk for transcription"""
        if not self.audio_ws:
            print("❌ Audio connection not established")
            return False

        try:
            # Convert audio data to base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')

            message = {
                "type": "audio_chunk",
                "session_id": self.session_id,
                "audio_data": audio_b64
            }
            await self.audio_ws.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"❌ Failed to send audio chunk: {e}")
            return False

    async def end_streaming_transcription(self):
        """End streaming transcription session"""
        if not self.audio_ws:
            print("❌ Audio connection not established")
            return False

        try:
            message = {
                "type": "end_stream",
                "session_id": self.session_id
            }
            await self.audio_ws.send(json.dumps(message))
            print(f"🛑 Ended streaming session: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to end streaming: {e}")
            return False

    def record_audio(self):
        """Record audio from microphone"""
        audio = pyaudio.PyAudio()

        # Open stream
        stream = audio.open(format=self.FORMAT,
                          channels=self.CHANNELS,
                          rate=self.RATE,
                          input=True,
                          frames_per_buffer=self.CHUNK)

        print("🎤 Recording... Press Ctrl+C to stop")

        self.is_recording = True
        try:
            while self.is_recording:
                data = stream.read(self.CHUNK)
                self.audio_queue.put(data)
        except KeyboardInterrupt:
            print("\n🛑 Recording stopped")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            self.is_recording = False

    async def process_audio_queue(self):
        """Process audio chunks from queue"""
        while self.is_recording or not self.audio_queue.empty():
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
            else:
                await asyncio.sleep(0.01)

    async def listen_for_responses(self):
        """Listen for transcription responses"""
        try:
            async for message in self.audio_ws:
                response = json.loads(message)

                if response.get("type") == "realtime_transcription":
                    result = response.get("result", {})
                    text = result.get("text", "")
                    confidence = result.get("confidence", 0)
                    timestamp = result.get("timestamp", "")
                    print(f"🎯 Real-time: {text} (confidence: {confidence:.2f})")

                elif response.get("type") == "transcription_result":
                    result = response.get("result", {})
                    text = result.get("text", "")
                    processing_time = result.get("processing_time", 0)
                    real_time_factor = result.get("real_time_factor", 0)
                    print(f"📝 Final: {text}")
                    print(f"⚡ Processing time: {processing_time:.2f}s, RTF: {real_time_factor:.2f}")
                elif response.get("type") == "error":
                    error_msg = response.get("message", "Unknown error")
                    print(f"❌ Error: {error_msg}")

        except websockets.exceptions.ConnectionClosed:
            print("🔌 Connection closed")
        except Exception as e:
            print(f"❌ Error listening for responses: {e}")

    async def run_streaming_example(self):
        """Run complete streaming transcription example"""
        print("🚀 Starting WebSocket ASR Client...")

        # Connect to servers
        if not await self.connect_control():
            return
        if not await self.connect_audio():
            return

        # Load model
        await self.load_model("biodatlab-medium-faster")

        # Get initial stats
        await self.get_stats()

        # Start streaming session
        if not await self.start_streaming_transcription():
            return

        # Start recording in background thread
        recording_thread = threading.Thread(target=self.record_audio)
        recording_thread.start()

        # Create tasks for processing
        audio_task = asyncio.create_task(self.process_audio_queue())
        listen_task = asyncio.create_task(self.listen_for_responses())

        try:
            # Wait for recording to finish
            while self.is_recording:
                await asyncio.sleep(0.1)

            # Stop recording
            recording_thread.join()

            # Wait a bit for remaining audio to be processed
            await asyncio.sleep(1)

            # End streaming session
            await self.end_streaming_transcription()

            # Wait for final results
            await asyncio.sleep(2)

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            # Cancel tasks
            audio_task.cancel()
            listen_task.cancel()

            # Close connections
            if self.control_ws:
                await self.control_ws.close()
            if self.audio_ws:
                await self.audio_ws.close()

            print("👋 Client shutdown complete")

    async def run_single_transcription_example(self, audio_file_path=None):
        """Run single transcription example"""
        print("🚀 Starting Single Transcription Example...")

        # Connect to audio server
        if not await self.connect_audio():
            return

        # Record or load audio
        if audio_file_path:
            print(f"📁 Loading audio from: {audio_file_path}")
            # Load audio file (implement if needed)
            audio_data = b""  # Placeholder
        else:
            print("🎤 Recording 5 seconds of audio...")
            audio = pyaudio.PyAudio()
            stream = audio.open(format=self.FORMAT,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              input=True,
                              frames_per_buffer=self.CHUNK)

            frames = []
            for i in range(0, int(self.RATE / self.CHUNK * 5)):
                data = stream.read(self.CHUNK)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            audio.terminate()

            audio_data = b''.join(frames)

        # Send for transcription
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        message = {
            "type": "transcribe",
            "audio_data": audio_b64,
            "config": {
                "language": "th",
                "beam_size": 1
            }
        }

        await self.audio_ws.send(json.dumps(message))
        print("📤 Sent transcription request")

        # Wait for response
        try:
            response = await self.audio_ws.recv()
            result = json.loads(response)

            if result.get("type") == "transcription_result":
                transcription = result.get("result", {}).get("text", "")
                print(f"📝 Transcription: {transcription}")
            else:
                print(f"❌ Unexpected response: {result}")

        except Exception as e:
            print(f"❌ Error receiving response: {e}")
        finally:
            if self.audio_ws:
                await self.audio_ws.close()

async def main():
    """Main function"""
    client = WebSocketASRClient()

    print("Select mode:")
    print("1. Streaming transcription (real-time)")
    print("2. Single transcription")
    print("3. Get server stats only")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        await client.run_streaming_example()
    elif choice == "2":
        audio_file = input("Enter audio file path (or press Enter for live recording): ").strip()
        audio_file = audio_file if audio_file else None
        await client.run_single_transcription_example(audio_file)
    elif choice == "3":
        if await client.connect_control():
            await client.get_stats()
            await client.control_ws.close()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    asyncio.run(main())
