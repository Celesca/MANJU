#!/usr/bin/env python3
"""
Test client for parallel transcription performance
Tests multiple concurrent connections to verify parallel processing
"""

import asyncio
import websockets
import json
import time
import numpy as np
import base64
from typing import List

# Generate synthetic audio data for testing
def generate_test_audio(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate synthetic audio data for testing"""
    # Generate a simple sine wave
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    frequency = 440  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    return audio_int16.tobytes()

async def test_client(client_id: int, num_requests: int = 3):
    """Test client that sends multiple audio requests"""
    uri = "ws://localhost:8765/audio"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Client {client_id}: Connected")
            
            # Send multiple audio chunks
            for i in range(num_requests):
                # Generate test audio
                audio_bytes = generate_test_audio(duration_seconds=2.0)
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Send audio message
                message = {
                    'type': 'audio_chunk',
                    'audio_data': audio_b64,
                    'sample_rate': 16000,
                    'channels': 1,
                    'chunk_id': f"client_{client_id}_chunk_{i}",
                    'is_final': i == num_requests - 1
                }
                
                start_time = time.time()
                await websocket.send(json.dumps(message))
                print(f"Client {client_id}: Sent chunk {i+1}/{num_requests}")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    response_data = json.loads(response)
                    response_time = time.time() - start_time
                    
                    print(f"Client {client_id}: Response {i+1} - "
                          f"Type: {response_data.get('type', 'unknown')}, "
                          f"Time: {response_time:.2f}s")
                    
                    if response_data.get('text'):
                        print(f"Client {client_id}: Text: {response_data['text']}")
                        
                except asyncio.TimeoutError:
                    print(f"Client {client_id}: Timeout waiting for response {i+1}")
                
                # Small delay between requests
                await asyncio.sleep(0.5)
            
            print(f"Client {client_id}: Completed all requests")
            
    except Exception as e:
        print(f"Client {client_id}: Error - {e}")

async def test_parallel_performance():
    """Test parallel processing with multiple clients"""
    print("Testing parallel transcription performance...")
    print("=" * 50)
    
    # Test configuration
    num_clients = 4
    requests_per_client = 3
    
    print(f"Starting {num_clients} concurrent clients")
    print(f"Each client will send {requests_per_client} requests")
    print(f"Total requests: {num_clients * requests_per_client}")
    print("-" * 50)
    
    # Start all clients concurrently
    start_time = time.time()
    
    tasks = []
    for i in range(num_clients):
        task = asyncio.create_task(test_client(i + 1, requests_per_client))
        tasks.append(task)
    
    # Wait for all clients to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    total_requests = num_clients * requests_per_client
    
    print("-" * 50)
    print(f"Test completed in {total_time:.2f} seconds")
    print(f"Average time per request: {total_time / total_requests:.2f}s")
    print(f"Requests per second: {total_requests / total_time:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_parallel_performance())
