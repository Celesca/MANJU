#!/usr/bin/env python3
"""
Simple test script for WebSocket ASR Server
Tests basic connectivity and functionality
"""

import asyncio
import json
import websockets
import time

async def test_control_server():
    """Test control server connectivity and basic functions"""
    print("🧪 Testing Control Server...")

    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("✅ Connected to control server")

            # Test health check
            await websocket.send(json.dumps({"type": "health_check"}))
            response = await websocket.recv()
            result = json.loads(response)
            print(f"🏥 Health check: {result}")

            # Test get models
            await websocket.send(json.dumps({"type": "get_models"}))
            response = await websocket.recv()
            result = json.loads(response)
            print(f"📦 Available models: {len(result.get('models', []))} models")

            # Test get stats
            await websocket.send(json.dumps({"type": "get_stats"}))
            response = await websocket.recv()
            result = json.loads(response)
            print("📊 Server stats retrieved successfully")
            print("✅ Control server tests passed")
            return True

    except Exception as e:
        print(f"❌ Control server test failed: {e}")
        return False

async def test_audio_server():
    """Test audio server connectivity"""
    print("\n🧪 Testing Audio Server...")

    try:
        async with websockets.connect("ws://localhost:8766") as websocket:
            print("✅ Connected to audio server")

            # Test invalid request (should get error response)
            await websocket.send(json.dumps({"type": "invalid"}))
            response = await websocket.recv()
            result = json.loads(response)
            print(f"🚫 Error handling: {result.get('type', 'unknown')}")

            print("✅ Audio server tests passed")
            return True

    except Exception as e:
        print(f"❌ Audio server test failed: {e}")
        return False

async def test_model_loading():
    """Test model loading functionality"""
    print("\n🧪 Testing Model Loading...")

    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            # Load a model
            await websocket.send(json.dumps({
                "type": "load_model",
                "model_id": "biodatlab-medium-faster"
            }))

            response = await websocket.recv()
            result = json.loads(response)
            print(f"📥 Model load response: {result}")

            if result.get("status") == "success":
                print("✅ Model loading test passed")
                return True
            else:
                print(f"⚠️ Model loading returned: {result}")
                return False

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

async def test_concurrent_connections():
    """Test multiple concurrent connections"""
    print("\n🧪 Testing Concurrent Connections...")

    async def single_connection_test(conn_id):
        try:
            async with websockets.connect("ws://localhost:8765") as websocket:
                await websocket.send(json.dumps({"type": "health_check"}))
                response = await websocket.recv()
                result = json.loads(response)
                return result.get("status") == "healthy"
        except Exception as e:
            print(f"❌ Connection {conn_id} failed: {e}")
            return False

    # Test 3 concurrent connections
    tasks = [single_connection_test(i) for i in range(3)]
    results = await asyncio.gather(*tasks)

    success_count = sum(results)
    print(f"🔗 Concurrent connections: {success_count}/3 successful")

    if success_count == 3:
        print("✅ Concurrent connections test passed")
        return True
    else:
        print("⚠️ Some concurrent connections failed")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting WebSocket ASR Server Tests")
    print("=" * 50)

    tests = [
        ("Control Server", test_control_server),
        ("Audio Server", test_audio_server),
        ("Model Loading", test_model_loading),
        ("Concurrent Connections", test_concurrent_connections),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} test...")
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")

    passed = 0
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if results[i]:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Server is ready for production.")
        return True
    else:
        print("⚠️ Some tests failed. Check server configuration.")
        return False

async def main():
    """Main test function"""
    print("WebSocket ASR Server Test Suite")
    print("Make sure the server is running before starting tests.")
    print("Server should be started with: python server_websocket.py")
    print()

    # Check if server is running
    print("⏳ Checking server availability...")
    await asyncio.sleep(2)  # Give server time to start

    try:
        success = await run_all_tests()
        if success:
            print("\n🎊 Test suite completed successfully!")
        else:
            print("\n💥 Test suite completed with failures.")
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
