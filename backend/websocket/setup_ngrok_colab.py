#!/usr/bin/env python3
"""
Google Colab ngrok setup for Thai ASR WebSocket Server
Sets up tunnels for both control (8765) and audio (8766) ports
"""

import ngrok
import time

def setup_ngrok_tunnels():
    """Set up ngrok tunnels for both WebSocket ports"""
    
    try:
        # Set up tunnel for control WebSocket (port 8765)
        control_tunnel = ngrok.connect(8765, proto="http", bind_tls=True)
        print("🔧 Control WebSocket tunnel:")
        print(f"   HTTP URL: {control_tunnel}")
        print(f"   WebSocket URL: {control_tunnel.replace('https://', 'wss://')}")
        
        # Set up tunnel for audio WebSocket (port 8766)  
        audio_tunnel = ngrok.connect(8766, proto="http", bind_tls=True)
        print("\n🎵 Audio WebSocket tunnel:")
        print(f"   HTTP URL: {audio_tunnel}")
        print(f"   WebSocket URL: {audio_tunnel.replace('https://', 'wss://')}")
        
        # Print JavaScript config for easy copy-paste
        control_ws_url = control_tunnel.replace('https://', 'wss://')
        audio_ws_url = audio_tunnel.replace('https://', 'wss://')
        
        print("\n" + "="*60)
        print("🔗 COPY THIS CONFIG TO YOUR HTML CLIENT:")
        print("="*60)
        print("const WEBSOCKET_CONFIG = {")
        print(f"    controlUrl: '{control_ws_url}',")
        print(f"    audioUrl: '{audio_ws_url}'")
        print("};")
        print("="*60)
        
        return {
            'control_url': control_ws_url,
            'audio_url': audio_ws_url,
            'control_tunnel': control_tunnel,
            'audio_tunnel': audio_tunnel
        }
        
    except Exception as e:
        print(f"❌ Error setting up tunnels: {e}")
        return None

if __name__ == "__main__":
    # For testing
    print("🚀 Setting up ngrok tunnels for Thai ASR WebSocket Server...")
    result = setup_ngrok_tunnels()
    
    if result:
        print(f"\n✅ Tunnels ready!")
        print("📝 Update your HTML client with the WebSocket URLs above")
        print("🔄 Keep this script running to maintain the tunnels")
        
        # Keep alive
        try:
            while True:
                time.sleep(60)
                print("🔄 Tunnels still active...")
        except KeyboardInterrupt:
            print("\n🛑 Shutting down tunnels...")
    else:
        print("❌ Failed to set up tunnels")
