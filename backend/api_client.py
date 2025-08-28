#!/usr/bin/env python3
"""
Example API client for the Multi-agent Call Center Backend
Demonstrates how to use the Thai ASR API endpoints
"""

import requests
import json
import time
from pathlib import Path
import argparse


class CallCenterAPIClient:
    """Client for the Multi-agent Call Center Backend API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self):
        """Check server health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Health check failed: {e}")
    
    def get_asr_info(self):
        """Get ASR model information"""
        try:
            response = self.session.get(f"{self.base_url}/api/asr/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"ASR info request failed: {e}")
    
    def transcribe_audio(self, audio_file_path, language="th", use_vad=True, beam_size=1):
        """
        Transcribe an audio file
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (default: 'th')
            use_vad: Use Voice Activity Detection (default: True)
            beam_size: Beam size for decoding (default: 1)
            
        Returns:
            Transcription result dictionary
        """
        audio_path = Path(audio_file_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        try:
            with open(audio_path, 'rb') as f:
                files = {
                    'file': (audio_path.name, f, 'audio/wav')
                }
                data = {
                    'language': language,
                    'use_vad': use_vad,
                    'beam_size': beam_size
                }
                
                print(f"🎵 Uploading {audio_path.name}...")
                start_time = time.time()
                
                response = self.session.post(
                    f"{self.base_url}/api/asr",
                    files=files,
                    data=data,
                    timeout=120  # 2 minutes timeout for large files
                )
                
                upload_time = time.time() - start_time
                response.raise_for_status()
                
                result = response.json()
                result['upload_time'] = upload_time
                return result
                
        except requests.RequestException as e:
            raise Exception(f"Transcription request failed: {e}")
    
    def transcribe_batch(self, audio_files, language="th", use_vad=True, beam_size=1):
        """
        Transcribe multiple audio files in batch
        
        Args:
            audio_files: List of audio file paths
            language: Language code (default: 'th')
            use_vad: Use Voice Activity Detection (default: True)
            beam_size: Beam size for decoding (default: 1)
            
        Returns:
            Batch transcription results
        """
        if len(audio_files) > 10:
            raise ValueError("Maximum 10 files per batch")
        
        files = []
        for audio_file in audio_files:
            audio_path = Path(audio_file)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
            files.append(
                ('files', (audio_path.name, open(audio_path, 'rb'), 'audio/wav'))
            )
        
        try:
            data = {
                'language': language,
                'use_vad': use_vad,
                'beam_size': beam_size
            }
            
            print(f"🎵 Uploading {len(audio_files)} files for batch processing...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/api/asr/batch",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout for batch
            )
            
            upload_time = time.time() - start_time
            response.raise_for_status()
            
            result = response.json()
            result['batch_upload_time'] = upload_time
            return result
            
        except requests.RequestException as e:
            raise Exception(f"Batch transcription failed: {e}")
        finally:
            # Close opened files
            for _, (_, file_obj, _) in files:
                try:
                    file_obj.close()
                except:
                    pass


def print_transcription_result(result):
    """Pretty print transcription result"""
    print("\n📝 Transcription Result:")
    print("=" * 50)
    print(f"Text: {result.get('text', 'N/A')}")
    print(f"Language: {result.get('language', 'N/A')}")
    print(f"Duration: {result.get('duration', 0):.2f}s")
    print(f"Processing time: {result.get('processing_time', 0):.2f}s")
    print(f"Upload time: {result.get('upload_time', 0):.2f}s")
    print(f"Speed ratio: {result.get('speed_ratio', 0):.1f}x realtime")
    print(f"Chunks processed: {result.get('chunks_processed', 0)}")
    print(f"Model: {result.get('model', 'N/A')}")
    print(f"Device: {result.get('device', 'N/A')}")
    print(f"Timestamp: {result.get('timestamp', 'N/A')}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Multi-agent Call Center API Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--health", action="store_true", help="Check server health")
    parser.add_argument("--info", action="store_true", help="Get ASR model info")
    parser.add_argument("--transcribe", metavar="AUDIO_FILE", help="Transcribe audio file")
    parser.add_argument("--batch", nargs="+", metavar="AUDIO_FILES", help="Batch transcribe audio files")
    parser.add_argument("--language", default="th", help="Language code (default: th)")
    parser.add_argument("--no-vad", action="store_true", help="Disable Voice Activity Detection")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for decoding (default: 1)")
    
    args = parser.parse_args()
    
    # Create API client
    client = CallCenterAPIClient(args.server)
    
    try:
        # Health check
        if args.health:
            print("🔍 Checking server health...")
            health = client.health_check()
            print(f"✅ Server status: {health.get('status')}")
            print(f"   ASR Model loaded: {health.get('asr_model_loaded')}")
            print(f"   Device: {health.get('device')}")
            print(f"   Uptime: {health.get('uptime', 0):.1f}s")
            return
        
        # ASR info
        if args.info:
            print("📋 Getting ASR model information...")
            info = client.get_asr_info()
            print("✅ ASR Model Info:")
            for key, value in info.items():
                print(f"   {key}: {value}")
            return
        
        # Single file transcription
        if args.transcribe:
            print(f"🎵 Transcribing: {args.transcribe}")
            result = client.transcribe_audio(
                args.transcribe,
                language=args.language,
                use_vad=not args.no_vad,
                beam_size=args.beam_size
            )
            print_transcription_result(result)
            return
        
        # Batch transcription
        if args.batch:
            print(f"🎵 Batch transcribing {len(args.batch)} files...")
            result = client.transcribe_batch(
                args.batch,
                language=args.language,
                use_vad=not args.no_vad,
                beam_size=args.beam_size
            )
            
            print(f"\n📝 Batch Results (Upload time: {result.get('batch_upload_time', 0):.2f}s):")
            print("=" * 70)
            
            for i, file_result in enumerate(result.get('results', [])):
                print(f"\n{i+1}. {file_result.get('filename', 'Unknown')}")
                if file_result.get('status') == 'success':
                    file_data = file_result.get('result', {})
                    print(f"   Text: {file_data.get('text', 'N/A')}")
                    print(f"   Duration: {file_data.get('duration', 0):.2f}s")
                    print(f"   Processing: {file_data.get('processing_time', 0):.2f}s")
                else:
                    print(f"   Error: {file_result.get('error', 'Unknown error')}")
            return
        
        # No specific action, show help
        parser.print_help()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main() or 0)
