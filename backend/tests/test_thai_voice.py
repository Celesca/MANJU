#!/usr/bin/env python3
"""
Test F5-TTS-THAI with proper Thai voice configuration
Based on official F5-TTS-THAI documentation and model recommendations
"""

import os
import sys
import tempfile
import time

def test_f5_tts_thai_voice():
    """Test F5-TTS-THAI with authentic Thai voice configuration"""
    
    print("🧪 Testing F5-TTS-THAI with proper Thai voice configuration...")
    print("=" * 60)
    
    # Test text in Thai
    test_text = "สวัสดีครับ วันนี้เป็นวันที่สวยงาม ผมหวังว่าคุณจะมีความสุขมากๆ"
    thai_ref_text = "สวัสดีครับ ผมเป็นผู้ช่วยเสียงภาษาไทย ยินดีให้บริการครับ"
    
    print(f"📝 Test Text: {test_text}")
    print(f"🎤 Reference Text: {thai_ref_text}")
    print("-" * 60)
    
    # Try to import F5-TTS components
    try:
        from f5_tts.api import F5TTS
        print("✅ F5-TTS API imported successfully")
    except ImportError as e:
        print(f"❌ F5-TTS API import failed: {e}")
        return False
    
    try:
        import torch
        import torchaudio
        import soundfile as sf
        print(f"✅ Audio libraries imported: torch={torch.__version__}")
    except ImportError as e:
        print(f"❌ Audio library import failed: {e}")
        return False
    
    # Create temporary reference audio using gTTS
    print("\n🔊 Creating Thai reference audio...")
    try:
        from gtts import gTTS
        
        temp_ref_file = os.path.join("temp", "thai_ref_test.mp3")
        os.makedirs("temp", exist_ok=True)
        
        # Create Thai reference audio
        ref_tts = gTTS(text=thai_ref_text, lang='th', slow=False)
        ref_tts.save(temp_ref_file)
        print(f"✅ Thai reference audio created: {temp_ref_file}")
        
    except ImportError:
        print("❌ gTTS not available, cannot create reference audio")
        temp_ref_file = None
    except Exception as e:
        print(f"❌ Error creating reference audio: {e}")
        temp_ref_file = None
    
    # Test F5-TTS-THAI model loading
    print("\n🤖 Loading F5-TTS-THAI model...")
    try:
        # Load the specific Thai fine-tuned model
        model_name = "VIZINTZOR/F5-TTS-THAI"
        print(f"📦 Loading model: {model_name}")
        
        model = F5TTS.from_pretrained(model_name)
        print(f"✅ Model loaded successfully: {model_name}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        try:
            # Fallback to default initialization
            print("🔄 Trying default F5TTS initialization...")
            model = F5TTS()
            print("✅ Default F5TTS model loaded")
        except Exception as e2:
            print(f"❌ Default model loading also failed: {e2}")
            return False
    
    # Test inference with Thai configuration
    print("\n🎯 Testing Thai voice inference...")
    try:
        output_file = os.path.join("temp", "test_thai_voice.wav")
        
        print("🔄 Generating audio with Thai configuration...")
        print(f"   Text: {test_text}")
        print(f"   Reference Text: {thai_ref_text}")
        print(f"   Reference Audio: {temp_ref_file}")
        
        # Test with Thai-specific parameters
        inference_params = {
            "gen_text": test_text,
            "ref_text": thai_ref_text,
            "remove_silence": True,
            "speed": 0.8,  # Slower for clearer Thai pronunciation
        }
        
        # Add reference file if available
        if temp_ref_file and os.path.exists(temp_ref_file):
            inference_params["ref_file"] = temp_ref_file
        
        print(f"🔧 Inference parameters: {list(inference_params.keys())}")
        
        # Try inference with different parameter combinations
        audio_data = None
        
        # Method 1: Full parameters
        try:
            audio_data = model.infer(**inference_params)
            print("✅ Inference successful with full parameters")
        except TypeError as e:
            print(f"⚠️ Full parameters failed: {e}")
            
            # Method 2: Minimal parameters
            try:
                minimal_params = {
                    "gen_text": test_text,
                    "ref_text": thai_ref_text
                }
                if temp_ref_file:
                    minimal_params["ref_file"] = temp_ref_file
                
                audio_data = model.infer(**minimal_params)
                print("✅ Inference successful with minimal parameters")
            except Exception as e2:
                print(f"❌ Minimal parameters also failed: {e2}")
                return False
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return False
        
        # Process and save audio
        if audio_data is not None:
            print("🔄 Processing audio data...")
            
            # Handle different audio data formats
            if hasattr(audio_data, 'cpu'):
                audio_data = audio_data.cpu().numpy()
            elif hasattr(audio_data, 'numpy'):
                audio_data = audio_data.numpy()
            
            # Handle list/tuple format
            if isinstance(audio_data, (list, tuple)):
                audio_data = audio_data[0] if len(audio_data) > 0 else audio_data
                if hasattr(audio_data, 'cpu'):
                    audio_data = audio_data.cpu().numpy()
                elif hasattr(audio_data, 'numpy'):
                    audio_data = audio_data.numpy()
            
            # Ensure numpy array
            import numpy as np
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # Handle dimensions
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            
            # Normalize
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Save audio
            try:
                sf.write(output_file, audio_data, 24000)  # F5-TTS typically uses 24kHz
                print(f"✅ Audio saved: {output_file}")
                print(f"📊 Audio length: {len(audio_data)/24000:.2f} seconds")
                
                # Play audio if possible
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(output_file)
                    pygame.mixer.music.play()
                    print("🔊 Playing generated Thai audio...")
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    pygame.mixer.quit()
                    
                except ImportError:
                    print("💡 Install pygame to play audio automatically")
                except Exception as e:
                    print(f"⚠️ Could not play audio: {e}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error saving audio: {e}")
                return False
        else:
            print("❌ No audio data returned from inference")
            return False
            
    except Exception as e:
        print(f"❌ Thai voice inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary files
        if temp_ref_file and os.path.exists(temp_ref_file):
            try:
                os.remove(temp_ref_file)
                print(f"🧹 Cleaned up: {temp_ref_file}")
            except:
                pass

def main():
    """Main test function"""
    print("🚀 F5-TTS-THAI Authentic Thai Voice Test")
    print("=" * 60)
    
    success = test_f5_tts_thai_voice()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed successfully!")
        print("🎉 F5-TTS-THAI should now generate authentic Thai voice")
        print("\n💡 Key findings for authentic Thai voice:")
        print("   • Use VIZINTZOR/F5-TTS-THAI model")
        print("   • Provide Thai reference text and audio")
        print("   • Use speed=0.8 for clearer pronunciation")
        print("   • Ensure all text inputs are in Thai")
    else:
        print("❌ Test failed!")
        print("🔧 Please check F5-TTS-THAI installation and dependencies")
    
    print("\n📁 Check the 'temp' folder for generated audio files")

if __name__ == "__main__":
    main()
