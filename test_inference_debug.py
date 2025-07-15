#!/usr/bin/env python3
"""
Quick test for F5-TTS-THAI model inference
"""

import os
import sys
import tempfile

def test_model_inference():
    """Test direct model inference"""
    print("🧪 Testing F5-TTS-THAI Model Inference")
    print("=" * 50)
    
    # Check for model
    model_paths = ["model_1000000.pt", "models/model_1000000.pt"]
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ Model not found")
        return False
    
    print(f"✅ Model found: {model_path}")
    
    try:
        # Import F5-TTS components
        import torch
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_model, infer_process
        
        print("✅ F5-TTS imports successful")
        
        # Load model
        F5TTS_model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, 
            text_dim=512, conv_layers=4
        )
        
        print("🔄 Loading model...")
        model = load_model(DiT, F5TTS_model_cfg, model_path, use_ema=True)
        print(f"✅ Model loaded: {type(model)}")
        
        # Check model attributes
        print(f"📋 Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        
        # Try to load vocoder
        try:
            from f5_tts.infer.utils_infer import load_vocoder
            vocoder = load_vocoder()
            print(f"✅ Vocoder loaded: {type(vocoder)}")
        except Exception as e:
            print(f"⚠️ Vocoder loading failed: {e}")
            vocoder = None
        
        # Test inference
        print("🔄 Testing inference...")
        
        test_text = "สวัสดีครับ"
        ref_text = "สวัสดีครับ ผมเป็นผู้ช่วยเสียงภาษาไทย"
        
        # Create a simple reference audio using gTTS
        ref_audio_path = None
        try:
            from gtts import gTTS
            temp_ref = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            tts = gTTS(text=ref_text, lang='th')
            tts.save(temp_ref.name)
            ref_audio_path = temp_ref.name
            print(f"✅ Reference audio created: {ref_audio_path}")
        except Exception as e:
            print(f"⚠️ Reference audio creation failed: {e}")
        
        # Test inference
        try:
            final_wave, final_sample_rate, _ = infer_process(
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                gen_text=test_text,
                model=model,
                vocoder=vocoder,
                nfe_step=16,  # Faster for testing
                speed=1.0,
                cfg_strength=2.0
            )
            
            print(f"✅ Inference successful!")
            print(f"📊 Audio shape: {final_wave.shape if hasattr(final_wave, 'shape') else type(final_wave)}")
            print(f"📊 Sample rate: {final_sample_rate}")
            
            # Save test output
            import soundfile as sf
            output_path = "test_inference_output.wav"
            
            # Convert to numpy if needed
            if hasattr(final_wave, 'cpu'):
                audio_array = final_wave.cpu().numpy()
            elif hasattr(final_wave, 'numpy'):
                audio_array = final_wave.numpy()
            else:
                audio_array = final_wave
            
            sf.write(output_path, audio_array, final_sample_rate)
            print(f"✅ Test output saved: {output_path}")
            
            # Clean up
            if ref_audio_path and os.path.exists(ref_audio_path):
                os.unlink(ref_audio_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            print(f"🔍 Full error: {traceback.format_exc()}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"🔍 Full error: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    if test_model_inference():
        print("\n🎉 Model inference test passed!")
    else:
        print("\n❌ Model inference test failed!")
