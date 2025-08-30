#!/usr/bin/env python3
"""
Model Manager for Multi-agent Call Center Backend
Handles different Thai ASR models and model switching
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'whisper'))

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Available model types"""
    FASTER_WHISPER = "faster_whisper"
    STANDARD_WHISPER = "standard_whisper"


@dataclass
class ModelInfo:
    """Information about available models"""
    name: str
    display_name: str
    model_type: ModelType
    model_path: str
    language: str
    description: str
    performance_tier: str  # "fast", "balanced", "accurate"
    recommended: bool = False


class ModelManager:
    """Manages different Thai ASR models"""
    
    def __init__(self):
        self.current_model = None
        self.current_model_info = None
        self.available_models = self._get_available_models()
    
    def _get_available_models(self) -> Dict[str, ModelInfo]:
        """Get list of available Thai ASR models"""
        models = {
            # Faster-whisper models (optimized for speed)
            "biodatlab-faster": ModelInfo(
                name="biodatlab-faster",
                display_name="Biodatlab Whisper Thai (Faster)",
                model_type=ModelType.FASTER_WHISPER,
                model_path="Vinxscribe/biodatlab-whisper-th-large-v3-faster",
                language="th",
                description="Optimized Thai model based on large-v3, 2-4x faster performance",
                performance_tier="fast",
                recommended=True
            ),
            
            "large-v3-faster": ModelInfo(
                name="large-v3-faster",
                display_name="Whisper Large-v3 (Faster)",
                model_type=ModelType.FASTER_WHISPER,
                model_path="large-v3",
                language="th",
                description="Standard large-v3 model with faster-whisper optimization",
                performance_tier="balanced"
            ),
            
            "medium-faster": ModelInfo(
                name="medium-faster",
                display_name="Whisper Medium (Faster)",
                model_type=ModelType.FASTER_WHISPER,
                model_path="openai/whisper-medium",
                language="th",
                description="Medium model with faster-whisper, good balance of speed and accuracy",
                performance_tier="fast"
            ),

            "biodatlab-small-combined": ModelInfo(
                name="biodatlab-small-combined",
                display_name="Biodatlab Whisper Thai Small Combined",
                model_type=ModelType.STANDARD_WHISPER,
                model_path="biodatlab/whisper-th-small-combined",
                language="th",
                description="Small combined Thai model from Biodatlab, optimized for speed and resource usage. Supports safetensors format.",
                performance_tier="fast",
                recommended=False
            ),
            
            # Standard Whisper models (higher accuracy)
            "pathumma-large": ModelInfo(
                name="pathumma-large",
                display_name="Pathumma Whisper Thai Large-v3",
                model_type=ModelType.STANDARD_WHISPER,
                model_path="nectec/Pathumma-whisper-th-large-v3",
                language="th",
                description="Thai-specific model from NECTEC, optimized for Thai language",
                performance_tier="accurate",
                recommended=True
            ),
            
            "large-v3-standard": ModelInfo(
                name="large-v3-standard",
                display_name="Whisper Large-v3 (Standard)",
                model_type=ModelType.STANDARD_WHISPER,
                model_path="openai/whisper-large-v3",
                language="th",
                description="Standard OpenAI large-v3 model with high accuracy",
                performance_tier="accurate"
            ),
            
            "medium-standard": ModelInfo(
                name="medium-standard",
                display_name="Whisper Medium (Standard)",
                model_type=ModelType.STANDARD_WHISPER,
                model_path="openai/whisper-medium",
                language="th",
                description="Standard medium model, good for general use",
                performance_tier="balanced"
            )
        }
        
        return models
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models for API response"""
        models_list = []
        
        for model_id, model_info in self.available_models.items():
            models_list.append({
                "id": model_id,
                "name": model_info.display_name,
                "type": model_info.model_type.value,
                "language": model_info.language,
                "description": model_info.description,
                "performance_tier": model_info.performance_tier,
                "recommended": model_info.recommended
            })
        
        return sorted(models_list, key=lambda x: (not x["recommended"], x["performance_tier"]))
    
    def load_model(self, model_id: str, config_overrides: Optional[Dict] = None) -> Any:
        """
        Load a specific model
        
        Args:
            model_id: ID of the model to load
            config_overrides: Optional configuration overrides
            
        Returns:
            Loaded model instance
        """
        if model_id not in self.available_models:
            raise ValueError(f"Unknown model ID: {model_id}")
        
        model_info = self.available_models[model_id]
        logger.info(f"🔄 Loading model: {model_info.display_name}")
        
        try:
            if model_info.model_type == ModelType.FASTER_WHISPER:
                model = self._load_faster_whisper_model(model_info, config_overrides)
            else:
                model = self._load_standard_whisper_model(model_info, config_overrides)
            
            self.current_model = model
            self.current_model_info = model_info
            
            logger.info(f"✅ Model loaded successfully: {model_info.display_name}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_info.display_name}: {e}")
            raise
    
    def _load_faster_whisper_model(self, model_info: ModelInfo, config_overrides: Optional[Dict] = None):
        """Load faster-whisper model"""
        try:
            from whisper.faster_whisper_thai import WhisperConfig, create_thai_asr
        except ImportError:
            from faster_whisper_thai import WhisperConfig, create_thai_asr
        
        # Create configuration
        config = WhisperConfig(
            model_name=model_info.model_path,
            language=model_info.language,
            device="auto",
            compute_type="int8_float16",
            beam_size=1,
            use_vad=True,
            chunk_length_ms=30000,
            overlap_ms=1000
        )
        
        # Apply config overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return create_thai_asr(config)
    
    def _load_standard_whisper_model(self, model_info: ModelInfo, config_overrides: Optional[Dict] = None):
        """Load standard Whisper model"""
        try:
            from whisper.whisper import OverlappingASRPipeline, AudioConfig, ProcessingConfig, SimplePipelineASR
        except ImportError:
            from whisper import OverlappingASRPipeline, AudioConfig, ProcessingConfig, SimplePipelineASR
        
        # Create configurations
        audio_config = AudioConfig(
            chunk_length_ms=27000,
            overlap_ms=2000,
            min_chunk_length_ms=1000,
            sample_rate=16000,
            channels=1
        )
        
        processing_config = ProcessingConfig(
            model_name=model_info.model_path,
            language=model_info.language,
            task="transcribe",
            batch_size=4,
            max_workers=2,
            use_gpu=True,
            use_faster_whisper=False  # Force standard whisper
        )
        
        # Apply config overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(processing_config, key):
                    setattr(processing_config, key, value)
                elif hasattr(audio_config, key):
                    setattr(audio_config, key, value)

        # Use the lightweight Transformers pipeline adapter by default
        return SimplePipelineASR(audio_config, processing_config)
    
    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about currently loaded model"""
        if self.current_model_info is None:
            return None
        
        return {
            "id": next((k for k, v in self.available_models.items() if v == self.current_model_info), "unknown"),
            "name": self.current_model_info.display_name,
            "type": self.current_model_info.model_type.value,
            "model_path": self.current_model_info.model_path,
            "language": self.current_model_info.language,
            "description": self.current_model_info.description,
            "performance_tier": self.current_model_info.performance_tier
        }
    
    def transcribe_with_current_model(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio with the currently loaded model"""
        if self.current_model is None:
            raise RuntimeError("No model is currently loaded")
        
        if self.current_model_info.model_type == ModelType.FASTER_WHISPER:
            return self.current_model.transcribe(audio_path)
        else:
            # For standard whisper, we need to use the process_file method
            result = self.current_model.process_file(audio_path)
            
            # Convert to consistent format
            return {
                "text": result.get("transcription", ""),
                "language": self.current_model_info.language,
                "duration": result.get("total_duration", 0),
                "processing_time": result.get("total_processing_time", 0),
                "speed_ratio": result.get("total_duration", 0) / max(result.get("total_processing_time", 1), 0.001),
                "chunks_processed": result.get("total_chunks", 0),
                "model": self.current_model_info.model_path,
                "device": getattr(self.current_model.asr, 'device', 'unknown')
            }


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    return model_manager
