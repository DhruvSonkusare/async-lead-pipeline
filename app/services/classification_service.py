"""
app/services/classification_service.py
=======================================
ML model management and classification service.

Handles:
  - Loading trained ML model
  - Classification predictions
  - Confidence scoring
  - Error handling for missing models
"""

import pickle
import logging
from typing import Tuple
from app.config import get_settings

logger = logging.getLogger(__name__)

class ClassificationService:
    """Service for classifying lead intent using ML model"""
    
    def __init__(self):
        """Initialize classification service"""
        self.settings = get_settings()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load trained ML model from disk"""
        model_path = self.settings.get_model_path()
        
        try:
            logger.info(f"Loading ML model from: {model_path}")
            
            # Check if model file exists
            import os
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                logger.warning("Model will be None. Train the model first using: python app/ml/train.py")
                self.model = None
                return
            
            # Load the model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Classification service will not work. Train the model first.")
            self.model = None
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def classify(self, message: str) -> Tuple[str, float]:
        """
        Classify message into intent (hot/warm/cold).
        
        Args:
            message: The lead message to classify
        
        Returns:
            Tuple of (intent, confidence)
            intent: "hot", "warm", "cold", or "unknown"
            confidence: float between 0.0 and 1.0
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_model_loaded():
            logger.error("Model not loaded. Cannot classify message.")
            logger.error("Train the model first: python app/ml/train.py")
            raise RuntimeError(
                "ML model not loaded. Train the model first using: python app/ml/train.py"
            )
        
        try:
            # Get prediction
            intent = self.model.predict([message])[0]
            
            # Get confidence score
            probabilities = self.model.predict_proba([message])[0]
            confidence = float(max(probabilities))
            
            logger.debug(f"Classified message: {intent} ({confidence:.2%})")
            
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Error classifying message: {e}")
            return "unknown", 0.0


# Create a singleton instance
try:
    classification_service = ClassificationService()
except Exception as e:
    logger.error(f"Failed to initialize classification service: {e}")
    classification_service = None