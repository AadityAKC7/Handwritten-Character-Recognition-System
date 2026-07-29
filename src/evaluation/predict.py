# predict.py - IMPROVED VERSION
"""
Prediction module for HCR System
Handles model loading and character prediction
"""

import tensorflow as tf
import numpy as np
import os
from typing import Dict, Tuple, Optional, List

class CharacterPredictor:
    """Handles character prediction using trained model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.class_mapping = {}
        self.model_type = None  # 'digits', 'letters', or 'combined'
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _create_digits_mapping(self) -> Dict[int, str]:
        """Create mapping for digits only (0-9)"""
        return {i: str(i) for i in range(10)}
    
    def _create_letters_mapping(self) -> Dict[int, str]:
        """Create mapping for letters only (A-Z)"""
        # EMNIST Letters uses 0-25 for A-Z
        return {i: chr(65 + i) for i in range(26)}  # 65 is 'A' in ASCII
    
    def _create_combined_mapping(self) -> Dict[int, str]:
        """Create mapping for combined digits + uppercase letters (36 classes)"""
        mapping = {}
        # Digits 0-9 (classes 0-9)
        for i in range(10):
            mapping[i] = str(i)
        # Uppercase Letters A-Z (classes 10-35)
        for i in range(26):
            mapping[10 + i] = chr(65 + i)
        return mapping
    
    def _detect_model_type(self, output_shape: int) -> str:
        """Detect model type based on output shape"""
        if output_shape == 10:
            return 'digits'
        elif output_shape == 26:
            return 'letters'
        elif output_shape == 36:
            return 'combined'
        elif output_shape == 47:  # EMNIST Balanced
            return 'emnist_balanced'
        else:
            print(f"⚠️ Unknown output shape: {output_shape}, assuming combined model")
            return 'combined'
    
    def load_model(self, model_path: str) -> bool:
        """Load trained TensorFlow model"""
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return False
            
            # Load the model
            self.model = tf.keras.models.load_model(model_path)
            
            # Get output shape to determine model type
            output_shape = self.model.output_shape[-1]
            
            # Set mapping based on output shape
            if output_shape == 10:
                self.model_type = 'digits'
                self.class_mapping = self._create_digits_mapping()
                print(f"✅ Loaded DIGITS model (10 classes: 0-9)")
                
            elif output_shape == 26:
                self.model_type = 'letters'
                self.class_mapping = self._create_letters_mapping()
                print(f"✅ Loaded LETTERS model (26 classes: A-Z)")
                print(f"   📌 Note: A=0, B=1, ..., Z=25")
                
            elif output_shape == 36:
                self.model_type = 'combined'
                self.class_mapping = self._create_combined_mapping()
                print(f"✅ Loaded COMBINED model (36 classes: 0-9 + A-Z)")
                
            else:
                self.model_type = 'unknown'
                # Create generic mapping
                self.class_mapping = {i: f"Class_{i}" for i in range(output_shape)}
                print(f"⚠️ Loaded model with {output_shape} classes (unknown type)")
            
            print(f"   📁 Path: {model_path}")
            print(f"   📊 Input shape: {self.model.input_shape}")
            print(f"   📊 Output shape: {self.model.output_shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict character from preprocessed image
        
        Args:
            image: Preprocessed image array - can be shape:
                   (28, 28), (1, 28, 28), or (1, 28, 28, 1)
        
        Returns:
            Tuple of (predicted_character, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Store original shape for debugging
        original_shape = image.shape
        
        # Ensure correct shape for model input
        if len(image.shape) == 2:  # (28, 28)
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            if image.shape[0] == 1 and image.shape[-1] == 1:  # (1, 28, 1)
                pass  # Already correct
            elif image.shape[-1] == 1:  # (28, 28, 1)
                image = image.reshape(1, 28, 28, 1)
            else:  # (1, 28, 28)
                image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 4:
            if image.shape[0] != 1:
                # Take first batch if multiple
                image = image[:1]
        
        # Final validation
        if image.shape != (1, 28, 28, 1):
            print(f"⚠️ Unexpected image shape: {original_shape} -> {image.shape}")
        
        # Make prediction
        try:
            predictions = self.model.predict(image, verbose=0)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise
        
        # Get highest probability class
        class_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get character from mapping
        character = self.class_mapping.get(class_index, "?")
        
        # Debug output
        print(f"   🎯 Prediction: class {class_index} -> '{character}' (confidence: {confidence:.4f})")
        
        return character, confidence
    
    def predict_top_k(self, image: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Get top k predictions with their probabilities
        
        Args:
            image: Preprocessed image array
            k: Number of top predictions to return
        
        Returns:
            List of dictionaries with character, confidence, and class_index
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Ensure correct shape
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3 and image.shape[0] == 1:
            image = image.reshape(1, 28, 28, 1)
        
        # Get predictions
        predictions = self.model.predict(image, verbose=0)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'character': self.class_mapping.get(int(idx), "?"),
                'confidence': float(predictions[idx]),
                'class_index': int(idx)
            })
        
        return results
    
    def predict_with_details(self, image: np.ndarray) -> Dict:
        """
        Predict and return detailed information including top predictions
        
        Returns:
            Dictionary with prediction details
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Ensure correct shape
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = image.reshape(1, 28, 28, 1)
        
        # Get predictions
        predictions = self.model.predict(image, verbose=0)[0]
        
        # Get top 5
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        
        top_predictions = []
        for idx in top_5_indices:
            top_predictions.append({
                'character': self.class_mapping.get(int(idx), "?"),
                'confidence': float(predictions[idx]),
                'class_index': int(idx)
            })
        
        # Best prediction
        best_idx = top_5_indices[0]
        
        return {
            'character': self.class_mapping.get(int(best_idx), "?"),
            'confidence': float(predictions[best_idx]),
            'class_index': int(best_idx),
            'top_predictions': top_predictions,
            'all_confidences': predictions.tolist()
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {
                'loaded': False,
                'model_type': None,
                'num_classes': 0
            }
        
        return {
            'loaded': True,
            'model_type': self.model_type,
            'num_classes': len(self.class_mapping),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'class_mapping': self.class_mapping
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None


# ==================== TEST FUNCTION ====================

def test_predictor():
    """Test the predictor with a dummy model"""
    print("\n" + "="*50)
    print("TESTING CHARACTER PREDICTOR")
    print("="*50)
    
    # Create a simple test model for digits
    from tensorflow.keras import layers, models
    
    print("\n1. Creating test model...")
    test_model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    test_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Save temporary model
    test_model.save('test_model.h5')
    
    # Test predictor
    print("\n2. Loading predictor...")
    predictor = CharacterPredictor('test_model.h5')
    
    # Test prediction
    print("\n3. Testing prediction...")
    test_input = np.random.rand(1, 28, 28, 1).astype('float32')
    character, confidence = predictor.predict(test_input)
    
    print(f"\n   ✅ Prediction successful!")
    print(f"   Character: {character}")
    print(f"   Confidence: {confidence:.4f}")
    
    # Test top k
    print("\n4. Testing top-k predictions...")
    top_k = predictor.predict_top_k(test_input, k=3)
    for i, pred in enumerate(top_k):
        print(f"   {i+1}. {pred['character']}: {pred['confidence']:.4f}")
    
    # Cleanup
    import os
    os.remove('test_model.h5')
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_predictor()