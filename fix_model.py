"""
Fix model loading issues and create a compatible version
"""
import tensorflow as tf
import os

def fix_model():
    model_path = r'C:\Users\Administrator\Downloads\GroomAI-model\model\groomai_skin_model.h5'
    
    try:
        # Try loading with compile=False to avoid optimizer issues
        print("Attempting to load model with compile=False...")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompile the model with current TensorFlow version
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the fixed model
        fixed_path = r'C:\Users\Administrator\Downloads\GroomAI-model\model\groomai_skin_model_fixed.h5'
        model.save(fixed_path)
        print(f"✅ Model successfully fixed and saved to: {fixed_path}")
        
        # Test the model
        import numpy as np
        test_input = np.random.rand(1, 224, 224, 3)
        predictions = model.predict(test_input)
        print(f"✅ Model test successful. Output shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix model: {e}")
        print("\nTrying alternative loading methods...")
        
        try:
            # Try with different loading options
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects=None,
                options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
            )
            print("✅ Alternative loading successful")
            return True
        except Exception as e2:
            print(f"❌ Alternative loading failed: {e2}")
            return False

if __name__ == "__main__":
    print("🔧 Fixing GroomAI model compatibility...")
    success = fix_model()
    if not success:
        print("\n💡 Consider retraining the model with current TensorFlow version")
