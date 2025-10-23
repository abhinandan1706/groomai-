import tensorflow as tf
import os

# Custom InputLayer to handle batch_shape parameter for backward compatibility
class CompatibleInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            # Convert batch_shape to input_shape
            if 'input_shape' not in kwargs:
                kwargs['input_shape'] = batch_shape[1:]
        # Remove batch_shape from kwargs as it's not supported in newer versions
        kwargs.pop('batch_shape', None)
        super().__init__(**kwargs)

def test_model():
    model_path = r'C:\Users\Administrator\Downloads\GroomAI-model\model\groomai_skin_model.h5'
    
    print("=== GroomAI Model Test ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        try:
            print("Loading model...")
            # Try loading with compile=False first
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Model loaded successfully (without compilation)!")
            except Exception as e1:
                print(f"Failed to load without compilation: {e1}")
                # Try with custom objects if needed
                try:
                    custom_objects = {'InputLayer': tf.keras.layers.InputLayer}
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                    print("✅ Model loaded successfully (with custom objects)!")
                except Exception as e2:
                    print(f"Failed to load with custom objects: {e2}")
                    raise e1
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            
            # Test prediction with dummy data
            import numpy as np
            dummy_input = np.random.rand(1, 224, 224, 3)
            print("Testing prediction with dummy data...")
            prediction = model.predict(dummy_input)
            print(f"✅ Prediction successful! Shape: {prediction.shape}")
            print(f"Prediction values: {prediction[0]}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Model file not found!")

if __name__ == "__main__":
    test_model()
