import tensorflow as tf

# Path to the directory containing saved_model.pb and variables/
saved_model_dir = "C:/Users/diogo/Desktop/python/ModelosRPi/mobilenetv2"

# Check if the SavedModel directory exists
import os
if not os.path.exists(saved_model_dir):
    raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")

# Check if saved_model.pb and variables/ exist
if not os.path.exists(os.path.join(saved_model_dir, "saved_model.pb")):
    raise FileNotFoundError(f"saved_model.pb not found in: {saved_model_dir}")
if not os.path.exists(os.path.join(saved_model_dir, "variables")):
    raise FileNotFoundError(f"variables folder not found in: {saved_model_dir}")

# Convert the SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "ModelosRPi/mobilenet_v2.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model successfully converted and saved to: {tflite_model_path}")