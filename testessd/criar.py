import tensorflow as tf

# Path to the SavedModel directory
saved_model_dir = 'C:/Users/diogo/Desktop/python/testessd/ssd-mobilenet-v2-tensorflow2-fpnlite-320x320-v1'


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the converted model
with open('ssd.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite format and saved as model.tflite")