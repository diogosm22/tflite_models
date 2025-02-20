import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})  # Adjust resolution for performance
picam2.configure(config)
picam2.start()

# Load the TensorFlow Lite model
model_path = "mobilenet_v2.tflite"  # Path to your TFLite model
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input shape of the model
input_shape = input_details[0]['shape']  # e.g., [1, 128, 128, 3]
input_height, input_width = input_shape[1], input_shape[2]

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the input size expected by the model
    resized_image = cv2.resize(image, (input_width, input_height))
    
    # Normalize the image (if required by the model)
    normalized_image = resized_image / 255.0  # Scale to [0, 1]
    
    # Add batch dimension and convert to the expected data type
    input_data = np.expand_dims(normalized_image, axis=0).astype(np.float32)
    return input_data

# Function to draw detection results on the image
def draw_results(image, output_data):
    # Example: Assuming output_data contains bounding boxes and labels
    # Replace this with your actual post-processing logic
    for detection in output_data:
        # Extract bounding box coordinates (adjust based on your model's output format)
        ymin, xmin, ymax, xmax = detection['bounding_box']
        
        # Convert normalized coordinates to pixel values
        height, width, _ = image.shape
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        ymin = int(ymin * height)
        ymax = int(ymax * height)
        
        # Draw the bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Draw the label (if available)
        if 'label' in detection:
            label = detection['label']
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Main loop for capturing and processing images
try:
    while True:
        # Capture an image from the camera
        image = picam2.capture_array()
        
        # Preprocess the image for the model
        input_data = preprocess_image(image)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Post-process the output (e.g., extract bounding boxes, labels, etc.)
        # Replace this with your actual post-processing logic
        detections = []  # Example: List of detections
        for detection in output_data[0]:  # Adjust based on your model's output format
            detections.append({
                'bounding_box': detection[:4],  # Example: [ymin, xmin, ymax, xmax]
                'label': f"Class {np.argmax(detection[4:])}"  # Example: Class label
            })
        
        # Draw detection results on the image
        draw_results(image, detections)
        
        # Display the image with OpenCV
        cv2.imshow("Object Detection", image)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    picam2.stop()
    cv2.destroyAllWindows()