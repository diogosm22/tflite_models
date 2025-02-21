#!/usr/bin/python3

import cv2
import numpy as np
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

def set_input_tensor(interpreter, image):
    tensor_index = input_details[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

while True:
    frame = picam2.capture_array()
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    input_image = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_image = np.expand_dims(input_image, axis=0)

    # Set the input tensor
    set_input_tensor(interpreter, input_image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)

    # Draw bounding boxes and labels on the image
    for i in range(len(scores)):
        if scores[i] > 0.5 and int(classes[i]) == 1:  # Confidence threshold and class 1 for person
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * image.shape[1])
            xmax = int(xmax * image.shape[1])
            ymin = int(ymin * image.shape[0])
            ymax = int(ymax * image.shape[0])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f'Person: {int(scores[i] * 100)}%'
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Camera", image)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()