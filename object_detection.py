import cv2
import numpy as np
import os

def calculate_traffic_signal(vehicle_count):
    """
    Calculate the appropriate traffic signal based on the number of vehicles detected.
    
    Args:
        vehicle_count (int): Number of vehicles detected
        
    Returns:
        tuple: (signal_color, duration_seconds)
            signal_color: 'green', 'yellow', or 'red'
            duration_seconds: How long the signal should remain in this state
    """
    if vehicle_count < 10:
        # Low traffic - short green light
        return 'green', 15
    elif vehicle_count < 20:
        # Medium traffic - medium green light
        return 'green', 30
    elif vehicle_count < 30:
        # Heavy traffic - long green light
        return 'green', 45
    else:
        # Very heavy traffic - extended green light
        return 'green', 60

def detect_objects(image_path, weights_path, config_path, output_path=None):
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    if not os.path.exists(weights_path):
        print(f"Error: YOLO weights file not found at {weights_path}")
        return
    if not os.path.exists(config_path):
        print(f"Error: YOLO config file not found at {config_path}")
        return
    
    # Load image
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape
    
    # Load YOLO model
    print("Loading YOLO model...")
    net = cv2.dnn.readNet(weights_path, config_path)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    try:
        # OpenCV 4.5.4+
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        # Older OpenCV versions
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Load class names
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Detecting objects
    print("Detecting objects...")
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on image
    class_ids = []
    confidences = []
    boxes = []
    
    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw boxes
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y + 30), font, 2, color, 2)

    # Count vehicles
    vehicle_classes = ['car', 'truck', 'bus', 'motorbike']
    vehicle_count = 0
    for i in range(len(class_ids)):
        if i in indexes:
            if classes[class_ids[i]] in vehicle_classes:
                vehicle_count += 1
    
    # Calculate traffic signal
    signal_color, duration = calculate_traffic_signal(vehicle_count)
    
    # Add traffic signal information to the image
    signal_info = f"Traffic: {vehicle_count} vehicles - Signal: {signal_color.upper()} for {duration}s"
    cv2.putText(frame, signal_info, (10, 30), font, 1.5, (0, 0, 255), 2)

    # Display result
    print(f"Found {len(indexes)} objects, including {vehicle_count} vehicles")
    print(f"Recommended traffic signal: {signal_color.upper()} for {duration} seconds")
    
    # Save or display the image
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"Result saved to {output_path}")
    else:
        cv2.imshow("Object Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage
    detect_objects(
        image_path='data/frame_0.webp',
        weights_path='yolov3.weights',
        config_path='yolov3.cfg',
        output_path='output.jpg'
    )