from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input image
image_path = "D:\\Python\\Vehicle Detection Project\\sample images\\OIP.jpg"

# Run detection
results = model(image_path)

# Process results
for result in results:

    boxes = result.boxes

    for box in boxes:

        # Class ID
        class_id = int(box.cls[0])

        # Class name
        class_name = model.names[class_id]

        # Confidence
        confidence = float(box.conf[0])

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print("--------------------------------")
        print("Object     :", class_name)
        print("Confidence :", f"{confidence:.2f}")
        print("Bounding Box:")
        print("x1 =", x1)
        print("y1 =", y1)
        print("x2 =", x2)
        print("y2 =", y2)




# Display result
for result in results:
    result.show()

# Save result
results[0].save(filename="phase1_result.jpg")