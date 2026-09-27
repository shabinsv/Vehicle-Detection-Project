from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input images
images = [
    r"D:\Python\Vehicle Detection Project\sample images\OIP.jpg",
    r"D:\Python\Vehicle Detection Project\sample images\OIP (1).jpg",
    r"D:\Python\Vehicle Detection Project\sample images\OIP (2).jpg"
]

# Output folder
output_folder = r"D:\Python\Vehicle Detection Project\Phase 1\output"

# Create output folder
os.makedirs(output_folder, exist_ok=True)


for image in images:

    print("\n==============================")
    print("Testing:", image)
    print("==============================")

    # Run YOLO
    results = model(image)

    for result in results:

        # Detect objects
        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            print(
                f"{class_name} -> "
                f"{confidence:.2f}"
            )

        # Get only filename
        image_name = os.path.basename(image)

        # Output filename
        output_path = os.path.join(
            output_folder,
            "detected_" + image_name
        )

        # Save detected image
        result.save(filename=output_path)

        print("Saved:", output_path)