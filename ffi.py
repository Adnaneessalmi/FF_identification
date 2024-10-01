import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')

# Define the team lists with abbreviations
Team_1 = ["ab", "cl", "lp", "br", "vt"]
Team_2 = ["t7", "t9", "bm", "bt"]

# Mapping from abbreviations to full names
abbreviation_to_name = {
    "ab": "M1 Abrams", "cl": "Challenger", "lp": "Leopard", 
    "t7": "T-72", "t9": "T-90",
    "bm": "BMP", "br": "Bradley",
    "bt": "BTR", "vt": "M113"
}

# Define vehicle type mapping using full names
vehicle_type = {
    "M1 Abrams": "Tank", "Challenger": "Tank", "Leopard": "Tank", 
    "T-72": "Tank", "T-90": "Tank",
    "BMP": "IFV", "Bradley": "IFV",
    "BTR": "APC", "M113": "APC"
}

# Ask the user for their team
team = input("Enter your team (Team_1 or Team_2): ")

# Load video
video_path = 'vid1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties for writing output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model.predict(source=frame, save=False, save_txt=False)

    # Iterate through the results
    for result in results:
        for box in result.boxes:
            # Get box information
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            class_id = box.cls[0]
            class_abbreviation = model.names[int(class_id.item())]  # Get class abbreviation (e.g., 'ab', 't7')

            # Convert abbreviation to full name
            class_name = abbreviation_to_name.get(class_abbreviation, "Unknown")

            # Determine if friend or enemy
            if (team == "Team_1" and class_abbreviation in Team_1) or (team == "Team_2" and class_abbreviation in Team_2):
                label = "Friend"
                color = (0, 255, 0)  # Green
            else:
                label = "Enemy"
                color = (0, 0, 255)  # Red

            # Get vehicle type from full name
            v_type = vehicle_type.get(class_name, "Unknown")

            # Draw a reversed triangle (pointing downwards)
            center_x = (x1 + x2) // 2
            top_y = y1 - 20
            triangle_points = [(center_x, top_y + 30), (center_x - 15, top_y), (center_x + 15, top_y)]
            cv2.drawContours(frame, [np.array(triangle_points)], 0, color, -1)

            # Put the label and vehicle type below the triangle
            text = f"{label}: {class_name} {v_type}"
            cv2.putText(frame, text, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write the frame to the output video
    output.write(frame)

# Release resources
cap.release()
output.release()
