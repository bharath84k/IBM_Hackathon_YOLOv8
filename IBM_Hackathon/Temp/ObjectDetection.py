import cv2
import numpy as np

# Load YOLOv5 pre-trained model
net = cv2.dnn.readNetFromDarknet('yolov5s.cfg', 'yolov5s.weights')

# Specify the classes that the model can detect
classes = ['person',]

# Replace the URL with the video stream link
url = "http://192.168.137.137:6677"+"/videofeed?username=&password="

# Open the video stream
cap = cv2.VideoCapture(url)

while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    if ret:
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Set the input to the model and perform a forward pass
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        # Loop over the outputs and filter detections based on class confidence
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in classes:
                    # Extract the bounding box coordinates and draw the box and label
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (x, y, w, h) = box.astype("int")
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = "{}: {:.4f}".format(classes[class_id], confidence)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame in a window
        cv2.imshow('Video Stream', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
