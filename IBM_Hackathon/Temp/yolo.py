from ultralytics import YOLO
url1 = "http://192.168.137.137:6677"+"/videofeed?username=&password="
url = "http://172.20.100.20:6677"+"/videofeed?username=&password="

# Load a model
#model = YOLO('yolov8n.pt')  # load an official detection model
model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
#model = YOLO('path/to/best.pt')  # load a custom model

# Track with the model
results = model.track(source=url, show=True) 
results = model.track(source=url, show=True, tracker="bytetrack.yaml") 
