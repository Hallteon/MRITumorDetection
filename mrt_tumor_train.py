from ultralytics import YOLO


model = YOLO("yolov8l.yaml")

model.train(data="data.yaml", epochs=3, device='0')
