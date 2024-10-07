from ultralytics import  YOLO

# Load model
model = YOLO("best.pt")

results = model("Test Holds/IMG_01793.jpg", show=True)

results.show()