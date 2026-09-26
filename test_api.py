import requests

test_files = [
    r"C:\Users\sujay\Downloads\sample\sample\img_00eeee4fc7754a82.JPG",
    r"C:\Users\sujay\Downloads\sample\sample\img_0e7929a4069c45bd.jpg"
]

for p in test_files:
    print(f"\nTesting: {p}")
    with open(p, "rb") as f:
        up = requests.post("http://localhost:5000/upload", files={"image": f}).json()
    res = requests.post("http://localhost:5000/execute", json=up).json()
    print("  Status:", res.get("status"))
    print("  Class :", res.get("primary_classification"))
    print("  Title :", res.get("classification_title"))
    print("  Conf  :", f"{res.get('classification_confidence')*100:.1f}%")
    print("  Boxes :", res.get("detection_count"))
