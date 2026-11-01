import os
import requests

BASE_URL = "http://127.0.0.1:8000"

def run_tests():

    print("1. Test GET /")
    res = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}\n")


    print("2. Test GET /health")
    res = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}\n")

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    print("3. Test POST /predict")
    image_path = os.path.join(CURRENT_DIR, "Data", "Collie.jpg") 

    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            files = {"file": (image_path, img, "image/jpeg")}
            res = requests.post(f"{BASE_URL}/predict", files=files)
            print(f"Status Code: {res.status_code}")
            print(f"Response: {res.json()}\n")
    else:
        print("Not found the image")

    print("4. Test POST /predict (Test file text.txt)")
    text_path = os.path.join(CURRENT_DIR, "text.txt")
    if os.path.exists(text_path):
        with open(text_path, "rb") as txt_file:
            files = {"file": ("text.txt", txt_file, "text/plain")}
            res = requests.post(f"{BASE_URL}/predict", files=files)
            print(f"Status Code: {res.status_code}")
            print(f"Response: {res.json()}\n")

if __name__ == "__main__":
    run_tests()