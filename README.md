# [LAB-2] ViT Image Classification Web API

This project deploys a deep learning **Vision Transformer (ViT)** model from Hugging Face as a Web API using **FastAPI**, developed for the Computational Thinking course



## 1. Student Information
- **Full Name:** Trần Vĩnh Kiệt
- **Student ID:** 25127399
- **Course:** Computational Thinking
- **Faculty:** Faculty of Information Technology
- **University:** University of Science, VNU-HCM



## 2. Model Selection
- **Model Name:** `google/vit-base-patch16-224`
- **Hugging Face Hub:** [https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- **Type**: Image Classification (Vision Transformer)  
- **Description**: A Vision Transformer model pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k. It classifies images into 1000 ImageNet classes and returns the top-k predictions with confidence scores.



## 3. System Architecture and Functionality
The system is built as a FastAPI web service that accepts an image file (JPEG, PNG) and returns the top-5 classification results from the ViT model in JSON format

### API Endpoints
| Method | Endpoint     | Description                          |
|--------|--------------|--------------------------------------|
| GET    | `/`          | Service introduction and available endpoints |
| GET    | `/health`    | Health check (model loaded status)   |
| POST   | `/predict`   | Upload an image and get classification results |




## 4. Clone Repository
Clone the repository:
```bash
git clone <https://github.com/BryantVinhz/Lab-Computational-Thinking>
```



## 5. Install requirement dependencies
```bash
pip install -r requirements.txt
```


## 6. Instructions for running the application
- Open Lab02.ipynb in VS Code or Jupyter Lab
- Execute all cells sequentially
- Result: The server will start on http://0.0.0.0:8000


## 7. Instructions for calling the API, including example requests and responses
```
python test_api.py
```



## 8. Project Structure
```text
Lab-Computational-Thinking/
├── Data/
│   └── Collie.jpg         # Sample image for valid testing
|   └── STF.jpg
├── text.txt               # Non-image text file for error-handling validation
├── config.yaml            # Model selection and Top-K configurations
├── Lab02.ipynb            # Jupyter Notebook for model loading and server execution
├── test_api.py            # Automated testing script using the requests library
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```


## 9. Link to Video: 
[https://drive.google.com/drive/folders/19Qdo1xF7iXEZFt3o5xtPYSLb_LPiis-J?usp=sharing]





