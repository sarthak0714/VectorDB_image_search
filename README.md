# Streamlit Image Search App

This is a web application built with Streamlit that allows users to upload an image and search for similar images using a pre-trained model and Zilliz API.

## Features

- Upload an image file (JPEG, JPG, or PNG).
- Extract image embeddings using a pre-trained model.
- Search for similar images using Zilliz API.
- Display the top 4 similar images.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git

2. Navigate to the project directory:

   ```bash
   cd your-repo
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
5. Create a .env file in the project directory and add the following lines:
   ```bash
   ZILLIZ_URI="URI"
ZILLIZ_API_KEY=YOUR_API_KEY


## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
2. Access the app in your browser at http://localhost:8501.
3. Upload an image file using the provided file uploader.
4. The app will display the uploaded image and show the top 4 similar images based on the uploaded image.
