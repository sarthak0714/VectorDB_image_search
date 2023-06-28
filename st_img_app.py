import streamlit as st
import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import requests
import base64
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()


# Define the TIMM model name
model_name = 'efficientnet_b0'

# Load the pre-trained TIMM model
model = timm.create_model(model_name, pretrained=True)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_embeddings(image_file):
    image = Image.open(image_file)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model.forward_features(input_tensor)

    # Flatten the output embedding and resize it to have a length of 2048
    embedding = output.squeeze().view(-1)[:2048].numpy()

    return embedding.tolist()

def search_similar_images(embeddings):
    url = os.environ.get('ZILLIZ_URI')
    payload = {
        "collectionName": "fruit_data",
        "limit": 10,
        "vector": embeddings
    }
    headers = {
        "Authorization": f"Bearer {os.environ.get('ZILLIZ_API_KEY')}"
    }
    response = requests.post(url, json=payload, headers=headers)
    res = response.json()['data']
    ids = [item['id'] for item in res][:4]

    # Load the data containing image information
    with open('./output1.json', 'r') as f:
        data = json.load(f)

    image_urls = []
    for id in ids:
        image_base64 = data["rows"][id]["image_base64"]
        decoded_image = base64.b64decode(image_base64)
        image = Image.open(BytesIO(decoded_image))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_url = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        image_urls.append(image_url)

    return image_urls

def main():
    st.title('Image Search')

    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        embeddings = get_embeddings(uploaded_file)
        similar_images = search_similar_images(embeddings)

        if similar_images:
            st.subheader('Similar Images From Vector DB')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(similar_images[0], use_column_width=True)
            with col2:
                st.image(similar_images[1], use_column_width=True)
            with col3:
                st.image(similar_images[2], use_column_width=True)
            with col4:
                st.image(similar_images[3], use_column_width=True)

if __name__ == '__main__':
    main()
