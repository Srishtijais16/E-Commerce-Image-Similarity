import os
import pickle
import numpy as np
import sqlite3
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision.models import mobilenet_v2
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Check for GPU/Metal backend
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load MobileNetV2 model
model = mobilenet_v2(pretrained=True).to(device)
model.eval()

# Allowed extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Database connection
def get_db_connection():
    conn = sqlite3.connect('ecommerce.db')
    conn.row_factory = sqlite3.Row
    return conn

# Function to extract features using the MobileNetV2 model
def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()
    return features

# Precompute features (called once)
if not os.path.exists('features.pkl'):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM products")
    products = cursor.fetchall()
    conn.close()

    features_dict = {}
    for product in products:
        image_path = f"static/images/{product['id']}.jpg"
        features_dict[product['id']] = extract_features(image_path)

    with open('features.pkl', 'wb') as f:
        pickle.dump(features_dict, f)

# Load precomputed features
with open('features.pkl', 'rb') as f:
    precomputed_features = pickle.load(f)

# Function to find similar images
def find_similar_images(uploaded_image_features):
    similarities = []
    for product_id, db_features in precomputed_features.items():
        similarity = cosine_similarity([uploaded_image_features], [db_features])[0][0]
        if similarity == 1.0:
            continue
        similarities.append((product_id, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    threshold = 0.8
    filtered_similarities = [sim for sim in similarities if sim[1] >= threshold]
    return filtered_similarities[:3]

# Routes
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filepath = os.path.join('temp', file.filename)
    file.save(filepath)

    uploaded_image_features = extract_features(filepath)
    os.remove(filepath)

    similar_images = find_similar_images(uploaded_image_features)
    
    # Fetch details for all similar images
    conn = get_db_connection()
    similar_products_details = []
    
    if similar_images:
        for sim in similar_images:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM products WHERE id = ?", (sim[0],))
            similar_product_detail = cursor.fetchone()  # Fetch each similar product's details
            if similar_product_detail:
                similar_products_details.append(similar_product_detail)  # Add to list of details
        
        # Fetch details for the most similar image
        most_similar_id = similar_images[0][0]  # Get the ID of the most similar product
        cursor.execute("SELECT * FROM products WHERE id = ?", (most_similar_id,))
        product = cursor.fetchone()  # Fetch product details for the most similar product
    else:
        product = None  # No product found if no similar images

    conn.close()

    # Pass all necessary variables to the template
    return render_template('results.html', similar_products=similar_products_details, product=product, uploaded_image_id=most_similar_id if most_similar_id else None)
@app.route('/products', methods=['GET'])
def view_products():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    conn.close()
    return render_template('view_products.html', products=products)

@app.route('/product/<int:product_id>')
def product_details(product_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
    product = cursor.fetchone()
    conn.close()

    if product:
        product_image_url = f"images/{product['id']}.jpg"
        return render_template('product_details.html', product=product, product_image_url=product_image_url)
    else:
        return "Product not found", 404

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, port=5002)
