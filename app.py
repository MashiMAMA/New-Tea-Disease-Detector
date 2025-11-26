from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import socket
import traceback
import cv2
import base64

app = Flask(__name__)
CORS(app)

# Load your trained model
MODEL_PATH = r"D:\Research\5 Preprocessed\tea_leaf_model5.h5"

print("=" * 60)
print("Loading TensorFlow model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Build the model with dummy input
    dummy_input = tf.zeros((1, 256, 256, 3))
    _ = model(dummy_input)
    
    print(f"✓ Model loaded successfully from: {MODEL_PATH}")
    print(f"✓ Model input shape: {model.input_shape}")
    print(f"✓ Model output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    traceback.print_exc()
    model = None

# Class names - MUST match training order EXACTLY
CLASS_NAMES = [
    "BrownBlight",
    "GrayBlight",
    "GreenMiridBug",
    "HealthyLeaf",
    "Helopeltis",
    "RedSpider",
    "TeaAlgalLeafSpot"
]

DISPLAY_NAMES = {
    "BrownBlight": "Brown Blight",
    "GrayBlight": "Gray Blight",
    "GreenMiridBug": "Green Mirid Bug",
    "HealthyLeaf": "Healthy Leaf",
    "Helopeltis": "Helopeltis",
    "RedSpider": "Red Spider",
    "TeaAlgalLeafSpot": "Tea Algal Leaf Spot"
}

# Disease information dictionary
DISEASE_INFO = {
    "BrownBlight": {
        "scientific_name": "Colletotrichum camelliae",
        "description": "Brown blight causes rapidly spreading brown patches on young leaves and shoots, leading to defoliation. Control measures: remove infected foliage immediately, improve plant ventilation, avoid water stress, use copper-based treatments, and ensure balanced fertilization. Most common during rainy seasons."
    },
    "GrayBlight": {
        "scientific_name": "Pestalotiopsis theae",
        "description": "Gray blight appears as gray-brown lesions with distinct margins, often affecting mature leaves. Management includes: pruning infected areas, reducing leaf wetness through proper spacing, applying protective fungicides, and maintaining plant vigor through adequate nutrition."
    },
    "GreenMiridBug": {
        "scientific_name": "Helopeltis theivora",
        "description": "Green mirid bug is a pest that feeds on tender shoots, leaves, and buds, causing necrotic lesions and stunted growth. Control: regular monitoring, removal of affected parts, use of insecticides like neem oil or synthetic pyrethroids, and maintaining natural predators."
    },
    "HealthyLeaf": {
        "scientific_name": "No disease or pest detected",
        "description": "The tea leaf appears to be healthy with no visible signs of disease or pest damage. Continue regular care and monitoring practices. Ensure proper watering, fertilization, pest management, and pruning."
    },
    "Helopeltis": {
        "scientific_name": "Helopeltis spp.",
        "description": "Helopeltis (tea mosquito bug) causes characteristic sunken necrotic lesions on stems, leaves, and young shoots. Management: regular inspection, pruning affected parts, apply approved insecticides, maintain shade levels, and encourage natural enemies."
    },
    "RedSpider": {
        "scientific_name": "Oligonychus coffeae",
        "description": "Red spider mite infestation causes bronzing of leaves, with fine webbing visible on undersides. Control: maintain adequate moisture, remove heavily infested leaves, use acaricides or miticides, encourage predatory mites, and avoid water stress."
    },
    "TeaAlgalLeafSpot": {
        "scientific_name": "Cephaleuros virescens",
        "description": "Tea algal leaf spot is caused by parasitic algae and appears as orange-brown to rusty-red circular spots on the upper leaf surface. Management: improve air circulation, reduce humidity, prune overcrowded branches, and apply copper-based fungicides if severe."
    }
}

# Grad-CAM Configuration
IMG_SIZE = (256, 256)
LAST_CONV_LAYER_NAME = "conv2d_2"  # Update if your model has different layer name

def get_local_ip():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "Unable to determine"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for a given image"""
    try:
        # Find the target layer
        last_conv_layer_idx = None
        for idx, layer in enumerate(model.layers):
            if layer.name == last_conv_layer_name:
                last_conv_layer_idx = idx
                break
        
        if last_conv_layer_idx is None:
            print(f"   Warning: Layer {last_conv_layer_name} not found")
            return None
        
        # Create model up to conv layer
        conv_model = keras.Model(
            inputs=model.layers[0].input,
            outputs=model.layers[last_conv_layer_idx].output
        )
        
        # Create model from conv layer to output
        classifier_input = keras.Input(shape=model.layers[last_conv_layer_idx].output.shape[1:])
        x = classifier_input
        for layer in model.layers[last_conv_layer_idx + 1:]:
            x = layer(x)
        classifier_model = keras.Model(inputs=classifier_input, outputs=x)
        
        # Get conv layer output and gradients
        with tf.GradientTape() as tape:
            conv_output = conv_model(img_array)
            tape.watch(conv_output)
            preds = classifier_model(conv_output)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_output)
        
        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get values
        conv_output_value = conv_output[0].numpy()
        pooled_grads_value = pooled_grads.numpy()
        
        # Weight each channel by its importance
        # CRITICAL FIX: Only use positive gradients (ReLU on gradients)
        # This ensures we only highlight features that SUPPORT the prediction
        for i in range(pooled_grads_value.shape[0]):
            conv_output_value[:, :, i] *= pooled_grads_value[i]
        
        # Create heatmap by averaging weighted channels
        heatmap = np.mean(conv_output_value, axis=-1)
        
        # CRITICAL FIX: Apply ReLU to focus on positive contributions only
        # This removes negative values that would show "anti-features"
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1] for better visualization
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        print(f"   Heatmap stats - min: {heatmap.min():.3f}, max: {heatmap.max():.3f}, mean: {heatmap.mean():.3f}")
        
        return heatmap
    
    except Exception as e:
        print(f"   Error generating Grad-CAM: {e}")
        traceback.print_exc()
        return None

def create_gradcam_overlay(img_array_original, heatmap, alpha=0.5):
    """Overlay Grad-CAM heatmap on original image"""
    try:
        # img_array_original is already 256x256x3 normalized to [0,1]
        img = (img_array_original * 255).astype(np.uint8)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
        
        # Convert heatmap to 0-255 range
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap (JET: blue=low importance, red=high importance)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend: more alpha = more heatmap visibility
        # Improved blending for better visibility
        superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        
        print(f"   Overlay created with alpha={alpha}")
        
        return superimposed_img
    
    except Exception as e:
        print(f"   Error creating overlay: {e}")
        traceback.print_exc()
        return None

def preprocess_image(image_bytes):
    """Preprocess image to match model input requirements"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        print(f"   Original image mode: {img.mode}, size: {img.size}")
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"   Converted to RGB")
        
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array_with_batch = np.expand_dims(img_array, axis=0)
        
        print(f"   Final shape: {img_array_with_batch.shape}")
        return img_array_with_batch, img_array
    
    except Exception as e:
        print(f"   ✗ Preprocessing error: {e}")
        traceback.print_exc()
        raise

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/')
def home():
    return jsonify({
        "message": "Tea Disease Detection API with Grad-CAM",
        "status": "running",
        "model_loaded": model is not None,
        "version": "2.0",
        "features": ["prediction", "grad_cam_visualization"],
        "endpoints": {
            "/predict": "POST - Upload image for disease detection",
            "/predict_with_gradcam": "POST - Upload image for detection with Grad-CAM",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_count": len(CLASS_NAMES),
        "gradcam_enabled": True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Regular prediction without Grad-CAM"""
    print("\n" + "="*60)
    print("PREDICTION REQUEST (No Grad-CAM)")
    print("="*60)
    
    try:
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500
        
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        print(f"📸 Received image: {image_file.filename}")
        
        image_bytes = image_file.read()
        processed_image, _ = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class = CLASS_NAMES[predicted_class_idx]
        display_name = DISPLAY_NAMES[predicted_class]
        
        print(f"✓ Prediction: {display_name} ({confidence*100:.2f}%)")
        
        disease_data = DISEASE_INFO.get(predicted_class, {
            "scientific_name": "Unknown",
            "description": "No information available."
        })
        
        response = {
            "success": True,
            "prediction": {
                "disease_name": display_name,
                "scientific_name": disease_data["scientific_name"],
                "description": disease_data["description"],
                "confidence": round(confidence * 100, 2)
            },
            "all_predictions": {
                DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(predictions[0][i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/predict_with_gradcam', methods=['POST'])
def predict_with_gradcam():
    """Prediction WITH Grad-CAM visualization"""
    print("\n" + "="*60)
    print("PREDICTION REQUEST (With Grad-CAM)")
    print("="*60)
    
    try:
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500
        
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        print(f"📸 Received image: {image_file.filename}")
        
        # Preprocess image
        image_bytes = image_file.read()
        processed_image, img_array_original = preprocess_image(image_bytes)
        
        # Make prediction
        print("\n🤖 Making prediction...")
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class = CLASS_NAMES[predicted_class_idx]
        display_name = DISPLAY_NAMES[predicted_class]
        
        print(f"✓ Prediction: {display_name} ({confidence*100:.2f}%)")
        
        # Generate Grad-CAM
        print("\n🔥 Generating Grad-CAM heatmap...")
        heatmap = make_gradcam_heatmap(processed_image, model, LAST_CONV_LAYER_NAME, predicted_class_idx)
        
        gradcam_image_base64 = None
        if heatmap is not None:
            overlay_img = create_gradcam_overlay(img_array_original, heatmap)
            if overlay_img is not None:
                gradcam_image_base64 = image_to_base64(overlay_img)
                print("✓ Grad-CAM generated successfully")
            else:
                print("⚠ Failed to create overlay")
        else:
            print("⚠ Failed to generate heatmap")
        
        # Get disease info
        disease_data = DISEASE_INFO.get(predicted_class, {
            "scientific_name": "Unknown",
            "description": "No information available."
        })
        
        # Prepare response
        response = {
            "success": True,
            "prediction": {
                "disease_name": display_name,
                "scientific_name": disease_data["scientific_name"],
                "description": disease_data["description"],
                "confidence": round(confidence * 100, 2)
            },
            "all_predictions": {
                DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(predictions[0][i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "gradcam_image": gradcam_image_base64  # Base64 encoded PNG
        }
        
        print(f"\n📤 Sending response (Grad-CAM included: {gradcam_image_base64 is not None})")
        return jsonify(response)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🍃 TEA DISEASE DETECTION API WITH GRAD-CAM")
    print("=" * 60)
    
    if model:
        print(f"✓ Model: Loaded successfully")
        print(f"✓ Classes: {len(CLASS_NAMES)} disease/pest types")
        print(f"✓ Class names: {CLASS_NAMES}")
        print(f"✓ Grad-CAM layer: {LAST_CONV_LAYER_NAME}")
    else:
        print("✗ Model: Failed to load!")
    
    print("\n" + "=" * 60)
    print("📱 CONNECTION INFORMATION")
    print("=" * 60)
    
    local_ip = get_local_ip()
    
    print(f"\n📍 For Android EMULATOR:")
    print(f"   http://10.0.2.2:5000")
    
    print(f"\n📍 For PHYSICAL DEVICE:")
    print(f"   http://{local_ip}:5000")
    
    print(f"\n🌐 Test in browser:")
    print(f"   http://localhost:5000")
    
    print("\n" + "=" * 60)
    print("Starting server...")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)



































# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import os
# import socket
# import traceback

# app = Flask(__name__)
# CORS(app)

# # Load your trained model
# MODEL_PATH = r"D:\Research\5 Preprocessed\tea_leaf_model5.h5"

# print("=" * 60)
# print("Loading TensorFlow model...")
# try:
#     # Load model without compiling first
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
#     # Manually compile the model
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     print(f"✓ Model loaded successfully from: {MODEL_PATH}")
#     print(f"✓ Model input shape: {model.input_shape}")
#     print(f"✓ Model output shape: {model.output_shape}")
#     print(f"✓ Expected input: {model.input_shape}")
# except Exception as e:
#     print(f"✗ Error loading model: {e}")
#     print("Full traceback:")
#     traceback.print_exc()
#     model = None

# # Define your class names - MUST match training order EXACTLY
# # These are the internal class names used by the model (no spaces, camelCase)
# CLASS_NAMES = [
#     "BrownBlight",
#     "GrayBlight",
#     "GreenMiridBug",
#     "HealthyLeaf",
#     "Helopeltis",
#     "RedSpider",
#     "TeaAlgalLeafSpot"
# ]

# # Display names for user interface (with spaces for readability)
# DISPLAY_NAMES = {
#     "BrownBlight": "Brown Blight",
#     "GrayBlight": "Gray Blight",
#     "GreenMiridBug": "Green Mirid Bug",
#     "HealthyLeaf": "Healthy Leaf",
#     "Helopeltis": "Helopeltis",
#     "RedSpider": "Red Spider",
#     "TeaAlgalLeafSpot": "Tea Algal Leaf Spot"
# }

# # Disease information dictionary (using display names with spaces)
# DISEASE_INFO = {
#     "BrownBlight": {
#         "scientific_name": "Colletotrichum camelliae",
#         "description": "Brown blight causes rapidly spreading brown patches on young leaves and shoots, leading to defoliation. Control measures: remove infected foliage immediately, improve plant ventilation, avoid water stress, use copper-based treatments, and ensure balanced fertilization. Most common during rainy seasons."
#     },
#     "GrayBlight": {
#         "scientific_name": "Pestalotiopsis theae",
#         "description": "Gray blight appears as gray-brown lesions with distinct margins, often affecting mature leaves. Management includes: pruning infected areas, reducing leaf wetness through proper spacing, applying protective fungicides, and maintaining plant vigor through adequate nutrition. Avoid mechanical damage to leaves."
#     },
#     "GreenMiridBug": {
#         "scientific_name": "Helopeltis theivora",
#         "description": "Green mirid bug is a pest that feeds on tender shoots, leaves, and buds, causing necrotic lesions and stunted growth. Control: regular monitoring, removal of affected parts, use of insecticides like neem oil or synthetic pyrethroids, and maintaining natural predators. Inspect regularly during active growth periods."
#     },
#     "HealthyLeaf": {
#         "scientific_name": "No disease or pest detected",
#         "description": "The tea leaf appears to be healthy with no visible signs of disease or pest damage. Continue regular care and monitoring practices. Ensure proper watering, fertilization, pest management, and pruning. Maintain optimal growing conditions for continued health."
#     },
#     "Helopeltis": {
#         "scientific_name": "Helopeltis spp.",
#         "description": "Helopeltis (tea mosquito bug) causes characteristic sunken necrotic lesions on stems, leaves, and young shoots. The damage appears as dark brown or black spots that can girdle stems. Management: regular inspection, pruning affected parts, apply approved insecticides, maintain shade levels, and encourage natural enemies like spiders and wasps."
#     },
#     "RedSpider": {
#         "scientific_name": "Oligonychus coffeae",
#         "description": "Red spider mite infestation causes bronzing of leaves, with fine webbing visible on undersides. Leaves may become dry and fall premataturately. Control: maintain adequate moisture, remove heavily infested leaves, use acaricides or miticides, encourage predatory mites, and avoid water stress. More common in dry conditions."
#     },
#     "TeaAlgalLeafSpot": {
#         "scientific_name": "Cephaleuros virescens",
#         "description": "Tea algal leaf spot is caused by parasitic algae and appears as orange-brown to rusty-red circular spots on the upper leaf surface. The spots have a velvety texture. Management: improve air circulation around plants, reduce humidity levels, prune overcrowded branches to increase sunlight penetration, and apply copper-based fungicides if infection is severe. More prevalent in shaded, humid conditions."
#     }
# }

# def get_local_ip():
#     """Get the local IP address of the machine"""
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         s.connect(("8.8.8.8", 80))
#         ip = s.getsockname()[0]
#         s.close()
#         return ip
#     except Exception:
#         return "Unable to determine"

# def preprocess_image(image_bytes):
#     """Preprocess image to match model input requirements"""
#     try:
#         # Open image
#         img = Image.open(io.BytesIO(image_bytes))
#         print(f"   Original image mode: {img.mode}, size: {img.size}")
        
#         # Convert to RGB if needed
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
#             print(f"   Converted to RGB")
        
#         # Resize to model input size (256x256)
#         img = img.resize((256, 256))
#         print(f"   Resized to: {img.size}")
        
#         # Convert to array
#         img_array = np.array(img)
#         print(f"   Array shape before normalization: {img_array.shape}")
#         print(f"   Array dtype: {img_array.dtype}")
#         print(f"   Array min/max: {img_array.min()}/{img_array.max()}")
        
#         # Normalize to [0, 1]
#         img_array = img_array.astype('float32') / 255.0
#         print(f"   After normalization min/max: {img_array.min()}/{img_array.max()}")
        
#         # Add batch dimension (1, 256, 256, 3)
#         img_array = np.expand_dims(img_array, axis=0)
#         print(f"   Final shape: {img_array.shape}")
        
#         return img_array
#     except Exception as e:
#         print(f"   ✗ Preprocessing error: {e}")
#         traceback.print_exc()
#         raise

# @app.route('/')
# def home():
#     return jsonify({
#         "message": "Tea Disease Detection API",
#         "status": "running",
#         "model_loaded": model is not None,
#         "version": "1.0",
#         "endpoints": {
#             "/predict": "POST - Upload image for disease detection",
#             "/health": "GET - Check API health",
#             "/classes": "GET - Get list of detectable diseases"
#         }
#     })

# @app.route('/health')
# def health():
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "classes_count": len(CLASS_NAMES)
#     })

# @app.route('/classes')
# def get_classes():
#     return jsonify({
#         "classes": CLASS_NAMES,
#         "count": len(CLASS_NAMES)
#     })

# @app.route('/predict', methods=['POST'])
# def predict():
#     print("\n" + "="*60)
#     print("NEW PREDICTION REQUEST")
#     print("="*60)
    
#     try:
#         # Check if model is loaded
#         if model is None:
#             print("✗ ERROR: Model not loaded")
#             return jsonify({
#                 "success": False,
#                 "error": "Model not loaded. Check server logs."
#             }), 500
        
#         # Check if image was sent
#         if 'image' not in request.files:
#             print("✗ ERROR: No image in request")
#             return jsonify({
#                 "success": False,
#                 "error": "No image provided. Please send image with key 'image'"
#             }), 400
        
#         # Get the image file
#         image_file = request.files['image']
        
#         if image_file.filename == '':
#             print("✗ ERROR: Empty filename")
#             return jsonify({
#                 "success": False,
#                 "error": "Empty filename"
#             }), 400
        
#         print(f"📸 Received image: {image_file.filename}")
        
#         # Read image bytes
#         image_bytes = image_file.read()
#         print(f"   Image size: {len(image_bytes)} bytes ({len(image_bytes)/1024:.2f} KB)")
        
#         # Preprocess image
#         print("\n🔄 Preprocessing image...")
#         try:
#             processed_image = preprocess_image(image_bytes)
#         except Exception as e:
#             print(f"✗ PREPROCESSING FAILED: {e}")
#             return jsonify({
#                 "success": False,
#                 "error": f"Image preprocessing failed: {str(e)}"
#             }), 500
        
#         # Make prediction
#         print("\n🤖 Making prediction...")
#         try:
#             predictions = model.predict(processed_image, verbose=0)
#             print(f"   Prediction array shape: {predictions.shape}")
#             print(f"   Prediction values: {predictions[0]}")
            
#             predicted_class_idx = np.argmax(predictions[0])
#             confidence = float(predictions[0][predicted_class_idx])
            
#             # Get predicted class name
#             predicted_class = CLASS_NAMES[predicted_class_idx]
#             display_name = DISPLAY_NAMES[predicted_class]
            
#             print(f"\n✓ SUCCESS!")
#             print(f"   Predicted class: {predicted_class} (displayed as: {display_name})")
#             print(f"   Confidence: {confidence*100:.2f}%")
#             print(f"   Class index: {predicted_class_idx}")
            
#         except Exception as e:
#             print(f"✗ PREDICTION FAILED: {e}")
#             traceback.print_exc()
#             return jsonify({
#                 "success": False,
#                 "error": f"Model prediction failed: {str(e)}"
#             }), 500
        
#         # Get disease information
#         disease_data = DISEASE_INFO.get(predicted_class, {
#             "scientific_name": "Unknown",
#             "description": "No information available for this disease."
#         })
        
#         # Prepare response
#         response = {
#             "success": True,
#             "prediction": {
#                 "disease_name": display_name,  # Send display name with spaces
#                 "scientific_name": disease_data["scientific_name"],
#                 "description": disease_data["description"],
#                 "confidence": round(confidence * 100, 2)
#             },
#             "all_predictions": {
#                 DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(predictions[0][i]) * 100, 2)
#                 for i in range(len(CLASS_NAMES))
#             }
#         }
        
#         print(f"\n📤 Sending response to client")
#         print("="*60 + "\n")
#         return jsonify(response)
    
#     except Exception as e:
#         print(f"\n✗ UNEXPECTED ERROR: {str(e)}")
#         traceback.print_exc()
#         print("="*60 + "\n")
#         return jsonify({
#             "success": False,
#             "error": f"Server error: {str(e)}"
#         }), 500

# if __name__ == '__main__':
#     print("=" * 60)
#     print("🍃 TEA DISEASE DETECTION API")
#     print("=" * 60)
    
#     if model:
#         print(f"✓ Model: Loaded successfully")
#         print(f"✓ Classes: {len(CLASS_NAMES)} disease/pest types")
#         print(f"✓ Class names: {CLASS_NAMES}")
#         print(f"✓ Input size: RGB images (256x256x3)")
#     else:
#         print("✗ Model: Failed to load!")
    
#     print("\n" + "=" * 60)
#     print("📱 CONNECTION INFORMATION")
#     print("=" * 60)
    
#     local_ip = get_local_ip()
    
#     print(f"\n📍 For Android EMULATOR, use in Flutter:")
#     print(f"   final String baseUrl = \"http://10.0.2.2:5000\";")
    
#     print(f"\n📍 For PHYSICAL DEVICE, use in Flutter:")
#     print(f"   final String baseUrl = \"http://{local_ip}:5000\";")
    
#     print(f"\n🌐 Test in browser:")
#     print(f"   http://localhost:5000")
#     print(f"   http://{local_ip}:5000")
    
#     print("\n" + "=" * 60)
#     print("Starting server...")
#     print("=" * 60 + "\n")
    
#     # Run on all interfaces so it's accessible from network
#     app.run(host='0.0.0.0', port=5000, debug=True)

























# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import os

# app = Flask(__name__)
# CORS(app)  # Enable CORS for Flutter app to access

# # Load your trained model
# MODEL_PATH = r"D:\Research\5 Preprocessed\tea_leaf_model5.h5"
# model = tf.keras.models.load_model(MODEL_PATH)

# # Define your class names (update these based on your actual classes)
# CLASS_NAMES = [
#     "Healthy",
#     "Algal Leaf Spot",
#     "Anthracnose",
#     "Bird Eye Spot",
#     "Brown Blight",
#     "Gray Blight",
#     "Red Leaf Spot",
#     "White Spot"
# ]

# # Disease information dictionary
# DISEASE_INFO = {
#     "Healthy": {
#         "scientific_name": "No disease detected",
#         "description": "The tea leaf appears to be healthy with no visible signs of disease. Continue regular care and monitoring."
#     },
#     "Algal Leaf Spot": {
#         "scientific_name": "Cephaleuros virescens",
#         "description": "Algal leaf spot causes orange-brown spots on leaves. Improve air circulation and reduce humidity. Apply copper-based fungicides if severe."
#     },
#     "Anthracnose": {
#         "scientific_name": "Colletotrichum spp.",
#         "description": "Anthracnose causes dark lesions on leaves and stems. Remove infected parts, improve drainage, and apply appropriate fungicides."
#     },
#     "Bird Eye Spot": {
#         "scientific_name": "Cercospora theae",
#         "description": "Characterized by circular spots with gray centers. Prune affected leaves and apply recommended fungicides. Ensure proper spacing between plants."
#     },
#     "Brown Blight": {
#         "scientific_name": "Colletotrichum camelliae",
#         "description": "Brown blight causes brown patches on leaves. Remove infected foliage, improve ventilation, and use copper-based treatments."
#     },
#     "Gray Blight": {
#         "scientific_name": "Pestalotiopsis theae",
#         "description": "Gray blight appears as gray-brown lesions. Prune infected areas, reduce leaf wetness, and apply appropriate fungicides."
#     },
#     "Red Leaf Spot": {
#         "scientific_name": "Cephaleuros parasiticus",
#         "description": "Red leaf spot causes reddish-brown spots. Improve plant nutrition, reduce humidity, and apply copper fungicides if needed."
#     },
#     "White Spot": {
#         "scientific_name": "Phoma sp.",
#         "description": "White spot disease causes white or pale spots on leaves. Remove affected leaves and improve air circulation around plants."
#     }
# }

# def preprocess_image(image_bytes):
#     """Preprocess image to match model input requirements"""
#     # Open image
#     img = Image.open(io.BytesIO(image_bytes))
    
#     # Convert to RGB if needed
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
    
#     # Resize to model input size
#     img = img.resize((256, 256))
    
#     # Convert to array and normalize
#     img_array = np.array(img)
#     img_array = img_array.astype('float32') / 255.0
    
#     # Add batch dimension
#     img_array = np.expand_dims(img_array, axis=0)
    
#     return img_array

# @app.route('/')
# def home():
#     return jsonify({
#         "message": "Tea Disease Detection API",
#         "status": "running",
#         "endpoints": {
#             "/predict": "POST - Upload image for disease detection",
#             "/health": "GET - Check API health"
#         }
#     })

# @app.route('/health')
# def health():
#     return jsonify({"status": "healthy", "model_loaded": model is not None})

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Check if image was sent
#         if 'image' not in request.files:
#             return jsonify({"error": "No image provided"}), 400
        
#         # Get the image file
#         image_file = request.files['image']
        
#         if image_file.filename == '':
#             return jsonify({"error": "Empty filename"}), 400
        
#         # Read image bytes
#         image_bytes = image_file.read()
        
#         # Preprocess image
#         processed_image = preprocess_image(image_bytes)
        
#         # Make prediction
#         predictions = model.predict(processed_image)
#         predicted_class_idx = np.argmax(predictions[0])
#         confidence = float(predictions[0][predicted_class_idx])
        
#         # Get predicted class name
#         predicted_class = CLASS_NAMES[predicted_class_idx]
        
#         # Get disease information
#         disease_data = DISEASE_INFO.get(predicted_class, {
#             "scientific_name": "Unknown",
#             "description": "No information available for this disease."
#         })
        
#         # Prepare response
#         response = {
#             "success": True,
#             "prediction": {
#                 "disease_name": predicted_class,
#                 "scientific_name": disease_data["scientific_name"],
#                 "description": disease_data["description"],
#                 "confidence": round(confidence * 100, 2)
#             },
#             "all_predictions": {
#                 CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
#                 for i in range(len(CLASS_NAMES))
#             }
#         }
        
#         return jsonify(response)
    
#     except Exception as e:
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500

# if __name__ == '__main__':
#     print("Starting Flask server...")
#     print(f"Model loaded from: {MODEL_PATH}")
#     print(f"Classes: {CLASS_NAMES}")
#     print("\nServer running on http://localhost:5000")
#     print("Use http://YOUR_LOCAL_IP:5000 for physical devices")
    
#     # Run on all interfaces so it's accessible from network
#     app.run(host='0.0.0.0', port=5000, debug=True)
