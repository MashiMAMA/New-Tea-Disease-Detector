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
# Configuration
LAST_CONV_LAYER_NAME = "conv2d_2"  # Default: last conv layer (64x64 resolution, 128 channels)
# Alternative options:
# - "conv2d_1": Higher resolution (128x128, 64 channels) - better for small features
# - "conv2d": Highest resolution (256x256, 32 channels) - very detailed but may be noisy


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
    """Generate Grad-CAM++ heatmap for a given image with improved accuracy"""
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
        
        # Get conv layer output and gradients using second-order derivatives for Grad-CAM++
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                conv_output = conv_model(img_array)
                tape2.watch(conv_output)
                preds = classifier_model(conv_output)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]
            
            # First order gradient
            grads = tape2.gradient(class_channel, conv_output)
        
        # Second order gradient (for Grad-CAM++)
        second_grads = tape1.gradient(grads, conv_output)
        
        # Get values
        conv_output_value = conv_output[0].numpy()
        grads_value = grads[0].numpy()
        
        # Apply ReLU to gradients (Guided Grad-CAM approach)
        # This ensures we only use positive gradients that support the prediction
        grads_value = np.maximum(grads_value, 0)
        
        # Grad-CAM++: Use second-order gradients for better weighting
        if second_grads is not None:
            second_grads_value = second_grads[0].numpy()
            third_grads_value = grads_value * grads_value * grads_value  # Third order approximation
            
            # Calculate alpha weights (spatial importance)
            global_sum = np.sum(conv_output_value, axis=(0, 1), keepdims=True)
            alpha_denom = second_grads_value * 2.0 + global_sum * third_grads_value + 1e-10
            alpha = second_grads_value / alpha_denom
            
            # Apply ReLU to alpha
            alpha = np.maximum(alpha, 0)
            
            # Weight by alpha and ReLU gradients
            weights = np.sum(alpha * np.maximum(grads_value, 0), axis=(0, 1))
        else:
            # Fallback to standard Grad-CAM with guided approach
            weights = np.mean(grads_value, axis=(0, 1))
        
        # Weight each channel by its importance
        for i in range(weights.shape[0]):
            conv_output_value[:, :, i] *= weights[i]
        
        # Create heatmap by averaging weighted channels
        heatmap = np.mean(conv_output_value, axis=-1)
        
        # Apply ReLU to focus on positive contributions only
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        heatmap_max = np.max(heatmap)
        heatmap_min = np.min(heatmap)
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            print(f"   Warning: Heatmap has no variation")
            return None
        
        # Apply power transformation to enhance high-activation regions
        # This makes the important regions stand out more
        heatmap = np.power(heatmap, 0.8)  # Less than 1 enhances mid-values
        
        # Apply slight smoothing to reduce noise
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        # Re-normalize after processing
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        print(f"   Heatmap stats - min: {heatmap.min():.3f}, max: {heatmap.max():.3f}, mean: {heatmap.mean():.3f}, std: {heatmap.std():.3f}")
        print(f"   Heatmap shape: {heatmap.shape}")
        print(f"   Using layer: {last_conv_layer_name}")
        
        return heatmap
    
    except Exception as e:
        print(f"   Error generating Grad-CAM: {e}")
        traceback.print_exc()
        return None

def create_gradcam_overlay(img_array_original, heatmap, original_image_array=None, alpha=0.6):
    """Overlay Grad-CAM heatmap on original image with improved visualization
    
    Args:
        img_array_original: 256x256 normalized image used for model inference
        heatmap: Grad-CAM heatmap (smaller than 256x256)
        original_image_array: Original image array with true aspect ratio (optional)
        alpha: Blending factor for overlay
    """
    try:
        # Use original aspect ratio image if provided, otherwise use 256x256
        if original_image_array is not None:
            img = original_image_array.copy()
            print(f"   Using original image shape: {img.shape}")
        else:
            # Fallback to 256x256 image
            img = (img_array_original * 255).astype(np.uint8)
            print(f"   Using model input image shape: {img.shape}")
        
        print(f"   Heatmap shape before resize: {heatmap.shape}")
        
        # Resize heatmap to match the target image size using high-quality interpolation
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Normalize heatmap to [0, 1] if not already
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()
        
        # Convert heatmap to 0-255 range
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap (JET: blue=low importance, red=high importance)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create the overlay with weighted blending
        # Higher alpha = more heatmap visibility
        superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Optional: Enhance contrast for better visibility
        # Clip values to ensure they're in valid range
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        print(f"   Overlay created - alpha={alpha}, output shape: {superimposed_img.shape}")
        print(f"   Output range: [{superimposed_img.min()}, {superimposed_img.max()}]")
        
        return superimposed_img
    
    except Exception as e:
        print(f"   Error creating overlay: {e}")
        traceback.print_exc()
        return None

def preprocess_image(image_bytes):
    """Preprocess image to match model input requirements, returns original image too"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        print(f"   Original image mode: {img.mode}, size: {img.size}")
        
        # Store original image
        original_img = img.copy()
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            original_img = original_img.convert('RGB')
            print(f"   Converted to RGB")
        
        # Resize to model input size (256x256) for inference
        img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        img_array = img_array.astype('float32') / 255.0
        img_array_with_batch = np.expand_dims(img_array, axis=0)
        
        # Also return original image as numpy array for proper aspect ratio overlay
        original_array = np.array(original_img)
        
        print(f"   Original size: {original_img.size}, Model input: {img_resized.size}")
        print(f"   Batch shape: {img_array_with_batch.shape}")
        return img_array_with_batch, img_array, original_array
    
    except Exception as e:
        print(f"   ✗ Preprocessing error: {e}")
        traceback.print_exc()
        raise
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
        processed_image, _, _ = preprocess_image(image_bytes)
        
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
        
        # Get optional layer parameter for experimentation
        # Options: conv2d (256x256), conv2d_1 (128x128), conv2d_2 (64x64 - default)
        layer_name = request.form.get('layer', LAST_CONV_LAYER_NAME)
        if layer_name not in ['conv2d', 'conv2d_1', 'conv2d_2']:
            layer_name = LAST_CONV_LAYER_NAME
        
        print(f"📸 Received image: {image_file.filename}")
        print(f"🎯 Using Grad-CAM layer: {layer_name}")
        
        # Preprocess image
        image_bytes = image_file.read()
        processed_image, img_array_original, original_img_array = preprocess_image(image_bytes)
        
        # Make prediction
        print("\n🤖 Making prediction...")
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class = CLASS_NAMES[predicted_class_idx]
        display_name = DISPLAY_NAMES[predicted_class]
        
        print(f"✓ Prediction: {display_name} ({confidence*100:.2f}%)")
        
        # Generate Grad-CAM with selected layer
        print(f"\n🔥 Generating Grad-CAM heatmap with {layer_name}...")
        heatmap = make_gradcam_heatmap(processed_image, model, layer_name, predicted_class_idx)
        
        gradcam_image_base64 = None
        if heatmap is not None:
            # Use original image array to preserve aspect ratio
            overlay_img = create_gradcam_overlay(img_array_original, heatmap, original_img_array)
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
    print(f"   http://10.0.2.2:5001")
    
    print(f"\n📍 For PHYSICAL DEVICE:")
    print(f"   http://{local_ip}:5001")
    
    print(f"\n🌐 Test in browser:")
    print(f"   http://localhost:5001")
    
    print("\n" + "=" * 60)
    print("Starting server...")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)



































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
