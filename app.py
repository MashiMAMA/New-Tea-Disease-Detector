# pyright: reportMissingImports=false

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
import cv2
import os

# Load the trained model
model = load_model("tea_leaf_model5.h5")

# Build the model
dummy_input = tf.zeros((1, 256, 256, 3))
_ = model(dummy_input)

# Load the class names exactly as in training
class_names = ['BrownBlight', 'GrayBlight', 'GreenMiridBug', 'HealthyLeaf', 'Helopeltis', 'RedSpider', 'TeaAlgalLeafSpot']

# Display names for GUI (with underscores)
display_names = ['Brown_Blight', 'Gray_Blight', 'Green_Mirid_Bug', 'Healthy_Leaf', 'Helopeltis', 'Red_Spider', 'Tea_Algal_Leaf_Spot']

# Grad-CAM configuration
IMG_SIZE = (256, 256)
LAST_CONV_LAYER_NAME = "conv2d_2"

# Global variables
image_path = None
original_image_label = None
heatmap_image_label = None
result_label = None
confidence_label = None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for a given image."""
    # Find the target layer and create submodels
    last_conv_layer_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            last_conv_layer_idx = idx
            break
    
    if last_conv_layer_idx is None:
        raise ValueError(f"Layer {last_conv_layer_name} not found in model")
    
    # Create model up to the conv layer
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
    
    # Get conv layer output
    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Compute gradients
    grads = tape.gradient(class_channel, conv_output)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Proper weighting for accurate heatmap
    conv_output_value = conv_output[0].numpy()
    pooled_grads_value = pooled_grads.numpy()
    
    # Weight each channel by the pooled gradients
    for i in range(pooled_grads_value.shape[0]):
        conv_output_value[:, :, i] *= pooled_grads_value[i]
    
    # Average over all channels to get the heatmap
    heatmap = np.mean(conv_output_value, axis=-1)
    
    # Apply ReLU and normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-10)
    
    return heatmap


def create_gradcam_overlay(img_path, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    # Load original image
    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img = keras.preprocessing.image.img_to_array(img)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    
    # Convert heatmap to RGB
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    superimposed_img = heatmap_colored * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img


def predict_with_gradcam(img_path):
    """Predict disease and generate Grad-CAM visualization."""
    try:
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)
        
        # Create overlay image
        overlay_img = create_gradcam_overlay(img_path, heatmap)
        
        return display_names[predicted_class], confidence, overlay_img
        
    except Exception as e:
        return f"Error: {str(e)}", 0, None


# GUI Functions
def upload_image():
    global image_path, original_image_label, heatmap_image_label, result_label, confidence_label
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        image_path = file_path
        
        # Display original image (use 256x256 to match model training)
        img = Image.open(file_path)
        img = img.resize((256, 256))
        img_tk = ImageTk.PhotoImage(img)
        
        if original_image_label is None:
            original_image_label = tk.Label(image_frame, image=img_tk)
            original_image_label.image = img_tk
            original_image_label.grid(row=0, column=0, padx=10, pady=10)
        else:
            original_image_label.config(image=img_tk)
            original_image_label.image = img_tk
        
        # Clear heatmap and results
        if heatmap_image_label:
            heatmap_image_label.grid_forget()
            heatmap_image_label = None
        if result_label:
            result_label.config(text="")
        if confidence_label:
            confidence_label.config(text="")


def detect_disease():
    global image_path, heatmap_image_label, result_label, confidence_label
    if image_path:
        disease, confidence, overlay_img = predict_with_gradcam(image_path)
        
        # Display result
        if result_label is None:
            result_label = tk.Label(result_frame, text=f"Prediction: {disease}", 
                                   font=("Arial", 14, "bold"), fg="blue")
            result_label.pack(pady=5)
        else:
            result_label.config(text=f"Prediction: {disease}")
        
        # Display confidence
        if confidence_label is None:
            confidence_label = tk.Label(result_frame, text=f"Confidence: {confidence:.2%}", 
                                       font=("Arial", 12), fg="green")
            confidence_label.pack(pady=5)
        else:
            confidence_label.config(text=f"Confidence: {confidence:.2%}")
        
        # Display Grad-CAM heatmap overlay
        if overlay_img is not None:
            overlay_pil = Image.fromarray(overlay_img)
            overlay_pil = overlay_pil.resize((256, 256))
            overlay_tk = ImageTk.PhotoImage(overlay_pil)
            
            if heatmap_image_label is None:
                heatmap_image_label = tk.Label(image_frame, image=overlay_tk)
                heatmap_image_label.image = overlay_tk
                heatmap_image_label.grid(row=0, column=1, padx=10, pady=10)
            else:
                heatmap_image_label.config(image=overlay_tk)
                heatmap_image_label.image = overlay_tk


def clear_all():
    global original_image_label, heatmap_image_label, result_label, confidence_label, image_path
    if original_image_label:
        original_image_label.destroy()
        original_image_label = None
    if heatmap_image_label:
        heatmap_image_label.destroy()
        heatmap_image_label = None
    if result_label:
        result_label.destroy()
        result_label = None
    if confidence_label:
        confidence_label.destroy()
        confidence_label = None
    image_path = None


# GUI Setup
window = tk.Tk()
window.title("Tea Leaf Disease Detection with Grad-CAM")
window.geometry("600x700")
window.configure(bg="#f0f0f0")

# Title
title_label = tk.Label(window, text="Tea Leaf Disease Detection", 
                       font=("Arial", 18, "bold"), bg="#f0f0f0")
title_label.pack(pady=15)

# Button Frame
button_frame = tk.Frame(window, bg="#f0f0f0")
button_frame.pack(pady=10)

upload_button = tk.Button(button_frame, text="Upload Image", command=upload_image, 
                         bg="green", fg="white", padx=15, pady=8, font=("Arial", 11))
upload_button.grid(row=0, column=0, padx=5)

detect_button = tk.Button(button_frame, text="Detect Disease", command=detect_disease, 
                         bg="blue", fg="white", padx=15, pady=8, font=("Arial", 11))
detect_button.grid(row=0, column=1, padx=5)

clear_button = tk.Button(button_frame, text="Clear", command=clear_all, 
                        bg="red", fg="white", padx=15, pady=8, font=("Arial", 11))
clear_button.grid(row=0, column=2, padx=5)

# Image Frame (for original and heatmap)
image_frame = tk.Frame(window, bg="#f0f0f0")
image_frame.pack(pady=10)

# Labels for images
tk.Label(image_frame, text="Original Image", font=("Arial", 11, "bold"), bg="#f0f0f0").grid(row=1, column=0)
tk.Label(image_frame, text="Grad-CAM Heatmap", font=("Arial", 11, "bold"), bg="#f0f0f0").grid(row=1, column=1)

# Result Frame
result_frame = tk.Frame(window, bg="#f0f0f0")
result_frame.pack(pady=20)

window.mainloop()




































# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow import keras
# import cv2
# import os

# # Load the trained model
# model = load_model("tea_leaf_model5.h5")

# # Build the model
# dummy_input = tf.zeros((1, 256, 256, 3))
# _ = model(dummy_input)

# # Load the class names exactly as in training
# class_names = ['BrownBlight', 'GrayBlight', 'GreenMiridBug', 'HealthyLeaf', 'Helopeltis', 'RedSpider', 'TeaAlgalLeafSpot']

# # Display names for GUI (with underscores)
# display_names = ['Brown_Blight', 'Gray_Blight', 'Green_Mirid_Bug', 'Healthy_Leaf', 'Helopeltis', 'Red_Spider', 'Tea_Algal_Leaf_Spot']

# # Grad-CAM configuration
# IMG_SIZE = (256, 256)
# LAST_CONV_LAYER_NAME = "conv2d_2"

# # Global variables
# image_path = None
# original_image_label = None
# heatmap_image_label = None
# result_label = None
# confidence_label = None

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     """Generate Grad-CAM heatmap for a given image."""
#     # Find the target layer and create submodels
#     last_conv_layer_idx = None
#     for idx, layer in enumerate(model.layers):
#         if layer.name == last_conv_layer_name:
#             last_conv_layer_idx = idx
#             break
    
#     if last_conv_layer_idx is None:
#         raise ValueError(f"Layer {last_conv_layer_name} not found in model")
    
#     # Create model up to the conv layer
#     conv_model = keras.Model(
#         inputs=model.layers[0].input,
#         outputs=model.layers[last_conv_layer_idx].output
#     )
    
#     # Create model from conv layer to output
#     classifier_input = keras.Input(shape=model.layers[last_conv_layer_idx].output.shape[1:])
#     x = classifier_input
#     for layer in model.layers[last_conv_layer_idx + 1:]:
#         x = layer(x)
#     classifier_model = keras.Model(inputs=classifier_input, outputs=x)
    
#     # Get conv layer output
#     with tf.GradientTape() as tape:
#         conv_output = conv_model(img_array)
#         tape.watch(conv_output)
#         preds = classifier_model(conv_output)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]
    
#     # Compute gradients
#     grads = tape.gradient(class_channel, conv_output)
    
#     # Global average pooling of gradients
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
#     # Proper weighting for accurate heatmap
#     conv_output_value = conv_output[0].numpy()
#     pooled_grads_value = pooled_grads.numpy()
    
#     # Weight each channel by the pooled gradients
#     for i in range(pooled_grads_value.shape[0]):
#         conv_output_value[:, :, i] *= pooled_grads_value[i]
    
#     # Average over all channels to get the heatmap
#     heatmap = np.mean(conv_output_value, axis=-1)
    
#     # Apply ReLU and normalize
#     heatmap = np.maximum(heatmap, 0)
#     heatmap = heatmap / (np.max(heatmap) + 1e-10)
    
#     return heatmap


# def create_gradcam_overlay(img_path, heatmap, alpha=0.4):
#     """Overlay Grad-CAM heatmap on original image."""
#     # Load original image
#     img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
#     img = keras.preprocessing.image.img_to_array(img)
    
#     # Resize heatmap to match image size
#     heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    
#     # Convert heatmap to RGB
#     heatmap_colored = np.uint8(255 * heatmap_resized)
#     heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
#     heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
#     # Overlay heatmap on original image
#     superimposed_img = heatmap_colored * alpha + img
#     superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
#     return superimposed_img


# def predict_with_gradcam(img_path):
#     """Predict disease and generate Grad-CAM visualization."""
#     try:
#         # Load and preprocess image
#         img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
#         img_array = keras.preprocessing.image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0
        
#         # Make prediction
#         predictions = model.predict(img_array, verbose=0)
#         predicted_class = np.argmax(predictions[0])
#         confidence = predictions[0][predicted_class]
        
#         # Generate Grad-CAM heatmap
#         heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)
        
#         # Create overlay image
#         overlay_img = create_gradcam_overlay(img_path, heatmap)
        
#         return display_names[predicted_class], confidence, overlay_img
        
#     except Exception as e:
#         return f"Error: {str(e)}", 0, None


# # GUI Functions
# def upload_image():
#     global image_path, original_image_label, heatmap_image_label, result_label, confidence_label
#     file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
#     if file_path:
#         image_path = file_path
        
#         # Display original image
#         img = Image.open(file_path)
#         img = img.resize((250, 250))
#         img_tk = ImageTk.PhotoImage(img)
        
#         if original_image_label is None:
#             original_image_label = tk.Label(image_frame, image=img_tk)
#             original_image_label.image = img_tk
#             original_image_label.grid(row=0, column=0, padx=10, pady=10)
#         else:
#             original_image_label.config(image=img_tk)
#             original_image_label.image = img_tk
        
#         # Clear heatmap and results
#         if heatmap_image_label:
#             heatmap_image_label.grid_forget()
#             heatmap_image_label = None
#         if result_label:
#             result_label.config(text="")
#         if confidence_label:
#             confidence_label.config(text="")


# def detect_disease():
#     global image_path, heatmap_image_label, result_label, confidence_label
#     if image_path:
#         disease, confidence, overlay_img = predict_with_gradcam(image_path)
        
#         # Display result
#         if result_label is None:
#             result_label = tk.Label(result_frame, text=f"Prediction: {disease}", 
#                                    font=("Arial", 14, "bold"), fg="blue")
#             result_label.pack(pady=5)
#         else:
#             result_label.config(text=f"Prediction: {disease}")
        
#         # Display confidence
#         if confidence_label is None:
#             confidence_label = tk.Label(result_frame, text=f"Confidence: {confidence:.2%}", 
#                                        font=("Arial", 12), fg="green")
#             confidence_label.pack(pady=5)
#         else:
#             confidence_label.config(text=f"Confidence: {confidence:.2%}")
        
#         # Display Grad-CAM heatmap overlay
#         if overlay_img is not None:
#             overlay_pil = Image.fromarray(overlay_img)
#             overlay_pil = overlay_pil.resize((250, 250))
#             overlay_tk = ImageTk.PhotoImage(overlay_pil)
            
#             if heatmap_image_label is None:
#                 heatmap_image_label = tk.Label(image_frame, image=overlay_tk)
#                 heatmap_image_label.image = overlay_tk
#                 heatmap_image_label.grid(row=0, column=1, padx=10, pady=10)
#             else:
#                 heatmap_image_label.config(image=overlay_tk)
#                 heatmap_image_label.image = overlay_tk


# def clear_all():
#     global original_image_label, heatmap_image_label, result_label, confidence_label, image_path
#     if original_image_label:
#         original_image_label.destroy()
#         original_image_label = None
#     if heatmap_image_label:
#         heatmap_image_label.destroy()
#         heatmap_image_label = None
#     if result_label:
#         result_label.destroy()
#         result_label = None
#     if confidence_label:
#         confidence_label.destroy()
#         confidence_label = None
#     image_path = None


# # GUI Setup
# window = tk.Tk()
# window.title("Tea Leaf Disease Detection with Grad-CAM")
# window.geometry("600x700")
# window.configure(bg="#f0f0f0")

# # Title
# title_label = tk.Label(window, text="Tea Leaf Disease Detection", 
#                        font=("Arial", 18, "bold"), bg="#f0f0f0")
# title_label.pack(pady=15)

# # Button Frame
# button_frame = tk.Frame(window, bg="#f0f0f0")
# button_frame.pack(pady=10)

# upload_button = tk.Button(button_frame, text="Upload Image", command=upload_image, 
#                          bg="green", fg="white", padx=15, pady=8, font=("Arial", 11))
# upload_button.grid(row=0, column=0, padx=5)

# detect_button = tk.Button(button_frame, text="Detect Disease", command=detect_disease, 
#                          bg="blue", fg="white", padx=15, pady=8, font=("Arial", 11))
# detect_button.grid(row=0, column=1, padx=5)

# clear_button = tk.Button(button_frame, text="Clear", command=clear_all, 
#                         bg="red", fg="white", padx=15, pady=8, font=("Arial", 11))
# clear_button.grid(row=0, column=2, padx=5)

# # Image Frame (for original and heatmap)
# image_frame = tk.Frame(window, bg="#f0f0f0")
# image_frame.pack(pady=10)

# # Labels for images
# tk.Label(image_frame, text="Original Image", font=("Arial", 11, "bold"), bg="#f0f0f0").grid(row=1, column=0)
# tk.Label(image_frame, text="Grad-CAM Heatmap", font=("Arial", 11, "bold"), bg="#f0f0f0").grid(row=1, column=1)

# # Result Frame
# result_frame = tk.Frame(window, bg="#f0f0f0")
# result_frame.pack(pady=20)

# window.mainloop()



