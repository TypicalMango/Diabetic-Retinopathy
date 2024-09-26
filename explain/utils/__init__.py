import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import vgg16, resnet50, efficientnet
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path, model_name):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    if model_name == 'vgg16':
        img_array = vgg16.preprocess_input(img_array)
    elif model_name == 'resnet50':
        img_array = resnet50.preprocess_input(img_array)
    elif model_name == 'efficientnet':
        img_array = efficientnet.preprocess_input(img_array)
    return img_array

# Function to generate Grad-CAM heatmap
def get_grad_cam_heatmap(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Function to overlay heatmap on original image
def overlay_heatmap_on_image(img_path, heatmap, intensity=0.5, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlayed_img = cv2.addWeighted(img, 1 - intensity, heatmap, intensity, 0)
    return overlayed_img

def layer_image_grad_cam(model, conv_layer_name_list, img_array, img_path, mod_factor):
    layer_image_list = []
    for i, layer in enumerate(conv_layer_name_list):
        # Output every nth layer
        if i%mod_factor != 0:
            continue
        
        # Generate Grad-CAM heatmap
        heatmap = get_grad_cam_heatmap(model, img_array, layer_name=layer)
        
        # Overlay heatmap on original image
        overlayed_img = overlay_heatmap_on_image(img_path, heatmap)
        layer_image_list.append(overlayed_img)
    return layer_image_list

def display_grad_cam(layer_image_list, conv_layer_name_list, rows, cols):
    
    # Create a figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    # Plot each image in the corresponding subplot
    for i in range(len(layer_image_list)):
        ax = axes[i]
        ax.imshow(cv2.cvtColor(layer_image_list[i], cv2.COLOR_BGR2RGB))
        ax.set_title(conv_layer_name_list[i])
        ax.axis('off')  # Hide axes
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def visualise_layer_outputs(model_name, model, img_path, conv_layer_name_list, rows=5, cols=5, mod_factor=1):
    img_array = preprocess_image(img_path, model_name)
    layer_image_list = layer_image_grad_cam(model, conv_layer_name_list, img_array, img_path, mod_factor)
    display_grad_cam(layer_image_list, conv_layer_name_list, rows, cols)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
