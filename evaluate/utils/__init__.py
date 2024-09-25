import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the data
def load_data(csv_file, base_image_dir, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_file)
    X = []
    y = []
    label_map = {
        0: 'No_DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferate_DR'
    }
    for index, row in df.iterrows():
        image_filename = row['id_code']
        label = row['diagnosis']
        image_path = os.path.join(base_image_dir, f"{label_map[label]}/{image_filename}.png")
        X.append(image_path)
        y.append(label)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Preprocess image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Get predictions
def model_predictions(model, X_test, y_test):
    predictions = []
    for img_path in X_test:
        img = preprocess_image(img_path)
        pred = model.predict(img, verbose=0)
        predictions.append(np.argmax(pred))
    
    return predictions

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def evaluate_model(model_path):
    # Parameters
    base_image_dir = '../archive/gaussian_filtered_images'
    labels_path = '../archive/train.csv'
    # model_path = '../models/efficientnet_model.keras'
    class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(labels_path, base_image_dir)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Get predictions
    y_pred = model_predictions(model, X_test, y_test)

    # Print classification report
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names)