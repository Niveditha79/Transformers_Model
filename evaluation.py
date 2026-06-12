# evaluation.py
import os
import logging

# Mute system, library logging, and oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Import generator from model_training script
from model_training import generate_trend_sequences

def main():
    model_path = 'transformer_model.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Run model_training.py first.")
        return

    print("Generating validation trend sequence data...")
    length = 50
    X_test, y_test = generate_trend_sequences(samples_per_class=200, length=length)
    y_test_categorical = to_categorical(y_test, 3)
    
    model = tf.keras.models.load_model(model_path)
    
    # 1. Base Evaluation
    print("\nEvaluating model on the test sequence dataset...")
    test_loss, test_acc = model.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 2. Detailed Classification Report
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    target_names = ['Up-Trend', 'Down-Trend', 'Cyclic']
    print("\nDetailed Sequence Classification Metrics:")
    print(classification_report(y_test, predicted_classes, target_names=target_names))
    
    # 3. Save Confusion Matrix Plot (using Teal/Cyan color map)
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(y_test, predicted_classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.YlGnBu, ax=ax)
    
    plot_filename = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Confusion matrix image saved as '{plot_filename}'.")

if __name__ == '__main__':
    main()