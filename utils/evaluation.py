import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, confusion_matrix
import numpy as np

def evaluate_kmeans_model(model, X):
    # Get the cluster labels
    labels = model.labels_

    # Calculate the silhouette score
    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score}")

    return labels

def plot_clusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.title("Customer Segmentation Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title ):
    cm = confusion_matrix(y_true, y_pred )

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
