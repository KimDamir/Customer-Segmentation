# Customer Segmentation for Marketing Campaign using Deep Learning

 This project demonstrates how to use deep learning to segment customers based on their demographic and purchasing behavior data. The goal is to classify customers into three different product tiers (Tier 1, Tier 2, Tier 3) to enable personalized marketing strategies.

## Project Overview

The goal of this project is to build a deep learning model that segments customers into one of three tiers based on features such as income, marital status, education, and product spending. This segmentation can be used to tailor marketing strategies and increase customer engagement.

Key steps:

- Preprocess and clean the dataset.
- Train a deep learning model using a feedforward neural network (fully connected) and also Kmeans without deep learning.
- Use hyperparameter tuning to optimize the model’s performance.
- Evaluate the model using metrics like accuracy, confusion matrix, and classification report.

## Data

The dataset contains customer demographics, purchasing behaviors, and responses to previous marketing campaigns.
You can find a similar dataset on Kaggle / use the Marketing Campaign Dataset.

## Model 

The deep learning model is a fully connected feedforward neural network with the following architecture:

Input Layer: Takes in 29 features (after preprocessing).

Hidden Layers:
First Hidden Layer: 64 neurons, ReLU activation.
Second Hidden Layer: 128 neurons, ReLU activation, Dropout (0.3).
Third Hidden Layer: 64 neurons, ReLU activation.

Output Layer: 3 neurons (softmax activation) for multi-class classification (Tier 1, Tier 2, Tier 3).

The model is optimized using the Adam optimizer and trained with the sparse categorical cross-entropy loss function.

## Hyperparameter Optimisation

To tune the hyperparameters (e.g., number of neurons, learning rate, dropout rate), used Keras Tuner as implemented in model.py

- The learning rate is set to 0.001 (default in Adam). This provides a good balance between convergence speed and model stability. A lower learning rate might slow down training but could lead to better long-term performance.

- The batch size is set to 32, which is commonly used for balanced performance. A larger batch size could result in faster training but might affect generalization.

- To prevent overfitting, dropout is added with a rate of 0.3, randomly dropping 30% of the neurons during training to regularize the model.

- 64 neurons in the first and third layers and 128 neurons in the second layer provide a balanced amount of complexity.

## Usage

Clone the repository
cd customer-segmentation
You can install the libraries using: pip install -r requirements.txt

Preprocess the data: Clean and preprocess the dataset using the functions provided in preprocessing.py.

Deep learning model: You can train and evaluate the deep learning model using the provided Jupyter notebook customer_segmentation_main.ipynb. And hyperparameter tuning for the deep learning model. Also added the k means to compare the performance with deep learning model.

There is a Jupyter notebook - jupyter_notebooks/customer_segmentation_know_your_data.ipynb to know your data more.

## Actionable Insights

- Data Collection: Acquire a more diverse dataset.

- Model Development: Experiment with more complex architectures to improve generalization and prevent overfitting.

- Deployment Considerations: Given the model’s current limitations, it should not be deployed in environments requiring high generalization without further refinement.

- Softmax works well because it provides probability estimates, which are important for customer segmentation. The model outputs can be interpreted as the likelihood of a customer belonging to a specific tier.

- The deep learning architecture is harder to interpret than simpler models. For example, it's difficult to understand the exact contribution of features like Income or Age to the final classification, which is often referred to as the black-box nature of deep learning models. This trade-off is common in deep learning: accuracy often comes at the cost of interpretability.

## Results

The model achieved following results:
- Test Accuracy: 85.86% when KMeans is used for Clustering
- Test Accuracy: 85.86% on Deep learning but saw 88.54% in some readings.
- Test Accuracy: 85.86% on model with Keras Tuner

However, this high accuracy raises concerns about overfitting, suggesting the model may not generalize well to new, unseen data. The perfect accuracy likely reflects the simplicity and limited variability of the dataset, making it easy for the model to overfit.

Key learnings:
- The dataset is too small and imbalanced to provide robust generalization.
- Although the model performs well on the given dataset, it may fail when exposed to more complex or varied data.
- Future improvements should include using a larger, more diverse dataset and potentially a more complex architecture to avoid overfitting.

## Conclusions

While the model performs perfectly on the current data, overfitting is a major concern due to the small, imbalanced dataset. This highlights the need for better data and more robust architectures for real-world applications. Future work should focus on improving the model's generalizability by introducing more varied data and refining the architecture.