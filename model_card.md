# Model Card

## Model Description

**Input:** 
The model takes marketing campaign dataset containing demographic and purchasing data for customers.
Age, Income, Marital_Status, Education, Total_Children, NumWebPurchases, etc.

**Output:**
The model outputs a probability distribution of 3-class classification (Tier 1, Tier 2, Tier 3) / cluster of customer segmentations.

**Hidden Layers:**
Layer 1: 64 neurons (ReLU activation)
Layer 2: 128 neurons (ReLU activation)
Layer 3: Dropout (0.3)
Layer 4: 64 neurons (ReLU activation)
Output Layer: 3 neurons (softmax activation)

**Training Data:**
Split: 70% training, 30% test split.

**Model Architecture:** 
Model displays output with and without deep learning and also hyperparameter tuning.

The architecture of the deep learning model used for the customer segmentation problem is a fully connected feedforward neural network (also known as a multi-layer perceptron). The goal of the model is to classify customers into three distinct tiers (Tier 1, Tier 2, Tier 3) based on demographic, behavioral, and spending data. 

Optimizer: Adam, which is widely used for its adaptive learning rate and efficiency.
Loss: Sparse Categorical Cross-Entropy, suitable for multi-class classification problems.

**Hyperparameters:**
- Learning Rate: 0.001 (Adam optimizer)
- Batch Size: 32
- Epochs: 50
- Dropout Rate: 0.3

## Performance

The model achieved following results:
- Test Accuracy: 85.86% when KMeans is used for Clustering
- Test Accuracy: 85.86% on Deep learning but saw 88.54% in some readings.
- Test Accuracy: 85.86% on model with Keras Tuner

- Confusion Matrix: Shows that most customers are correctly classified into their respective tiers.
- Loss: Sparse Categorical Cross-Entropy loss.

## Limitations

- Overfitting Risk:
There can be overfitting, especially given the relatively small and imbalanced dataset. The model may not perform well on unseen or more varied data.
- Dataset Bias:
Data Bias: The model may be biased due to the distribution of income levels or product purchase categories.
Static Data: As the dataset is static, the model may not capture evolving customer behavior over time.
- Limited Generalization:
The model was trained on a specific marketing campaign dataset and may not generalize well to different regions or industries. The static nature of the data may lead to inaccurate predictions if customer behavior changes significantly over time.

## Trade-offs

- The trade-offs between model complexity, interpretability, and overfitting are key. By using dropout, Adam optimizer, and ReLU activations, the model is built to learn from data effectively while minimizing the risks of overfitting.

## Ethical Considerations:

- Fairness: There is a risk of unfair treatment for certain customer segments based on income or marital status, leading to skewed product offerings.
- Bias Mitigation: Regular checks should be performed to ensure fairness, and model retraining should be done with updated data.
- Privacy: Ensure that the personal data used (e.g., income, spending) complies with data privacy laws (e.g., GDPR).

## Future Improvements
- Retraining: The model should be retrained periodically with updated data to account for changing customer behavior.
- Monitoring: Implement ongoing model monitoring to check for drift and update the model when necessary.