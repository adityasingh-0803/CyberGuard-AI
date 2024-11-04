# CyberGuard-AI

## By Aditya Singh & Mohak Gupta

## Introduction
This project involves building a cybercrime classification model using a BERT-based architecture. The model is designed to classify text data into various categories and subcategories related to cybercrime.

## Dependencies
The following libraries are required to run the project:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `torch`: The PyTorch library for building and training the model.
- `transformers`: Hugging Face's library for state-of-the-art natural language processing models.
- `nltk`: For natural language processing tasks.
- `sklearn`: For machine learning utilities and metrics.

## Dataset Preparation
The `CybercrimeDataset` class is used to prepare the data for training. It handles tokenization and encoding of the input text.

## Model Architecture
The `CybercrimeClassifier` class defines the BERT-based model. It consists of:
- **BERT Backbone**: The model uses a pretrained BERT model to extract features from the input text.
- **Dropout Layer**: Applied to reduce overfitting.
- **Hidden Layers**: Fully connected layers to process BERT embeddings.
- **Output Layers**: Separate classifiers for predicting categories and subcategories of cybercrime.

## Training Process
The model is trained using the `train` method in the `CybercrimeNLP` class. The training involves:
- Preparing the data into batches.
- Running a loop for a specified number of epochs to update model weights.
- Calculating the loss and updating the model using backpropagation.

## Evaluation
The `evaluate` method assesses the model's performance on a validation dataset. It calculates the average loss and can be expanded to include metrics such as accuracy and F1 score.

## Inference
The `predict` method allows the model to make predictions on new text. It outputs the predicted categories along with confidence scores.

## Saving and Loading the Model
The model can be saved to disk using the `save_model` method and loaded back using the `load_model` method, ensuring that the model state and label encoders are preserved.

## Example Usage
In the `main` function, the model is trained and tested on sample data. Here’s an example of how to make a prediction:
```python
test_text = "I received a call from an unknown number..."
prediction = classifier.predict(test_text)
print(prediction)


### Conclusion
```markdown
## Conclusion
This project demonstrates the implementation of a BERT-based model for cybercrime text classification, highlighting the potential of deep learning in tackling cybersecurity issues.
