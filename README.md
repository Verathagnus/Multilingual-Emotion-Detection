# Indic Multilingual Emotion Detection

## Introduction

Indic Multilingual Emotion Detection is a machine learning project aimed at identifying emotions in text across multiple Indic languages, starting with Hindi. This repository hosts the Streamlit code for easy deployment and usage. Download model files at [https://uploadnow.io/f/dc4f6Hb](https://uploadnow.io/f/dc4f6Hb)

## Project Structure

- `models/`: Directory containing trained models for emotion detection. Download model files at [https://uploadnow.io/f/dc4f6Hb](https://uploadnow.io/f/dc4f6Hb)
- `app.py`: Streamlit application code for interacting with the models.
- `requirements.txt`: List of Python dependencies required to run the Streamlit app.

## Tokenizers

The dataset text is converted to embeddings using the following tokenizers and models:

- **GPT-2 Tokenizer and GPT2 Model**: A tokenizer based on the GPT-2 model by OpenAI, which uses Byte Pair Encoding (BPE) to handle text and generate embeddings suitable for downstream tasks. The tokenized sentences are then provided to the GPT2 model to get the embeddings for the input sentences.
- **BERT Tokenizer and BERT Model**: A tokenizer based on the BERT model by Google, which uses WordPiece tokenization to handle text and create embeddings for use in various NLP tasks. The tokenized sentences are then provided to the BERT model to get the embeddings for the input sentences.

## Model Performance

The following table summarizes the performance of each model on both training and test data:

| Model               | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|---------------------|----------------|---------------|-----------|--------|----------|
| Naive Bayes         | 0.2597         | 0.25          | 0.35      | 0.25   | 0.27     |
| Logistic Regression | 0.4424         | 0.42          | 0.36      | 0.42   | 0.35     |
| KNN                 | 0.4800         | 0.35          | 0.31      | 0.35   | 0.31     |
| K-Means             | 0.1671         | 0.17          | 0.21      | 0.17   | 0.18     |
| SVM                 | 0.7941         | 0.46          | 0.44      | 0.46   | 0.43     |
| Decision Tree       | 0.5398         | 0.34          | 0.27      | 0.34   | 0.29     |
| Random Forest       | 0.9971         | 0.42          | 0.35      | 0.42   | 0.33     |

### Detailed Model Descriptions

#### Naive Bayes
- **Description**: Naive Bayes is a probabilistic classifier based on Bayes' Theorem, with the assumption of independence between features. It's particularly effective for text classification tasks.
- **Performance**: Naive Bayes has a training accuracy of 0.2597 and a test accuracy of 0.25. It performs moderately well for some classes but struggles with others, indicating that the independence assumption may not hold well in this context.

#### Logistic Regression
- **Description**: Logistic Regression is a linear model used for binary classification, but it can be extended to multiclass classification using techniques like one-vs-rest. It predicts the probability of a sample belonging to a particular class.
- **Performance**: Logistic Regression shows better performance than Naive Bayes with a training accuracy of 0.4424 and a test accuracy of 0.42. It handles some classes well but has poor performance for others, especially for minority classes.

#### K-Nearest Neighbors (KNN)
- **Description**: KNN is a non-parametric method used for classification and regression. It classifies a sample based on the majority class among its k-nearest neighbors in the feature space.
- **Performance**: KNN has a training accuracy of 0.4800 and a test accuracy of 0.35. It performs reasonably well for the major classes but poorly for the minority classes. It is sensitive to the choice of k and the distance metric.

#### K-Means
- **Description**: K-Means is a clustering algorithm that partitions the data into k clusters based on the feature space. It is not typically used for classification tasks but can be adapted for such purposes.
- **Performance**: K-Means has a training accuracy of 0.1671 and a test accuracy of 0.17, indicating it is not well-suited for this classification task. It shows poor precision and recall for most classes.

#### Support Vector Machine (SVM)
- **Description**: SVM is a powerful classification algorithm that finds the hyperplane which best separates the classes in the feature space. It can be used for linear and non-linear classification.
- **Performance**: SVM has a training accuracy of 0.7941 and a test accuracy of 0.46, the highest among the models tested. It performs well overall but still has difficulties with minority classes, which affects its macro and weighted averages.

#### Decision Tree
- **Description**: Decision Trees are non-linear models that split the data based on feature values, creating a tree-like structure of decisions. They are easy to interpret but can overfit the training data.
- **Performance**: Decision Tree has a training accuracy of 0.5398 and a test accuracy of 0.34. It performs decently for some classes but struggles with others, especially the minority classes, indicating potential overfitting.

#### Random Forest
- **Description**: Random Forest is an ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction. It reduces overfitting compared to a single decision tree.
- **Performance**: Random Forest has a training accuracy of 0.9971 and a test accuracy of 0.42, indicating it fits the training data very well. However, its test accuracy is much lower, suggesting overfitting. It struggles with minority classes, impacting its overall performance metrics.

## Setup

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/<yourusername>/indic-multilingual-emotion-detection.git
   cd indic-multilingual-emotion-detection
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Download model files at [https://uploadnow.io/f/dc4f6Hb](https://uploadnow.io/f/dc4f6Hb) and place them in a folder named models inside the project directory.**
5. **Run the Streamlit application:**
   ```sh
   streamlit run streamlit_app.py
   ```

## Usage

Once the Streamlit app is running, you can interact with the models through a web interface. You can input text in Hindi (or other supported Indic languages) and select the model you wish to use for emotion detection. The app will display the detected emotion along with confidence scores. Using fasttext-langdetect and iso639 packages, the input language is found and then appropriate models list are selected for that language.  

<!-- ## Example

![Streamlit App Screenshot](screenshot.png)  # Add a screenshot of your Streamlit app here -->

## Future Work

- **Extend Language Support**: Increase the number of Indic languages supported by training the models on additional language datasets.
- **Improve Model Performance**: Experiment with advanced models and techniques to enhance the accuracy and robustness of emotion detection.
- **User Feedback**: Incorporate user feedback to continuously improve the application and its usability.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- This project utilizes embeddings generated using GPT-2 and BERT tokenizers.
- Special thanks to the contributors and the open-source community for their invaluable resources and support.

---
