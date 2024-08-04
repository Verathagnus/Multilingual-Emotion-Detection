from transformers import (GPT2Config,
                          GPT2Tokenizer,
                          GPT2Model,
                          BertTokenizer, 
                          BertModel)
import torch
import pickle
import streamlit as st
from ftlangdetect import detect
import iso639
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
class_names = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
import os
# gpt2_model = GPT2Model.from_pretrained("gpt2")
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# gpt2_tokenizer.padding_side = "left"
# gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
# Define preprocessing function with smaller max length
def tokenize_sample(texts, tokenizer="bert"):
    if tokenizer == "gpt2":
        return gpt2_tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt', max_length=128)
    return bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
def get_embeddings(text, model_type="bert"):
    tokenized_text = tokenize_sample(text, model_type)
    if model_type =="gpt2":
        outputs = gpt2_model(**tokenized_text)
    else:
        outputs = bert_model(**tokenized_text)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()  # Get the embeddings for [CLS] token
    return embeddings
path_to_models = os.environ['RAILWAY_VOLUME_MOUNT_PATH']+"/storage"
classifier_map={
"Naive Bayes":f"{path_to_models}/models/naive_bayes_model.sav",
"Logistic Regression":f"{path_to_models}/models/logistic_regression_model.sav",
"KNN":f"{path_to_models}/models/knn_model.sav",
"KMeans":f"{path_to_models}/models/kmeans_model.sav",
"SVM":f"{path_to_models}/models/svm_model.sav",
"Decision Tree":f"{path_to_models}/models/decision_tree_model.sav",
"Random Forest":f"{path_to_models}/models/random_forest_model.sav"
}
# print(os.listdir())
# print(os.environ["RAILWAY_VOLUME_MOUNT_PATH"])
# print(os.listdir(os.environ["RAILWAY_VOLUME_MOUNT_PATH"]+"/storage"))
models=dict()
for i in classifier_map:
    with open(classifier_map[i], 'rb') as file:
        models[i] = pickle.load(file)
def get_prediction(input, model_name):
    if model_name in models:
        return class_names[models[model_name].predict(get_embeddings(input))[0]]
    else:
        raise ValueError("Model type should be of the types: {}".format(", ".join(list(models.keys()))))


def main():
    # Title of the web app
    st.title('Multilingual Emotion Prediction from Text')
    # print(os.listdir())
    # print(os.environ["RAILWAY_VOLUME_MOUNT_PATH"])
    # print(os.listdir(os.environ["RAILWAY_VOLUME_MOUNT_PATH"]))
    
    # Input text from the user
    input_sentence = st.text_input('Enter a sentence')

    # Model selection
    model_option = st.selectbox('Select the model', list(models.keys()))

    # Result initialization
    result = None
    error = None
    langlist = {"hi": "Hindi"}
    # Prediction button
    if st.button('Predict Emotion'):
        lang = detect(text=input_sentence, low_memory=False)['lang']
        if lang in langlist:
            result = get_prediction(input_sentence, model_option)
        else:
            error = f"{iso639.Language.from_part1(lang).name} is not supported.\n List of supported languages: {', '.join(langlist.values())}"

    # Display the result
    if result:
        st.success(f'Prediction: {result}')
    if error:
        st.error(f'Error: {error}')
    # Credits
    st.markdown("---")  # Separator
    st.markdown("**Mentored by Dr. Sahinur Rahaman**")

if __name__ == '__main__':
    main()
