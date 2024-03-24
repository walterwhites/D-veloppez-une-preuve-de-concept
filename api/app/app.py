#!/my_env/bin/python3

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import zipfile
import text_preprocessing
import joblib
import os
import numpy as np
import torch
from collections import Counter, OrderedDict
from nltk.corpus import stopwords
from wordcloud import WordCloud
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from torch.nn.parallel import DataParallel

# Nom du fichier zip contenant les modèles
models_zip_file = 'models_src.zip'

# Charger les modèles depuis le fichier zip
with zipfile.ZipFile(models_zip_file, 'r') as zip_ref:
    if not os.path.exists('models_src'):
        zip_ref.extractall()

# Définition du nooveaau modèle
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', max_length=128)
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=100)

nltk_data_dir = os.path.abspath("nltk_data")
nltk.data.path.append(nltk_data_dir)

class XLNetPipeline:
    def __init__(self, model):
        self.model = model
        self.threshold = 0.3
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def fit(self, X, y, epochs=6, batch_size=4):
        if torch.cuda.device_count() > 1:
            print("Utilisation de", torch.cuda.device_count(), "GPUs pour l'entraînement.")
            self.model = DataParallel(self.model)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(X), batch_size):
                batch_texts = X[i:i+batch_size]
                batch_labels = torch.tensor(y[i:i+batch_size], dtype=torch.float32)

                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                labels = batch_labels

                self.optimizer.zero_grad()  # Reset le gradient
                outputs = self.model(**inputs)
                logits = outputs.logits

                loss = self.loss_fn(logits, labels)
                loss.backward()  # Backpropagation
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")


    def predict(self, X):
        predictions = []
        class_counts = Counter(mlb.classes_)
        N = 100
        top_classes = [class_name for class_name, _ in class_counts.most_common(N)]
        with torch.no_grad():
            for text in X:
                inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits).detach().numpy()

                specific_class_indices = [mlb.classes_.tolist().index(cls) for cls in top_classes]
                specific_class_probabilities = probabilities[:, specific_class_indices]
                predictions.append(specific_class_probabilities)
        return np.squeeze(predictions)

# Download NLTK data only if it doesn't exist
if not os.path.exists(nltk_data_dir):
    try:
        nltk.download('punkt', force=True)
        nltk.download('wordnet', force=True)
        nltk.download('stopwords', force=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
else:
    print("NLTK data already exists. Skipping download.")

# Charger les modèles
mlb = joblib.load('models_src/mlb_model.joblib')
pipeline_xlnet_joblib = joblib.load('models_src/XLNet_custom_classification_layer_model.joblib')
pipeline_multinomial_naive_bayes_joblib = joblib.load('models_src/oneVsRestClassifier_mlb_model.joblib')

@st.cache_data
def generate_wordclouds(data):
    all_words = [word for words_list in data['title_lemmatized'] + data['body_lemmatized'] for word in str(words_list).split()]
    title_words = [word for words_list in data['title_lemmatized'] for word in str(words_list).split()]
    body_words = [word for words_list in data['body_lemmatized'] for word in str(words_list).split()]

    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(all_words))
    title_wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(title_words))
    body_wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(body_words))

    return wordcloud, title_wordcloud, body_wordcloud

@st.cache_data
def generate_outliers_graph(data):
    nltk_stopwords = set(stopwords.words('english'))
    title_and_body_lemmatized = data['title_lemmatized'] + data['body_lemmatized']
    all_words = [word for words_list in title_and_body_lemmatized for word in str(words_list).split()]
    all_words_without_stopwords = [word for word in all_words if word.lower() not in nltk_stopwords]
    word_freq = Counter(all_words_without_stopwords)
    outliers = [word for word, freq in word_freq.items()]
    word_freq_ordered = OrderedDict(word_freq)
    sorted_outliers = sorted(outliers, key=lambda word: word_freq_ordered[word], reverse=True)
    top_outliers = sorted_outliers[:30]
    fig_outliers, ax_outliers = plt.subplots()
    ax_outliers.bar(top_outliers, [word_freq[word] for word in top_outliers])
    ax_outliers.set_title("Analyse des mots outliers", rotation=45)
    ax_outliers.set_xticklabels(top_outliers, rotation=70, ha='right')
    return fig_outliers

@st.cache_data
def generate_sentence_lengths_graph(data):
    sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in data['title_lemmatized'] + data['body_lemmatized']]
    bins = range(0, 1001, 100)
    fig_sentence, ax_sentence = plt.subplots()
    ax_sentence.hist(sentence_lengths, bins=bins, color='skyblue', edgecolor='black')
    ax_sentence.set_title("Distribution des longueurs de phrases")
    ax_sentence.set_xlabel("Longueur de phrase")
    ax_sentence.set_ylabel("Nombre de phrases")
    return fig_sentence

# Interface utilisateur avec Streamlit
st.title("Analyse de texte et prédiction supervisée")

# Charger les données
data = pd.read_csv("https://raw.githubusercontent.com/walterwhites/D-veloppez-une-preuve-de-concept/main/models/dataset_cleaned.csv")

# Affichage des graphiques
wordcloud, title_wordcloud, body_wordcloud = generate_wordclouds(data)
st.subheader("Nuage de mots pour titre et body")
st.image(wordcloud.to_array(), use_column_width=True, caption='Nuage de mots pour titre et body')

st.subheader("Nuage de mots pour le titre")
st.image(title_wordcloud.to_array(), use_column_width=True, caption='Nuage de mots pour le titre')

st.subheader("Nuage de mots pour le body")
st.image(body_wordcloud.to_array(), use_column_width=True, caption='Nuage de mots pour le body')

fig_outliers = generate_outliers_graph(data)
st.title("Analyse des mots outliers")
st.pyplot(fig_outliers)

fig_sentence = generate_sentence_lengths_graph(data)
st.title("Analyse des longueurs de phrases")
st.pyplot(fig_sentence)


####################################################################################################

def original_multinomial_naive_bayes_predict(title: str, body: str):
    content = title + ' ' + body
    processed_question = text_preprocessing.preprocess_text(content)
    content_as_array = [title, body]
    print(content_as_array)
    predictions_proba_combined = pipeline_multinomial_naive_bayes_joblib.predict_proba(content_as_array)

    n_top_classes = 5

    # itérer sur chaque prédiction
    for i, question in enumerate(processed_question):
        top_classes_indices = predictions_proba_combined.argsort(axis=1)[:, -n_top_classes:][i]
        top_classes_probabilities = predictions_proba_combined[i, top_classes_indices]

        # Tri des classes et des probabilités par ordre décroissant de probabilité
        sorted_indices = np.argsort(top_classes_probabilities)[::-1]
        top_tags_combined_sorted = mlb.classes_[top_classes_indices[sorted_indices]]
        top_classes_probabilities_sorted = top_classes_probabilities[sorted_indices]

        print(f"Tags associés pour la question '{question}':", list(zip(top_tags_combined_sorted, top_classes_probabilities_sorted)))
        print("\n")

    return {"prediction":  list(zip(top_tags_combined_sorted, top_classes_probabilities_sorted))}


# Fonction de prédiction
def multinomial_naive_bayes_predict(title: str, body: str):
    url = 'http://127.0.0.1:8000/models/multinomial_naive_bayes/predict/'
    data = {
        "title": title,
        "body": body
    }

    response = requests.post(url, params=data)
    print(response)

    if response.status_code == 200:
        return response.json().get('prediction', [])
    else:
        print("La requête a échoué avec le code:", response.status_code)
        return []

# Interface utilisateur avec Streamlit
st.title("Prédiction supervisée")

title = st.text_input('Titre de la question:')
body = st.text_area('Corps de la question:', height=200)

if st.button('Prédire'):
    response = original_multinomial_naive_bayes_predict(title, body)
    predictions = response.get('prediction', [])
    st.write("Résultats de la prédiction :")
    for i, (tag, proba) in enumerate(predictions):
        st.write(f"Prédiction {i + 1}:")
        st.write(f"Tag: {tag}, Probabilité: {proba}")

            
####################################################################################################

def original_XLNet_predict(title: str, body: str):
    content = title + ' ' + body
    processed_question = text_preprocessing.preprocess_text(content)
    content_as_array = [title, body]

    predictions_XLNet = pipeline_xlnet_joblib.predict(content_as_array)

    n_top_classes = 5
    result = []

    # Iterate over each prediction
    for i, question in enumerate(processed_question):
        top_classes_indices = predictions_XLNet.argsort(axis=1)[:, -n_top_classes:][i]
        top_classes_probabilities = predictions_XLNet[i, top_classes_indices]

        # Sort classes and probabilities by descending probability
        sorted_indices = np.argsort(top_classes_probabilities)[::-1]
        top_tags_combined_sorted = mlb.classes_[top_classes_indices[sorted_indices]].tolist()  # Convert to list
        top_classes_probabilities_sorted = top_classes_probabilities[sorted_indices].tolist()  # Convert to list

        # Append predictions to result list
        result.append(list(zip(top_tags_combined_sorted, top_classes_probabilities_sorted)))

    return {"prediction": result}

def XLNet_predict(title: str, body: str):
    url = 'http://127.0.0.1:8000/models/xlnet/predict/'
    data = {
        "title": title,
        "body": body
    }
    response = requests.post(url, params=data)

    if response.status_code == 200:
        return response.json().get('prediction', [])
    else:
        print("La requête a échoué avec le code:", response.status_code)
        return []

# Interface utilisateur avec Streamlit
st.title("XLNet - Prédiction supervisée")

title = st.text_input('XLNet - Titre de la question:')
body = st.text_area('XLNet - Corps de la question:', height=200)

if st.button('Prédire avec le modèle XLNet'):
    response = original_XLNet_predict(title, body)
    predictions = response.get('prediction', [])
    st.write("Résultats de la prédiction :")
    for i, pred_list in enumerate(predictions):
        st.write(f"Prédiction {i + 1}:")
        for tag, proba in pred_list:
            st.write(f"Tag: {tag}, Probabilité: {proba}")
