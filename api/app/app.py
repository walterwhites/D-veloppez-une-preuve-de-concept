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
from torch.nn.parallel import DataParallel

nltk_data_dir = os.path.abspath("nltk_data")
nltk.data.path.append(nltk_data_dir)

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
    predictions = multinomial_naive_bayes_predict(title, body)
    st.write("Résultats de la prédiction :")
    for i, pred in enumerate(predictions):
        st.write(f"Tag: {pred[0]}, Probabilité: {pred[1]}")

####################################################################################################

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
    predictions = XLNet_predict(title, body)
    st.write("Résultats de la prédiction:")
    for i, pred in enumerate(predictions):
        st.write(f"Tag: {pred[0]}, Probabilité: {pred[1]}")

