#!/my_env/bin/python3

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import zipfile
import numpy as np
import preprocessing
import joblib
import os
from collections import Counter, OrderedDict
from nltk.corpus import stopwords
from wordcloud import WordCloud


####################################################################################################

# Nom du fichier zip contenant les modèles
models_zip_file = 'models_src.zip'

# Charger les modèles depuis le fichier zip
with zipfile.ZipFile(models_zip_file, 'r') as zip_ref:
    if not os.path.exists('models_src'):
        zip_ref.extractall()

relative_path_to_nltk_data = "nltk_data"
relative_path_to_models = "models_src"
nltk_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path_to_nltk_data))
models__path = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path_to_models))

nltk.data.path.append(nltk_data_path)


# Charger les données à partir du fichier CSV
data = pd.read_csv("https://raw.githubusercontent.com/walterwhites/D-veloppez-une-preuve-de-concept/main/models/dataset_cleaned.csv")

# Données pour créer le WordCloud
all_words = [word for words_list in data['title_lemmatized'] + data['body_lemmatized'] for word in str(words_list).split()]
title_words = [word for words_list in data['title_lemmatized'] for word in str(words_list).split()]
body_words = [word for words_list in data['body_lemmatized'] for word in str(words_list).split()]

# Création des WordClouds
wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(all_words))
title_wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(title_words))
body_wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(body_words))

# Affichage des WordClouds dans Streamlit
st.title("Analyse des mots avec WordCloud")
st.subheader("Nuage de mots pour titre et body")
st.image(wordcloud.to_array(), use_column_width=True, caption='Nuage de mots pour titre et body')

st.subheader("Nuage de mots pour le titre")
st.image(title_wordcloud.to_array(), use_column_width=True, caption='Nuage de mots pour le titre')

st.subheader("Nuage de mots pour le body")
st.image(body_wordcloud.to_array(), use_column_width=True, caption='Nuage de mots pour le body')


####################################################################################################


# Graph outlier 

# Définir les stopwords
nltk_stopwords = set(stopwords.words('english'))

# Lemmatization et suppression des stopwords
title_and_body_lemmatized = data['title_lemmatized'] + data['body_lemmatized']
all_words = [word for words_list in title_and_body_lemmatized for word in str(words_list).split()]
all_words_without_stopwords = [word for word in all_words if word.lower() not in nltk_stopwords]

# Calcul des fréquences de mots
word_freq = Counter(all_words_without_stopwords)
outliers = [word for word, freq in word_freq.items()]
word_freq_ordered = OrderedDict(word_freq)
sorted_outliers = sorted(outliers, key=lambda word: word_freq_ordered[word], reverse=True)
top_outliers = sorted_outliers[:30]

# Création du graphique
fig, ax = plt.subplots()
ax.bar(top_outliers, [word_freq[word] for word in top_outliers])
ax.set_title("Analyse des mots outliers", rotation=45)

# Rotation des étiquettes sur l'axe des x
ax.set_xticklabels(top_outliers, rotation=70, ha='right')

# Affichage des mots outliers dans Streamlit
st.title("Analyse des mots outliers")
st.pyplot(fig)


####################################################################################################

# Calcul des longueurs des phrases
sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in title_and_body_lemmatized]

# Définition des bins de 100 en 100 jusqu'à 1000
bins = range(0, 1001, 100)

# Création du graphique pour les longueurs de phrases
fig_sentence, ax_sentence = plt.subplots()
ax_sentence.hist(sentence_lengths, bins=bins, color='skyblue', edgecolor='black')
ax_sentence.set_title("Distribution des longueurs de phrases")
ax_sentence.set_xlabel("Longueur de phrase")
ax_sentence.set_ylabel("Nombre de phrases")

# Affichage du graphique des longueurs de phrases dans Streamlit
st.title("Analyse des longueurs de phrases")
st.pyplot(fig_sentence)



####################################################################################################


# Charger les modèles
combined_pipeline = joblib.load('models_src/oneVsRestClassifier_mlb_model.joblib')
mlb = joblib.load('models_src/mlb_model.joblib')
pipeline_xlnet = joblib.load('models_src/XLNet_custom_classification_layer_model.joblib')

# Fonction de prédiction
def supervised_predict(title: str, body: str):
    content = title + ' ' + body
    processed_question = preprocessing.preprocess_text(content)
    content_as_array = [title, body]

    predictions_proba_combined = combined_pipeline.predict_proba(content_as_array)

    n_top_classes = 5
    result = []

    # itérer sur chaque prédiction
    for i, question in enumerate(processed_question):
        top_classes_indices = predictions_proba_combined.argsort(axis=1)[:, -n_top_classes:][i]
        top_classes_probabilities = predictions_proba_combined[i, top_classes_indices]

        # Tri des classes et des probabilités par ordre décroissant de probabilité
        sorted_indices = np.argsort(top_classes_probabilities)[::-1]
        top_tags_combined_sorted = mlb.classes_[top_classes_indices[sorted_indices]]
        top_classes_probabilities_sorted = top_classes_probabilities[sorted_indices]

        result.append(list(zip(top_tags_combined_sorted, top_classes_probabilities_sorted)))

    return result

# Interface utilisateur avec Streamlit
st.title("Prédiction supervisée")

title = st.text_input('Titre de la question:')
body = st.text_area('Corps de la question:', height=200)

if st.button('Prédire'):
    predictions = supervised_predict(title, body)
    st.write("Résultats de la prédiction :")
    for i, pred in enumerate(predictions):
        st.write(f"Prédiction {i + 1}:")
        for tag, proba in pred:
            st.write(f"Tag: {tag}, Probabilité: {proba}")

            
####################################################################################################

def XLNet_predict(title: str, body: str):
    content = title + ' ' + body
    processed_content = preprocessing.preprocess_text(content)

    predictions_XLNet = pipeline_xlnet.predict(processed_content)

    n_top_classes = 5
    result = []

    for i, content in enumerate(processed_content):
        top_classes_indices = np.argsort(predictions_XLNet[i])[-n_top_classes:]
        top_tags_combined = mlb.classes_[top_classes_indices]
        result.append(top_tags_combined)

    return result

# Interface utilisateur avec Streamlit
st.title("XLNet - Prédiction supervisée")

title = st.text_input('XLNet - Titre de la question:')
body = st.text_area('XLNet - Corps de la question:', height=200)

if st.button('Prédire avec le modèle XLNet'):
    predictions = XLNet_predict(title, body)
    st.write("Résultats de la prédiction :")
    for i, pred in enumerate(predictions):
        st.write(f"Prédiction {i + 1}: {', '.join(pred)}")