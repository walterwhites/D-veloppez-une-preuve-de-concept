import numpy as np
import nltk
import zipfile
import os
import api.app.text_preprocessing
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

####################################################################################################

app = FastAPI()

# Nom du fichier zip contenant les modèles
models_zip_file = 'api/app/models_src.zip'

# Charger les modèles depuis le fichier zip
with zipfile.ZipFile(models_zip_file, 'r') as zip_ref:
    if not os.path.exists('api/app/models_src'):
        zip_ref.extractall()

# Charger les modèles
mlb = joblib.load('api/app/models_src/mlb_model.joblib')
pipeline_xlnet_joblib = joblib.load('api/app/models_src/XLNet_custom_classification_layer_model.joblib')
pipeline_multinomial_naive_bayes_joblib = joblib.load('api/app/models_src/oneVsRestClassifier_mlb_model.joblib')

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    prediction: str


@app.post("/models/xlnet/predict/")
def xlnet_predict(title: str, body: str):
    try:
        nltk.download('punkt', force=True)
        nltk.download('wordnet', force=True)
        nltk.download('stopwords', force=True)
    except Exception as e:
        print(f"Erreur lors du téléchargement des données NLTK : {e}")

    content = title + ' ' + body
    processed_question = api.app.text_preprocessing.preprocess_text(content)
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

@app.post("/models/multinomial_naive_bayes/predict/")
def multinomial_naive_bayes_predict(title: str, body: str):
    try:
        nltk.download('punkt', force=True)
        nltk.download('wordnet', force=True)
        nltk.download('stopwords', force=True)
    except Exception as e:
        print(f"Erreur lors du téléchargement des données NLTK : {e}")

    content = title + ' ' + body
    processed_question = api.app.text_preprocessing.preprocess_text(content)
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

# %%
