#!/bin/sh
'''exec' "python" "$0" "$@"
' '''
# -*- coding: utf-8 -*-
import re
import sys
from streamlit.web.cli import main

import numpy as np
from collections import Counter, OrderedDict
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import joblib
from torch.nn.parallel import DataParallel
import torch

# Définition du nooveaau modèle
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', max_length=128)
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=100)

mlb = joblib.load('models_src/mlb_model.joblib')
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

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())

