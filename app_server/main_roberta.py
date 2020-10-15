import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib
from transformers import AutoTokenizer, AutoModel, pipeline
from flask import Flask
from flask import request


app = Flask(__name__)


MODEL = None
TOKENIZER = None
EMBEDDINGS = None
MODEL_PCA = None
DATA_PATH = None


def load_model():
    tokenizer = AutoTokenizer.from_pretrained('youscan/ukr-roberta-base')
    model = AutoModel.from_pretrained('youscan/ukr-roberta-base')
    return tokenizer, model


def get_roberta_embedding(text, tokenizer, model):
    text = text.lower()

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    outputs = model(input_ids, output_hidden_states=True)
    emb = outputs[2]
    emb = np.array([i[0].detach().numpy() for i in emb])[:]
    emb = emb.mean(axis=(0, 1))
    return emb


def create_embeddings(tokenizer, model):
    train_data = pd.read_csv(DATA_PATH, index_col=0)
    train_data['description'] = train_data['description'].str.lower()
    labels_list = train_data['word'].values
    descriptions_list = train_data['description'].values
    embeddings_dict = {}

    for label, descr in zip(labels_list, descriptions_list):
        tmp = embeddings_dict.get(label, [])
        tmp.append(
            get_roberta_embedding(descr, tokenizer, model)
        )
        embeddings_dict[label] = tmp

    for label, embeddings in embeddings_dict.items():
        embeddings_dict[label] = np.mean(embeddings, axis=0)

    return embeddings_dict


def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def get_pca_model(x):
    return PCA(n_components=2).fit(x)


@app.route('/pca_centroids', methods=['POST'])
def get_pca_centroids():
    emb_matrix = np.array(list(EMBEDDINGS.values()))
    x = MODEL_PCA.transform(emb_matrix)
    return {'x': x.tolist(), 'labels': list(EMBEDDINGS.keys())}


@app.route('/model_prediction', methods=['POST'])
def get_model_prediction():
    input_text = request.form['input_text']

    input_embedding = get_roberta_embedding(input_text, TOKENIZER, MODEL)
    labels = []
    distances = []
    for label, emb in EMBEDDINGS.items():
        labels.append(label)
        distances.append(float(distance(emb, input_embedding)))
    
    pca_values = MODEL_PCA.transform(input_embedding.reshape(1, -1))
    
    return {
        'labels': labels, 
        'distances': distances, 
        'pca_values': pca_values.tolist()
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_port', type=int, default=5000, help='the server port')
    parser.add_argument('--data_path', type=str, default="server/data/index.csv", help='the server port')
    arguments = parser.parse_args()
    port = arguments.server_port
    DATA_PATH = arguments.data_path

    print('Loading model...')
    TOKENIZER, MODEL = load_model()

    print('Creating embeddings...')
    EMBEDDINGS = create_embeddings(TOKENIZER, MODEL)
    emb_matrix = np.array(list(EMBEDDINGS.values()))
    
    print('Creating PCA model...')
    MODEL_PCA = get_pca_model(emb_matrix)

    print('Starting server...')
    app.run(host='0.0.0.0', port=port)