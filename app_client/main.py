import argparse

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests

SERVEL_URL = None

def display_main_header():
    st.title("Text classifier")
    st.header("Try your own description")
    input_text = st.text_input('')
    return input_text

def create_scatter_plot(points, labels, new_point=None):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(points[:, 0], points[:, 1], label='centroids', color='violet')
    for i, txt in enumerate(labels):
        ax.annotate(txt, points[i] + 0.05)

    if new_point is not None:
        ax.scatter(
            new_point[:, 0], new_point[:, 1], label='new description',
            marker='X', s=100, color='purple'
        )

    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

    return fig

def create_bar_plot(labels, distances):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hlines(xmin=0, xmax=distances, y=labels, color='violet')
    ax.plot(distances, labels, "D", color='purple')
    return fig

def streamlit_display_pca(pca_points, pca_labels, new_pca_point=None, new_label=None):
    st.header('PCA visualization (2 dimensions instead of 768)')
    fig = create_scatter_plot(pca_points, pca_labels, new_pca_point)
    st.pyplot(fig)

def streamlit_display_bar(bar_labels, bar_distances):
    indices = list(range(len(bar_distances)))
    indices.sort(key=lambda x: -bar_distances[x])
    
    st.header('Vector distances')
    fig = create_bar_plot(bar_labels[indices], bar_distances[indices])
    st.pyplot(fig)

def display_info(input_text):
    pca_data = requests.post(f"{SERVEL_URL}/pca_centroids").json()
    pca_points = np.array(pca_data['x'])
    pca_labels = pca_data['labels']

    if input_text == '':
        streamlit_display_pca(pca_points, pca_labels)
        return

    pred_data = requests.post(
        f"{SERVEL_URL}/model_prediction", 
        data={'input_text': input_text}
    ).json()
    bar_distances = np.array(pred_data['distances'])
    bar_labels = np.array(pred_data['labels'])
    new_pca_point = np.array(pred_data['pca_values'])

    predicted_class = bar_labels[bar_distances.argmin()]
    st.header(f"Predicted class: {predicted_class}")
    streamlit_display_pca(pca_points, pca_labels, new_pca_point)
    streamlit_display_bar(bar_labels, bar_distances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server_url', 
        type=str, 
        default="http://127.0.0.1:5000", 
        help='url of the server'
    )
    arguments = parser.parse_args()
    SERVEL_URL = arguments.server_url

    input_text = display_main_header()
    display_info(input_text)