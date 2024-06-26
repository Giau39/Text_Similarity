import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Model(object):
    def __init__(self, model_path):
        super(Model, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)




  def fit_kmeans(self, corpus: List[str], n_clusters: int):
    # compute embeddings
    corpus_embeddings = self.mbert_model.encode(corpus)
    # compute cosine similarity
    similarity_matrix = cosine_similarity(self.get_embeddings(corpus_embeddings))
    
    # cluster
    clustering_model = KMeans(n_clusters)
    clustering_model.fit(similarity_matrix)
    
    # perform PCA
    n_components = min(len(corpus), 3)
    pca = PCA(n_components)
    X_reduced = pca.fit_transform(similarity_matrix)

    # plot corpus in 3d scatter plot
    df = pd.DataFrame({
        'sent': corpus,
        'cluster': clustering_model.labels_.astype(str),
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'z': X_reduced[:, 2] if X_reduced.shape[1] > 2 else np.zeros(X_reduced.shape[0])
    })

    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='cluster', hover_name='sent',
                        range_x=[df.x.min() - 1, df.x.max() + 1],
                        range_y=[df.y.min() - 1, df.y.max() + 1],
                        range_z=[df.z.min() - 1, df.z.max() + 1])

    fig.update_traces(hovertemplate='<b>%{hovertext}</b>')

    # convert graph to html and replace its id
    graph = fig.to_html(full_html=False, include_plotlyjs=False)

    re_graph = r"Plotly\.newPlot\(\s*'(.*?)',.*?\)"
    groups_html = re.search(re_graph, graph, re.DOTALL)
    result = groups_html[0].replace(groups_html[1], 'plotly')
    
    return result

# Assuming 'en_vi_mbert/mbert_model' is the directory where the model was saved
model_path = 'en_vi_mbert/mbert_model'

# Load the model
model = Model(model_path)

def get_model():
    return model