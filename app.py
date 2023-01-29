from pathlib import Path
import os
import pandas as pd
import numpy as np
import openai
#from openai.embeddings_utils import get_embedding
#import requests
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import matplotlib
import joblib
import gradio as gr
from flask import Flask, request
#plt.style.use('seaborn-poster')

EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY = os.path.join(Path.cwd(), "openai_api_key.txt")
openai.api_key_path = os.path.join(Path.cwd(), OPENAI_API_KEY)

def get_embedding(text: str, model=EMBEDDING_MODEL) -> list[float]:
    """
    Gets a text as an input and the embedding model used from Openai
    Returns the embeddings of that blurb of text
    """
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]



def get_plot(website: str) -> matplotlib.figure.Figure:
    """
    Has a blurb of text as input and returns a plot with the of that 
    blurb of text marked on the plot with black.It uses an unsupervised method 
    of classification called T-distributed Stochastic Neighbor Embedding.
    """
    df = pd.read_csv("data_plot.csv")
    matrix = np.array(df.embeddings.apply(eval).to_list())
    website_embed = get_embedding(website, model = EMBEDDING_MODEL)
    website_embed = np.array(website_embed)
    matrix = np.append(matrix, website_embed.reshape(1, -1), axis = 0)
    mapping = {"benign": 0, "defacement": 1, "phishing": 2, "malware": 3}
    df["type"] = df["type"].map(mapping)
    colors = ["red", "darkorange", "gold", "turquoise"]
   
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    
    x = [x for x,y in vis_dims]
    y = [y for x,y in vis_dims]
    
    color_indices = df.type.values - 1
    fig = plt.figure()
    colormap = matplotlib.colors.ListedColormap(colors)
    plt.scatter(x[:1000], y[:1000], c=color_indices, cmap=colormap, alpha=0.3)
    plt.scatter(x[-1:], y[-1:], c="black", alpha = 0.8)
    plt.text(-71, 52, "Benign", fontsize = 15, bbox = dict(facecolor = 'red', alpha = 0.5))
    plt.text(-71, 46, "Defacement", fontsize = 15, bbox = dict(facecolor = 'darkorange', alpha = 0.5))
    plt.text(-71, 40, "Phishing", fontsize = 15, bbox = dict(facecolor = 'gold', alpha = 0.5))
    plt.text(-71, 34, "Malware", fontsize = 15, bbox = dict(facecolor = 'turquoise', alpha = 0.5))
    plt.text(-71, 28, "Our Data", fontsize = 15, bbox = dict(facecolor = 'black', alpha = 0.5))
    

    return fig

def predict_label(website: str) -> str:
    """
    It takes the blurb of text and predicts whether it is malicious or not

    """
    loaded_model = joblib.load("model.joblib")
    embedding = get_embedding(website, model = EMBEDDING_MODEL)
    embedding = np.array(embedding)
    y_predicted = loaded_model.predict(embedding.reshape(1, -1))
    if y_predicted[0] == "benign":
        return "This website is most probably safe."
    elif y_predicted[0] != "benign":
        return "This website is most probably malicious." 
    
#def my_app(website: str):
#   return (get_plot(website), predict_label(website))

#get_plot(website = "https://www.youtube.com/watch?v=RiCQzBluTxU")
#print(predict_label(website = "https://www.youtube.com/watch?v=RiCQzBluTxU"))

interface = gr.Interface(
    fn = predict_label,
    inputs=["text"],
    outputs=[ "text"],
    live= True,
    title= "Malicious Website Detector",
    description= "This website comes as a helping tool for those that want to surf safely on the internet.\n Attention: Not all predictions are true and this should be taken as a demo for now.",
    
)
interface.launch(share=True)

"""app = Flask(__name__)
@app.route("/result", methods = ["POST", "GET"])
def result():
    output = request.get_data()
    website = str(output)
    
    return predict_label(website=website)
if __name__ == "__main__":
    app.run(debug = True, port = 2000)"""




