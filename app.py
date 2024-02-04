from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import openai
from sklearn.manifold import TSNE
import joblib
import gradio as gr
from typing import Optional
import altair as alt

# Load environment variables and set API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load resources once
EMBEDDING_MODEL = "text-embedding-ada-002"
df = pd.read_csv("data_plot.csv")
matrix = np.array(df.embeddings.apply(eval).to_list())
loaded_model = joblib.load("model.joblib")

def get_embedding(text: str, model=EMBEDDING_MODEL) -> list[float]:
    """
    Gets a text as an input and the embedding model used from OpenAI.
    Returns the embeddings of that blurb of text.
    """
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def get_plot(website: Optional[str], matrix=matrix, df=df) -> alt.Chart:
    if website:
        website_embed = get_embedding(website, model=EMBEDDING_MODEL)
        website_embed = np.array(website_embed)
        updated_matrix = np.append(matrix, website_embed.reshape(1, -1), axis=0)

        tsne = TSNE(n_components=2, perplexity=50, random_state=42, init='random', learning_rate=200)
        vis_dims = tsne.fit_transform(updated_matrix)

        # Create a new DataFrame for visualization
        df_vis = pd.DataFrame(vis_dims, columns=['x', 'y'])
        df_vis['type'] = df['type'].tolist() + ['Our Data']  # Convert to list and append directly
        df_vis["url"] = df["url"].tolist() + [website]  # Convert to list and append directly

        # Ensure a unique index after appending by resetting the DataFrame's index
        df_vis.reset_index(drop=True, inplace=True)

        scale = alt.Scale(domain=['benign', 'defacement', 'phishing', 'malware', 'Our Data'],
                          range=['red', 'darkorange', 'gold', 'turquoise', 'black'])

        scatter_plot = alt.Chart(df_vis).mark_circle(size=60).encode(
            x='x',
            y='y',
            color=alt.Color('type', scale=scale),
            tooltip=['type', 'url']
        ).interactive()
        return scatter_plot
    else:
        return None

def predict_label(website: Optional[str] = "") -> str:
    if website:
        embedding = get_embedding(website, model=EMBEDDING_MODEL)
        embedding = np.array(embedding)
        y_predicted = loaded_model.predict(embedding.reshape(1, -1))
        return "This website is most probably safe." if y_predicted[0] == "benign" else "This website is most probably malicious."
    else:
        return "Please enter a website URL."

def gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Malicious Website Detector")
        gr.Markdown("This tool helps you identify potentially malicious websites. \n **Note:** This is a demonstration and results may not be accurate.")
        website_input = gr.Textbox(label="Enter website URL")
        predict_button = gr.Button("Predict")
        prediction_output = gr.Textbox(label="Prediction", interactive=True)  # Ensure the output is interactive
        plot_output = gr.Plot(label="Website Embedding Plot", height=500, width=750)
        
        def update_output(website):
            prediction = predict_label(website)
            plot = get_plot(website) if website else None
            # Instead of trying to update the components, just return the values.
            # Gradio will automatically update the components with these returned values.
            return prediction, plot

        predict_button.click(update_output, inputs=website_input, outputs=[prediction_output, plot_output])

    demo.launch()

if __name__ == "__main__":
    gradio_app()