import csv
import os
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import umap

from cryptography.fernet import Fernet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define lists for OpenAI and HuggingFace models
OPENAI_MODELS = ["gpt-3.5-turbo"]
HUGGINGFACE_MODELS = ["gpt2", "distilgpt2"]

# Global variables for models and embeddings to improve execution time
GLOBAL_SENTENCE_MODEL = None
GLOBAL_HUGGINGFACE_MODELS = {}
GLOBAL_OPENAI_CLIENT = None
GLOBAL_OPENAI_MODEL_NAME = None
GLOBAL_UMAP_EMBEDDINGS = None
GLOBAL_UMAP_REDUCER = None


def decrypt_file(file_path, key):
    """
    Decrypt chunked docs file and return path to temporary decrypted file.

    Args:
        file_path (str): path to encrypted file.
        key (bytes): encryption key.

    Returns:
        str: path to temporary decrypted file.
    """
    fernet = Fernet(key)

    with open(file_path, "rb") as enc_file:
        encrypted = enc_file.read()

    decrypted = fernet.decrypt(encrypted)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    with open(temp_file.name, "wb") as dec_file:
        dec_file.write(decrypted)

    return temp_file.name


def load_sentence_transformer_model(model_name="all-MiniLM-L6-v2"):
    """
    Load SentenceTransformer model globally if not already loaded.

    Args:
        model_name (str, optional): name of SentenceTransformer model. Defaults to "all-MiniLM-L6-v2".
    """
    global GLOBAL_SENTENCE_MODEL
    if GLOBAL_SENTENCE_MODEL is None:
        GLOBAL_SENTENCE_MODEL = SentenceTransformer(model_name)


def load_huggingface_model(model_name):
    """
    Load HuggingFace model and tokenizer globally if not already loaded.

    Args:
        model_name (str): name of HuggingFace model.
    """
    global GLOBAL_HUGGINGFACE_MODELS
    if model_name not in GLOBAL_HUGGINGFACE_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        GLOBAL_HUGGINGFACE_MODELS[model_name] = (model, tokenizer)


def load_umap_embeddings(path="document_umap_embeddings.pkl"):
    """
    Load UMAP embeddings and reducer globally if not already loaded.

    Args:
        path (str, optional): path to UMAP embeddings file. Defaults to "document_umap_embeddings.pkl".

    Raises:
        FileNotFoundError: If the embeddings file does not exist.
    """
    global GLOBAL_UMAP_EMBEDDINGS, GLOBAL_UMAP_REDUCER
    if GLOBAL_UMAP_EMBEDDINGS is None or GLOBAL_UMAP_REDUCER is None:
        if os.path.exists(path):
            with open(path, "rb") as f:
                GLOBAL_UMAP_EMBEDDINGS, GLOBAL_UMAP_REDUCER = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"{path} does not exist. Run UMAP embedding creation first."
            )


def create_umap_embeddings(
    document_embeddings, path="/data/document_umap_embeddings.pkl"
):
    """
    Create and save UMAP embeddings and reducer.

    Args:
        document_embeddings (np.ndarray): document embeddings.
        path (str, optional): path to save the UMAP embeddings file. Defaults to "/data/document_umap_embeddings.pkl".

    Returns:
        tuple: UMAP embeddings and the reducer.
    """
    umap_reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=3, random_state=42
    )
    document_umap_embeddings = umap_reducer.fit_transform(document_embeddings)
    with open(path, "wb") as f:
        pickle.dump((document_umap_embeddings, umap_reducer), f)
    return document_umap_embeddings, umap_reducer


def init_openai_client(api_key):
    """
    Initialize and return OpenAI client.

    Args:
        api_key (str): API key for OpenAI.

    Returns:
        OpenAI: initialized OpenAI client.
    """
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def initialize_model(model_name):
    """
    Initialize model based on its type and load it globally if not already loaded.

    Args:
        model_name (str): name of the model to initialize.

    Returns:
        tuple: model, tokenizer, and model type.
    """
    global GLOBAL_OPENAI_CLIENT, GLOBAL_OPENAI_MODEL_NAME

    if model_name in OPENAI_MODELS:
        if GLOBAL_OPENAI_CLIENT is None or GLOBAL_OPENAI_MODEL_NAME != model_name:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API key for OpenAI is not set.")
            GLOBAL_OPENAI_CLIENT = init_openai_client(api_key)
            GLOBAL_OPENAI_MODEL_NAME = model_name
        return GLOBAL_OPENAI_CLIENT, None, "openai"
    elif model_name in HUGGINGFACE_MODELS:
        if model_name not in GLOBAL_HUGGINGFACE_MODELS:
            load_huggingface_model(model_name)
        return (
            GLOBAL_HUGGINGFACE_MODELS[model_name][0],
            GLOBAL_HUGGINGFACE_MODELS[model_name][1],
            "huggingface",
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")


def generate_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate sentence embeddings for a list of texts.

    Args:
        texts (list): list of texts to generate embeddings for.
        model_name (str, optional): name of the SentenceTransformer model. Defaults to "all-MiniLM-L6-v2".

    Returns:
        np.ndarray: generated embeddings.
    """
    if GLOBAL_SENTENCE_MODEL is None:
        load_sentence_transformer_model(model_name)
    embeddings = GLOBAL_SENTENCE_MODEL.encode(texts, convert_to_tensor=True)
    return (
        embeddings.cpu().numpy()
    )  # Move to CPU and convert to NumPy array to address error


def compute_similarities(query_embedding, document_embeddings):
    """
    Compute cosine similarities between the query embedding and document embeddings.

    Args:
        query_embedding (np.ndarray): query embedding.
        document_embeddings (np.ndarray): document embeddings.

    Returns:
        np.ndarray: computed similarities.
    """
    if query_embedding.shape[1] != document_embeddings.shape[1]:
        raise ValueError(
            f"Incompatible dimension for X and Y matrices: X.shape[1] == {query_embedding.shape[1]} while Y.shape[1] == {document_embeddings.shape[1]}"
        )
    similarities = cosine_similarity(query_embedding, document_embeddings)
    return similarities.flatten()


def retrieve_top_k_documents(
    query, chunked_df, document_embeddings, top_k=10, model_name="all-MiniLM-L6-v2"
):
    """
    Retrieve the top_k documents based on similarity scores to query.

    Args:
        query (str): query text.
        chunked_df (pd.DataFrame): chunked DataFrame of documents.
        document_embeddings (np.ndarray): document embeddings.
        top_k (int, optional): number of top documents to retrieve. Defaults to 10.
        model_name (str, optional): name of the SentenceTransformer model. Defaults to "all-MiniLM-L6-v2".

    Returns:
        tuple: top_k documents and their similarity scores.
    """
    query_embedding = generate_embeddings([query], model_name=model_name)
    similarities = compute_similarities(query_embedding, document_embeddings)
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    top_k_similarities = similarities[top_k_indices]
    top_k_documents = chunked_df.iloc[top_k_indices]
    return top_k_documents, top_k_similarities


def plot_similarity_scores(query, top_k_documents, top_k_similarities, output_path):
    """
    Plot cosine similarity scores for top_k documents.

    Args:
        query (str): query text.
        top_k_documents (pd.DataFrame): top_k documents.
        top_k_similarities (np.ndarray): similarity scores.
        output_path (str): path to save the plot.
    """
    group_names = top_k_documents["GroupName"].tolist()
    wrapped_group_names = [wrap_text(name) for name in group_names]
    plt.figure(figsize=(8, 8))
    bars = plt.bar(range(len(top_k_similarities)), top_k_similarities, color="blue")
    plt.xlabel("Document")
    plt.ylabel("Cosine Similarity Score")
    plt.title(f'Cosine Similarity Scores for Query: \n"{query}"')
    plt.xticks(
        range(len(top_k_similarities)), wrapped_group_names, rotation=45, ha="right"
    )
    for bar, score in zip(bars, top_k_similarities):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{score:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
        )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def wrap_text(text, n=6):
    """
    Wrap text every n words.

    Args:
        text (str): text to wrap.
        n (int, optional): number of words per line. Defaults to 6.

    Returns:
        str: wrapped text.
    """
    words = text.split()
    wrapped_text = "\n".join(
        [" ".join(words[i : i + n]) for i in range(0, len(words), n)]
    )
    return wrapped_text


def generate_response_with_augmentation(
    model, tokenizer, query, top_k_documents, model_name
):
    """
    Generate a response with document augmentation.

    Args:
        model: model to generate the response.
        tokenizer: tokenizer for the model.
        query (str): query text.
        top_k_documents (pd.DataFrame): top_k documents.
        model_name (str): name of the model.

    Returns:
        str: generated response.
    """
    messages = [{"role": "user", "content": query}]
    for doc in top_k_documents["chunked_text"]:
        messages.append({"role": "user", "content": doc})
    if model_name in OPENAI_MODELS:
        try:
            response = GLOBAL_OPENAI_CLIENT.chat.completions.create(
                model=model_name, messages=messages, max_tokens=550
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            print("Failed to generate response:", e)
            return None
    elif model_name in HUGGINGFACE_MODELS:
        try:
            augmented_query = (
                query + " " + " ".join([doc for doc in top_k_documents["chunked_text"]])
            )
            inputs = tokenizer(
                augmented_query, return_tensors="pt", truncation=True, max_length=1024
            )
            outputs = model.generate(
                inputs["input_ids"], max_length=550, num_return_sequences=1
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
        except Exception as e:
            print("Failed to generate response:", e)
            return None
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def generate_response_without_augmentation(model, tokenizer, query, model_name):
    """
    Generate a response without document augmentation.

    Args:
        model: model to generate the response.
        tokenizer: tokenizer for the model.
        query (str): query text.
        model_name (str): name of the model.

    Returns:
        str: generated response.
    """
    if model_name in OPENAI_MODELS:
        try:
            messages = [{"role": "user", "content": query}]
            response = GLOBAL_OPENAI_CLIENT.chat.completions.create(
                model=model_name, messages=messages, max_tokens=550
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            print("Failed to generate response:", e)
            return None
    elif model_name in HUGGINGFACE_MODELS:
        try:
            inputs = tokenizer(
                query, return_tensors="pt", truncation=True, max_length=1024
            )
            outputs = model.generate(
                inputs["input_ids"], max_length=550, num_return_sequences=1
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
        except Exception as e:
            print("Failed to generate response:", e)
            return None
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def generate_umap_plot(
    query,
    chunked_df,
    document_embeddings,
    top_k_documents,
    model_name="all-MiniLM-L6-v2",
):
    """
    Generate and return a UMAP plot using precomputed UMAP embeddings.

    Args:
        query (str): query text.
        chunked_df (pd.DataFrame): chunked DataFrame of documents.
        document_embeddings (np.ndarray): document embeddings.
        top_k_documents (pd.DataFrame): top_k documents.
        model_name (str, optional): name of SentenceTransformer model. Defaults to "all-MiniLM-L6-v2".

    Returns:
        plotly.graph_objs._figure.Figure: generated UMAP plot.
    """
    query_embedding = generate_embeddings([query], model_name=model_name)[0]

    # Create UMAP embeddings if not already loaded
    global GLOBAL_UMAP_EMBEDDINGS, GLOBAL_UMAP_REDUCER
    if GLOBAL_UMAP_EMBEDDINGS is None or GLOBAL_UMAP_REDUCER is None:
        GLOBAL_UMAP_EMBEDDINGS, GLOBAL_UMAP_REDUCER = create_umap_embeddings(
            document_embeddings
        )

    query_umap_embedding = GLOBAL_UMAP_REDUCER.transform(query_embedding.reshape(1, -1))
    all_umap_embeddings = np.vstack([GLOBAL_UMAP_EMBEDDINGS, query_umap_embedding])

    chunked_df_copy_umap = chunked_df.copy()
    chunked_df_copy_umap["UMAP1"] = all_umap_embeddings[:-1, 0]
    chunked_df_copy_umap["UMAP2"] = all_umap_embeddings[:-1, 1]
    chunked_df_copy_umap["UMAP3"] = all_umap_embeddings[:-1, 2]

    query_row_umap = pd.DataFrame(
        {
            "GroupName": ["Query"],
            "UMAP1": [all_umap_embeddings[-1, 0]],
            "UMAP2": [all_umap_embeddings[-1, 1]],
            "UMAP3": [all_umap_embeddings[-1, 2]],
            "chunked_text": [query],  # Add query text
            "is_retrieved": [False],
            "is_query": [True],  # Add flag for query point
        }
    )
    chunked_df_copy_umap["is_query"] = (
        False  # Set is_query for all documents False to start
    )
    chunked_df_copy_umap = pd.concat(
        [chunked_df_copy_umap, query_row_umap], ignore_index=True
    )

    retrieved_indices_umap = [
        chunked_df_copy_umap.index[
            chunked_df_copy_umap["chunked_text"] == doc["chunked_text"]
        ][0]
        for i, doc in top_k_documents.iterrows()
    ]
    chunked_df_copy_umap["is_retrieved"] = chunked_df_copy_umap.index.isin(
        retrieved_indices_umap
    )
    chunked_df_copy_umap["is_retrieved"] = chunked_df_copy_umap["is_retrieved"].astype(
        bool
    )  # Explicitly cast to bool to address warning
    chunked_df_copy_umap["is_query"] = chunked_df_copy_umap["is_query"].astype(
        bool
    )  # Explicitly cast to bool to address warning
    chunked_df_copy_umap.loc[
        chunked_df_copy_umap["GroupName"] == "Query", "is_retrieved"
    ] = "Query"

    x_range = [chunked_df_copy_umap["UMAP1"].min(), chunked_df_copy_umap["UMAP1"].max()]
    y_range = [chunked_df_copy_umap["UMAP2"].min(), chunked_df_copy_umap["UMAP2"].max()]
    z_range = [chunked_df_copy_umap["UMAP3"].min(), chunked_df_copy_umap["UMAP3"].max()]

    camera = dict(eye=dict(x=1.25, y=1.25, z=1.25))  # Adjust to set view

    fig_umap = px.scatter_3d(
        chunked_df_copy_umap,
        x="UMAP1",
        y="UMAP2",
        z="UMAP3",
        color="is_retrieved",
        color_discrete_map={False: "blue", True: "red", "Query": "black"},
        custom_data=[
            "GroupName",
            "chunked_text",
            "is_query",
        ],  # Add is_query to custom_data
        title="UMAP Projection of Embeddings in 3D with GroupName Tooltip",
        labels={
            "UMAP1": "UMAP Dimension 1",
            "UMAP2": "UMAP Dimension 2",
            "UMAP3": "UMAP Dimension 3",
        },
    )

    for trace in fig_umap.data:
        if "Query" in trace.customdata[:, 0]:
            trace.hovertemplate = (
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
            )
        else:
            trace.hovertemplate = "<b>%{customdata[0]}</b><extra></extra>"

    fig_umap.update_layout(
        scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            camera=camera,  # Initial camera position
        ),
        width=700,
        height=600,
        title={
            "text": "Pan, Zoom, and Drag to Explore Distances",
            "y": 0.90,
            "x": 0.2,
            "xanchor": "left",
            "yanchor": "bottom",
        },
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Show All",
                        "method": "update",
                        "args": [
                            {"visible": [True] * len(fig_umap.data)},
                            {"scene.camera": camera},
                        ],
                    },
                    {
                        "label": "Show Red and Black Only",
                        "method": "update",
                        "args": [
                            {
                                "visible": [
                                    (trace.name == "True") or (trace.name == "Query")
                                    for trace in fig_umap.data
                                ]
                            },
                            {"scene.camera": camera},
                        ],
                    },
                ],
                "direction": "down",
                "showactive": True,
                "x": 0,  # Position dropdown horizontal
                "xanchor": "left",
                "y": 1.0,  # Position dropdown vertical
                "yanchor": "top",
            }
        ],
        legend=dict(
            title_text="Documents Retrieved",
            x=0.01,  # Positioning legend on left
            xanchor="left",
            y=0.90,  # Positioning legend below dropdown
            yanchor="top",
        ),
    )

    fig_umap.update_traces(
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.1)",  # 0.1 Transparency - did not work, research if possible
            font_color="black",
            align="left",
        ),
        marker=dict(size=4),
    )

    return fig_umap


def initialize_csv(path):
    """
    Initialize a CSV file with headers if it doesn't exist.

    Args:
        path (str): The path to the CSV file.
    """
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "generation_model",
                    "submitted_query_text",
                    "num_docs",
                    "retrieved_group_names",
                    "augmented_response",
                    "unaugmented_response",
                ]
            )


def append_to_csv(path, model_name, query, num_docs, group_names, aug_resp, unaug_resp):
    """
    Append a new row to user query tracking CSV file.

    Args:
        path (str): path to the CSV file.
        model_name (str): name of the model used.
        query (str): query text.
        num_docs (int): number of documents retrieved.
        group_names (list): list of retrieved group names.
        aug_resp (str): augmented response.
        unaug_resp (str): unaugmented response.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                timestamp,
                model_name,
                query,
                num_docs,
                ",".join(group_names),
                aug_resp,
                unaug_resp,
            ]
        )


def main(model_name, num_docs, query, input_csv_path, embeddings_path):
    """
    Main function to run doc retrieval and response generation pipeline.

    Args:
        model_name (str): name of the model to use.
        num_docs (int): number of documents to retrieve.
        query (str): query text.
        input_csv_path (str): path to the input CSV file.
        embeddings_path (str): path to the embeddings file.

    Returns:
        tuple: chunked DataFrame, document embeddings, top_k documents, top_k similarities, augmented response,
               unaugmented response, path to static similarity plot, and interactive plot figure.
    """
    model, tokenizer, model_type = initialize_model(model_name)

    # Load and decrypt the CSV file
    encryption_key = os.getenv("ENCRYPTION_KEY").encode()
    decrypted_csv_path = decrypt_file(input_csv_path, encryption_key)
    chunked_df = pd.read_csv(decrypted_csv_path)

    # Check if the embeddings file exists, create embeddings if it doesn't
    # Save embeddings to /data in Hugging Face environment
    if not os.path.exists(embeddings_path):
        print(f"{embeddings_path} does not exist. Creating embeddings.")
        document_texts = chunked_df["chunked_text"].tolist()
        document_embeddings = generate_embeddings(document_texts)
        with open(embeddings_path, "wb") as f:
            pickle.dump(document_embeddings, f)
    else:
        with open(embeddings_path, "rb") as f:
            document_embeddings = pickle.load(f)
        if document_embeddings.shape[1] != 384:
            raise ValueError(
                f"Expected document embeddings of dimension 384, but got {document_embeddings.shape[1]}"
            )

    # Ensure UMAP embeddings are created or loaded
    # Save UMAP embeddings to /data in Hugging Face environment
    umap_path = "/data/document_umap_embeddings.pkl"
    if not os.path.exists(umap_path):
        GLOBAL_UMAP_EMBEDDINGS, GLOBAL_UMAP_REDUCER = create_umap_embeddings(
            document_embeddings, umap_path
        )
    else:
        load_umap_embeddings(umap_path)

    with ThreadPoolExecutor() as executor:
        # Retrieve top_k documents in parallel
        retrieval_future = executor.submit(
            retrieve_top_k_documents, query, chunked_df, document_embeddings, num_docs
        )
        top_k_documents, top_k_similarities = retrieval_future.result()

        # Generate responses in parallel
        aug_response_future = executor.submit(
            generate_response_with_augmentation,
            model,
            tokenizer,
            query,
            top_k_documents,
            model_name,
        )
        unaug_response_future = executor.submit(
            generate_response_without_augmentation,
            model,
            tokenizer,
            query,
            model_name,
        )
        augmented_response = aug_response_future.result()
        unaugmented_response = unaug_response_future.result()

        # Generate plots in parallel
        static_plot_future = executor.submit(
            plot_similarity_scores,
            query,
            top_k_documents,
            top_k_similarities,
            "static_similarity_plot.png",
        )
        umap_plot_future = executor.submit(
            generate_umap_plot,
            query,
            chunked_df,
            document_embeddings,
            top_k_documents,
            "all-MiniLM-L6-v2",
        )
        static_plot_future.result()
        interactive_plot_fig = umap_plot_future.result()

    # Expanded returns to make csv tracking work
    return (
        chunked_df,
        document_embeddings,
        top_k_documents,
        top_k_similarities,
        augmented_response,
        unaugmented_response,
        "static_similarity_plot.png",
        interactive_plot_fig,
    )
