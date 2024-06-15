import csv
import os

import gradio as gr

from main import append_to_csv, initialize_csv, main


# This demo code relies on the HuggingFace persistent memory feature and use of its /data directory

# Paths for HuggingFace file environment
input_csv_path = "./chunked_groupnames_with_metadata.csv"  # Loaded at start, encrypted
embeddings_path = "/data/embeddings.pkl"  # Created in hf persist mem if missing

# Path to CSV file in HF space persistent memory
record_csv_path = "/data/user_queries_responses.csv"

# Initialize CSV file
initialize_csv(record_csv_path)

# OpenAI API key - environment variable HuggingFace secrets
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. Please check your HuggingFace secrets."
    )


def update_outputs(model_name, num_docs, query):
    """
    Update outputs based on the query and selected model.

    Args:
        model_name (str): The name of generation model.
        num_docs (int): The number of documents to retrieve.
        query (str): The query text.

    Returns:
        tuple: augmented response, unaugmented response, path to static similarity plot, and interactive plot figure.
    """
    (
        chunked_df,
        document_embeddings,
        top_k_documents,
        top_k_similarities,
        aug_resp,
        unaug_resp,
        static_img,
        inter_plot_fig,
    ) = main(model_name, num_docs, query, input_csv_path, embeddings_path)
    # Append row to CSV
    group_names = top_k_documents["GroupName"].tolist()
    append_to_csv(
        record_csv_path, model_name, query, num_docs, group_names, aug_resp, unaug_resp
    )
    return aug_resp, unaug_resp, static_img, inter_plot_fig


def list_files_in_data():
    """
    List all files in /data directory.

    Returns:
        list: A list of file paths in /data directory.
    """
    data_dir = "/data"
    files = []
    if os.path.exists(data_dir):
        files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))
        ]
    return files


def display_files():
    """
    Display a list of files in /data directory for download.

    Returns:
        list: List of gr.File components for each file in /data directory.
    """
    files = list_files_in_data()
    return [gr.File(label=os.path.basename(f), value=f) for f in files]


def download_csv():
    """
    Serve CSV file for download.

    Returns:
        gr.File: The Gradio File component for CSV file.
    """
    return gr.File(record_csv_path)


# Markdown for instructions
instructions = """
# Enron RAG Demo: Compare Augmented Responses of GPT 3.5, GPT 2, and light version GPT 2 model
1. **Select Generation Model**: Choose the model to generate responses.
2. **Select Number of Documents**: Slider to select number of top documents to retrieve (1-5).
3. **Enter Your Query**: Provide a 1-3 sentence question in Query box.
4. **Submit**: After typing question press [shift]+[enter] in the Query box:
   - **Augmented Response**: The response generated with document augmentation.
   - **Unaugmented Response**: The response generated without document augmentation.
   - **Static Image**: A plot of cosine similarity scores between the query and documents.
   - **Interactive Plot**: 3D UMAP plot with pan, zoom, and dropdown for filtering.
   - **Documents**: Enron SEC complaints, annual reports, and case studies - chunked.
   - **Chunking Method**: RecursiveCharacterTextSplitter with 400 character chunks, 100 overlap.
   - **Embeddings Model:** SentenceTransformer all-MiniLM-L6-v2
   - **Generation Models:** gpt-3.5-turbo, gpt2, distilgpt2
6. **Download CSV**: Click the button below to download the log of user queries and responses.
"""

with gr.Blocks() as demo:
    gr.Markdown(instructions)
    with gr.Row():
        model_name = gr.Dropdown(
            choices=["gpt-3.5-turbo", "gpt2", "distilgpt2"],
            value="gpt-3.5-turbo",
            label="Generation Model",
        )
        num_docs = gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            value=5,
            label="Number of Documents for Retrieval",
        )
    query = gr.Textbox(
        lines=2, placeholder="Enter your query here (1 or 2 sentences)", label="Query"
    )

    with gr.Row():
        augmented_response = gr.Textbox(label="Augmented Response", lines=10)
        unaugmented_response = gr.Textbox(label="Unaugmented Response", lines=10)

    with gr.Row():
        static_image = gr.Image(label="Doc to Query Similarity Scores")
        interactive_plot = gr.Plot(
            label="3D UMAP Interactive Plot of Similarity Distances"
        )

    with gr.Row():
        download_button = gr.Button(
            "Click Here to Get Download Link of Full Queries History to the Right ==>"
        )
        download_file = gr.File(label="Download Link for Queries History CSV")
        download_button.click(fn=download_csv, outputs=download_file)

    query.submit(
        fn=update_outputs,
        inputs=[model_name, num_docs, query],
        outputs=[
            augmented_response,
            unaugmented_response,
            static_image,
            interactive_plot,
        ],
    )

    # Uncomment below to display files in hf /data directory as debug check
    # with gr.Row():
    #     gr.Markdown("## Files in /data Directory:")
    #     file_list = gr.Group(display_files())

demo.launch()
