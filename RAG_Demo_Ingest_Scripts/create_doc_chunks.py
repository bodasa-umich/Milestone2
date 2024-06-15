import os
import argparse
import pickle

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.text import partition_text


def parse_elements_to_dataframe(elements):
    """
    Convert a list of elements into a DataFrame, grouping text by title.
    Unstructured.io parsing of Enron 10-K Annual Reports created the elements.
    This is an experimental method to quickly parse these text files into our format,
    using the Title elements as the GroupName for NarrativeText element sections.

    Args:
        elements (list): List of elements with categories and text.

    Returns:
        DataFrame: DataFrame with GroupName and grouped_text columns.
    """
    current_group_name = None
    current_group_text = ""
    df_list = []

    for element in elements:
        if element.category == "Title":
            current_group_name = element.text
            current_group_text = ""
        elif element.category == "NarrativeText":
            current_group_text += (" " if current_group_text else "") + element.text

        if current_group_name and current_group_text:
            df_list.append(
                pd.DataFrame(
                    {
                        "GroupName": [current_group_name],
                        "grouped_text": [current_group_text],
                    }
                )
            )
            current_group_name = None
            current_group_text = ""

    df = pd.concat(df_list, ignore_index=True)
    return df


def process_fin_documents(document_paths, output_dir=None):
    """
    Process financial documents and save grouped text into CSV files.
    After parsing the 10-Ks based on elements, we create the CSVs.

    Args:
        document_paths (list): List of paths to financial documents.
        output_dir (str): Directory to save output files.
    """
    for document_path in document_paths:
        filename = os.path.splitext(os.path.basename(document_path))[0]
        output_file = (
            os.path.join(output_dir, f"{filename}_df_group.csv")
            if output_dir
            else f"{filename}_df_group.csv"
        )
        elements = partition_text(filename=document_path)
        parsed_df = parse_elements_to_dataframe(elements)
        parsed_df.to_csv(output_file, index=False)
        print(f"Parsed document: {document_path}, saved as: {output_file}")


def load_and_process_fin_documents(document_paths):
    """
    Load and process financial document CSV files, filtering and formatting group names.
    The Title elements used as GroupNames can be used as filters to remove unwanted sections.
    We removed sections that did not seem helpful or relevant to retrieval augmentation.

    Args:
        document_paths (list): List of paths to processed financial document CSV files.

    Returns:
        DataFrame: Combined and processed DataFrame.
    """
    dfs = []
    words_to_exclude = [
        "CONSENT OF INDEPENDENT PUBLIC ACCOUNTANTS",
        "POWER OF ATTORNEY",
    ]
    for path in document_paths:
        df = pd.read_csv(path)
        df["GroupName"] = df["GroupName"].astype(str)
        df["grouped_text"] = df["grouped_text"].astype(str)
        mask = df["GroupName"].apply(
            lambda x: not any(x.startswith(word) for word in words_to_exclude)
        )
        df_filtered = df.loc[mask].copy()
        filename = os.path.basename(path).split(".")[0][:4]
        prepend_str = f"{filename} Annual Report: "
        df_filtered["GroupName"] = prepend_str + df_filtered["GroupName"]
        dfs.append(df_filtered)
    fin_combined_df = pd.concat(dfs, ignore_index=True)
    return fin_combined_df


def load_and_process_case_documents(document_paths):
    """
    Load and process case document CSV files, filtering specific GroupNames.
    Similar to how we filtered out sections of 10-Ks, we also filter these docs.

    Args:
        document_paths (list): List of paths to case document CSV files.

    Returns:
        DataFrame: Combined and processed DataFrame.
    """
    dfs = []
    words_to_exclude = ["Violation", "Violations", "Aiding"]
    for path in document_paths:
        df = pd.read_csv(path)
        mask = df["GroupName"].apply(
            lambda x: not any(x.startswith(word) for word in words_to_exclude)
        )
        df_filtered = df.loc[mask].copy()
        dfs.append(df_filtered)
    case_combined_df = pd.concat(dfs, ignore_index=True)
    return case_combined_df


def load_and_concatenate_dfs(
    case_path,
    fin_path,
    enron_pickle_path,
    enron_case_study_path,
    accounting_lessons_path,
    output_path,
):
    """
    Load multiple CSV and pickle files, concatenate into single DataFrame, and save result.
    This is the DataFrame which serves as the input to the chunking method.

    Args:
        case_path (str): Path to case document CSV file.
        fin_path (str): Path to financial document CSV file.
        enron_pickle_path (str): Path to Enron pickle file.
        enron_case_study_path (str): Path to Enron case study CSV file.
        accounting_lessons_path (str): Path to accounting lessons CSV file.
        output_path (str): Path to save combined DataFrame.

    Returns:
        DataFrame: Combined DataFrame.
    """
    case_df = pd.read_csv(case_path, encoding="utf-8")
    fin_df = pd.read_csv(fin_path, encoding="utf-8")
    enron_case_study_df = pd.read_csv(enron_case_study_path, encoding="utf-8")
    accounting_lessons_df = pd.read_csv(accounting_lessons_path, encoding="utf-8")

    with open(enron_pickle_path, "rb") as f:
        enron_df = pd.read_pickle(f)

    combined_df = pd.concat(
        [case_df, fin_df, accounting_lessons_df, enron_case_study_df, enron_df],
        ignore_index=True,
    )
    combined_df.to_csv(output_path, index=False)
    with open(output_path.replace(".csv", ".pkl"), "wb") as f:
        pickle.dump(combined_df, f)
    return combined_df


def process_enron_txt_to_pickle(enron_txt_path, enron_pickle_path):
    """
    Process Enron text file and save as pickle file.
    This text file needed to be converted from pipe-delimited to our format.

    Args:
        enron_txt_path (str): Path to Enron text file.
        enron_pickle_path (str): Path to save Enron pickle file.
    """
    df = pd.read_csv(
        enron_txt_path,
        delimiter="|",
        header=None,
        names=["GroupName", "grouped_text"],
        skip_blank_lines=False,
        encoding="utf-8",
        engine="python",
    )
    with open(enron_pickle_path, "wb") as f:
        pickle.dump(df, f)
    print(f"Processed Enron data and saved to {enron_pickle_path}")


def main(args):
    """
    Main function to process documents, concatenate and chunk data.
    Output is the complete chunked doc data, for input to the RAG demo app.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    # Process Enron text file to pickle
    process_enron_txt_to_pickle(args.enron_txt_path, args.enron_pickle_path)

    # Process financial docs
    process_fin_documents(args.fin_doc_paths, args.output_dir)

    # Load and process financial documents
    fin_combined_df = load_and_process_fin_documents(args.fin_output_paths)
    fin_combined_df.to_csv(args.fin_combined_output, index=False)

    # Load and process case docs
    case_combined_df = load_and_process_case_documents(args.case_output_paths)
    case_combined_df.to_csv(args.case_combined_output, index=False)

    # Load and concatenate dataframes
    final_df = load_and_concatenate_dfs(
        args.case_combined_output,
        args.fin_combined_output,
        args.enron_pickle_path,
        args.enron_case_study_path,
        args.accounting_lessons_path,
        args.final_combined_output,
    )

    # Load final combined dataframe
    final_df = pd.read_pickle(args.final_combined_output.replace(".csv", ".pkl"))

    # Combine GroupName and grouped_text for chunking
    final_df["combined_text"] = final_df["GroupName"] + ": " + final_df["grouped_text"]

    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    # Chunk text and keep track of GroupName as metadata
    chunked_texts, metadata = [], []
    for _, row in final_df.iterrows():
        chunks = text_splitter.split_text(row["combined_text"])
        chunked_texts.extend(chunks)
        metadata.extend([row["GroupName"]] * len(chunks))

    # Create DataFrame with chunked texts and metadata
    chunked_df = pd.DataFrame({"chunked_text": chunked_texts, "GroupName": metadata})
    chunked_df.to_csv(args.chunked_output, index=False)

    print(f"Chunked DataFrame saved to {args.chunked_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process financial and case documents."
    )
    parser.add_argument(
        "--fin_doc_paths",
        nargs="+",
        required=True,
        help="Paths to financial document files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--fin_output_paths",
        nargs="+",
        required=True,
        help="Paths to processed financial document output files.",
    )
    parser.add_argument(
        "--case_output_paths",
        nargs="+",
        required=True,
        help="Paths to case document output files.",
    )
    parser.add_argument(
        "--enron_txt_path", type=str, required=True, help="Path to Enron text file."
    )
    parser.add_argument(
        "--enron_pickle_path",
        type=str,
        required=True,
        help="Path to Enron pickle file.",
    )
    parser.add_argument(
        "--enron_case_study_path",
        type=str,
        required=True,
        help="Path to Enron Case Study CSV file.",
    )
    parser.add_argument(
        "--accounting_lessons_path",
        type=str,
        required=True,
        help="Path to Accounting Lessons CSV file.",
    )
    parser.add_argument(
        "--fin_combined_output",
        type=str,
        required=True,
        help="Path to save combined financial output file.",
    )
    parser.add_argument(
        "--case_combined_output",
        type=str,
        required=True,
        help="Path to save combined case output file.",
    )
    parser.add_argument(
        "--final_combined_output",
        type=str,
        required=True,
        help="Path to save final combined output file.",
    )
    parser.add_argument(
        "--chunked_output",
        type=str,
        required=True,
        help="Path to save chunked DataFrame output file.",
    )
    args = parser.parse_args()

    main(args)
