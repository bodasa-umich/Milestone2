import os
import subprocess


"""
The subprocess library in Python allows execution of shell commands and scripts from within Python code.
It enables spawning new processes, connecting to their input/output/error pipes, and obtaining their return codes.
This is useful for running external programs, automating system tasks, and integrating Python scripts with other tools.
"""


def main():
    """
    Set up paths and arguments, then run create_doc_chunks.py script using subprocess.
    """
    # Define paths relative to data subdirectory
    data_dir = "data"

    fin_doc_paths = [
        os.path.join(data_dir, "2000_10k.txt"),
        os.path.join(data_dir, "1999_10k.txt"),
        os.path.join(data_dir, "1998_10k.txt"),
        os.path.join(data_dir, "1997_10k.txt"),
        os.path.join(data_dir, "1996_10k.txt"),
        os.path.join(data_dir, "1995_10k.txt"),
        os.path.join(data_dir, "1994_10k.txt"),
        os.path.join(data_dir, "1993_10k.txt"),
    ]

    fin_output_paths = [
        os.path.join(data_dir, "2000_10k_df_group.csv"),
        os.path.join(data_dir, "1999_10k_df_group.csv"),
        os.path.join(data_dir, "1998_10k_df_group.csv"),
        os.path.join(data_dir, "1997_10k_df_group.csv"),
        os.path.join(data_dir, "1996_10k_df_group.csv"),
        os.path.join(data_dir, "1995_10k_df_group.csv"),
        os.path.join(data_dir, "1994_10k_df_group.csv"),
        os.path.join(data_dir, "1993_10k_df_group.csv"),
    ]

    case_output_paths = [
        os.path.join(data_dir, "18435_df_group.csv"),
        os.path.join(data_dir, "18776_df_group.csv"),
        os.path.join(data_dir, "20058_df_group.csv"),
        os.path.join(data_dir, "20441_df_group.csv"),
    ]

    enron_txt_path = os.path.join(data_dir, "Enron_Good_Bad.txt")
    enron_pickle_path = os.path.join(data_dir, "EnronGoodBad.pkl")
    fin_combined_output = os.path.join(data_dir, "fin_combined_df.csv")
    case_combined_output = os.path.join(data_dir, "case_combined_df.csv")
    final_combined_output = os.path.join(data_dir, "final_combined_df.csv")
    chunked_output = os.path.join(data_dir, "chunked_groupnames_with_metadata.csv")

    # Copyrighted files to include for RAG demo, do not include for public (academic fair use)
    enron_case_study_path = os.path.join(data_dir, "Enron_Case_Study_Chapters.csv")
    accounting_lessons_path = os.path.join(
        data_dir, "Accounting_Lessons_Chapters_Corrected.csv"
    )

    # Path to .venv python needed for subprocess library correct functioning
    # Full path to Python interpreter from 'where python' at command line
    python_interpreter = (
        r"c:\Users\gary_\OneDrive\Dev Folder\Milestone_2\.venv\Scripts\python.exe"
    )
    # Command to execute
    command = [
        python_interpreter,
        "create_doc_chunks.py",
        "--fin_doc_paths",
        *fin_doc_paths,
        "--output_dir",
        data_dir,
        "--fin_output_paths",
        *fin_output_paths,
        "--case_output_paths",
        *case_output_paths,
        "--enron_txt_path",
        enron_txt_path,
        "--enron_pickle_path",
        enron_pickle_path,
        "--enron_case_study_path",
        enron_case_study_path,
        "--accounting_lessons_path",
        accounting_lessons_path,
        "--fin_combined_output",
        fin_combined_output,
        "--case_combined_output",
        case_combined_output,
        "--final_combined_output",
        final_combined_output,
        "--chunked_output",
        chunked_output,
    ]

    # Execute command
    subprocess.run(command)


if __name__ == "__main__":
    main()
