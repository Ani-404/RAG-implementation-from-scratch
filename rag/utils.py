# utils script - useful function(s) to use with the RAG pipeline 

import os

DEFAULT_DOCS_PATH = os.path.join(os.path.dirname(__file__), "documents")

def get_documents(doc_path: str = DEFAULT_DOCS_PATH):
    """
    Retrieves all .txt documents from the specified directory and stores them
    in a list of dictionaries with id, name, and text.

    Args:
        doc_path (str): Path to the directory containing .txt files.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - "id": A unique identifier for the document (starting from 1).
              - "name": The name of the .txt file (without extension).
              - "text": The content of the .txt file.
    """
    documents = []

    supported_extensions = ('.txt', '.md', '.markdown')
    try:
        # a list of all supported files in the directory
        files = sorted([f for f in os.listdir(doc_path) if f.lower().endswith(supported_extensions)])
        
        # iterate through all the .txt files
        for idx, file_name in enumerate(files, start=1):
            file_path = os.path.join(doc_path, file_name)
            
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Adding document details to the list
            documents.append({
                "id": idx,
                "name": os.path.splitext(file_name)[0], # this removes the .txt extension
                "text": content
            })


    except FileNotFoundError:
        print(f"Error: The directory '{doc_path}' does not exist.")


    except Exception as e:
        print(f"An error occurred: {e}")     


    return documents

