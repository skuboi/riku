import pandas as pd
import os
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

# Chunking into smaller chunks, beacuse we're hitting the token limit

## this allows for 10% overlap; adjust with the decimal
def chunk_text2(text, max_chars):
    overlap = int(max_chars * 0.1)
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars - overlap)]

def chunk_text(text, max_chars):
    chunker = SentenceTransformersTokenTextSplitter(
        chunk_overlap=5,  # Overlap between chunks
        tokens_per_chunk=256  # Number of tokens per chunk, try some values like 256
    )

    new_chunks = chunker.split_text(text)
    print(new_chunks)
    return new_chunks

def process_csv(file_path, max_chars, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the CSV into a DataFrame
    df = pd.read_csv(file_path, encoding = 'unicode_escape')

    # Prepare a list to store the chunks
    chunks = []

    # Iterate over each row
    for row_index, row in df.iterrows():
        # Store the first column's name and value
        first_col_name = df.columns[0]
        first_col_value = row[first_col_name]

        # Iterate over the rest of the columns
        for col_name in df.columns[1:]:
            # Get the text from the column
            text = str(row[col_name])

            # Split the text into chunks
            text_chunks = chunk_text(text, max_chars)

            # For each chunk, store the metadata and the chunk itself
            for chunk_index, chunk in enumerate(text_chunks):
                chunks.append({
                    'first_col_name': first_col_name,
                    'first_col_value': first_col_value,
                    'col_name': col_name,
                    'chunk': chunk,
                })

                # Write the chunk to a file
                filename = f"{output_dir}/class_{first_col_value}_{col_name}_row_{row_index}_chunk_{chunk_index}.txt"
                with open(filename, 'w') as f:
                    f.writelines(first_col_name + ": " + first_col_value + '\n')
                    f.writelines("Column: " + col_name + '\n')
                    f.writelines("Contents: " + chunk)

    # Return the chunks
    return chunks

process_csv(file_path='/Users/kuboi/fandom-chatbot/ScrapeFandom-main/chunks_src/skills_v0.csv', max_chars=1000, output_dir='/Users/kuboi/fandom-chatbot/ScrapeFandom-main/chunks_1/skills_0/')