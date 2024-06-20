## CONFIGURED FOR XC3 SKILLS CSV ##

import pandas as pd
import os

# Chunking into smaller chunks, beacuse we're hitting the token limit

## this allows for 10% overlap; adjust with the decimal
def chunk_text(text, max_chars):
    overlap = int(max_chars * 0.15)
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars - overlap)]

def process_csv(file_path, max_chars, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the CSV into a DataFrame
    df = pd.read_csv(file_path, encoding = 'utf-8-sig')

    # Prepare a list to store the chunks
    chunks = []

    # Iterate over each row
    for row_index, row in df.iterrows():
        # Store the first column's name and value
        first_col_name = df.columns[0]
        first_col_value = row[first_col_name]

        second_col_name = df.columns[1]
        second_col_value = row[second_col_name]

        third_col_name = df.columns[2]
        third_col_value = row[third_col_name]

        fourth_col_name = df.columns[3]
        fourth_col_value = row[fourth_col_name]

        fifth_col_name = df.columns[4]
        fifth_col_value = row[fifth_col_name]

        sixth_col_name = df.columns[5]
        sixth_col_value = row[sixth_col_name]

       
        # Get the text from the column
        text = str(row)

        # Split the text into chunks
        text_chunks = chunk_text(text, max_chars)

        # For each chunk, store the metadata and the chunk itself
        for chunk_index, chunk in enumerate(text_chunks):
            chunks.append({
                'first_col_name': first_col_name,
                'first_col_value': first_col_value,
                'second_col_name': second_col_name,
                'second_col_value': second_col_value,
                'third_col_name': third_col_name,
                'third_col_value': third_col_value,
                'fourth_col_name': fourth_col_name,
                'fourth_col_value': fourth_col_value,
                'fifth_col_name': fifth_col_name,
                'fifth_col_value': fifth_col_value,
                'sixth_col_name': sixth_col_name,
                'sixth_col_value': sixth_col_value,
                'chunk': chunk
            })
            
            print(chunk)

            # Write the chunk to a file
            filename = f"{output_dir}/class_overview_{first_col_value}_CRITICAL_INFO_row_{row_index}_chunk_{chunk_index}.txt"
            with open(filename, 'w') as f:
                f.writelines(second_col_name + ": " + second_col_value + '\n')
                f.writelines('\n')
                f.writelines(first_col_name + ": " + first_col_value + '\n')
                f.writelines('\n')
                f.writelines(third_col_name + ": " + third_col_value + '\n')
                f.writelines('\n')
                f.writelines(fourth_col_name + ": " + fourth_col_value + '\n')
                f.writelines('\n')
                f.writelines(fifth_col_name + ": " + fifth_col_value + '\n')
                f.writelines('\n')
                f.writelines(str(sixth_col_name) + ": " + str(sixth_col_value) + '\n')
                f.writelines('\n')



    # Return the chunks
    return chunks

process_csv(file_path='/Users/kuboi/riku/riku/chunks_src/classes_v0.csv', max_chars=2500, output_dir='/Users/kuboi/riku/riku/chunks_1/classes_1/')