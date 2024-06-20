import pandas as pd
import os

# Chunking into smaller chunks, beacuse we're hitting the token limit

## this allows for 10% overlap; adjust with the decimal
def chunk_text(text, max_chars):
    overlap = int(max_chars * 0.1)
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars - overlap)]

def process_csv(file_path, max_chars, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the CSV into a DataFrame
    df = pd.read_csv(file_path, encoding = 'utf-8')

    # Prepare a list to store the chunks
    chunks = []

    # Iterate over each row
    for row_index, row in df.iterrows():
        # Store the first column's name and value
        first_col_name = df.columns[0]
        first_col_value = row[first_col_name]

        # Store the second column's name and value
        second_col_name = df.columns[1]
        second_col_value = row[second_col_name]

        # Store the third column's name and value
        third_col_name = df.columns[2]
        third_col_value = row[third_col_name]

        _4_col_name = df.columns[3]
        _4_col_value = row[_4_col_name]

        _5_col_name = df.columns[4]
        _5_col_value = row[_5_col_name]

        _6_col_name = df.columns[5]
        _6_col_value = row[_6_col_name]

        _7_col_name = df.columns[6]
        _7_col_value = row[_7_col_name]

        _8_col_name = df.columns[7]
        _8_col_value = row[_8_col_name]

        _9_col_name = df.columns[8]
        _9_col_value = row[_9_col_name]

        _10_col_name = df.columns[9]
        _10_col_value = row[_10_col_name]


        # Get the text from the row
        text = str(row)

        # Split the text into chunks
        text_chunks = chunk_text(text, max_chars)

        # For each chunk, store the metadata and the chunk itself
        for chunk in enumerate(text_chunks):
            chunks.append({
                'first_col_name': first_col_name,
                'first_col_value': first_col_value,
                'second_col_name': second_col_name,
                'second_col_value': second_col_value,
                'third_col_name': third_col_name,
                'third_col_value': third_col_value,
                'chunk': chunk,
            })

            # Write the chunk to a file
            filename = f"{output_dir}/class_{first_col_value}_recommended_build.txt"
            with open(filename, 'w') as f:
                f.writelines(first_col_name + ": " + first_col_value + '\n')
                f.writelines('\n')
                f.writelines(second_col_name + ": " + second_col_value + '\n')
                f.writelines('\n')
                f.writelines(third_col_name + ": " + third_col_value + '\n')
                f.writelines('\n')
                f.writelines(_4_col_name + ": " + _4_col_value + '\n')
                f.writelines('\n')
                f.writelines(_5_col_name + ": " + _5_col_value + '\n')
                f.writelines('\n')
                f.writelines(_6_col_name + ": " + _6_col_value + '\n')
                f.writelines('\n')
                f.writelines(_7_col_name + ": " + _7_col_value + '\n')
                f.writelines('\n')
                f.writelines(_8_col_name + ": " + _8_col_value + '\n')
                f.writelines('\n')
                f.writelines(_9_col_name + ": " + _9_col_value + '\n')
                


    # Return the chunks
    return chunks

process_csv(file_path='/Users/kuboi/fandom-chatbot/ScrapeFandom-main/chunks_src/class_build_recommendations_enel.csv', max_chars=1000, output_dir='/Users/kuboi/fandom-chatbot/ScrapeFandom-main/chunks/build_recommendations_1/')