
import pandas as pd
import os
from pathlib import Path
from argparse import ArgumentParser
import re

def clean_transcription(transcription):
    """
    Clean transcription by removing non-alphabetical characters.
    
    Parameters:
    transcription (str): Input text to be cleaned.
    
    Returns:
    str: Cleaned transcription with only alphabetical characters and spaces.
    """
    return re.sub(r'[^a-zA-ZñÑáéíóúÁÉÍÓÚüÜ\s]', '', transcription)

def get_excel_col_letter(df, col_name):
    """
    Convert a zero-based column index to an Excel-style column letter.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the columns.
    col_name (str): The name of the column for which the letter is retrieved.
    
    Returns:
    str: Corresponding Excel column letter.
    """
    # Find the position of the column in the DataFrame
    col_idx = df.columns.get_loc(col_name)
    
    letters = ''
    while col_idx >= 0:
        letters = chr(col_idx % 26 + 65) + letters
        col_idx = col_idx // 26 - 1
    return letters

def word_count(text):
    """
    Count the number of words in a given text.
    
    Parameters:
    text (str): Input text for word count.
    
    Returns:
    int: The number of words in the text.
    """
    try:
        cleaned_text = clean_transcription(text).strip()
        return len(cleaned_text.split())
    except Exception as e:
        print(f"Error in word_count: {e}")
        return 0

def word_count_unique(text):
    """
    Count the number of unique words in a given text.
    
    Parameters:
    text (str): Input text for unique word count.
    
    Returns:
    int: The number of unique words in the text.
    """
    try:
        cleaned_text = clean_transcription(text).strip()
        unique_words = set(cleaned_text.lower().split())
        return len(unique_words)
    except Exception as e:
        print(f"Error in word_count_unique: {e}")
        return 0

if __name__ == '__main__':
    parser = ArgumentParser(description="Script for processing transcription data.")
    parser.add_argument('input_file', type=str, help='Path to the input file containing transcription data.')
    args = parser.parse_args()

    # Read the input file (assumes the file is a CSV)
    if os.path.exists(args.input_file):
        df = pd.read_csv(args.input_file)
        
        # Example usage: Clean the transcription in a column named 'transcription'
        if 'transcription' in df.columns:
            df['cleaned_transcription'] = df['transcription'].apply(clean_transcription)
            df['word_count'] = df['cleaned_transcription'].apply(word_count)
            df['unique_word_count'] = df['cleaned_transcription'].apply(word_count_unique)
        
        # Save the cleaned data to a new file
        output_file = Path(args.input_file).stem + '_cleaned.csv'
        df.to_csv(output_file, index=False)
        print(f"Processed file saved to {output_file}")
    else:
        print(f"File {args.input_file} does not exist.")
