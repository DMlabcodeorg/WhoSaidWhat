# Transcription Data Reliability Analysis Processor

This Python script processes transcription data, cleaning it and providing word count statistics. It is particularly useful for cleaning up text data from transcriptions, removing non-alphabetical characters, and calculating both total word counts and unique word counts.

## Features
- Clean transcriptions by removing non-alphabetical characters.
- Convert a column index to Excel-style column letters.
- Count the total number of words in a transcription.
- Count the number of unique words in a transcription.

## Requirements

- Python 3.x
- pandas
- argparse

Install the required Python libraries using:
```bash
pip install pandas argparse
```

## Usage

Run the script using the command line:
```bash
python S_Tran_Whisper.py <input_file>
```

- `<input_file>`: Path to the CSV file containing the transcription data.

### Example

If your CSV file has a column named `transcription`, the script will:
1. Clean the transcription by removing non-alphabetical characters.
2. Count the total number of words.
3. Count the unique words.

The output will be saved in a new Excel file with the name `Trans_Stat.xlsx`, as well as some detailed segments level analysis CSV files.
