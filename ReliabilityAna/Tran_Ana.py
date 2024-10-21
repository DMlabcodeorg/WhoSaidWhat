import pandas as pd
import os
from pathlib import Path
from argparse import ArgumentParser
import re

def clean_transcription(transcription):
    return re.sub(r'[^a-zA-ZñÑáéíóúÁÉÍÓÚüÜ\s]', '', transcription)

def get_excel_col_letter(df, col_name):
    """Convert zero-based column index to Excel column letter."""
    # Find the position of the 'mean' column in the DataFrame
    col_idx = df.columns.get_loc(col_name)

    letters = ''
    while col_idx >= 0:
        letters = chr(col_idx % 26 + 65) + letters
        col_idx = col_idx // 26 - 1
    return letters

def word_count(s):
    try:
        # Remove all non-character symbols
        cleaned_str = clean_transcription(s).strip()
        return len(cleaned_str.split())
    except:
        return 0

def word_count_unq(s):
    try:
        # Remove all non-character symbols and convert to lowercase
        cleaned_str = clean_transcription(s).strip()
        # Use a set to keep track of unique words
        unique_words = set(cleaned_str.split())
        return len(unique_words)
    except Exception as e:
        print(f"Error: {e}")
        return 0

def calculate_sums_and_ratios(file_path):
    # Read the Excel file
    # print(file_path)
    df = pd.read_excel(file_path)

    df = df.dropna(subset=[df.columns[10], df.columns[12], df.columns[14], df.columns[6]])

    # Function to check if a value is numeric
    def is_numeric(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # Identify the rows where the specific columns are not numeric
    mask = (df.iloc[:, 10].apply(is_numeric)) & (df.iloc[:, 12].apply(is_numeric))

    # Keep only the rows where the specific columns are numeric
    df_filtered = df[mask]
    df_filtered['sentence_count'] = 1

    # Get the indices of rows to drop
    positional_indices_to_drop = check_repeated_values_with_positional_index(df_filtered['sentence'])

    # Drop the identified rows
    df_filtered = df_filtered.drop(df_filtered.index[positional_indices_to_drop]).reset_index(drop=True)

    df_filtered = df_filtered[df_filtered.iloc[:, 7].notna()]

    df_filtered.iloc[:, 12] = df_filtered['sentence'].astype(str).apply(word_count)

    # Selecting the required columns by their positions (K, M, O, G)
    k_col = df_filtered.iloc[:, 10].astype(float)  # 11th column
    #m_col = df_filtered.iloc[:, 12].astype(float)  # 13th column
    o_col = df_filtered.iloc[:, 14].apply(lambda x: str(x).strip().upper())  # 15th column
    g_col = df_filtered.iloc[:, 6].apply(lambda x: str(x).strip())  # 7th column
    h_col = df_filtered.iloc[:, 7].astype(float)  # 8th column
    #Sen_col = df_filtered.iloc[:, 4].apply(lambda x: str(x).strip())  # 5th column
    #HumSen_col = df_filtered.iloc[:, 8].apply(lambda x: str(x).strip())  # 9th column
    sc_col = df_filtered['sentence_count'].astype(float)

    #df_filtered.iloc[:, 8].fillna(df_filtered['sentence'], inplace=True)
    df_filtered.loc[df_filtered.iloc[:, 7].astype(float) == 1, df_filtered.columns[8]] = df_filtered['sentence']
    df_filtered['JL_WC'] = df_filtered.iloc[:, 8].astype(str).apply(word_count)
    df_filtered['WH_WC'] = df_filtered.iloc[:, 4].astype(str).apply(word_count)
    df_filtered['human_sentence_count'] = df_filtered.iloc[:, 8].apply(lambda x: 1 if pd.notna(x) else 0)
    hsc_col = df_filtered['human_sentence_count'].astype(float)

    # Grouping by columns O and G
    grouped_data = pd.DataFrame({
        'K': k_col,
        #'M': m_col,
        'M': df_filtered['WH_WC'],
        'O': o_col,
        'G': g_col,
        'SF': h_col,
        'SC': sc_col,
        'HSC': hsc_col,
        'JL_WC': df_filtered['JL_WC']
    }).groupby(['O', 'G']).sum()

    # print(k_col, m_col)
    # Calculating the ratio of K over M for each group
    grouped_data['Ratio_K_M'] = grouped_data['K'] / grouped_data['M']

    return grouped_data

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]

def set_column_width(df, worksheet):
    """Set the width of Excel columns based on the maximum width of DataFrame column values."""
    for idx, col in enumerate(df.columns):
        # Get the max length in this column
        max_len = max(df[col].astype(str).apply(len).max(),  # max length in column
                      len(str(col)))  # length of column name/header
        worksheet.set_column(idx, idx, max_len)  # set column width


# Function to check for repeated values
def check_repeated_values_with_positional_index(series):
    count = 1
    to_drop = []

    for i in range(1, len(series)):
        if series.iloc[i] == series.iloc[i - 1]:
            count += 1
        else:
            count = 1

        # If a value is repeated more than two times continuously, note its position
        if count > 2:
            to_drop.append(i)

    return to_drop

def process_folder(excel_folder_path, output_path, vc=''):

    # Integrated results DataFrame
    integrated_results = pd.DataFrame()
    integrated_LD = pd.DataFrame()

    # Iterating through Excel files and processing them
    for file_name in os.listdir(excel_folder_path):
        if file_name.endswith('.xlsx') and 'Stat' not in file_name and 'Agg' not in file_name and 'WhisperCoding' in file_name:
            if ('v' in vc and vc in file_name) or vc == '':

                print(file_name)

                file_path = os.path.join(excel_folder_path, file_name)

                df_LD = pd.read_excel(file_path)

                #print(df_LD)

                # Compute the Levenshtein distance in letter level using index-based referencing
                df_LD['Levenshtein Distance'] = df_LD.apply(
                    lambda row: 0 if pd.isna(row[8]) and not pd.isna(row[7]) else
                    levenshtein_distance(str(row[4]), str(row[8])),
                    axis=1
                )
                df_LD = df_LD.iloc[:, [1, 2, 4, 6, 7, 8, 10, 12, 14, -1]]
                df_LD.columns = ['start_sec', 'end_sec', 'Sentence', 'Language', 'Flag', 'Correct_Words', 'LD_Word', 'WordCount', 'Speaker',
                                'Levenshtein Distance']
                df_LD['Levenshtein Distance'] = df_LD['LD_Word']

                #print(df_LD)

                df_LD = df_LD[df_LD['WordCount'].notna()]
                df_LD = df_LD[df_LD['Language'].notna()]
                df_LD['WordCount'] = df_LD['WordCount'].astype(int)
                df_LD['Speaker'] = df_LD['Speaker'].str.strip().str.upper()

                # Calculating the required statistics
                file_result = calculate_sums_and_ratios(file_path)

                # Adding file name as a new column
                file_result['File Name'] = ('_'.join(file_name.split('_')[2:])).split('.')[0]
                df_LD['File Name'] = ('_'.join(file_name.split('_')[2:])).split('.')[0]

                # Appending to the integrated results
                integrated_results = integrated_results._append(file_result.reset_index(), ignore_index=True)
                integrated_LD = integrated_LD._append(df_LD.reset_index(), ignore_index=True)

    #Update to keep missing sentence from Whisper
    integrated_LD = integrated_LD[pd.notna(integrated_LD.iloc[:, 5])]

    # Get the indices of rows to drop
    positional_indices_to_drop = check_repeated_values_with_positional_index(integrated_LD['Sentence'])

    # Drop the identified rows
    df_cleaned = integrated_LD.drop(integrated_LD.index[positional_indices_to_drop]).reset_index(drop=True)

    # Reset the index of the cleaned dataframe
    df_cleaned.reset_index(drop=True, inplace=True)

    df_cleaned.loc[df_cleaned['Flag'].astype(int) == 1, 'Correct_Words'] = df_cleaned['Sentence']

    df_cleaned['JL_WC'] = df_cleaned['Correct_Words'].apply(word_count)
    df_cleaned['WH_WC'] = df_cleaned['Sentence'].apply(word_count)

    df_cleaned['JL_LexDiv'] = df_cleaned['Correct_Words'].apply(word_count_unq)
    df_cleaned['WH_LexDiv'] = df_cleaned['Sentence'].apply(word_count_unq)

    #df_cleaned['Correct_Words'].fillna(df_cleaned['Sentence'], inplace=True)

    df_cleaned.to_csv('Agg_Sen_LD.csv')

    df_filtered_ADU = df_cleaned[df_cleaned['Speaker'] == 'FEM']

    df_filtered_CHI = df_cleaned[df_cleaned['Speaker'].isin(['CHI', 'KCHI'])]

    df_filtered_KCHI = df_cleaned[df_cleaned['Speaker'] == 'KCHI']

    df_filtered_ADU.to_csv('Agg_Sen_LD_ADU.csv')

    df_filtered_CHI.to_csv('Agg_Sen_LD_CHI.csv')

    df_filtered_KCHI.to_csv('Agg_Sen_LD_KCHI.csv')

    calculate_MLU_df = df_filtered_ADU[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    average_jl_wc = df_non_zero_jl_wc.groupby('File Name')['JL_WC'].mean().reset_index()
    average_wh_wc = df_non_zero_wh_wc.groupby('File Name')['WH_WC'].mean().reset_index()

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_ADU = pd.merge(average_jl_wc, average_wh_wc, on='File Name')

    combined_averages_ADU.to_csv('ADU_MLU.csv')

    calculate_MLU_df = df_filtered_CHI[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    average_jl_wc = df_non_zero_jl_wc.groupby('File Name')['JL_WC'].mean().reset_index()
    average_wh_wc = df_non_zero_wh_wc.groupby('File Name')['WH_WC'].mean().reset_index()

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_CHI = pd.merge(average_jl_wc, average_wh_wc, on='File Name')

    combined_averages_CHI.to_csv('CHI_MLU.csv')


    #Question

    df_filtered_ADU_cleaned = df_filtered_ADU.dropna(subset=['Correct_Words'])

    df_filtered_question = df_filtered_ADU_cleaned[
        df_filtered_ADU_cleaned['Correct_Words'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]

    df_filtered_ADU_cleaned = df_filtered_ADU.dropna(subset=['Sentence'])

    df_filtered_question = df_filtered_ADU_cleaned[
        df_filtered_ADU_cleaned['Sentence'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    average_jl_wc = df_non_zero_jl_wc.groupby('File Name')['JL_WC'].mean().reset_index()
    average_wh_wc = df_non_zero_wh_wc.groupby('File Name')['WH_WC'].mean().reset_index()

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_ADU = pd.merge(average_jl_wc, average_wh_wc, on='File Name')

    combined_averages_ADU.to_csv('ADU_MLU_Q.csv')

    df_filtered_CHI_cleaned = df_filtered_CHI.dropna(subset=['Correct_Words'])

    df_filtered_question = df_filtered_CHI_cleaned[
        df_filtered_CHI_cleaned['Correct_Words'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]

    df_filtered_CHI_cleaned = df_filtered_CHI.dropna(subset=['Sentence'])

    df_filtered_question = df_filtered_CHI_cleaned[
        df_filtered_CHI_cleaned['Sentence'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    average_jl_wc = df_non_zero_jl_wc.groupby('File Name')['JL_WC'].mean().reset_index()
    average_wh_wc = df_non_zero_wh_wc.groupby('File Name')['WH_WC'].mean().reset_index()

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_CHI = pd.merge(average_jl_wc, average_wh_wc, on='File Name')

    combined_averages_CHI.to_csv('CHI_MLU_Q.csv')


    #Non-Question

    df_filtered_ADU_cleaned = df_filtered_ADU.dropna(subset=['Correct_Words'])

    df_filtered_question = df_filtered_ADU_cleaned[
        ~df_filtered_ADU_cleaned['Correct_Words'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]

    df_filtered_ADU_cleaned = df_filtered_ADU.dropna(subset=['Sentence'])

    df_filtered_question = df_filtered_ADU_cleaned[
        ~df_filtered_ADU_cleaned['Sentence'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    average_jl_wc = df_non_zero_jl_wc.groupby('File Name')['JL_WC'].mean().reset_index()
    average_wh_wc = df_non_zero_wh_wc.groupby('File Name')['WH_WC'].mean().reset_index()

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_ADU = pd.merge(average_jl_wc, average_wh_wc, on='File Name')

    combined_averages_ADU.to_csv('ADU_MLU_NQ.csv')

    df_filtered_CHI_cleaned = df_filtered_CHI.dropna(subset=['Correct_Words'])

    df_filtered_question = df_filtered_CHI_cleaned[
        ~df_filtered_CHI_cleaned['Correct_Words'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]

    df_filtered_CHI_cleaned = df_filtered_CHI.dropna(subset=['Sentence'])

    df_filtered_question = df_filtered_CHI_cleaned[
        ~df_filtered_CHI_cleaned['Sentence'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    average_jl_wc = df_non_zero_jl_wc.groupby('File Name')['JL_WC'].mean().reset_index()
    average_wh_wc = df_non_zero_wh_wc.groupby('File Name')['WH_WC'].mean().reset_index()

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_CHI = pd.merge(average_jl_wc, average_wh_wc, on='File Name')

    combined_averages_CHI.to_csv('CHI_MLU_NQ.csv')

    # Question

    df_filtered_ADU_cleaned = df_filtered_ADU.dropna(subset=['Correct_Words'])

    df_filtered_question = df_filtered_ADU_cleaned[
        df_filtered_ADU_cleaned['Correct_Words'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]

    df_filtered_ADU_cleaned = df_filtered_ADU.dropna(subset=['Sentence'])

    df_filtered_question = df_filtered_ADU_cleaned[
        df_filtered_ADU_cleaned['Sentence'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    count_jl_wc = df_non_zero_jl_wc.groupby('File Name').size().reset_index(name='JL_WC_Count')
    count_wh_wc = df_non_zero_wh_wc.groupby('File Name').size().reset_index(name='WH_WC_Count')

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_ADU = pd.merge(count_jl_wc, count_wh_wc, on='File Name')

    combined_averages_ADU.to_csv('ADU_NU_Q.csv')

    df_filtered_CHI_cleaned = df_filtered_CHI.dropna(subset=['Correct_Words'])

    df_filtered_question = df_filtered_CHI_cleaned[
        df_filtered_CHI_cleaned['Correct_Words'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]

    df_filtered_CHI_cleaned = df_filtered_CHI.dropna(subset=['Sentence'])

    df_filtered_question = df_filtered_CHI_cleaned[
        df_filtered_CHI_cleaned['Sentence'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    count_jl_wc = df_non_zero_jl_wc.groupby('File Name').size().reset_index(name='JL_WC_Count')
    count_wh_wc = df_non_zero_wh_wc.groupby('File Name').size().reset_index(name='WH_WC_Count')

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_CHI = pd.merge(count_jl_wc, count_wh_wc, on='File Name')

    combined_averages_CHI.to_csv('CHI_NU_Q.csv')

    # Non-Question

    df_filtered_ADU_cleaned = df_filtered_ADU.dropna(subset=['Correct_Words'])

    df_filtered_question = df_filtered_ADU_cleaned[
        ~df_filtered_ADU_cleaned['Correct_Words'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]

    df_filtered_ADU_cleaned = df_filtered_ADU.dropna(subset=['Sentence'])

    df_filtered_question = df_filtered_ADU_cleaned[
        ~df_filtered_ADU_cleaned['Sentence'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    # Calculating the average for JL_WC and WH_WC separately
    count_jl_wc = df_non_zero_jl_wc.groupby('File Name').size().reset_index(name='JL_WC_Count')
    count_wh_wc = df_non_zero_wh_wc.groupby('File Name').size().reset_index(name='WH_WC_Count')

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_ADU = pd.merge(count_jl_wc, count_wh_wc, on='File Name')

    combined_averages_ADU.to_csv('ADU_NU_NQ.csv')

    df_filtered_CHI_cleaned = df_filtered_CHI.dropna(subset=['Correct_Words'])

    df_filtered_question = df_filtered_CHI_cleaned[
        ~df_filtered_CHI_cleaned['Correct_Words'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_jl_wc = calculate_MLU_df[calculate_MLU_df['JL_WC'] != 0]

    df_filtered_CHI_cleaned = df_filtered_CHI.dropna(subset=['Sentence'])

    df_filtered_question = df_filtered_CHI_cleaned[
        ~df_filtered_CHI_cleaned['Sentence'].str.strip().str.endswith('?')]

    calculate_MLU_df = df_filtered_question[['JL_WC', 'WH_WC', 'File Name']]

    # Filtering and calculating the averages separately for JL_WC and WH_WC
    df_non_zero_wh_wc = calculate_MLU_df[calculate_MLU_df['WH_WC'] != 0]

    # Calculating the average for JL_WC and WH_WC separately
    count_jl_wc = df_non_zero_jl_wc.groupby('File Name').size().reset_index(name='JL_WC_Count')
    count_wh_wc = df_non_zero_wh_wc.groupby('File Name').size().reset_index(name='WH_WC_Count')

    # Integrating the separately calculated averages into a single DataFrame
    combined_averages_CHI = pd.merge(count_jl_wc, count_wh_wc, on='File Name')

    combined_averages_CHI.to_csv('CHI_NU_NQ.csv')


    # Load the CSV file
    integrated_LD = pd.read_csv('Agg_Sen_LD.csv')
    # integrated_LD_save = integrated_LD.copy()

    integrated_LD['Language'] = integrated_LD['Language'].apply(lambda x: x.strip())

    # Calculate normalized Levenshtein Distance
    # integrated_LD['Normalized LD'] = integrated_LD['Levenshtein Distance'] / integrated_LD['WordCount']

    # Group by 'File Name' and 'Speaker (KCHI,FEM, CHI, MAL)' and compute mean and std
    grouped_speaker_LD = integrated_LD.groupby(['File Name', 'Speaker'])['Levenshtein Distance'].agg(
        ['mean', 'std']).reset_index()

    # Group by 'File Name' and 'Human_Language (en = English, es = Spanish)' and compute mean and std
    grouped_language_LD = integrated_LD.groupby(['File Name',
                                                 'Language'])[
        'Levenshtein Distance'].agg(['mean', 'std']).reset_index()

    grouped_detailed_LD = integrated_LD.groupby(['Speaker', 'Language', 'File Name'])[
        'Levenshtein Distance'].agg(['mean', 'std']).reset_index()

    # Renaming columns
    integrated_results.rename(columns={
        'O': 'Speaker',
        'G': 'Language',
        'K': 'Incorrect Words',
        'M': 'Total Words',
        'Ratio_K_M': 'Ratio',
        'SF': 'Sentence Correct',
        'SC': 'Total Whisper Sentences',
        'HSC': 'Total Human Sentences',
        'JL_WC': 'Total Human Words'
    }, inplace=True)

    #integrated_results['Speaker'] = integrated_results['Speaker'].str.strip().str.upper()

    integrated_results['Weighted Ratio'] = integrated_results['Ratio'] * integrated_results['Total Whisper Sentences']

    # Saving the integrated results to a CSV file
    # integrated_results.to_csv('Trans_Stat_Detail.csv', index=False)

    grouped_by_speaker = integrated_results.groupby('Speaker').agg(
        Incorrect_Words_Sum=('Incorrect Words', 'sum'),
        Total_Words_Sum=('Total Words', 'sum'),
        Correct_Sen_Sum=('Sentence Correct', 'sum'),
        Total_Sentences_Sum=('Total Whisper Sentences', 'sum'),
        Total_Human_Sentences_Sum = ('Total Human Sentences', 'sum')
    ).reset_index()
    grouped_by_speaker['Incorrect Words Ratio'] = grouped_by_speaker['Incorrect_Words_Sum'] / grouped_by_speaker['Total_Words_Sum']
    grouped_by_speaker['Accuracy (Words)'] = 1 - grouped_by_speaker['Incorrect Words Ratio']
    grouped_by_speaker['Correct (%) Utterance'] = grouped_by_speaker['Correct_Sen_Sum'] / grouped_by_speaker['Total_Sentences_Sum'] * 100

    #grouped_by_speaker.to_csv('Trans_Stat_Speaker.csv', index=False)

    grouped_by_language = integrated_results.groupby('Language').agg(
        Incorrect_Words_Sum=('Incorrect Words', 'sum'),
        Total_Words_Sum=('Total Words', 'sum'),
        Correct_Sen_Sum=('Sentence Correct', 'sum'),
        Total_Sentences_Sum=('Total Whisper Sentences', 'sum'),
        Total_Human_Sentences_Sum=('Total Human Sentences', 'sum')
    ).reset_index()
    grouped_by_language['Incorrect Words Ratio'] = grouped_by_language['Incorrect_Words_Sum'] / grouped_by_language[
        'Total_Words_Sum']
    grouped_by_language['Accuracy (Words)'] = 1 - grouped_by_language['Incorrect Words Ratio']
    grouped_by_language['Correct (%) Utterance'] = grouped_by_language['Correct_Sen_Sum'] / grouped_by_language[
        'Total_Sentences_Sum'] * 100

    #grouped_by_language.to_csv('Trans_Stat_Lang.csv', index=False)

    # Calculating the sum of columns "C" and "D" for each unique file name
    sums_per_file = integrated_results.groupby('File Name').agg(
        sum_C=pd.NamedAgg(column='Incorrect Words', aggfunc='sum'),
        sum_D=pd.NamedAgg(column='Total Words', aggfunc='sum'),
        sum_SC=pd.NamedAgg(column='Sentence Correct', aggfunc='sum'),
        sum_TS=pd.NamedAgg(column='Total Whisper Sentences', aggfunc='sum'),
    )

    # Calculating the ratio of the sum of column "C" over the sum of column "D" for each unique file name
    sums_per_file['Ratio'] = sums_per_file['sum_C'] / sums_per_file['sum_D']
    sums_per_file['Correct (%)'] = sums_per_file['sum_SC'] / sums_per_file['sum_TS']

    # Calculating the overall ratio of the total sum of column "C" over the total sum of column "D"
    overall_Ratio = sums_per_file['sum_C'].sum() / sums_per_file['sum_D'].sum()
    overall_Correct = sums_per_file['sum_SC'].sum() / sums_per_file['sum_TS'].sum()

    sums_per_file.loc['Overall'] = [sums_per_file['sum_C'].sum(), sums_per_file['sum_D'].sum(),
                                    sums_per_file['sum_SC'].sum(), sums_per_file['sum_TS'].sum(),
                                    overall_Ratio, overall_Correct]

    sums_per_file.rename(columns={
        'sum_C': 'Incorrect Words',
        'sum_D': 'Total Words',
        'sum_SC': 'Correct Sentences',
        'sum_TS': 'Total Whisper Sentences'
    }, inplace=True)

    sums_per_file['Accuracy'] = 1 - sums_per_file['Ratio']

    #sums_per_file.to_csv('Overall_Trans_Stat.csv')
    sums_per_file['Type'] = sums_per_file.index.to_series().apply(
        lambda x: 'Child' if 'Child' in x else ('Adult' if 'Adult' in x else ' '))
    grouped_speaker_LD['Type'] = grouped_speaker_LD['File Name'].apply(
        lambda x: 'Child' if 'Child' in x else ('Adult' if 'Adult' in x else ' '))
    grouped_language_LD['Type'] = grouped_language_LD['File Name'].apply(
        lambda x: 'Child' if 'Child' in x else ('Adult' if 'Adult' in x else ' '))
    grouped_detailed_LD['Type'] = grouped_detailed_LD['File Name'].apply(
        lambda x: 'Child' if 'Child' in x else ('Adult' if 'Adult' in x else ' '))
    integrated_results['Type'] = integrated_results['File Name'].apply(
        lambda x: 'Child' if 'Child' in x else ('Adult' if 'Adult' in x else ' '))

    merged_df = pd.merge(integrated_results, grouped_detailed_LD,
                         on=["Speaker", "Language", "File Name", "Type"],
                         how="outer")

    merged_df_rounded = merged_df.round({column: 4 for column in merged_df.select_dtypes(['float64']).columns})

    # Sort the dataframe by Speaker, Type, and then Language
    sorted_df = merged_df_rounded.sort_values(by=["Speaker", "Type", "Language"])

    custom_order = {
        "CHI": 0,
        "KCHI": 1,
        "FEM": 2
    }

    # Apply the custom ordering and then sort by the other columns
    sorted_df['speaker_order'] = sorted_df['Speaker'].map(lambda x: custom_order.get(x, 3))
    sorted_df_custom = sorted_df.sort_values(by=['speaker_order', 'Type', 'Language']).drop('speaker_order', axis=1)

    sorted_df_filled = sorted_df_custom.fillna(0)

    sorted_df_filled['Length (min)'] = sorted_df_filled['File Name'].str.split('_').str[-1].astype(int)
    sorted_df_filled['Whisper Words # (/min)'] = (sorted_df_filled['Total Words'] /
                                                sorted_df_filled['Length (min)']).round(4)
    sorted_df_filled['Human Words # (/min)'] = (sorted_df_filled['Total Human Words'] /
                                                sorted_df_filled['Length (min)']).round(4)
    sorted_df_filled['Correct (%) Words'] = ((1 - sorted_df_filled['Ratio']) * 100).round(2)
    sorted_df_filled['Proportion Incorrect Words'] = (sorted_df_filled['Incorrect Words'] /
                                                sorted_df_filled['Total Words']).round(4)
    sorted_df_filled['Correct (%) Utterance'] = ((sorted_df_filled['Sentence Correct'] /
                                                  sorted_df_filled['Total Whisper Sentences']) * 100).round(2)
    sorted_df_filled['Average whisper words per utterance'] = (sorted_df_filled['Total Words'] /
                                                  sorted_df_filled['Total Whisper Sentences']).round(2)
    sorted_df_filled['Average human words per utterance'] = (sorted_df_filled['Total Human Words'] /
                                                             sorted_df_filled['Total Whisper Sentences']).round(2)
    LD_E_R_W_W = 'LD EXPRESSED AS A RATIO OF MEAN Whisper WORDS: {} / {}'.format(
        get_excel_col_letter(sorted_df_filled, 'mean'),
        get_excel_col_letter(sorted_df_filled,
                             'Average whisper words per utterance'))
    LD_E_R_H_W = 'LD EXPRESSED AS A RATIO OF MEAN Human WORDS: {} / {}'.format(
        get_excel_col_letter(sorted_df_filled, 'mean'),
        get_excel_col_letter(sorted_df_filled,
                             'Average human words per utterance'))
    sorted_df_filled[LD_E_R_W_W] = (
                sorted_df_filled['mean'] /
                sorted_df_filled['Average whisper words per utterance']).round(4)
    sorted_df_filled[LD_E_R_H_W] = (
            sorted_df_filled['mean'] /
            sorted_df_filled['Average human words per utterance']).round(4)

    LD_R_W_W_W = 'LD RATIO WEIGHTED BY TOTAL Whisper WORDS: {} * {}'.format(
        get_excel_col_letter(sorted_df_filled, LD_E_R_W_W),
        get_excel_col_letter(sorted_df_filled, 'Total Words'))

    LD_R_W_H_W = 'LD RATIO WEIGHTED BY TOTAL Human WORDS: {} * {}'.format(
        get_excel_col_letter(sorted_df_filled, LD_E_R_H_W),
        get_excel_col_letter(sorted_df_filled, 'Total Human Words'))

    sorted_df_filled[LD_R_W_W_W] = (
                sorted_df_filled[LD_E_R_W_W] *
                sorted_df_filled['Total Words']).round(4)

    sorted_df_filled[LD_R_W_H_W] = (
            sorted_df_filled[LD_E_R_H_W] *
            sorted_df_filled['Total Words']).round(4)

    sorted_df_filled.rename(columns={
        'mean': 'Average Word LD in Utterance',
        'std': 'Std Word LD in Utterance',
        'Total Words': 'Total Whisper Words'
    }, inplace=True)

    sorted_df_filled = sorted_df_filled.reset_index()

    # Create a list to hold the dataframes including the blank rows
    dfs_with_blanks = []

    # Iterate through the sorted dataframe
    prev_speaker, prev_type = None, None
    for idx, row in sorted_df_filled.iterrows():
        current_speaker, current_type = row['Speaker'], row['Type']
        #print(idx, current_speaker, current_type)

        # If the group has changed (either speaker or type), insert a blank row
        if idx > 0 and (current_speaker != prev_speaker or current_type != prev_type):
            blank_row = pd.Series({col: '' for col in sorted_df_filled.columns}, name='blank')
            dfs_with_blanks.append(blank_row)
            dfs_with_blanks.append(blank_row)

        dfs_with_blanks.append(row)

        prev_speaker, prev_type = current_speaker, current_type

    # Concatenate all the dataframes (including the blank rows)
    df_with_blanks = pd.concat(dfs_with_blanks, axis=1).T
    df_with_blanks = df_with_blanks.drop('index', axis=1)

    # Create an empty DataFrame to store the results with sums
    df_with_sums = pd.DataFrame()

    #print(df_with_blanks)

    # Iterate over the DataFrame to identify categories distinguished by blank rows
    start_index = 0
    p_idx = 0
    sum_flag = True
    c_count = 1
    for idx, row in df_with_blanks.iterrows():
        # Check for a blank row (using the 'Type' column as a reference)
        if row['Type'] == '':
            if sum_flag:
                # Extract the subset of data for the current category
                category_df = df_with_blanks.iloc[start_index:(p_idx+c_count)]
                #print(category_df)

                # Calculate the sum for the "LD RATIO WEIGHTED BY TOTAL WORDS:" column
                sum_ld_w_ratio = category_df[LD_R_W_W_W].sum()
                sum_ld_h_ratio = category_df[LD_R_W_H_W].sum()

                # Calculate the sum for the "Total Whisper Words" column
                sum_total_whisper = category_df['Total Whisper Words'].sum()

                # Calculate the sum for the "Total Whisper Words" column
                sum_total_sen = category_df['Total Whisper Sentences'].sum()

                # Calculate the sum for the "Total Whisper Words" column
                sum_total_hsen = category_df['Total Human Sentences'].sum()

                # Calculate the sum for the "Weighted Ratio" column
                sum_total_weightedratio = category_df['Weighted Ratio'].sum()

                # Calculate the sum for the "Total Human Words" column
                sum_total_human = category_df['Total Human Words'].sum()

                # Create a new row with the sum values and append it to the category data
                sum_row = pd.DataFrame({
                    LD_R_W_W_W: [sum_ld_w_ratio],
                    LD_R_W_H_W: [sum_ld_h_ratio],
                    'Total Whisper Words': [sum_total_whisper],
                    'Total Whisper Sentences': [sum_total_sen],
                    'Total Human Sentences': [sum_total_hsen],
                    'Weighted Ratio': [sum_total_weightedratio],
                    'Total Human Words': [sum_total_human]
                })
                sum_row['Whisper Stat_Ratio: {} / {}'.format(
                    get_excel_col_letter(category_df, LD_R_W_W_W),
                    get_excel_col_letter(category_df, 'Total Whisper Words'))] = (
                            sum_row[LD_R_W_W_W] /
                            sum_row['Total Whisper Words']).round(4)

                sum_row['Human Stat_Ratio: {} / {}'.format(
                    get_excel_col_letter(category_df, LD_R_W_H_W),
                    get_excel_col_letter(category_df, 'Total Human Words'))] = (
                            sum_row[LD_R_W_H_W] /
                            sum_row['Total Human Words']).round(4)

                sum_row['Ratio of incorrect Utterances Weighted by Total Whisper Utterances: {} / {}'.format(
                    get_excel_col_letter(category_df, 'Weighted Ratio'),
                    get_excel_col_letter(category_df, 'Total Whisper Sentences'))] \
                    = (sum_row['Weighted Ratio'] / sum_row['Total Whisper Sentences']).round(4)

                sum_row['Ratio of incorrect Utterances Weighted by Total Human Utterances: {} / {}'.format(
                    get_excel_col_letter(category_df, 'Weighted Ratio'),
                    get_excel_col_letter(category_df, 'Total Human Sentences'))] \
                    = (sum_row['Weighted Ratio'] / sum_row['Total Human Sentences']).round(4)

                category_df = category_df._append(sum_row, ignore_index=True)

                # Append the category data (with the sum) to the results DataFrame
                df_with_sums = df_with_sums._append(category_df, ignore_index=True)
                #print(category_df)
                blank_row = pd.Series({col: '' for col in sorted_df_filled.columns}, name='blank')
                df_with_sums = df_with_sums._append(blank_row, ignore_index=True)

                # Update the start index for the next category
                start_index = p_idx + c_count + 2
                c_count += 2

                sum_flag = False
            else:
                sum_flag = True
        else:
            p_idx = idx

    # Handle any remaining data after the last blank row
    if start_index < df_with_blanks.shape[0]:
        remaining_df = df_with_blanks.iloc[start_index:]
        df_with_sums = df_with_sums._append(remaining_df, ignore_index=True)

    df_with_sums = df_with_sums.drop('index', axis=1)

    excel_writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

    # DataFrames to write to the Excel file (assuming they exist)
    dfs = {
        'Overall_Trans_Stat': sums_per_file,
        'Trans_Stat_Speaker': grouped_by_speaker,
        'Trans_Stat_Lang': grouped_by_language,
        'LD_Stat_Speaker': grouped_speaker_LD,
        'LD_Stat_Lang': grouped_language_LD,
        'LD_Stat_Detail': grouped_detailed_LD,
        'Trans_Stat_Detail': integrated_results,
        'All_Stat_Detail': df_with_sums
    }

    # Write each DataFrame to a different sheet and adjust column widths
    for sheet_name, df in dfs.items():
        df.to_excel(excel_writer, sheet_name=sheet_name, index=(sheet_name == 'Overall_Trans_Stat'))
        worksheet = excel_writer.sheets[sheet_name]
        set_column_width(df, worksheet)

    # Save the Excel file
    excel_writer._save()

if __name__ == "__main__":

    parser = ArgumentParser(description="Generate Transcription Statistical Results")

    # Get path's from argument
    parser.add_argument("--trans_coding_pth", type=str, default="./", help="The filepath to the Transcrption Coding files")
    parser.add_argument("--output_file", type=str, default="Trans_Stat.xlsx", help="The name of the output file")
    parser.add_argument("--version_control", type=str, default="", help="The version of files want to run")

    # Parse args
    args = parser.parse_args()

    process_folder(Path(args.trans_coding_pth), args.output_file, args.version_control)