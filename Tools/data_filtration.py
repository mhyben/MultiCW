import pandas as pd
import re

def filter_dataframe(df):
    # Remove rows where 'text' column is empty or contains NaN values
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip().astype(bool)]

    # Remove rows where 'text' column contains more than 5000 characters
    df = df[df['text'].str.len() <= 5000]

    # Detect incorrectly parsed CSV files
    def is_incorrectly_parsed(text):
        # Check for repeating patterns of indexes, language codes, and texts separated by a repeating symbol
        pattern = re.compile(r'(\d+,\w{2},.*?)(;|\||\t)')
        return bool(pattern.search(text))

    df = df[~df['text'].apply(is_incorrectly_parsed)]

    return df

# Example usage
data = {
    'text': [
        'This is a valid text.',
        '',
        'Another valid text with less than 5000 characters.',
        'a' * 5001,  # Invalid: more than 5000 characters
        '123456',  # Invalid: no alphabet characters
        '1,en,This is incorrectly parsed;2,fr,Ceci est incorrectement analysé',  # Invalid: incorrectly parsed
    ]
}

if __name__ == '__main__':
    df = pd.DataFrame(data)
    cleaned_df = filter_dataframe(df)
    print(cleaned_df)