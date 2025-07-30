from os.path import join

import ollama
import pandas as pd
from deep_translator import GoogleTranslator
from pandas import Series
from tqdm.auto import tqdm
from langdetect import detect

# ANSI Highlighting: https://stackoverflow.com/a/21786287
h_red = '\x1b[1;30;41m'
h_green = '\x1b[1;30;42m'
h_yellow = '\x1b[1;30;43m'
h_stop = '\x1b[0m'


# Convert to structured style
def to_structured(claims: Series, model='qwen2.5:7b', show_progress=True):
    prompt = ("Summarize the following text into a single, grammatically correct English sentence in a structured news-writing style, no longer than 300 characters. "
              "The sentence will be in English, but the named entities will be in original language. "
              "Replace any URLs with '[HTTP_LINK]'. "
              "Expand hashtags, abbreviations, and slang. "
              "Do not include empty lines or commentary.")

    # Show progress or not
    iterator = tqdm(claims) if show_progress else claims

    results = []
    for row in iterator:
        # Extract accumulate topics from each claim
        attempts = 0
        while attempts < 2:
            attempts += 1

            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"'{row}'"}
                    ],
                    options={
                        "temperature": 0.1
                    }
                )
                output = response['message']['content'].strip()
                # Translate the model response if necessary
                src_lang = detect(text=output)
                if src_lang != 'en':
                    output = GoogleTranslator(source=src_lang, target='en').translate(output)

                # print(f'{h_yellow}Sentence: {row}{h_stop}')
                print(f'{h_green}Response:{h_stop} {output}')
                results.append(output)
                break

            # In case of error just try again
            except Exception as err:
                print(f"Error: {str(err)}")

    # If no category was found after all the attempts, consider it 'unknown'
    return results

if __name__=='__main__':
    # Load CSVs
    noisy = pd.read_csv(join('..', 'Final-dataset', 'noisy_part.csv'))
    struc = pd.read_csv(join('..', 'Final-dataset', 'struc_part.csv'))

    # Get 20 rows from noisy that contain a hashtag
    noisy = noisy[noisy['text'].str.contains('#', na=False)].head(20)
    # Get 20 rows from structured that contain more than one dot
    struc = struc[struc['text'].str.count(r'\. ') > 1].head(20)

    # Convert to structured style
    print('Normalizing noisy sentences:')
    to_structured(noisy['text'])

    print('Normalizing struc sentences:')
    to_structured(struc['text'])
