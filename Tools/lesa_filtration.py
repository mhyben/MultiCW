import glob
import requests
import pandas as pd

from tqdm import tqdm
from os.path import join
from pandas import DataFrame

from ollama import chat
from ollama import ChatResponse

from Tools.preprocess import h_green

# Enable tqdm for pandas
tqdm.pandas()

# ANSI Highlighting: https://stackoverflow.com/a/21786287
h_red = '\x1b[1;30;41m'
h_green = '\x1b[1;30;42m'
h_yellow = '\x1b[1;30;43m'
h_stop = '\x1b[0m'

# Point to the Ollama server
ollama_url = "http://localhost:11434/api/generate"
model='qwen2.5:7b'

def is_claim_with_context(sample: str):
    """
    Classifies if a claim is relevant in terms of being a claims or has enough context.
    """

    messages = [
        {"role": "user", "content": f"Your task is to respond 'Yes' only if the sample contains a verifiable claim with nontrivial content, "
                                    f"including those with URLs, references to external content or minor typographical errors. "
                                    f"Respond with 'No' otherwise. Do not provide any explanations."
                                    f"Sample: {sample}"}
    ]

    response: ChatResponse = chat(
        model=model,
        messages=messages,
    )

    response_content = response.message.content.strip().lower()
    if 'yes' in response_content:
        return True
    else:
        return False

def test():
    cases = [
        'And to what end ?',
        'come to england',
        'I like it',
        'I fully agree',
        'Additionally , THIS IS NOT my soap box .',
        'Depends what you mean by recent . ',
        'Both theories have been proven false before .',
        'Myron Walker , husband of Utah governor Olene Walker uses the term " first lad , " dropping the " y " from lady .',
        'RT @PirateAtLaw: No no no. Corona beer is the cure not the disease https://t.co/fnba2fr2m2',
        '@FLOTUS Melania, do you approve of ingesting bleach and shining a bright light in the rectal area as a quick cure for #COVID19 ? #BeBest'
        'Hi again ! ',
        '86 . 3 . 136 . 130   ( talk   ·' ,
        'I have completed a draft ',
        '@lula_reh If/when I receive that 💩 I will douse it in Clorox to cure it of #COVID19. #COVIDIOTS'
    ]
    # Test cases
    for case in cases:
        worthy = is_claim_with_context(case)
        print(f'{h_green if worthy else h_red}[{worthy}]{h_stop}: {case}')

if __name__ == '__main__':
    test()

