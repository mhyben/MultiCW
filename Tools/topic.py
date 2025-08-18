# Topic extraction
import ollama
import pandas as pd
from pandas import Series
from tqdm.auto import tqdm


def extract_topics(claims: Series, model='qwen2.5:7b', show_progress=True):
    prompt = "Classify the following text into one of the categories: [health, politics, environment, science, sport, entertainment]. Respond in English and provide strictly one category from the list, without any additional commentary. If the text does not match any category, respond with 'unknown."
    categories = ['health', 'politics', 'environment', 'science', 'sport', 'entertainment', 'history']

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
                        {"role": "user", "content": row}
                    ],
                    options={
                        "temperature": 0.1
                    }
                )
                result = response['message']['content'].strip().lower()

                # Store any category found in the result
                detected = 'unknown'
                for category in categories:
                    if category in result:
                        detected = category
                        break
                results.append(detected)
                break

            # In case of error just try again
            except Exception as err:
                print(f"Error: {str(err)}")

    # If no category was found after all the attempts, consider it 'unknown'
    return results

if __name__=='__main__':
    # Topic extraction
    claim = pd.Series(['Dr. Wingsley just finished the operation', 'Dr. Wingsley just finished the operation'], copy=False)
    print(extract_topics(claim))