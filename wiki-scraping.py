import re
import urllib.parse
import warnings
from os.path import join
from typing import List

import pandas as pd
import wikipediaapi
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from wikipedia import wikipedia
from yaspin import yaspin

import time
from wtpsplit import SaT
from concurrent.futures import ThreadPoolExecutor, as_completed
from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload, NotValidLength, TooManyRequests, TranslationNotFound

warnings.filterwarnings('ignore')

splitter = None
# ANSI Highlighting: https://stackoverflow.com/a/21786287
h_stop = '\x1b[0m'
rh_start = '\x1b[1;30;41m'
gh_start = '\x1b[1;30;42m'
yh_start = '\x1b[1;30;43m'


def cooldown():
    print("Waiting 60 seconds.")
    time.sleep(60)

def chunk_by_words(text, max_length=4000):
    """ Splits a single long sentence into words and groups them back into pseudo-sentences (or chunks) 
    where each chunk is ≤ 4000 characters."""
    
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += " " + word if current_chunk else word
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def chunk_by_sentences(text, lang: str, max_length=4000):
    """ Splits a single long text into sentences and groups them back into pseudo-sentences (or chunks) 
    where each chunk is ≤ 4000 characters."""

    # Initialize sentence splitter in order to mitigate the GLiNER model context window length issues
    splitter = SaT("sat-3l", style_or_domain="ud", language=lang)
    sentences = splitter.split(text)
    
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > max_length:
            word_chunks = chunk_by_words(sentence)
            chunks.extend(word_chunks)
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def parallel_deeptranslate(content, src_lang: str, dst_lang: str, show_progress=True, max_workers: int = 4):
    def translate_text(index, row, src_lang, dst_lang):
        attempts = 0
        success = False
        
        while not success:
            if not row:
                text = ''
                break
            
            attempts += 1
            try:
                text = GoogleTranslator(source=src_lang, target=dst_lang).translate(str(row))
                success = True
                
            except NotValidPayload as e:
                # print(f"Invalid input:")
                text = deeptranslate([str(row)], src_lang, dst_lang, False)
                break

            except TranslationNotFound as e:
                # print(f"Could not find a translation:")
                text = ''
                break

            except NotValidLength as e:
                # print(f"Invalid input length:")
                # If there are no words in the text, it's probably a nonsense text
                if not ' ' in row:
                    text = ''
                    break

                sentences = chunk_by_sentences(row, src_lang)

                # Translate the sentences separately
                sentences = deeptranslate(sentences, src_lang, dst_lang, False)
                # Merge the translated sentences into a single sample
                text = ' '.join(sentences)
                break

            except TooManyRequests as e:
                print(f"Rate limit exceeded:")
                cooldown()
                success = False

            except Exception as err:
                if attempts >= 10:
                    print(f"Error: {str(err)}")
                    # print(f"Text: {str(row)}")
                    text = row
                    break
                else:
                    success = False
                    
        return index, text

    result = [None] * len(content)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(translate_text, index, row, src_lang, dst_lang): index for index, row in enumerate(content)}
        iterator = tqdm(as_completed(future_to_index), total=len(content)) if show_progress else as_completed(future_to_index)
        for future in iterator:
            index = future_to_index[future]
            try:
                _, translated_text = future.result()
                result[index] = translated_text
            except Exception as e:
                print(f"Translation failed for row: {content[index]} with error: {e}")
                result[index] = content[index]  # Append the original text if translation fails
    return result


def deeptranslate(content, src_lang: str, dst_lang: str, show_progress=True):
    splitter = None
    
    # Show progress or not
    iterator = tqdm(content) if show_progress else content

    # Iterate through the data
    result = []
    for row in iterator:
        if not row:
            result.append('')
            continue
            
        attempts = 0
        success = False
        while not success:
            attempts += 1
            try:
                text = GoogleTranslator(source=src_lang, target=dst_lang).translate(row)
                result.append(text)
                break
                
            except NotValidPayload as e:
                # print(f"Invalid input:")
                result.append(deeptranslate([str(row)], src_lang, dst_lang, False))
                break
                
            except TranslationNotFound as e:
                # print(f"Could not find a translation:")
                result.append('')
                break
                
            except NotValidLength as e:
                # print(f"Invalid input length:")
                # If there are no words in the text, it's probably a nonsense text
                if not ' ' in row:
                    result.append('')
                    break

                sentences = chunk_by_sentences(row, src_lang)

                # Translate the sentences separately
                sentences = deeptranslate(sentences, src_lang, dst_lang, False)
                # Merge the translated sentences into a single sample
                text = ' '.join(sentences)
                result.append(text)
                break
                
            except TooManyRequests as e:
                print(f"Rate limit exceeded:")
                cooldown()
                success = False
            
            except Exception as err:
                if attempts >= 10:
                    print(f"Error: {str(err)}")
                    # print(f"Text: {str(row)}")
                    result.append(row)
                    break
                else:
                    success = False

    return result

class WikipediaExtractor:
    def __init__(self):
        self.splitter = SaT("sat-3l", style_or_domain="ud", language='en')

        self.agent = 0
        self.wiki_api = wikipediaapi.Wikipedia(user_agent=f'veraAI (user{self.agent}@kinit.sk)',
                                               language='en',
                                               extract_format=wikipediaapi.ExtractFormat.WIKI)

        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # multi-language model

    def switch_user_agent(self):
        self.agent += 1
        self.wiki_api = wikipediaapi.Wikipedia(
            user_agent=f'veraAI (user{self.agent}@kinit.sk)',
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        print(f"Switched to user agent: veraAI (user{self.agent}@kinit.sk)")

    def match_entity_with_url(self, url: str, entity: str, threshold_high: float = 0.7,
                              threshold_low: float = 0.2) -> bool:
        """
        Checks if a given entity matches a given URL based on exact match and cosine similarity criteria.

        Parameters:
        url (str): The URL to check.
        entity (str): The entity text to match against the URL.
        threshold_high (float): Cosine similarity threshold for non-exact matches.
        threshold_low (float): Cosine similarity threshold for exact matches.

        Returns:
        bool: True if the entity matches the URL based on defined criteria, otherwise False.
        """
        # Decode URL and replace underscores for better matching
        url_decoded = urllib.parse.unquote(url).replace('_', ' ')

        # Remove Wikipedia prefix to isolate the page title
        url_cleaned = pd.Series(url_decoded).replace(r'https:\/\/\S+\.wikipedia\.org\/wiki\/', '', regex=True)[0]

        # Check for an exact match: entity text as a substring within the cleaned URL
        exact_match = entity in url_cleaned

        # Calculate cosine similarity between entity and URL text embeddings
        embedding_entity = self.model.encode(entity, convert_to_tensor=False)
        embedding_url = self.model.encode(url_cleaned, convert_to_tensor=False)
        cosine_score = util.cos_sim(embedding_entity, embedding_url).item()

        # Return True if it meets either the high cosine similarity threshold (for non-exact matches)
        # or a moderate threshold (for exact matches)
        return (cosine_score >= threshold_high and not exact_match) or (cosine_score >= threshold_low and exact_match)

    def additional_filtering(self, url: str, entity: str, threshold_high: float = 0.85,
                          threshold_mid: float = 0.7, threshold_low: float = 0.2) -> bool:
        """
        Checks if a given entity matches a given URL based on word match, cosine similarity,
        and exact match criteria.

        Parameters:
        url (str): The URL to check.
        entity (str): The entity text to match against the URL.
        threshold_high (float): Used when the number of words between the URL and entity matches. Set higher than 0.7 to account for observed content mismatches.
        threshold_mid (float): Applied when the number of words differs between the URL and entity, and there are no exact matches. Set at 0.7 based on observations of content alignment in such cases.
        threshold_low (float): Used when exact matches fail to resolve mismatches between the entity and wiki article content.

        Returns:
        bool: True if the entity matches the URL based on defined criteria, otherwise False.
        """
        # Decode URL and replace underscores for better matching
        url_decoded = urllib.parse.unquote(url).replace('_', ' ')

        # Remove Wikipedia prefix to isolate the page title
        url_cleaned = pd.Series(url_decoded).replace(r'https:\/\/\S+\.wikipedia\.org\/wiki\/', '', regex=True)[0]

        # Check for word-by-word match (number of words in URL and entity must be the same)
        word_match = len(url_cleaned.split()) == len(entity.split())

        # Check for an exact match: entity text as a substring within the cleaned URL
        exact_match = entity in url_cleaned

        # Calculate cosine similarity between entity and URL text embeddings
        embedding_entity = self.model.encode(entity, convert_to_tensor=False)
        embedding_url = self.model.encode(url_cleaned, convert_to_tensor=False)
        cosine_score = util.cos_sim(embedding_entity, embedding_url).item()

        # Apply logic based on thresholds:
        # - For exact matches, use the lowest threshold.
        # - For word matches, use the high threshold.
        # - For general matches (non-word, non-exact), use the mid-threshold.
        if exact_match and cosine_score < threshold_low:
            return False
        elif word_match and cosine_score < threshold_high:
            return False
        elif not exact_match and cosine_score < threshold_mid:
            return False
        else:
            return True


    @yaspin(text="Searching relevant wikipedia articles:")
    def search_wikipedia(self, entity: str, lang: str) -> List[wikipedia.WikipediaPage]:
        """
        Searches Wikipedia for pages related to an entity in a specified language and
        filters pages where the entity matches the page URL.

        Parameters:
        entity (str): The entity text to search for.
        lang (str): Language code for Wikipedia (e.g., 'en' for English).

        Returns:
        List[wikipedia.WikipediaPage]: List of WikipediaPage objects that match the entity.
        """

        # Set the language for Wikipedia
        wikipedia.set_lang(lang)

        matched_pages = []
        try:
            # Check if entity is in uppercase, suggesting an abbreviation
            if entity.isupper():
                try:
                    # Fetch the page directly without auto-suggestions
                    page = wikipedia.page(entity, auto_suggest=False)
                    matched_pages.append(page)
                except wikipedia.PageError:
                    # Skip if page does not exist
                    print(f"Page error: {entity} not found")
                except Exception as e:
                    print(f"Page error: {e}")
            else:
                success = False
                while not success:
                    try:
                        # Standard search for lowercase or mixed-case entity
                        search_results = wikipedia.search(query=entity, results=5)
                        success = True
                    except wikipedia.WikipediaException:
                        cooldown()

                for result in search_results:
                    # Try to load each page in search results
                    try:
                        page = wikipedia.page(result)
                        # Add pages that match the entity filter
                        if self.match_entity_with_url(page.url, entity):
                            matched_pages.append(page)
                    except wikipedia.PageError:
                        # Skip if page does not exist
                        print(f"Page error: {result} not found")

        except wikipedia.DisambiguationError as e:
            # Handle disambiguation by choosing the first option if abbreviation
            for option in e.options:
                try:
                    # Treat as abbreviation, take the first valid option
                    if entity.isupper():
                        page = wikipedia.page(option)
                        matched_pages.append(page)
                        break

                except wikipedia.DisambiguationError as e:
                    # Continue to next option if this page does not exist
                    print(f"Disambiguation error: {option} may refer to: {e.options}")
                except wikipedia.PageError:
                    # Continue to next option if this page does not exist
                    print(f"Page error: {option} not found")

        return matched_pages

    def apply_exclusion_rules(self, text: str, lang: str) -> tuple:
        # In case of empty text
        if not text:
            return [], []

        try:
            # Translate to English for splitting and cleanup
            target_lang = 'zh-CN' if lang == 'zh' else lang
            text_en = deeptranslate([text], target_lang, 'en', False)[0]

            # Remove in-line citations and references section
            text_en = re.sub(r'\[.*?]', '', text_en)

            # In case of empty translation
            if not text_en:
                return [], []

            # Ignore the page's References
            if 'References' in text_en:
                return [], []

            # Split into sentences and translate back
            sentences_en = self.splitter.split(text_en)
            # Split by lines and filter out empty ones, then join back
            sentences_en = [s for s in sentences_en if s and s.strip()]
            # Translate split sentences back to the original language
            if sentences_en:
                sentences = parallel_deeptranslate(sentences_en, 'en', target_lang, False)
                return sentences, sentences_en
            else:
                return [], []

        except Exception as e:
            print(f"Error in apply_exclusion_rules: {e}")
            return [], []

    def extract_page(self, entity: str, lang: str) -> pd.DataFrame:
        # Find the best match of the entity on the wikipedia
        self.wiki_api.language = lang
        relevant_pages = self.search_wikipedia(entity, lang)

        # Process all the found pages
        urls = []
        extracted_data = []
        for page in tqdm(relevant_pages, desc='Processing scrapped articles:'):
            # Do not process any duplicate urls
            if page.url in urls:
                continue
            else:
                urls.append(page.url)

            # Extract the summary of each article
            sentences, sentences_en = self.apply_exclusion_rules(text=page.summary, lang=lang)

            # Format the results
            if sentences:
                # Use DataFrame format. Label samples as non-check-worthy (class 0).
                samples = pd.DataFrame({'lang': lang,
                                        'topic': None,
                                        'style': 'structured',
                                        'label': 0,
                                        'text': sentences,
                                        'text_en': sentences_en,
                                        'entitities': entity,
                                        'url': page.url})

                # Add the extracted samples to the results
                # Only extract URLs that really match with the named entities
                if extractor.additional_filtering(page.url, entity):
                    extracted_data.append(samples)

        # Merge the results
        if extracted_data:
            extracted_data = pd.concat(extracted_data)

            # Drop potential duplicate sentences
            extracted_data = extracted_data.drop_duplicates(subset='text', keep='first')
            return extracted_data
        else:
            print(f"Page for entity '{entity}' does not exist or has no content.")
            return pd.DataFrame()

    def extract_wiki(self, entity: str, lang: str) -> pd.DataFrame:
        # Extract all the sentences from the entity-specific wiki page
        try:
            return self.extract_page(entity, lang)
        except:
            # In case of rate limit exceeded, switch the user
            self.switch_user_agent()
            return self.extract_page(entity, lang)


import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Script to get a language code from the command line.")

# Add the argument for the language code
parser.add_argument(
    "--lang", "-l",
    type=str,
    required=True,
    help="Language code (e.g., 'en' for English, 'fr' for French, etc.)"
)

# Parse the arguments
args = parser.parse_args()

# Access the language code
lang = args.lang

extractor = WikipediaExtractor()
multicw_path = join('Final dataset')
wiki_path = join(multicw_path, lang, 'wiki.csv')

# Load histogram
histogram = pd.read_csv(join(multicw_path, lang, 'histogram.csv'))
histogram = histogram.sort_values(by=['frequency'], ascending=False)

# In case of not enough entities in the histogram, supply them from translated English histogram
if histogram.shape[0] < 1000:
    print('Extending the histogram with English named entities:')
    histogram_en = pd.read_csv(join(multicw_path, 'en', 'histogram.csv'))[:500]
    histogram_en = histogram_en.sort_values(by=['frequency'], ascending=False)
    histogram_en['entity'] = parallel_deeptranslate(histogram_en['entity'], 'en', lang)
    histogram = pd.concat([histogram, histogram_en])
    
print(f'{yh_start}Extracting Wikipedia samples using found named entities: {lang}{h_stop}')
extracted_data = []
existing_entities = []
for index, row in tqdm(histogram.iterrows(), total=histogram.shape[0]):
    entity = row['entity']

    if not entity:
        continue

    # Ignore the entities that were already collected
    if entity in existing_entities:
        continue
    else:
        existing_entities.append(entity)

    # Extract the sentences for each entity.
    extracted_data.append(extractor.extract_wiki(entity, lang))

    # Save the results continuously
    result = pd.concat(extracted_data)
    if result.shape[0] <= 3000:
        result.to_csv(wiki_path, index=False)
    else:
        break
