import re
import traceback
import urllib.parse
import warnings
from os.path import join
from typing import List

import pandas as pd
import wikipediaapi
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
from wikipedia import wikipedia
from yaspin import yaspin

from topic import extract_topics
from translation import init_splitter, cooldown, deeptranslate, parallel_deeptranslate

warnings.filterwarnings('ignore')

splitter = None
# ANSI Highlighting: https://stackoverflow.com/a/21786287
h_stop = '\x1b[0m'
h_red = '\x1b[1;30;41m'
h_green = '\x1b[1;30;42m'
h_yellow = '\x1b[1;30;43m'


class WikipediaExtractor:
    def __init__(self, verbose=True):
        self.verbose = verbose
        print('WikiExtractor: Init splitter')
        self.splitter = init_splitter(lang='en')

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
        if self.verbose:
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
                    if self.verbose:
                        print(f"Page error: {entity} not found")
                except Exception as e:
                    if self.verbose:
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
                    if self.verbose:
                        print(f"Disambiguation error: {option} may refer to: {e.options}")
                except wikipedia.PageError:
                    # Continue to next option if this page does not exist
                    if self.verbose:
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
            if self.verbose:
                print(f"Error in apply_exclusion_rules: {e}")
                traceback.print_exc()
            return [], []

    def extract_page(self, entity: str, lang: str) -> pd.DataFrame:
        # Find the best match of the entity on the wikipedia
        self.wiki_api.language = lang
        relevant_pages = self.search_wikipedia(entity, lang)

        # Process all the found pages
        urls = []
        extracted_data = []
        iterator = tqdm(relevant_pages, desc='Processing scrapped articles:') if self.verbose else relevant_pages
        for page in iterator:
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
                                        'entities': entity,
                                        'url': page.url})

                # Add the extracted samples to the results
                # Only extract URLs that really match with the named entities
                if self.additional_filtering(page.url, entity):
                    extracted_data.append(samples)

        # Merge the results
        if extracted_data:
            extracted_data = pd.concat(extracted_data)

            # Drop potential duplicate sentences
            extracted_data = extracted_data.drop_duplicates(subset='text', keep='first')
            return extracted_data
        else:
            if self.verbose:
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

# ====================== Uncomment this part for use in terminal =======================================
# This section can be used for parallel and verbose wikipedia samples scraping. Usage:
# ```
# cd MultiCW
# python Tools/wiki_scraping.py --lang='<LANGUAGE CODE>'
# ```
# *Replace <LANGUAGE CODE> with your specific language code.*

if __name__ == '__main__':
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
    multicw_path = join('Final-dataset')
    wiki_path = join(multicw_path, lang, 'wiki.csv')
    
    # Load language specific named entity histogram
    noisy_hist = pd.read_csv(join(multicw_path, lang, 'noisy_histogram.csv'))
    struc_hist = pd.read_csv(join(multicw_path, lang, 'struc_histogram.csv'))
    histogram = pd.concat([noisy_hist, struc_hist]).drop_duplicates(subset='entity', keep='first')
    histogram = histogram.sort_values(by=['frequency'], ascending=False)
    
    # In case of not enough entities in the histogram, supply them from translated English histogram
    if histogram.shape[0] < 1000:
        print('Extending the histogram with English named entities:')
        histogram_en = pd.read_csv(join(multicw_path, 'en', 'struc_histogram.csv'))[:1000]
        histogram_en = histogram_en.sort_values(by=['frequency'], ascending=False)
        histogram_en['entity'] = parallel_deeptranslate(histogram_en['entity'], 'en', lang)
        histogram = pd.concat([histogram, histogram_en])
        
    print(f'{h_yellow}Extracting Wikipedia samples using found named entities: {lang}{h_stop}')
    extracted_data = []
    existing_entities = []
    for index, row in tqdm(histogram.iterrows(), total=histogram.shape[0]):
        entity = row['entity']
    
        if not entity:
            continue
    
        # Ignore the entities that were already collected
        # if entity in existing_entities:
        #     continue
        # else:
        #     existing_entities.append(entity)
    
        # Extract the sentences for each entity.
        wiki = extractor.extract_wiki(entity, lang)

        if not wiki.empty and 'text' in wiki.columns:
            # Translate the wiki sample also to English
            if lang != 'en':
                print(f'Extract Wikipedia sample translation to English:')
                wiki['text_en'] = parallel_deeptranslate(wiki['text'], lang, 'en')
            else:
                wiki['text_en'] = wiki['text']

            # Extract the topic of the wiki sentence
            print(f'Extracting Wikipedia samples topics:')
            wiki['topic'] = extract_topics(wiki['text_en'])

            wiki['origin'] = 'wikipedia'
            wiki['style'] = 'struc'

            extracted_data.append(wiki)
        else:
            continue

        # Save the results continuously
        result = pd.concat(extracted_data)
        if result.shape[0] > 2000:
            break
        else:
            result.to_csv(wiki_path, index=False)
