import Levenshtein
from gliner import GLiNER
from collections import defaultdict

from tqdm.auto import tqdm
from pandas import Series, DataFrame
from translation import chunk_by_sentences, init_splitter

gliner_categories = [
    'name',
    'nationality',
    'religious or political group',
    'building',
    'location',
    'company',
    'agency',
    'institution',
    'country',
    'city',
    'state',
    'event']


def levenshtein_dist(claim1, claim2, threshold=0.2):
    """
    Check if two claims are similar based on Levenshtein distance ratio.
    """
    distance = Levenshtein.distance(claim1, claim2)
    max_length = max(len(claim1), len(claim2))
    return distance / max_length <= threshold

# Load the model
def init_gliner():
    print('\nLoading Named Entity Recognition model (GLiNER):')
    gliner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to('cuda')

    return gliner


# Use the model
def gliner_entities(claims: Series, lang: str, model=None, show_progress=True) -> tuple:
    # Initialize model if it wasn't provided
    if not model:
        model = init_gliner()

    # Initialize sentence splitter model for the given language
    print('Entities: Init splitter')
    splitter = init_splitter(lang)

    # Initialize the 'entities' column with empty lists
    entity_lists = []
    text_lang_pairs = defaultdict(int)

    # Show progress or not
    iterator = tqdm(claims) if show_progress else claims

    # If empty list of claims
    if claims.shape[0] == 0:
        return entity_lists, None

    # Extract accumulate entities from each claim
    for text in iterator:
        extracted_entities = []

        # Chunk the text as Gliner is only capable of processing up to 384 characters
        if len(text) > 384:
            # print('Text too long. Chunking:')
            text = chunk_by_sentences(text, lang, splitter, 384)
        
        try:
            entities = model.predict_entities(text, gliner_categories)
            entities = [entity['text'] for entity in entities]
            extracted_entities.extend(entities)

        except:
            extracted_entities.extend([])

        # Save the named entities relevant to this row
        entity_lists.append(extracted_entities)

        # Extract the list of unique named entities and their frequency
        for entity in extracted_entities:
            is_unique = True
            for pair in text_lang_pairs:
                if levenshtein_dist(entity, pair[0]) and lang == pair[1]:
                    text_lang_pairs[pair] += 1
                    is_unique = False
                    break
            if is_unique:
                text_lang_pairs[(entity, lang)] += 1

    # Create histogram DataFrame
    entity_lang_freq = [(entity[0], entity[1], freq) for entity, freq in text_lang_pairs.items()]
    hist = DataFrame(entity_lang_freq, columns=['entity', 'language', 'frequency'])
    hist = hist.sort_values('frequency', ascending=False)

    return entity_lists, hist


if __name__=='__main__':
    # Load the named entity extraction model
    df = DataFrame({'text': ['President Joe Biden returned to Washington.']})
    if not 'gliner' in locals():
        gliner = init_gliner()
    entities, _ = gliner_entities(df['text'], 'en', gliner)
    print(entities)
    print('Done')

