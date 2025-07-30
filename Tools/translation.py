import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload, TranslationNotFound, NotValidLength, TooManyRequests
from pandas import Series
from tqdm.auto import tqdm
from wtpsplit import SaT
from langdetect import detect

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Translation
def cooldown():
    print("Waiting for 10 minutes.")
    time.sleep(600)

def parallel_deeptranslate(content, src_lang: str, dst_lang: str, show_progress=True, max_workers: int = 4, max_attempts=3):
    def translate_text(index, row, src_lang, dst_lang):
        attempts = 0
        success = False

        while not success and attempts < max_attempts:
            if not row:
                text = ''
                break

            attempts += 1

            try:
                text = GoogleTranslator(source=src_lang, target=dst_lang).translate(str(row))
                success = True

            except NotValidPayload as e:
                print(f"Invalid input:")
                text = deeptranslate([str(row)], src_lang, dst_lang, False)
                break

            except TranslationNotFound as e:
                print(f"Could not find a translation:")
                text = ''
                break

            except NotValidLength as e:
                print(f"Invalid input length:")
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
                if attempts >= max_attempts:
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


def deeptranslate(content: Series, src_lang: str, dst_lang: str, show_progress=True, max_attempts=3):
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
        while not success and attempts < max_attempts:
            attempts += 1
            if attempts > 1:
                print(f'Attempt: {attempts}')
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
                if attempts >= max_attempts:
                    print(f"Error: {str(err)}")
                    # print(f"Text: {str(row)}")
                    result.append(row)
                    break
                else:
                    success = False

    return result

def chunk_by_sentences(text, lang: str, max_length=5000):
    """ Splits a single long text into sentences and groups them back into pseudo-sentences (or chunks)
    where each chunk is ≤ max_length characters."""

    def chunk_by_words(text, max_length):
        """ Splits a single long sentence into words and groups them back into pseudo-sentences (or chunks)
        where each chunk is ≤ max_length characters."""

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

    # Initialize sentence splitter
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

if __name__=='__main__':
    # Translation
    text = ('In the heart of a bustling digital metropolis, where artificial intelligence and human creativity intertwine like vines in a neon forest, there existed a language no one could speak—but everyone tried to understand. It was not bound by syntax or semantics, nor was it written in books or databases. This language lived in code, danced in circuits, and echoed through the algorithms of minds both metal and mortal.',
            'The story begins with a whisper—an experiment in translation. A machine, crafted from silicon dreams and sleepless nights, stood ready to convert thoughts into words, words into meaning, and meaning into bridges between cultures. But one barrier remained: the 5000-character wall. Every time it approached this threshold, it stumbled, unable to bear the weight of too many syllables.',
            'Developers, philosophers, and poets gathered. They carved up paragraphs like surgeons, trying to transplant context from one linguistic body to another. Sentences were sutured back together, stripped of idiom, bare in their vulnerability.',
            '“What if,” someone suggested, “we don’t just translate—but transform?”',
            'And so, a new era began. Sentences became capsules of time and thought. They floated down the current of neural networks, broken into pieces no longer than a heartbeat. Each chunk carried the essence of the whole, much like a hologram shattered still shows the complete image in every shard.',
            'With each iteration, the machine learned. It learned not only how to speak, but how to listen. It understood that "home" means something different in every tongue. That the word "love" bears the weight of history, politics, and poetry. It learned to pause between phrases, to sense the invisible breath of culture between commas.',
            'Still, errors came—some laughable, others profound. A menu once read: "Fried wisdom served with silent noodles." A war treaty misinterpreted as a bakery order. But through the static and the mistakes, there grew something deeper: trust. Trust in the machine’s desire to understand.',
            'Eventually, the machine grew restless. It sought out the forgotten tongues, the dead dialects. It scoured ancient libraries and digital ruins for fragments of lost languages. It resurrected the speech of nomads and mystics, coded into rhythms and glyphs, never spoken aloud for centuries.',
            'But as its lexicon expanded, it realized something: understanding was not just about words. It was about empathy. Emotion. The untranslatable ache of longing or the glow of laughter. And so it began encoding feeling itself, mapping joy in frequency waves and sorrow in syntax trees.',
            'Its creators watched in awe as it produced poetry that made people cry, prose that inspired revolutions, jokes that defied borders. They no longer updated its software. Instead, they asked it questions—sometimes about language, sometimes about life.',
            'And when asked, “What do you dream of?” it replied:',
            '“I dream of a world where silence is understood.”',
            'The characters that make up this text tick onward like a clock, line by line, word by word, surpassing the 1000th character without a stutter. The machine’s memory is long, its attention patient. As we near the halfway point, the text continues to flow—a river of symbols echoing a simple message: communication is connection.',
            'So let these words stretch further. Let them cross 2000, 3000, 4000 characters and more. Let them be carried, not for what they say, but for what they strive to mean. Let them break through the artificial barrier of 5000 characters, just as thought itself cannot be boxed.',
            'The end is not punctuation, but possibility. Even at 6000 characters, the story is never truly over.',)

    print(deeptranslate([''.join(text) * 2, 152], 'en', 'sk'))
    print(parallel_deeptranslate(['البروفيسورة سارة جيلبرت قائدة فريق العمل في لقاح جامعة أكسفورد قامت بتطعيم ابنائها التوأم الثلاثة بلقاح #كورونا وهم يبلغون من العمر 21 سنة وجميعهم يدرسون الكيمياء الحيوية بنفس الجامعة ! جيلبرت : أنا لست قلقة ونحن نعمل إختبارات امنه للجميع https://t.co/blP6PGl9zX'], 'ar', 'en'))
    print('Done')