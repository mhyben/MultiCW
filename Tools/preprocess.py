from os.path import join

import ollama
import pandas as pd
from pandas import Series
from tqdm.auto import tqdm
from translation import cooldown, deeptranslate, parallel_deeptranslate
from langdetect import detect

# ANSI Highlighting: https://stackoverflow.com/a/21786287
h_red = '\x1b[1;30;41m'
h_green = '\x1b[1;30;42m'
h_yellow = '\x1b[1;30;43m'
h_stop = '\x1b[0m'


# Convert to structured style
def to_structured(claims: Series, model='qwen2.5:7b', show_progress=True, verbose=False):
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
                lang = detect(row)
                if lang != 'en':
                    output = deeptranslate(content=pd.Series([output]), src_lang='en', dst_lang=lang, show_progress=False)[0]

                # print(f'{h_yellow}Sentence: {row}{h_stop}')
                if verbose:
                    print(f'{h_green}Response:{h_stop} {output}')
                results.append(output)
                break

            # In case of error just try again
            except Exception as err:
                print(f"Error: {str(err)}")

    # If no category was found after all the attempts, consider it 'unknown'
    return results

# Convert to structured style
def to_noisy(claims: Series, model='qwen2.5:7b', max_length=2500, show_progress=True, verbose=False):
    import pandas as pd

    prompt = (f"Summarize the following text into a single English sentence in a noisy social media-writing style, no longer than {max_length} characters. "
              "Do not include empty lines or commentary.")

    long_claims = claims[claims.str.len() > max_length]
    short_claims = claims[claims.str.len() <= max_length]

    # Translate short claims in parallel
    if not short_claims.empty:
        short_translated = parallel_deeptranslate(short_claims, src_lang='auto', dst_lang='en', show_progress=show_progress)
        short_results = pd.Series(short_translated, index=short_claims.index)
    else:
        short_results = pd.Series(dtype=str)

    # Translate long claims using loop with ollama and deeptranslate
    long_results = pd.Series(index=long_claims.index, dtype=str)
    iterator = tqdm(long_claims.items()) if show_progress else long_claims.items()
    
    for idx, row in iterator:
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
                    options={"temperature": 0.1}
                )
                output = response['message']['content'].strip()

                output = deeptranslate(content=pd.Series([output]), src_lang='en', dst_lang=lang, show_progress=False)[0]

                if verbose:
                    print(f'{h_yellow}Sentence: {row}{h_stop}')
                    print(f'{h_green}Response:{h_stop} {output}')

                long_results.at[idx] = output
                break

            except Exception as err:
                print(f"Error on idx {idx}: {str(err)}")
                if attempts == 2:
                    long_results.at[idx] = ''

    # Combine back in original order
    full_results = pd.concat([short_results, long_results]).sort_index()
    return full_results.tolist()


if __name__=='__main__':
    text = 'V dnešnom globalizovanom svete, kde technológie napredujú rýchlejšie než kedykoľvek predtým a spoločnosti sa musia neustále prispôsobovať novým výzvam, zmenám a inováciám v rôznych oblastiach – od priemyslu cez vzdelávanie až po zdravotníctvo – je nesmierne dôležité, aby jednotlivci, organizácie aj vlády dokázali nielen sledovať najnovšie trendy, ale aj rozumieť ich dôsledkom, analyzovať potenciálne riziká a využívať nové príležitosti v prospech rozvoja, efektivity a udržateľnosti, pričom schopnosť komunikovať naprieč kultúrami, jazykmi a disciplínami hrá kľúčovú úlohu v úspešnom prekonávaní hraníc, búraní predsudkov a vytváraní prostredia otvoreného pre spoluprácu, inovácie a spoločenský pokrok, a preto je nevyhnutné, aby sme investovali do vzdelávania, podporovali kritické myslenie, rozvíjali digitálne zručnosti, motivovali mladých ľudí k aktívnej participácii na spoločenskom dianí, chránili naše životné prostredie, podporovali diverzitu a inklúziu, hľadali nové riešenia prostredníctvom interdisciplinárnych prístupov a zároveň si uvedomovali hodnotu ľudských práv, demokracie a solidarity, pretože iba spoločne – ako vedomí a zodpovední občania – môžeme čeliť klimatickej kríze, sociálnym nerovnostiam, politickej nestabilite či technologickým hrozbám, ktoré síce prinášajú neistotu a výzvy, no zároveň v sebe skrývajú potenciál na premenu nášho sveta na lepšie, spravodlivejšie a udržateľnejšie miesto pre nás všetkých aj pre budúce generácie, a preto je čas konať – premýšľať, diskutovať, spájať sa, tvoriť a meniť svet nie len slovami, ale aj činmi, ktoré reflektujú naše hodnoty, ciele a nádej, že každá snaha, akokoľvek malá, má zmysel, keď je vedená úprimným úmyslom zlepšiť život iným a posilniť základy spoločnosti postavenej na dôvere, empatii a spoločnom úsilí o lepšiu budúcnosť.'

    # Convert to structured style
    print('Normalizing noisy sentences:')
    to_noisy(pd.Series(text), 'en', verbose=False)

    print('Normalizing struc sentences:')
    to_structured(pd.Series(text), 'en', verbose=False)
