# Multi-Check-Worthy Dataset (MultiCW)

This repository contains the code and datasets for the paper "A Real-World Multi-Domain Dataset of Check-Worthy Claims." The project introduces the Multi-Check-Worthy dataset (MultiCW), a benchmarking dataset of check-worthy claims that spans multiple languages, topics, and writing styles.

## Abstract

We introduce the Multi-Check-Worthy dataset (MultiCW), a benchmarking dataset of check-worthy claims that spans multiple languages, topics, and writing styles. MultiCW was created by merging four existing datasets (CLEF-2022, CLEF-2023, MultiClaim, Ru22Facts) and extending them with novel samples specifically collected for the task using the MONANT system. To ensure balance, additional samples were added through translations of English samples from the LESA dataset into target languages and curated Wikipedia samples, guided by language-specific named entity extraction to preserve topical and linguistic diversity. The final dataset comprises 112,852 samples in 16 languages, evenly distributed between noisy and structured writing styles, with balanced representation across all languages in both classes.

We evaluated MultiCW by fine-tuning three state-of-the-art language models (XLM-RoBERTa, mDeBERTa, and LESA) and testing six large language models (LLMs), including both commercial and open-source, in zero-shot settings. The results show that MultiCW enhances performance, as indicated by improved F1 scores, and strengthens generalization across languages and topics. It surpasses the models trained on the original LESA dataset tested on the NewsClaims dataset, which was used as an out-of-distribution benchmark. This work highlights MultiCW as a valuable benchmarking dataset for improving the accuracy of automated fact-checking systems, particularly in detecting check-worthy claims.

## Project structure

- **Source datasets**: Contains the datasets used to create the MultiCW dataset.

- **Final dataset**: Contains the partial files used to compile the MultiCW dataset together with the final dataset and the train, validation, and test sets all in CSV format.

- **MultiCW dataset**: Notebook for setting up and exploring the MultiCW dataset.

- **Models fine-tuning**: Notebook for fine-tuning models on the MultiCW dataset.

- **Named entity benchmark**: Notebook for benchmarking named entity recognition on the MultiCW dataset.

## Conda environment setup

### MultiCW Dataset Notebook

```bash
conda create --name MultiCW-dataset python=3.10
conda activate MultiCW-dataset
pip install jupyterlab
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=MultiCW-dataset
conda install -c conda-forge gliner
conda install -n base -c conda-forge jupyterlab_widgets
jupyter labextension install js
```

### MultiCW Fine-Tuning Notebook

```bash
conda create --name MultiCW-finetune python=3.10
conda activate MultiCW-finetune
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=MultiCW-finetune
conda install -n base -c conda-forge jupyterlab_widgets
pip install -r requirements-finetune.txt
python -m spacy download en_core_web_sm
```

```bash
conda create --name MultiCW-lesa python=3.10
conda activate MultiCW-lesa
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=MultiCW-lesa
conda install -n base -c conda-forge jupyterlab_widgets
pip install -r requirements-lesa.txt
python -m spacy download en_core_web_sm
```

## Related Work

The MultiCW dataset builds upon existing datasets and methodologies for check-worthy claim detection described in the paper [Multilingual and Multi-topical Benchmark of Fine-tuned Language models and Large Language Models for Check-Worthy Claim Detection](https://arxiv.org/abs/2311.06121). 

## References

* [CLEF2022-CheckThat! Lab dataset](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab)
* [CLEF2023-CheckThat! Lab dataset](https://gitlab.com/checkthat_lab/clef2023-checkthat-lab)
* [LESA dataset (2021)](https://github.com/LCS2-IIITD/LESA-EACL-2021)
* [MultiClaim dataset](https://zenodo.org/records/7737983)
* [Ru22Fact dataset](https://paperswithcode.com/paper/ru22fact-optimizing-evidence-for-multilingual)