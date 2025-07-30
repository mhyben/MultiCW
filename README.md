# Multi-Check-Worthy Dataset (MultiCW)

This repository contains the code and datasets for the paper "A Real-World Multi-Domain Dataset of Check-Worthy Claims." The project introduces the Multi-Check-Worthy dataset (MultiCW), a benchmarking dataset of check-worthy claims that spans multiple languages, topics, and writing styles.

## Abstract

We introduce Multi-Check-Worthy dataset (MultiCW), a multilingual benchmarking dataset for check-worthy claim detection, covering 16 languages, 6 topical domains, and 2 writing styles. The dataset consists of 123,722 samples, evenly distributed between noisy and structured texts, with balanced representation of check-worthy and non-check-worthy classes across all languages. We evaluate MultiCW using (a) three fine-tuned state-of-the-art multilingual language models and (b) a variety of large language models (LLMs) in zero-shot settings, including both commercial and open-source models, using diverse sizes and prompting strategies. Results show that models fine-tuned on MultiCW consistently outperform zero-shot LLMs in binary classification of check-worthy claims. Moreover, these fine-tuned models demonstrate strong generalization across languages, topics, and writing styles, as evidenced by cross-domain experiments on previously unseen dataset. Our findings position MultiCW as a valuable resource for advancing multilingual and cross-domain automated fact-checking systems.

## Project structure

- **Source-datasets**: Contains the datasets used to create the MultiCW dataset.

- **Final-dataset**: Contains the partial files used to compile the MultiCW dataset together with the final dataset and the train, validation, and test sets all in CSV format.

- **1-MultiCW-dataset**: Notebook for setting up and exploring the MultiCW dataset.

- **2-Models-fine-tuning**: Notebook for fine-tuning models on the MultiCW dataset and their evaluation.

## Conda environment setup
There are two jupyter notebooks in this project. However, out of three models fine-tuned and evaluated in the ***2-Models-fine-tuning*** notebook, we need to create a specific conda environment for two models and a separate conda environment for the third one. We therefore need to create overall three conda environments for the following use cases:
* The whole ***1-MultiCW-dataset notebook***
* mDeBERTa & XLM-RoBERTa models of ***2-Models-fine-tuning notebook***
* LESA model of ***2-Models-fine-tuning notebook***

To make the creation and setup of the conda environments as simple as possible, we have prepared the shell script to automate the process. 
To run the shell script simply run the following commands:

```bash
cd /<your-path>/MultiCW
chmod +x setup.sh
./setup.sh
```
When prompted, enter your path to your Conda installation, i.e.:
```bash
/home/yourname/miniconda3
```

### Manual installation of the conda environments
In case you want to install the conda environments manually, you can follow the following steps.  

#### MultiCW Dataset Notebook

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

#### MultiCW Fine-Tuning Notebook - mDeBERTa & xlm-RoBERTa models

```bash
conda create --name MultiCW-finetune python=3.10
conda activate MultiCW-finetune
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=MultiCW-finetune
conda install -n base -c conda-forge jupyterlab_widgets
pip install -r requirements-finetune.txt
python -m spacy download en_core_web_sm
```

#### MultiCW Fine-Tuning Notebook - LESA model

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