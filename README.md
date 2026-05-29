Query Circuits
### On finding the sparse computation of the model for a single query 

## This work is accepted to ICML'26 as a main conference paper.

## Install
```
git clone https://github.com/tony10101105/Query-Circuit-Discovery.git
cd EAP-IG
pip install .
```

## Env Config
```
cp env.template .env
# put in the necessary environment variables, such as your OPENAI_API_KEY and HF_TOKEN, to .env
```

## Data Download

### Download and Process Data for Circuit Discovery
Below we use mmlu as an example. You can run scripts of different datasets based on your need.
```
cd probing_dataset
python mmlu_data_download_and_format.py --category marketing # to transform to a format suitable as circuit dataset. This will create mmlu_marketing_Llama-32-1B.py; we have provided that so you don't need to rerun this script
python mmlu_rephrase_only_stem.py # to generate paraphrases for each query. This will create mmlu_marketing_Llama-32-1B_gpt4o_paraphrases_only_stem.csv; we have provided that so you don't need to rerun this script
```

We provide intermediate data (e.g., score matrix) in [HF data repo](https://huggingface.co/datasets/tony10101105/Query-Circuit-Dataset) useful for fast replication. Follow the following steps to download it:
```
apt-get update
apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/tony10101105/Query-Circuit-Dataset
```
The whole dataset will occupy ~366GB. You can download a part of it if you don't have enought disk space.

### Download SAE data
```
cd sae_data
bash download_sae.sh
python data_unzipper.py
```

## Credit
The codebase was revised from [EAP-IG](https://github.com/hannamw/eap-ig). Thanks for their great work!

## Cite
```
@article{wu2025query,
  title={Query circuits: Explaining how language models answer user prompts},
  author={Wu, Tung-Yu and Barez, Fazl},
  journal={arXiv preprint arXiv:2509.24808},
  year={2025}
}
```