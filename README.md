# Query Circuits: Explaining How Language Models Answer User Prompts

Also refer to [Project Page](https://tony10101105.github.io/query-circuit/)

## Install
```
git clone https://github.com/tony10101105/Query-Circuit-Discovery.git
cd Query-Circuit-Discovery/EAP-IG
pip install .
```

## Environment Config
```
cp env.template .env
```
Then put in necessary environment variables, such as your OPENAI_API_KEY and HF_TOKEN, to .env

## Download Data

### Download and Process Dataset for Circuit Discovery
Below we use mmlu as an example. You can run scripts of different datasets based on your need.
```
cd probing_dataset
python mmlu_data_download_and_format.py --category marketing
```
The above transforms the MMLU marketing dataset into the format required for circuit analysis and generates `mmlu_marketing_Llama-32-1B.py`. We have already provided this file, so rerunning the script is unnecessary.
```
python mmlu_rephrase_only_stem.py
```
The above generates paraphrases for each query and produces `mmlu_marketing_Llama-32-1B_gpt4o_paraphrases_only_stem.csv`. The generated file is already provided, so you don't need to rerun this step.

### Download and Process Score Matrix
You need to generate score matrices before doing BoN or any other analyses. This can be done by running scripts under `save_score_matrix/`. For example, this generates score matrix:
```
cd save_score_matrix
python arcc_save_score_matrix.py
```

We provide intermediate data (e.g., score matrix) in [HF data repo](https://huggingface.co/datasets/tony10101105/Query-Circuit-Dataset) useful for fast replication. Follow the following steps to download it:
```
apt-get update
apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/tony10101105/Query-Circuit-Dataset
```
The dataset occupies ~366GB. You can download a part of it if you don't have enough disk space.

### Download SAE data
Download labeled SAE features if you want to apply SAEs on discovered circuits.
```
cd sae_data
bash download_sae.sh
python data_unzipper.py
```

## Credit
The codebase was revised from [EAP-IG](https://github.com/hannamw/eap-ig).

## Cite
```
@inproceedings{wu2026query,
  title={Query Circuits: Explaining How Language Models Answer User Prompts},
  author={Tung-Yu Wu and Fazl Barez},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=7F0sragazb}
}
```