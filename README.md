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

## Download and Preprocess Data

### Dataset for Circuit Discovery
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

### Score Matrix
You need to generate score matrices before doing BoN or any other analyses. This can be done by running scripts under `save_score_matrix/`, e.g., 
```
cd save_score_matrix
python arcc_save_score_matrix.py
```

We provide pre-generated score matrix in [HF data repo](https://huggingface.co/datasets/tony10101105/Query-Circuit-Dataset) for fast replication. Follow the following steps to download it:
```
apt-get update
apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/tony10101105/Query-Circuit-Dataset
```
The dataset occupies ~346GB.

Then merge `arc_challenge_1/` and `arc_challenge_2/` into `arc_challenge/`:
```
cd Query-Circuit-Dataset
bash merge_arc_challenge.sh
```

## Query Circuit Analysis
With the score matrices, you can analyze the query circuits they identify. The `query_circuit_analysis/` directory contains the subfolders `main/` (Figure 6), `para_num/` (Figure 7), `complement/` (Figure A13), and `different_bon/` (Figure A17), each of which contains Python scripts for running the corresponding analysis. For example,
```
cd query_circuit_analysis/main
python ioi_query_circuit_analysis.py
```
will give you Figure 6(a).

## Misc Analysis
The `misc_analysis/` directory contains self-contained analysis scripts. Each corresponds to a specific figure or table in the paper.

| Script | Figure / Table | Description |
|--------|---------------|-------------|
| `plot_score_matrix.py` | Figure A9, A10 | Visualizes score matrix heatmaps for each task and model. |
| `query_circuit_vs_task_circuit_performance.py` | Figure 2(b) | Compares query circuit faithfulness against task (capability) circuit faithfulness on GPT-2 Small. |
| `mmlu_NFS_instability_examples.py` | Figure 2(a) | Plots NFS curves for specific MMLU queries to illustrate NFS instability. |
| `mmlu_NFS_NDF_comparison.py` | Figure 3 | Side-by-side comparison of NFS and NDF faithfulness metrics across individual MMLU queries in a grid layout. |
| `gender_bias_greedy_dijkstra_comparison.py` | Figure A15 | Compares greedy top-N edge selection against Dijkstra-like circuit construction on the gender bias task with GPT-2 Small. |
| `mmlu_runtime_analysis.py` | Table A4 | Measures and compares runtime for single-query vs. best-of-N circuit discovery on MMLU. |

Run any script from `misc_analysis/`, e.g.:
```
cd misc_analysis
python mmlu_NFS_NDF_comparison.py
```

## Query Circuit with SAEs

### Download SAE Data
Download labeled SAE features if you want to apply SAEs on discovered circuits.
```
cd sae_data
bash download_sae.sh
python data_unzipper.py
```

### Run SAEs on Query Circuits
Feed activations of nodes in a circuit into corresponding SAEs and examine parsed concepts:

First save the detailed graph (circuit) data:
```
cd gender_bias_sae_analysis
python gender_bias_save_graph_data.py
```
which will create a `graph_data` folder.

Then, enrich the data with SAE features:
```
python gender_bias_sae_analysis.py
```
which will create a `sae_analysis_data` folder.

You can manually inspect interesting feature connections, or use coding agent/print script to help summarize, e.g., `gender_bias_sae_circuit_print.py`.

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