# Query Circuit Behavior Analysis

> The original raw code for this experiment were lost. The scripts in this directory were reproduced by a coding agent from the paper's described experimental setup and steps, not recovered from the original implementation. Resulting figures mostly match the paper, with minor differences.

Reproduces the gender-feature steering experiments (Tables 2 & A7 in the paper) on GPT-2 Small with the Gender Bias dataset. Run the scripts in order; each saves its intermediate data under `data/`.

| Script | Step | Description |
|--------|------|-------------|
| `step1_collect_biased_samples.py` | 1 | Collects samples with P(stereotypical) - P(anti-stereotypical) > 0.5 (50 / 986). |
| `step2_unpaired_circuit_discovery.py` | 2 | EAP-IG circuit (topn=200) on each sample; splits into high-NDF (> 0.8, 26) and low-NDF (24) groups. |
| `step2_paired_circuit_discovery.py` | 2 | EAP-IG circuits (topn=150) on all samples; for each sample, evaluates its own circuit plus 9 random others' circuits and records the best/worst by NDF; keeps samples with best NDF > 0.8 (32). |
| `step3_gender_feature_stats.py` | 3 | Runs each circuit on its target query, extracts top-5 SAE features per token per node, and counts gender-related features (keyword regex). |
| `step4_paired_steering.py` | 4 | Zero-ablates gender features on circuit MLPs and compares best vs. worst circuits with one-sided Wilcoxon signed-rank tests and Rosenthal's r (Table 2). |
| `step4_unpaired_steering.py` | 4 | Same ablation comparing high-NDF vs. low-NDF groups with one-sided Mann-Whitney U tests and Cohen's d (Table A7). |

Notes:
- Only MLP SAE features are ablated; attention-out SAEs are not head-granular, so attentional features are not guaranteed to lie in the circuit.
- Shared SAE helpers (SAE loading, circuit feature extraction, gender keyword matching, feature ablation) live in `src/eap/sae_utils.py`.
