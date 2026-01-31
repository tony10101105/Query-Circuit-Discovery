import time
import os
import time
import json
import pandas as pd
from tqdm import tqdm
from typing import List
from openai import OpenAI
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------
# Global Counter
# --------------------
stats = {"larger_cnt": 0, "fewer_cnt": 0, "sample_fewer_cnt": 0}

# --------------------
# Config
# --------------------
CATEGORY = 'professional_medicine' # marketing, professional_medicine, astronomy, college_biology, high_school_computer_science, logical_fallacies, nutrition, international_law, management
INPUT_CSV  = f"mmlu_{CATEGORY}_Llama-32-1B.csv"
MODEL      = "gpt-4o"
OUTPUT_CSV = f"mmlu_{CATEGORY}_Llama-32-1B_{MODEL.replace('-', '')}_paraphrases_all_query.csv"
TEMP       = 0 # Temperature for diversity (0.0 = deterministic, 1.0 = creative)
N_PARAPHRASES = 9
MAX_RETRIES = 2
SLEEP_BETWEEN_CALLS = 1.0  # seconds (tune for your rate limits)

# --------------------
# Prompt templates
# --------------------
SYSTEM_INSTRUCTION = f"""You are given a multiple-choice question (MCQ) prompt that includes a question stem,
four answer options labeled (A), (B), (C), and (D), and ends with 'Answer: ('.
Generate exactly {N_PARAPHRASES} distinct paraphrases of the entire question, preserving:

1) The meaning of the question.
2) The existence and labels of the four options (A), (B), (C), and (D), with the same option texts (you may lightly rephrase wording but must NOT change the meaning or swap which label maps to which text).
3) The final line 'Answer: (' exactly as-is at the end of each paraphrase.
4) Do NOT add or remove options.
5) Do NOT reveal or change the correct answer.
6) Keep the format readable and MCQ-like; you can rephrase the stem and option wordings, but keep (A)-(D) labels.

Output format:
Return the {N_PARAPHRASES} paraphrases of the question separated by the delimiter ||||.

As a result, your outputs should contain {N_PARAPHRASES-1} |||| that separate the {N_PARAPHRASES} paraphrases. It is important that you return exactly {N_PARAPHRASES} paraphrases.

Do not include any extra commentary before or after. Each paraphrase must be a complete MCQ block that ends with 'Answer: ('.
"""

# a tiny example to anchor the pattern (uses a generic dummy MCQ)
EXAMPLE_USER = """Example input MCQ:
What color is the sky on a clear day?
(A) Green.
(B) Blue.
(C) Red.
(D) Yellow.
Answer: ("""

EXAMPLE_OUTPUT = """What is typically the sky's color on a cloudless day?
(A) The sky is green.
(B) The sky is blue.
(C) The sky is red.
(D) The sky is yellow.
Answer: (
||||
On a bright, cloud-free day, the sky usually appears what color?
(A) Green in color.
(B) Blue in color.
(C) Red in color.
(D) Yellow in color.
Answer: ("""

def build_messages(question_text: str):
    """Builds the messages for the Chat Completions API."""
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": EXAMPLE_OUTPUT},
        {"role": "user", "content": f"Now paraphrase this MCQ into exactly {N_PARAPHRASES} versions:\n\n{question_text.strip()}"}
    ]

def parse_paraphrases(raw: str, n_expected: int) -> List[str]:
    """Parse the model output into N paraphrases using '||||' as the separator.
       Also trims whitespace and discards empty chunks."""
    # print('raw: ', raw)
    parts = [p.strip() for p in raw.strip().strip('|').split('||||')]
    parts = [p for p in parts if p]  # drop empties
    # print('parts: ', parts)
    # print(f"Parsed {len(parts)} parts from the model output.")
    # Soft validation: ensure each ends with 'Answer: (' (as requested).
    valid = []
    for p in parts:
        # tolerate trailing spaces/newlines
        if p.rstrip().endswith("Answer: ("):
            valid.append(p)
        else:
            # Try to coerce if the model printed extra whitespace after 'Answer: ('
            idx = p.rfind("Answer: (")
            if idx != -1:
                coerced = p[:idx + len("Answer: (")]
                if coerced.rstrip().endswith("Answer: ("):
                    valid.append(coerced.strip())
    # print('len(valid): ', len(valid))
    # exit(0)
    # Truncate or pad to exactly N
    if len(valid) == n_expected:
        return valid
    elif len(valid) > n_expected:
        print(f"Warning: Got {len(valid)} paraphrases, expected {n_expected}. Truncating to {n_expected}.")
        stats["larger_cnt"] += 1
        return valid[:n_expected]
    else:
        # If fewer, just return what we have (caller may retry)
        print(f"Warning: Got {len(valid)} paraphrases, expected {n_expected}. Returning fewer.")
        stats["fewer_cnt"] += 1
        return valid

def get_paraphrases_for_mcq(mcq_text: str) -> List[str]:
    """Call the model with retries and return exactly N_PARAPHRASES strings (or fewer if all retries fail)."""
    for attempt in range(MAX_RETRIES + 1):
        msg = build_messages(mcq_text)
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=TEMP,
            messages=msg,
        )
        content = resp.choices[0].message.content
        
        paras = parse_paraphrases(content, N_PARAPHRASES)
        if len(paras) == N_PARAPHRASES:
            return paras
        
        print(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: Got {len(paras)} paraphrases, expected {N_PARAPHRASES}. Retrying...")
        # brief backoff before retry
        time.sleep(1)
    # Final fallback: if still fewer, pad with copies of the original MCQ (discouraged but keeps shape)
    if len(paras) < N_PARAPHRASES:
        stats['sample_fewer_cnt'] += 1
        print(f"Final attempt: got {len(paras)} paraphrases, padding with original MCQ to reach {N_PARAPHRASES}. Failed number: {stats['sample_fewer_cnt']}")
        while len(paras) < N_PARAPHRASES:
            paras.append(mcq_text.strip())
    return paras

def main():
    df = pd.read_csv(INPUT_CSV)
    if "clean" not in df.columns:
        raise ValueError("Input CSV must contain a 'clean' column with the MCQ text.")

    # Prepare output columns
    out_cols = [f"paraphrase{i}" for i in range(1, N_PARAPHRASES + 1)]
    for c in out_cols:
        if c not in df.columns:
            df[c] = ""

    # Iterate rows and paraphrase each MCQ prompt in 'clean'
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        mcq = str(row["clean"])
        if not mcq.strip():
            # leave paraphrases blank
            continue

        paras = get_paraphrases_for_mcq(mcq)

        # Write into columns
        for i, col in enumerate(out_cols):
            df.at[idx, col] = paras[i]

        # polite rate limiting
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Save result
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved paraphrases to {OUTPUT_CSV}")
    
    print(f"Total larger counts: {stats['larger_cnt']}, fewer counts: {stats['fewer_cnt']}, sample fewer counts: {stats['sample_fewer_cnt']}")

if __name__ == "__main__":
    main()
