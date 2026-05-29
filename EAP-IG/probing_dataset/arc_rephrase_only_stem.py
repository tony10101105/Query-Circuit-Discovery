import os
import re
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------
# Global Counter
# --------------------
stats = {"larger_cnt": 0, "fewer_cnt": 0, "sample_fewer_cnt": 0}

# --------------------
# Prompt templates
# --------------------
EXAMPLE_USER = """Example input MCQ:
What color is the sky on a clear day?
(A) Green.
(B) Blue.
(C) Red.
(D) Yellow.
Answer: ("""

EXAMPLE_OUTPUT = """What is typically the sky's color on a cloudless day?
||||
On a bright, cloud-free day, the sky usually appears what color?
||||
[Remaining 7 paraphrases of the question stem]
"""

def build_messages(question_text: str, system_instruction: str, n_paraphrases: int):
    """Builds the messages for the Chat Completions API."""
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": EXAMPLE_OUTPUT},
        {"role": "user", "content": f"Now paraphrase the question stem of this MCQ into exactly {n_paraphrases} versions:\n\n{question_text.strip()}"}
    ]

def parse_paraphrases(mcq_text: str, raw: str, n_expected: int) -> List[str]:
    """Parse the model output into N paraphrases of the question stem using '||||' as the separator.
       Also trims whitespace and discards empty chunks."""

    parts = [p.strip() for p in raw.strip().strip('|').split('||||')]
    parts = [p for p in parts if p]  # drop empties

    # Regex explanation:
    # ^(.*?)\n   -> capture the first line (the description) up to the first newline
    # (?=\(A\))  -> look ahead to ensure the next line starts with (A)
    pattern = r"^(.*?)\n(?=\((A|1)\))"
    try:
        valid = [re.sub(pattern, f"{p}\n", mcq_text, flags=re.DOTALL) for p in parts]
    except:
        print('parts: ', parts)
        valid = ['X' for _ in parts]
    
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

def get_paraphrases_for_mcq(mcq_text: str, args, system_instruction: str) -> List[str]:
    """Call the model with retries and return exactly args.n_paraphrases strings (or fewer if all retries fail)."""
    for attempt in range(args.max_retries + 1):
        msg = build_messages(mcq_text, system_instruction, args.n_paraphrases)
        resp = client.chat.completions.create(
            model=args.model,
            temperature=args.temp,
            messages=msg,
        )
        content = resp.choices[0].message.content
        
        paras = parse_paraphrases(mcq_text, content, args.n_paraphrases)
        if len(paras) == args.n_paraphrases:
            return paras
        
        print(f"Attempt {attempt + 1}/{args.max_retries + 1}: Got {len(paras)} paraphrases, expected {args.n_paraphrases}. Retrying...")
        # brief backoff before retry
        time.sleep(1)
    
    # Final fallback: if still fewer, pad with copies of the original MCQ (discouraged but keeps shape)
    if len(paras) < args.n_paraphrases:
        stats['sample_fewer_cnt'] += 1
        print(f"Final attempt: got {len(paras)} paraphrases, padding with original MCQ to reach {args.n_paraphrases}. Failed number: {stats['sample_fewer_cnt']}")
        while len(paras) < args.n_paraphrases:
            paras.append(mcq_text.strip())
    
    return paras

def main():
    parser = argparse.ArgumentParser(description='Generate paraphrases for ARC dataset using OpenAI')
    parser.add_argument('--category', type=str, default='easy', choices=['easy', 'challenge'],
                        help='ARC category')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='OpenAI model to use')
    parser.add_argument('--temp', type=float, default=0.0,
                        help='Sampling temperature')
    parser.add_argument('--n_paraphrases', type=int, default=9,
                        help='Number of paraphrases to generate')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Max retries on failure')
    parser.add_argument('--sleep', type=float, default=1.0,
                        help='Seconds to sleep between API calls')
    args = parser.parse_args()

    input_csv = f"arc_{args.category}_Llama-32-1B.csv"
    output_csv = f"arc_{args.category}_Llama-32-1B_{args.model.replace('-', '')}_paraphrases_only_stem.csv"
    system_instruction = f"""You are given a multiple-choice question (MCQ) prompt that includes a question stem,
    four answer options labeled (A), (B), (C), and (D), and ends with 'Answer: ('.
    Generate exactly {args.n_paraphrases} distinct paraphrases of the question stem, preserving:

    1) The meaning of the original question stem.
    2) The rephrased question stem SHOULD NOT reveal or change the correct answer.

    Output format:
    Return the {args.n_paraphrases} paraphrases of the question stem separated by the delimiter ||||.

    As a result, your outputs should contain {args.n_paraphrases - 1} |||| that separate the {args.n_paraphrases} paraphrases of the question stem. It is important that you return exactly {args.n_paraphrases} paraphrases.

    Do not include any extra commentary before or after.
    """

    df = pd.read_csv(input_csv)
    if "clean" not in df.columns:
        raise ValueError("Input CSV must contain a 'clean' column with the MCQ text.")

    # Prepare output columns
    out_cols = [f"paraphrase{i}" for i in range(1, args.n_paraphrases + 1)]
    for c in out_cols:
        if c not in df.columns:
            df[c] = ""

    # Iterate rows and paraphrase each MCQ prompt in 'clean'
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        mcq = str(row["clean"])
        if not mcq.strip():
            # leave paraphrases blank
            continue

        paras = get_paraphrases_for_mcq(mcq, args, system_instruction)

        # Write into columns
        for i, col in enumerate(out_cols):
            df.at[idx, col] = paras[i]

        # polite rate limiting
        time.sleep(args.sleep)

    # Save result
    df.to_csv(output_csv, index=False)
    print(f"Saved paraphrases to {output_csv}")
    print(f"Total larger counts: {stats['larger_cnt']}, fewer counts: {stats['fewer_cnt']}, sample fewer counts: {stats['sample_fewer_cnt']}")


if __name__ == '__main__':
    main()