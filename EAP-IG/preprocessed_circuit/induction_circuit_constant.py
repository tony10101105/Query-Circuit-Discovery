import random
random.seed(2025)


GPT2_top10 = [(5, 0), (5, 1), (5, 5), (6, 9), (7, 2), (7, 10), (9, 6), (9, 9), (10, 1), (10, 7)]

all_gpt2_heads = [(layer, head) for layer in range(12) for head in range(12)]

induction = [head for head in all_gpt2_heads if head in GPT2_top10]
non_induction = [head for head in all_gpt2_heads if head not in GPT2_top10]

# as non_induction heads are many, we only sample len(GPT2_top10) to balance the dataset
non_induction = random.sample(non_induction, len(GPT2_top10))

# INDUCTION CIRCUIT is the whole GPT2-small for now
INDUCTION_CIRCUIT = {
    "induction": induction,
    "non-induction": non_induction,
}

# Shuffle and split induction heads
induction_shuffled = induction[:]
random.shuffle(induction_shuffled)
mid_ind = len(induction_shuffled) // 2
induction_A = induction_shuffled[:mid_ind]
induction_B = induction_shuffled[mid_ind:]

# Shuffle and split non-induction heads
non_induction_shuffled = non_induction[:]
random.shuffle(non_induction_shuffled)
mid_non = len(non_induction_shuffled) // 2
non_induction_A = non_induction_shuffled[:mid_non]
non_induction_B = non_induction_shuffled[mid_non:]

INDUCTION_CIRCUIT_GROUP_A = {
    "induction": sorted(induction_A),
    "non-induction": sorted(non_induction_A),
}

GROUP_A_IDX_2_CLASS = {
    head: "induction" if head in induction_A else "non-induction"
    for head in induction_A + non_induction_A
}

INDUCTION_CIRCUIT_GROUP_B = {
    "induction": sorted(induction_B),
    "non-induction": sorted(non_induction_B),
}

GROUP_B_IDX_2_CLASS = {
    head: "induction" if head in induction_B else "non-induction"
    for head in induction_B + non_induction_B
}


non_induction_layers = list(set([head[0] for head in non_induction]))
head_2_layer = {
    "induction": [5, 6, 7, 9, 10],
    "non-induction": non_induction_layers
}