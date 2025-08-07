import random

GPT2_top10 = [(5, 0), (5, 1), (5, 5), (6, 9), (7, 2), (7, 10), (9, 6), (9, 9), (10, 1), (10, 7)]

all_gpt2_heads = [(layer, head) for layer in range(12) for head in range(12)]

induction = [head for head in all_gpt2_heads if head in GPT2_top10]
non_induction = [head for head in all_gpt2_heads if head not in GPT2_top10]

# as non_induction heads are many, we only sample len(GPT2_top10) to balance the dataset
non_induction = random.sample(non_induction, len(GPT2_top10))

# INDUCTION CIRCUIT is the whole GPT2-small for now
INDUCTION_CIRCUIT_GPT2SMALL = {
    "induction": induction,
    "non-induction": non_induction,
}

GT_CIRCUIT_GPT2SMALL = {
    "0305": [(0, 3), (0, 5)], # read in YY
    "01": [(0, 1)], # read in YY
    "MEARLY": [(0, None), (1, None), (2, None), (3, None)], # read in YY
    "AMID": [(5, 5), (6, 1), (6, 9), (7, 10), (8, 11), (9, 1)], # create logit spike at YY
    "MLATE": [(8, None), (9, None), (10, None), (11, None)], # boost logits for yy > YY
}

IOI_CIRCUIT_GPT2SMALL = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}
# # stricter one
# IOI_CIRCUIT = {
#     "name mover": [(9, 9), (10, 0), (9, 6)],
#     "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
#     "negative": [(10, 7), (11, 10)],
#     "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
#     "induction": [(5, 5), (6, 9)],
#     "duplicate token": [(0, 1), (3, 0)],
#     "previous token": [(2, 2), (4, 11)],
# }

induction_description = "Induction heads attend from the second occurrence of each token to the token after its first occurrence."

previous_token_description = "Previous token heads copy information from the previous token into the next token."

duplicate_token_description = "Duplicate token heads identify tokens that have already appeared in the sentence. They are active\
                                at the S2 token, attend primarily to the S1 token, and signal that token duplication has occurred\
                                by writing the position of the duplicate token."

name_mover_description = "Name mover heads are active at END token (the final token position for prediction), attend to previous names\
                            in the sentence, and copy the names they attend to. Due to the S-inhibition heads, they attend to\
                            the IO token over the S1 and S2 tokens."

s_inhibition_description = "S-inhibition heads remove duplicate tokens from name mover heads attention. They are active\
                            at the END token, attend to the S2 token, and write in the query of the name mover heads,\
                            inhibiting their attention to S1 and S2 tokens."

backup_name_mover_description = "Backup name mover heads copy names to the correct position in the output, \
                            but only when regular name mover heads are ablated."

negative_name_mover_description = "Negative name mover heads share all the same properties as name mover heads \
                            except they (1) write in the opposite direction of names they attend to and (2) have \
                            a large negative copy score (the copy score calculated with the negative of the OV matrix, \
                            98 percent compared to 12 percent for an average head)."