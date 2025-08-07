GT_CIRCUIT = {
    "0305": [(0, 3), (0, 5)], # read in YY
    "01": [(0, 1)], # read in YY
    "MEARLY": [(0, None), (1, None), (2, None), (3, None)], # read in YY
    "AMID": [(5, 5), (6, 1), (6, 9), (7, 10), (8, 11), (9, 1)], # create logit spike at YY
    "MLATE": [(8, None), (9, None), (10, None), (11, None)], # boost logits for yy > YY
}

GT_CIRCUIT_MERGED = {
    "010305": [(0, 1), (0, 3), (0, 5)], # read in YY
    "MEARLY": [(0, None), (1, None), (2, None), (3, None)], # read in YY
    "AMID": [(5, 5), (6, 1), (6, 9), (7, 10), (8, 11), (9, 1)], # create logit spike at YY
    "MLATE": [(8, None), (9, None), (10, None), (11, None)], # boost logits for yy > YY
}

# for experiment 1: gpt2-small vs. gpt2-small
GT_CIRCUIT_GROUP_B = {
    "0305": [(0, 3)], # read in YY
    "01": [(0, 1)], # read in YY
    "MEARLY": [(0, None), (1, None)], # read in YY
    "AMID": [(5, 5), (6, 1), (6, 9)], # create logit spike at YY
    "MLATE": [(8, None), (9, None)], # boost logits for yy > YY
}

GROUP_B_IDX_2_CLASS = {
    (0, 3):   "0305",
    (0, 1):   "01",
    (0, None): "MEARLY",
    (1, None): "MEARLY",
    (5, 5):   "AMID",
    (6, 1):   "AMID",
    (6, 9):   "AMID",
    (8, None): "MLATE",
    (9, None): "MLATE",
}

GT_CIRCUIT_GROUP_A = {
    "0305": [(0, 5)], # read in YY
    "MEARLY": [(2, None), (3, None)], # read in YY
    "AMID": [(7, 10), (8, 11), (9, 1)], # create logit spike at YY
    "MLATE": [(10, None), (11, None)], # boost logits for yy > YY
}

GROUP_A_IDX_2_CLASS = {
    (0, 5):    "0305",
    (2, None): "MEARLY",
    (3, None): "MEARLY",
    (7, 10):   "AMID",
    (8, 11):   "AMID",
    (9, 1):    "AMID",
    (10, None): "MLATE",
    (11, None): "MLATE",
}

head_2_layer = {
    "010305": [0],
    "AMID": [5, 6, 7, 8, 9]
}

# head_2_layer = {
#     "010305": [0],
#     "AMID": [5, 6, 7, 8, 9],
#     "MEARLY": [0, 1, 2, 3],
#     "MLATE": [8, 9, 10, 11],
# }