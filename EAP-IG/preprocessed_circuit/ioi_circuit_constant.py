IOI_CIRCUIT = {
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


# for experiment 1: gpt2-small vs. gpt2-small
IOI_CIRCUIT_GROUP_A = {
    "name mover": [(9, 9), (10, 0)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1)],
    "negative": [(10, 7)],
    "s2 inhibition": [(7, 3), (7, 9)],
    "induction": [(5, 5), (5, 8)],
    "duplicate token": [(0, 1)],
    "previous token": [(2, 2)],
}

GROUP_A_IDX_2_CLASS = {
    (0, 1): "duplicate token",
    (2, 2): "previous token",
    (5, 5): "induction",
    (5, 8): "induction",
    (7, 3): "s2 inhibition",
    (7, 9): "s2 inhibition",
    (9, 9): "name mover",
    (10, 0): "name mover",
    (10, 1): "backup name mover",
    (10, 2): "backup name mover",
    (10, 6): "backup name mover",
    (10, 7): "negative",
    (10, 10): "backup name mover",
}

IOI_CIRCUIT_GROUP_B = {
    "name mover": [(9, 6)],
    "backup name mover": [(11, 2), (9, 7), (9, 0), (11, 9)],
    "negative": [(11, 10)],
    "s2 inhibition": [(8, 6), (8, 10)],
    "induction": [(5, 9), (6, 9)],
    "duplicate token": [(0, 10), (3, 0)],
    "previous token": [(4, 11)],
}

GROUP_B_IDX_2_CLASS = {
    (0, 10): "duplicate token",
    (3, 0): "duplicate token",
    (4, 11): "previous token",
    (5, 9): "induction",
    (6, 9): "induction",
    (8, 6): "s2 inhibition",
    (8, 10): "s2 inhibition",
    (9, 0): "backup name mover",
    (9, 6): "name mover",
    (9, 7): "backup name mover",
    (11, 2): "backup name mover",
    (11, 9): "backup name mover",
    (11, 10): "negative",
}

head_2_layer = {
    "duplicate token": [0, 3],
    "previous token": [2, 4],
    "induction": [5, 6],
    "s2 inhibition": [7, 8],
    "name mover": [9, 10],
    "backup name mover": [9, 10, 11],
    "negative": [10, 11]
}