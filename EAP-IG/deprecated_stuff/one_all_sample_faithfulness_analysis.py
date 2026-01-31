import json
import numpy as np
import matplotlib.pyplot as plt

task = 'ioi'
method = 'eap-ig-inputs'

# Load all-sample data
all_sample_filepath = f'{task}_{method}_5steps_all_sample_data.json'
with open(all_sample_filepath, 'r') as file:
    all_sample_data1 = json.load(file)

all_sample_filepath = f'{task}_{method}_15steps_all_sample_data.json'
with open(all_sample_filepath, 'r') as file:
    all_sample_data2 = json.load(file)

# List of IG steps to load
ig_steps_list = [5, 10, 15, 20, 30, 50, 100, 300]
one_sample_faithfulness_all = []

# Load and process each one-sample file
for steps in ig_steps_list:
    filepath = f'{task}_{method}_{steps}steps_one_sample_data.json'
    with open(filepath, 'r') as file:
        one_sample_data = json.load(file)
    faithfulness = [d['circuit_faithfulness'] for d in one_sample_data]
    faithfulness = np.mean(faithfulness, axis=0).tolist()
    one_sample_faithfulness_all.append((steps, one_sample_data[0]['topns'], faithfulness))

# Plotting
plt.plot(all_sample_data1['topns'], all_sample_data1['circuit_faithfulness'], label='All Sample IG-5', marker='s')
plt.plot(all_sample_data2['topns'], all_sample_data2['circuit_faithfulness'], label='All Sample IG-15', marker='s')

# Sort by IG steps before plotting
one_sample_faithfulness_all.sort()

for steps, topns, faithfulness in one_sample_faithfulness_all:
    plt.plot(topns, faithfulness, label=f'One Sample IG-{steps}', marker='o')

plt.ylim(-0.1, 1.1)
plt.xlabel('Top-K Edges')
plt.ylabel('Circuit Faithfulness')
plt.title(f'{task.upper()} Circuit Faithfulness vs Top-K Edges')
plt.legend()

plt.savefig(f'{task}_{method}_one_sample_faithfulness_multi_steps.png', dpi=500)