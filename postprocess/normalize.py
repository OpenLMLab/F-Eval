import json
import csv

import numpy as np

# the opencompass dir of the origin results
dir = ''

datasets = [
    'word-diversity',
    'rule-following',
    'word-diversity',
    'emotion',
    'commonsense_triple',
    'commonsenseqa',
    'textbookqa',
    'story',
    'instruction',
    'icl',
    'cot',
    'fallacy_attack',
    'contradiction',
    'coreference',
    'anaomaly_detection'
]

with open('../data/param.json', 'r') as f:
    param = json.load(f)


def normalize(score, dataset, gamma=2.5):
    lower_bound = param[dataset][0]
    upper_bound = param[dataset][1]

    transformed_scores = ((score - lower_bound) / (upper_bound - lower_bound)) * gamma - (gamma / 2)

    # Apply the logistic function
    rescaled_scores = 1 / (1 + np.exp(-transformed_scores))

    # Scale to the desired range
    rescaled_scores = np.round(rescaled_scores * 100, 2)

    return rescaled_scores


with open(f'{dir}/results.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    all_results = []
    for row in reader:
        all_results.append(row)

for i in range(len(all_results)):
    for j in range(1, len(all_results[i])):
        all_results[i][j] = normalize(float(all_results[i][j]), datasets[i])

with open(f'{dir}/results_normalized.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in all_results:
        writer.writerow(row)
