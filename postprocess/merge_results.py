import json
import csv

# the opencompass dir of the origin results
dir = ''

# all test datasets
combined_datasets = [
    'word-diversity',
    'rule-following',
    'word-diversity',
    'emotion'
]
split_datasets = [
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

# all test models
models = [
    'LLama2-7B', 'LLama2-13B',
    'baichuan2_7b', 'baichuan2_13b',
    'qwen2-7B', 'qwen2-14B', 'qwen2-72B',
    'chatglm2', 'chatglm3-base',
    'deepseek-llm-7b-base', 'deepseek-llm-67b-base',
    'GPT3.5', 'GPT4.0'
]

all_results = []
for model in models:
    all_scores = []

    for dataset in combined_datasets:
        res_path = f'{dir}/results/{model}/{dataset}.json'
        with open(res_path, 'r', encoding='utf-8') as f:
            cur_data = json.load(f)
        all_scores.append(round(cur_data['score'], 2))

    for dataset in split_datasets:
        scores = []
        for lang in ['zh', 'en']:
            res_path = f'{dir}/results/{model}/{dataset}-{lang}.json'
            pred_path = f'{dir}/predictions/{model}/{dataset}-{lang}.json'
            with open(res_path, 'r', encoding='utf-8') as f:
                cur_data = json.load(f)
            with open(pred_path, 'r', encoding='utf-8') as f:
                cur_pred = json.load(f)
            if dataset == 'icl':
                icl_res_path = f'{dir}/results/{model}/{dataset}-{lang}-4shot.json'
                with open(icl_res_path, 'r', encoding='utf-8') as f:
                    icl_data = json.load(f)
                cur_score = (1 / 3) * (icl_data['score'] - cur_data['score']) + (2 / 3) * icl_data['score']
                scores.extend([cur_score] * len(cur_pred))
            elif 'score' in cur_data.keys():
                scores.extend([cur_data['score']] * len(cur_pred))
            elif 'accuracy' in cur_data.keys():
                scores.extend([cur_data['accuracy']] * len(cur_pred))
            elif 'contradiction_score' in cur_data.keys():
                scores.extend([cur_data['contradiction_score']] * len(cur_pred))
        # print(model, data, scores)
        cur_mean = sum(scores) / len(scores)
        all_scores.append(round(cur_mean, 2))

    all_results.append(all_scores)

path = f'{dir}/results.csv'
with open(path, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 一个数据一行
    writer.writerow([' '] + models)
    writer.writerows(zip(combined_datasets + split_datasets, *all_results))
