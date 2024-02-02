# *F-Eval: Asssessing Fundamental Abilities with Refined Evaluation Methods*

F-Eval is a bilingual evaluation benchmark to evaluate the fundamental abilities, including expression, commonsense and
logic. It consists of 2211 instances in both English and Chinese. Please visit
our [paper](https://arxiv.org/abs/2401.14869) for more details.

The code of F-Eval will be released in the later version
of [OpenCompass 2.0](https://github.com/open-compass/opencompass). This repo only contains the dataset, the backend and
the postprocess code of F-Eval.

<img src="https://s11.ax1x.com/2024/01/26/pFmvkUx.png" width="65%" height="70%">

## Dataset

The statistics of the datasets.

| Dimension   | Sub-dataset         | #Samples English | #Samples Chinese | Task Format                | Settings         |
|-------------|---------------------|------------------|------------------|----------------------------|------------------|
| Expression  | Word Diversity      | 51               | 102              | reference-free subjective  | zero-shot        |
| Expression  | Informative         | 72               | 111              | reference-free subjective  | zero-shot        |
| Expression  | Rule Following      | 66               | 75               | open-ended objective       | zero-shot        |
| Expression  | Emotion Consistency | 70               | 80               | reference-free subjective  | zero-shot        |
| Commonsense | Commonsense Triple  | 84               | 66               | reference-based subjective | few-shot (k=5)   |
| Commonsense | CommonsenseQA       | 74               | 76               | multi-choice objective     | zero-shot        |
| Commonsense | TextbookQA          | 75               | 76               | reference-based subjective | zero-shot        |
| Commonsense | Story               | 75               | 75               | multi-choice objective     | zero-shot        |
| Commonsense | Instruction         | 80               | 70               | reference-based subjective | zero-shot        |
| Logic       | ICL                 | 75               | 75               | open-ended objective       | few-shot (k=0,4) |
| Logic       | COT                 | 80               | 80               | open-ended objective       | zero-shot        |
| Logic       | Fallacy Attack      | 52               | 52               | reference-based subjective | zero-shot        |
| Logic       | Contradiction       | 75               | 75               | reference-free subjective  | zero-shot        |
| Logic       | Coreference         | 57               | 58               | open-ended objective       | few-shot (k=4)   |
| Logic       | Anomaly Detection   | 79               | 75               | multi-choice objective     | zero-shot        |

An example of the rule-following dataset.

```text
Prompt: last chance,last minute,last name,last laugh,last resort
Output: last word,last straw,last minute
```

## Results

Below are the overall results of F-Eval across three dimensions. More details of the results in each sub-dataset can be
found in our [paper](https://arxiv.org/abs/2401.14869).

<img src="https://s11.ax1x.com/2024/01/26/pFnNkwD.png" width="95%" height="95%">

## How to evaluate on F-Eval

### Getting Started

**Step 1. Run the backend server.**

Before running evaluation file in OpenCompass, please ensure a backend server is running.

```shell
python backend/freq_flask.py
```

**Step 2. Prepare the dataset.**

The overall dataset with 2211 samples is in the `data/language_quality` folder. The selected dataset that is used for
analysis is in `data/select_data`.

Please download the dataset from the github repo and put it in the `data` folder under OpenCompass folder.

**Step 3. Run the evaluation file in OpenCompass.**

The main evaluation program entry is `configs/eval_language_quality/eval_language_quality.py` in the later version of
OpenCompass. Detailed usage of evaluation on OpenCompass can be found in
the [OpenCompass](https://github.com/open-compass/opencompass) repo.

```shell
python -u run.py configs/eval_language_quality/eval_language_quality.py -p llm2_t -s -r
```

**Step 4. Postprocess the results.**

After getting original results by OpenCompass, you should first
run `postprocess/merge_results.py` to get the merged results of each dataset (merge English and Chinese).
Then you can run `postprocess/normalize.py` to get the uniform results of each dataset.

```shell
python postprocess/merge_results.py
python postprocess/normalize.py
```

## Citation

If you find this repo useful, please cite with the following bibtex:

```
@misc{sun2024feval,
      title={F-Eval: Asssessing Fundamental Abilities with Refined Evaluation Methods}, 
      author={Yu Sun and Keyu Chen and Shujie Wang and Qipeng Guo and Hang Yan and Xipeng Qiu and Xuanjing Huang and Dahua Lin},
      year={2024},
      eprint={2401.14869},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```