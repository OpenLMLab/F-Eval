# *F-Eval: Asssessing Fundamental Abilities with Refined Evaluation Methods*

F-Eval is a bilingual evaluation benchmark to evaluate the fundamental abilities, including expression, commonsense and
logic. It consists of 2211 instances in both English and Chinese. Please visit
our [paper](https://arxiv.org/abs/2401.14869) for more details.

The code of F-Eval will be released in the later version
of [OpenCompass 2.0](https://github.com/open-compass/opencompass). This repo only contains the dataset, the backend and
the postprocess code of F-Eval.

<img src="https://s11.ax1x.com/2024/01/26/pFmvkUx.png" width="63%" height="63%">

## Dataset

The statistics of the datasets.

<div style="text-align:center;">
<img src="https://s11.ax1x.com/2024/03/04/pFDA1OJ.png" width="90%" height="90%">
</div>

An example of the rule-following dataset.

```text
Prompt: last chance,last minute,last name,last laugh,last resort
Output: last word,last straw,last minute
```

## Results

Below are the overall results of F-Eval across three dimensions. More details of the results in each sub-dataset can be
found in our [paper](https://arxiv.org/abs/2401.14869).

<img src="https://s11.ax1x.com/2024/01/26/pFnNkwD.png" width="95%" height="95%">

The following is a comparison of the correlation coefficients between subjective evaluation methods used in F-Eval and
other subjective evaluation methods.

<div style="text-align:center;">
<img src="https://s11.ax1x.com/2024/03/04/pFDANY6.png" width="90%" height="90%">
</div>

## How to evaluate on F-Eval

### Getting Started

**Step 1. Prepare the dataset.**

The overall dataset with 2211 samples is in the `data/f_eval` folder. The selected dataset that is used for
analysis is in `data/select_data`.

Please download the dataset from the github repo and put it in the `data` folder under OpenCompass folder.

**Step 2. Run the backend server.**

Before running evaluation files in OpenCompass, please ensure a backend server is running.

```shell
python backend/freq_flask.py
```

**Step 3. Run the evaluation file in OpenCompass.**

The main evaluation python files are in the `configs/eval_f_eval` folder in
OpenCompass. `f_eval_api.py` is used to evaluate the reference-based subjective datasets which are evaluated by API
models. `f_eval_other.py` is used to evaluate the other datasets.

You can directly run the following commands to get the results of F-Eval. Detailed usage of evaluation on OpenCompass
can be found in the [OpenCompass](https://github.com/open-compass/opencompass) repo.

```shell
python -u run.py configs/eval_f_eval/f_eval_api.py -s -r
python -u run.py configs/eval_f_eval/f_eval_other.py -s -r
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