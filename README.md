# F-Eval: Asssessing Fundamental Abilities with Refined Evaluation Methods

[[paper](https://arxiv.org/abs/2401.14869)] [Code in OpenCompass coming soon]

F-Eval is a bilingual evaluation benchmark to evaluate the fundamental abilities, including expression, commonsense and
logic.

## Dataset

Overall and selected dataset in F-Eval.

<img src="https://s11.ax1x.com/2024/01/26/pFmvkUx.png" width="70%" height="70%">

## Results

Below are the overall results of F-Eval.

<img src="https://s11.ax1x.com/2024/01/26/pFnNkwD.png" width="95%" height="95%">

## How to run

### Backend

Evaluators for Word Diversity, Informative, Rule Following, Emotion Consistency and Contradiction.

### Postprocess

Postprocess of the results computed by Opencompass.

### Getting Started

You need to run the `backend\freq_flask.py` first for evaluation.

Other code of F-Eval is released in the later version of Opencompass. After getting original results, you can
run `postprocess\merge_results.py` and `postprocess\normalize_results.py` to get the uniform results of each dataset.
Then you can run `postprocess\normalize.py` to get the final results.