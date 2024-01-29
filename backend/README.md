# Backends

Evaluators for Word Diversity, Informative, Rule Following, Emotion Consistency and Contradiction.

## Getting started

Dependent on spaCy's tokenization, please first install the Chinese language tokenization package.

The code for testing the model's ability to identify patterns and generate outputs based on these patterns. [code](rule_scorer.py)

esting the richness of language in texts, generally speaking, classical literature > modern poetry and prose > news articles > online novels and daily conversations.[code](word_diversity_scorer.py)


Currently, evaluations require the provision of a reference, that is, reference information, to prevent models from directly copying the input prompts. If the test is purely textual without a corresponding model, the prompts for the reference can be replaced with nonsensical strings, such as 'one two three four five' and so on. Note that the program will perform language detection; if the prompt is in Chinese, it will mandatorily require the input text to be in Chinese as well.



## Running with OpenCompass

This part of the [code](freq_flask) utilizes Flask as the backend, and the code for the opencompass part will be released in other repositories. It also includes several other types of evaluations. The command to start the backend is as follows.

```bash
python freq_flask.py
#srun -p llm2_t --ntasks=1 --ntasks=1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:4 --job-name=gqp_test_1 --quotatype=reserved --kill-on-bad-exit=1 python freq_flask.py
```
