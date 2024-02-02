# Backends

Evaluators for Word Diversity, Informative, Rule Following, Emotion Consistency and Contradiction.

## Getting started

Dependent on spaCy's tokenization, please first install the Chinese language tokenization package.

The code for testing the model's ability to identify patterns and generate outputs based on these
patterns. [code](rule_scorer.py)

Testing the richness of language in texts, generally speaking, classical literature > modern poetry and prose > news
articles > online novels and daily conversations.[code](word_diversity_scorer.py)

Currently, evaluations require the provision of a reference, that is, reference information, to prevent models from
directly copying the input prompts. If the test is purely textual without a corresponding model, the prompts for the
reference can be replaced with nonsensical strings, such as 'one two three four five' and so on. Note that the program
will perform language detection; if the prompt is in Chinese, it will mandatorily require the input text to be in
Chinese as well.

## Running the backend

This part of the [code](freq_flask) utilizes Flask as the backend. The command to start the backend is as follows. After
starting the backend, you should copy the address of the backend and set export OPENCOMPASS_LQ_BACKEND = 'address' in
the terminal. The backend will be running on port 5000 by default. If you want to change the port, you can modify the
code in freq_flask.py.

```bash
python freq_flask.py
```
