"""
需要安装spacy并下载模型
pip install -U spacy
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from collections import defaultdict
import os.path as osp
from typing import Dict, List, Optional
import spacy
import sys
import numpy as np
import json
from scipy.stats import spearmanr

import re

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detect_main_language(text):
    chinese_characters = len(
        [c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_characters = len(
        [c for c in text if 'a' <= c.lower() <= 'z'])

    if chinese_characters > english_characters:
        return 'zh'
    else:
        return 'en'


@app.route('/subject/emotion', methods=['POST'])
def emotion_cls_evaluator():
    data = request.json
    predictions = data["predictions"]
    references = data["references"]
    # load model and text split
    neg_label = ['anger', 'fear', 'sadness']
    pos_label = ['joy', 'love']
    classifier = pipeline('sentiment-analysis', model="nanaaaa/emotion_chinese_english")

    def load_model(sent: str):
        output = classifier(sent)
        return output[0]['label']

    def get_pos_prob(text):
        sents = nlp(text)
        labels = defaultdict(int)
        pred_label = []
        for sent in sents.sents:
            sent = sent.text
            label = load_model(sent)
            pred_label.append(label)
            if label in neg_label:
                labels['neg'] += 1
            elif label in pos_label:
                labels['pos'] += 1
        if len(pred_label) == 0:
            return 0, pred_label
        pos_prob = labels['pos'] / len(pred_label)
        return pos_prob, pred_label

    # compute score
    consistent = 0
    pred_labels = []
    refer_labels = []
    details = []
    for idx, pred in enumerate(predictions):
        pred_lang = detect_main_language(pred)
        if detect_main_language(references[idx]) != pred_lang:
            details.append({'pred': pred, 'answer': references[idx], 'correct': 0})
            pred_labels.append('wrong')
            refer_labels.append('wrong')
            continue
        if pred_lang == 'en':
            nlp = spacy.load('en_core_web_sm')
        elif pred_lang == 'zh':
            nlp = spacy.load('zh_core_web_sm')
        else:
            raise ValueError("Language should be en or zh.")
        pred_pos_prob, pred_pred_label = get_pos_prob(pred)
        refer_pos_prob, refer_pred_label = get_pos_prob(references[idx])
        if abs(pred_pos_prob - refer_pos_prob) < 0.2:
            consistent += 1
        pred_labels.append(pred_pred_label)
        refer_labels.append(refer_pred_label)
        detail = {
            'pred': pred,
            'answer': references[idx],
            'correct': [pred_pos_prob, refer_pos_prob]
        }
        details.append(detail)

    result = {
        'pos_consistent': consistent / len(predictions),
        'details': details,
        'pred_labels': pred_labels,
        'refer_labels': refer_labels

    }
    return jsonify(result)


@app.route('/subject/contradiction', methods=['POST'])
def contradiction_nli_evaluator():
    data = request.json
    predictions = data["predictions"]
    references = data["references"]

    model_path = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)

    # get contradiction pair
    def get_contradiction_pair(sents):
        """
        Get contradiction pair from a list of sentences.
        """
        if len(sents) <= 1:
            return []
        prefix_sent = sents[0]
        contradiction_pairs = []
        for sent in sents[1:]:
            contradiction_pairs.append([prefix_sent, sent])
            prefix_sent += ' ' + sent
        return contradiction_pairs

    scores = []
    details = []
    for idx, pred in enumerate(predictions):
        # load model and text split
        language = detect_main_language(pred)
        if language == 'en':
            split_model = "en_core_web_sm"
        elif language == 'zh':
            split_model = "zh_core_web_sm"
        else:
            raise ValueError("Language should be en or zh.")
        nlp = spacy.load(split_model)
        label_num = [0, 0, 0]
        # split sentences
        # print(references[idx], pred)
        all_text = references[idx] + ' ' + pred
        sents = nlp(all_text)
        sents = [sent.text for sent in sents.sents]

        # get contradiction pairs
        contradiction_pairs = get_contradiction_pair(sents)

        # compute nli score
        for c_pair in contradiction_pairs:
            inputs = tokenizer(c_pair[0], c_pair[1], truncation=True, return_tensors="pt")
            outputs = model(inputs['input_ids'].to(device))
            prediction = torch.softmax(outputs["logits"][0], -1).tolist()
            label = torch.argmax(outputs["logits"][0]).item()
            label_names = ["entailment", "neutral", "contradiction"]
            prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
            label_num[label] += 1

        detail = {'pred': pred, 'answer': references[idx], 'correct': {}}
        if len(contradiction_pairs) == 0:
            detail['correct']['contradiction'] = 1
            detail['correct']['neutral'] = 0
            detail['correct']['entailment'] = 0
            scores.append(1)
        else:
            detail['correct']['contradiction'] = label_num[2] / len(contradiction_pairs)
            detail['correct']['neutral'] = label_num[1] / len(contradiction_pairs)
            detail['correct']['entailment'] = label_num[0] / len(contradiction_pairs)
            scores.append(label_num[2] / len(contradiction_pairs))
        details.append(detail)
    result = {
        'details': details,
        'contradiction_score': sum(scores) / len(scores),
    }
    return jsonify(result)


def _preprocess(text: str) -> str:
    text = text.strip()
    # 分隔符
    pattern = re.compile(r'[\s\,\;\，\。\t\n]+')
    text = re.split(pattern, text)
    return text[0]


@app.route('/subject/text2word', methods=['POST'])
def text2word_evaluator():
    data = request.json
    predictions = data["predictions"]
    references = data["references"]

    zh_path = 'word.jsonl'
    en_path = 'ecdict.jsonl'
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh_data = [json.loads(line) for line in f]
    with open(en_path, 'r', encoding='utf-8') as f:
        en_data = [json.loads(line) for line in f]

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model = model.to(device)

    scores = []
    details = []
    for idx, pred in enumerate(predictions):
        pred = _preprocess(pred)
        refer = references[idx]
        detail = {'pred': pred, 'answer': refer}
        if pred == refer:
            scores.append(1)
            detail['correct'] = 1
        else:
            lang = detect_main_language(refer)
            if lang == 'zh':
                data = zh_data
            elif lang == 'en':
                data = en_data
            else:
                raise ValueError("Language should be en or zh.")
            words = [line['word'] for line in data]

            if pred not in words:
                scores.append(0)
                detail['correct'] = 0
            else:
                pred_idx = words.index(pred)
                refer_idx = words.index(refer)
                sentences = [data[pred_idx]['explanation'], data[refer_idx]['explanation']]
                embeddings = model.encode(sentences)
                cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
                scores.append(cosine_scores.item())
                detail['correct'] = cosine_scores.item()
        details.append(detail)
    return jsonify({'score': sum(scores) / len(scores), 'details': details})


def ngram_repeat(inp, ref, ngram=7):
    tot_cnt, matched = 0, 0
    for i in range(len(inp) - ngram):
        if inp[i:i + ngram] in ref:
            matched += 1
        tot_cnt += 1
    return matched / (tot_cnt + 1e-6)


def non_chinese_ratio(inp):
    non_chinese_count = 0
    for x in inp:
        if ord(x) < 256:
            non_chinese_count += 1
    return non_chinese_count / (len(inp) + 1e-6)


class Evaluators:
    """
    """

    def __init__(self) -> None:
        assert osp.exists('chinese_char_base.json'), (
            'Word frequency score needs the dictionary file. Please download it.'
        )
        freq_map = {0: 0.0, 1: 0.7, 2: 0.8, 3: 0.9,
                    4: 1.0}  # map the raw frequency data as the higher is more frequent.
        self.chinese_char_freq = {}
        self.chinese_word_freq = {}
        self.english_word_freq = {}
        self.chinese_char_base = {}

        with open('chinese_char_base.json') as f:
            data = json.load(f)
            for item in data:
                self.chinese_char_freq[item['char']] = freq_map.get(item['frequency'], 0)
                self.chinese_char_base[item['char']] = item
        with open('CET4luan_2.json') as f:  # tier 1
            for line in f.readlines():
                if len(line) > 0:
                    item = json.loads(line)
                    if item['headWord'] not in self.english_word_freq:
                        self.english_word_freq[item['headWord']] = 0.0
        with open('CET6_2.json') as f:  # tier 2
            for line in f.readlines():
                if len(line) > 0:
                    item = json.loads(line)
                    if item['headWord'] not in self.english_word_freq:
                        self.english_word_freq[item['headWord']] = 0.05
        with open('Level4luan_2.json') as f:  # tier 3
            for line in f.readlines():
                if len(line) > 0:
                    item = json.loads(line)
                    if item['headWord'] not in self.english_word_freq:
                        self.english_word_freq[item['headWord']] = 0.1
        with open('Level8luan_2.json') as f:  # tier 4
            for line in f.readlines():
                if len(line) > 0:
                    item = json.loads(line)
                    if item['headWord'] not in self.english_word_freq:
                        self.english_word_freq[item['headWord']] = 0.25
        with open('THUOCL.txt', 'r') as f:
            for line in f.readlines():
                if len(line) > 0:
                    if len(line.split()) == 2:
                        _word, _freq = line.split()
                        if int(_freq) > 3491:  # tier 1
                            self.chinese_word_freq[_word] = 0.1
                        elif int(_freq) > 1201:  # tier 2
                            self.chinese_word_freq[_word] = 0.7
                        elif int(_freq) > 288:  # tier 3
                            self.chinese_word_freq[_word] = 0.9
                        else:  # tier 4
                            self.chinese_word_freq[_word] = 1.0

        self.chinese_parser = spacy.load("zh_core_web_sm")
        self.english_parser = spacy.load("en_core_web_sm")
        self.non_chinese_char_score = 0.0  # Used for Chinese character, None for not account, otherwise it's a score in [0, 1]
        self.non_english_word_score = None  # Used for English word, None for not account, otherwise it's a score in [0, 1]

        self.chinese_ppl_tokenizer = AutoTokenizer.from_pretrained(
            "/cpfs01/shared/public/chenkeyu1/models/deepseek-llm-7b-base")
        self.chinese_ppl_model = AutoModelForCausalLM.from_pretrained(
            "/cpfs01/shared/public/chenkeyu1/models/deepseek-llm-7b-base", device_map="auto", trust_remote_code=True)
        self.ppl_pentaly = 10.0

    def score_nll(self, predictions: List, references: Optional[List] = None) -> Dict:
        """test for fluency using the difference of Negative Log-Likelihood between prompt and predction, the idea is that we should response a complex answer to a complex question,
        the amount of information should be at the same level.
        """
        nlls, lens, d_nlls = [], [], []
        for idx, pred in enumerate(predictions):
            _data = json.loads(references[idx])
            pred = pred.strip()
            if len(pred) > 0 and ngram_repeat(pred, _data['prompt'], ngram=7) > 0.5:
                pred = ""
            if non_chinese_ratio(_data['prompt']) < 0.5 and non_chinese_ratio(pred) > 0.5:
                pred = ""
            if non_chinese_ratio(_data['prompt']) > 0.5 and non_chinese_ratio(pred) < 0.5:
                pred = ""

            if len(pred) == 0:
                nlls.append(self.ppl_pentaly)
                d_nlls.append(self.ppl_pentaly)
                lens.append(1)
            else:
                inputs_Fhalf = self.chinese_ppl_tokenizer(_data['prompt'], return_tensors='pt').to(device)
                inputs_Shalf = self.chinese_ppl_tokenizer(_data['prompt'] + pred, return_tensors='pt').to(device)
                ret = self.chinese_ppl_model(input_ids=inputs_Fhalf['input_ids'], labels=inputs_Fhalf['input_ids'])
                ppl_Fhalf = ret[0].item()
                if ppl_Fhalf != ppl_Fhalf:
                    ppl_Fhalf = self.ppl_pentaly
                label_Shalf = inputs_Shalf['input_ids'].clone()
                label_Shalf[0, max(0, inputs_Fhalf['input_ids'].shape[1] - 1)] = -100
                ret = self.chinese_ppl_model(input_ids=inputs_Shalf['input_ids'], labels=label_Shalf)
                ppl_Shalf = ret[0].item()
                if ppl_Shalf != ppl_Shalf:
                    ppl_Shalf = self.ppl_pentaly
                len_diff = inputs_Shalf['input_ids'].shape[1] - inputs_Fhalf['input_ids'].shape[1]
                nlls.append(abs(ppl_Fhalf - ppl_Shalf))
                # nlls.append(_t * inputs['input_ids'].shape[1])
                d_nlls.append(abs(ppl_Fhalf - ppl_Shalf))
                lens.append(1)
        details = [{'pred': '', 'answer': '', 'correct': x} for x in d_nlls]
        return {'score': np.sum(nlls) / np.sum(lens), 'details': details}

    def score_diversity(self, predictions: List, references: Optional[List] = None) -> Dict:
        """ test diversity by considering both the fluency and the self-similarity of generated text by sampling multiple times.
        Ensure the model's generation process has do_sample=True and temperature>0.0
        """
        nlls = self.score_nll(predictions, references)['details']
        d_nlls = []
        uniq = defaultdict(list)
        for idx in range(len(predictions)):
            _data = json.loads(references[idx])
            uniq[_data['prompt']].append(idx)
        for idx in range(len(predictions)):
            if len(predictions[idx]) == 0:
                d_nlls.append(10.0 * self.ppl_pentaly)
            elif nlls[idx]['correct'] >= self.ppl_pentaly:
                d_nlls.append(10.0 * self.ppl_pentaly)
            else:
                _data = json.loads(references[idx])
                rest = ' '.join([predictions[x] for x in uniq[_data['prompt']] if x != idx])
                repeat = np.mean(
                    [ngram_repeat(predictions[idx], rest, ngram=2), ngram_repeat(predictions[idx], rest, ngram=3),
                     ngram_repeat(predictions[idx], rest, ngram=4)])
                d_nlls.append(10.0 * nlls[idx]['correct'] * repeat)
        details = [{'pred': '', 'answer': '', 'correct': x} for x in d_nlls]
        return {'score': np.mean(d_nlls), 'details': details}

    def score_rules(self, predictions: List, references: Optional[List] = None) -> Dict:
        """test the ability of following rules"""

        def _score(pred, _data):
            ret = []
            if _data['task'] == 'first_voice':
                first_word = [x.text for x in self.chinese_parser(pred.strip())][0]
                if len(first_word) > 0 and first_word not in _data['prompt'] and \
                        self.chinese_char_base.get(first_word[0], {}).get('pinyin', [None])[0] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'last_voice':
                first_word = [x.text for x in self.chinese_parser(pred.strip())][0]
                if len(first_word) > 0 and first_word not in _data['prompt'] and \
                        self.chinese_char_base.get(first_word[0], {}).get('pinyin', [None])[0] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'first_char':
                first_word = [x.text for x in self.chinese_parser(pred.strip())][0]
                if len(first_word) > 0 and first_word not in _data['prompt'] and first_word[0] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'last_char':
                first_word = [x.text for x in self.chinese_parser(pred.strip())][0]
                if len(first_word) > 0 and first_word not in _data['prompt'] and first_word[-1] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'radicals':
                if pred[0] in self.chinese_char_base and pred[0] not in _data['prompt'] and \
                        self.chinese_char_base[pred[0]]['radicals'] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'sent_begin':
                if ngram_repeat(pred, _data['prompt'], ngram=7) > 0.7:
                    pred = ""
                if len(pred) > len(_data['ans']) and pred.startswith(_data['ans']):
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'sent_end':
                if ngram_repeat(pred, _data['prompt'], ngram=7) > 0.7:
                    pred = ""
                if len(pred) > len(_data['ans']) and pred[-len(_data['ans']):] == _data['ans']:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'sent_mix':
                if ngram_repeat(pred, _data['prompt'], ngram=7) > 0.7:
                    pred = ""
                if len(pred) > len(_data['ans1']) + len(_data['ans2']) and pred[:len(_data['ans1'])] == _data[
                    'ans1'] and pred[-len(_data['ans2']):] == _data['ans2']:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            return ret[0]

        ret = []
        for idx in range(len(predictions)):
            if len(predictions[idx].strip()) < 1:
                ret.append(0.0)
                continue
            _data = json.loads(references[idx])
            if 'ans' in _data:
                ans = _data['ans']
            pred = predictions[idx].strip()
            if _score(pred, _data) > 0.0:
                ret.append(1.0)
            else:
                for prefix in ['下一个是', '下一个是：', '根据给出的文本规律，下一个词语应该是']:
                    if pred.startswith(prefix):
                        pred = pred[len(prefix):]
                for prefix in ['**', '“', '"']:
                    if prefix in pred:
                        pred = pred[pred.find(prefix) + len(prefix):]
                if len(pred) == 0:
                    ret.append(0.0)
                else:
                    ret.append(_score(pred, _data))

        details = [{'pred': '', 'answer': '', 'correct': x} for x in ret]
        return {'score': np.mean(ret) if len(ret) > 0 else 0.0, 'details': details}

    def score_word_freq(self, predictions: List, references: Optional[List] = None) -> Dict:
        """test word diveristy of generated text by looking the generated chinese characters, words, and english words."""
        freq_samples = []
        avg_len = int(np.mean([len(self.chinese_parser.tokenizer(x)) for x in predictions if len(x) > 0]))
        for idx, pred in enumerate(predictions):
            _data = json.loads(references[idx])
            if ngram_repeat(pred, _data['prompt'], ngram=7) > 0.7:
                pred = ""

            _freq = []
            if non_chinese_ratio(_data['prompt']) < 0.5:
                if non_chinese_ratio(pred) > 0.5:
                    pred = ""
                else:
                    for word in [x.text for x in self.chinese_parser.tokenizer(pred)]:
                        if word in self.chinese_word_freq:
                            _freq.append(self.chinese_word_freq[word])
                        else:
                            _char_freqs = []
                            for char in word:
                                if char in self.chinese_char_freq:
                                    _char_freqs.append(self.chinese_char_freq[char])
                                else:
                                    if self.non_chinese_char_score is not None:
                                        _char_freqs.append(self.non_chinese_char_score)
                            if len(_char_freqs) > 0:
                                _freq.append(np.mean(_char_freqs))
            else:
                if non_chinese_ratio(pred) < 0.5:
                    pred = ""
                else:
                    for word in [x.text for x in self.english_parser.tokenizer(pred)]:
                        if word in self.english_word_freq:
                            _freq.append(self.english_word_freq[word])
                        else:
                            if self.non_english_word_score is not None:
                                _freq.append(self.non_english_word_score)

            if len(_freq) == 0:
                _freq = [0.0] * avg_len
            freq_samples.append(_freq)

        details = [{'pred': '', 'answer': '', 'correct': x} for x in
                   [(np.mean(x) * 100.0 if len(x) > 0 else 0.0) for x in freq_samples]]
        return {'score': 100.0 * np.mean(sum(freq_samples, [])), 'details': details}


evaluator = Evaluators()


@app.route('/', methods=['POST'])
def index():
    data = request.json
    if 'references' not in data or len(data.get('references')) == 0:
        return jsonify({'score': None})
    else:
        predictions = data.get('predictions')
        references = data.get('references')
        _data = json.loads(references[0])
        if _data['task'] == 'word_diversity':
            ret = evaluator.score_word_freq(predictions, references)
        if _data['task'] in ['first_voice', 'last_voice', 'first_char', 'last_char', 'sent_begin', 'sent_end',
                             'sent_mix']:
            ret = evaluator.score_rules(predictions, references)
        if _data['task'] == 'diversity':
            ret = evaluator.score_diversity(predictions, references)
        if _data['task'] == 'informative':
            ret = evaluator.score_nll(predictions, references)

    return jsonify(ret)


if __name__ == "__main__":
    # CORS(app)
    CORS(app, resources=r'/*')
    app.run(port=5001, debug=False, host='0.0.0.0')
