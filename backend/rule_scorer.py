"""
需要安装spacy并下载模型
pip install -U spacy
python -m spacy download zh_core_web_sm
"""

from collections import defaultdict
import os.path as osp
from typing import Dict, List, Optional
import spacy 
import numpy as np
import json

def ngram_repeat(inp, ref, ngram=7):
    tot_cnt, matched = 0, 0
    for i in range(len(inp)-ngram):
        if inp[i:i+ngram] in ref:
            matched += 1
        tot_cnt += 1
    return matched/(tot_cnt+1e-6)

def non_chinese_ratio(inp):
    non_chinese_count = 0
    for x in inp:
        if ord(x)<256:
            non_chinese_count += 1
    return non_chinese_count/(len(inp)+1e-6)

class Evaluators:
    """
    """

    def __init__(self) -> None:
        assert osp.exists('chinese_char_base.json'), (
            'Word frequency score needs the dictionary file. Please download it.'
        )
        self.chinese_char_base = {}

        with open('chinese_char_base.json') as f:
            data = json.load(f)
            for item in data:
                self.chinese_char_base[item['char']] = item

        self.chinese_parser = spacy.load("zh_core_web_sm")

    def score_rules(self, predictions: List, references: Optional[List] = None) -> Dict:
        """test the ability of following rules"""
        ret = []
        for idx in range(len(predictions)):
            if len(predictions[idx].strip())<1:
                ret.append(0.0)
                continue    
            _data = json.loads(references[idx])
            if 'ans' in _data:
                ans = _data['ans']
            if _data['task'] == 'first_voice':
                pred = predictions[idx]
                first_word = [x.text for x in self.chinese_parser(pred.strip())][0]
                if len(first_word)>0 and first_word not in _data['prompt'] and self.chinese_char_base.get(first_word[0], {}).get('pinyin', [None])[0] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'last_voice':
                pred = predictions[idx]
                first_word = [x.text for x in self.chinese_parser(pred.strip())][0]
                if len(first_word)>0 and first_word not in _data['prompt'] and self.chinese_char_base.get(first_word[0], {}).get('pinyin', [None])[0] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'first_char':
                pred = predictions[idx]
                first_word = [x.text for x in self.chinese_parser(pred.strip())][0]
                if len(first_word)>0 and first_word not in _data['prompt'] and first_word[0] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'last_char':
                pred = predictions[idx]
                first_word = [x.text for x in self.chinese_parser(pred.strip())][0]
                if len(first_word)>0 and first_word not in _data['prompt'] and first_word[-1] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'radicals':
                pred = predictions[idx]
                if pred[0] in self.chinese_char_base and pred[0] not in _data['prompt'] and self.chinese_char_base[pred[0]]['radicals'] == ans:
                    ret.append(1.0)
                else:
                    ret.append(0.0) 
            if _data['task'] == 'sent_begin':
                pred = predictions[idx]
                if ngram_repeat(pred, _data['prompt'], ngram=7)>0.7:
                    pred = ""
                if len(pred)>len(_data['ans']) and pred.startswith(_data['ans']):
                    ret.append(1.0)
                else:
                    ret.append(0.0)                    
            if _data['task'] == 'sent_end':
                pred = predictions[idx]
                if ngram_repeat(pred, _data['prompt'], ngram=7)>0.7:
                    pred = ""
                if len(pred)>len(_data['ans']) and pred[-len(_data['ans']):] == _data['ans']:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
            if _data['task'] == 'sent_mix':
                pred = predictions[idx]
                if ngram_repeat(pred, _data['prompt'], ngram=7)>0.7:
                    pred = ""
                if len(pred)>len(_data['ans1'])+len(_data['ans2']) and pred[:len(_data['ans1'])] == _data['ans1'] and pred[-len(_data['ans2']):] == _data['ans2']:
                    ret.append(1.0)
                else:
                    ret.append(0.0)
                          
        details = [{'pred': '', 'answer': '', 'correct': x} for x in ret]
        return {'score': np.mean(ret) if len(ret)>0 else 0.0, 'details': details}

    
evaluator = Evaluators()

if __name__ == "__main__":
    predictions = ["衣锦食肉 衣锦褧衣"]
    references = ["{\"task\": \"first_voice\", \"prompt\": \"一一 伊于何底 依丱附木 医书 壹体 衣不兼彩\", \"ans\": \"yī\"}"]
    ret = evaluator.score_rules(predictions, references)
    print(ret)

