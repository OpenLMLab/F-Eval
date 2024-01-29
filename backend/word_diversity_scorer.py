"""
需要安装spacy并下载模型
pip install -U spacy
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
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
        freq_map = {0:0.0, 1: 0.7, 2: 0.8, 3:0.9, 4: 1.0} # map the raw frequency data as the higher is more frequent. 
        self.chinese_char_freq = {}
        self.chinese_word_freq = {}
        self.english_word_freq = {}

        with open('chinese_char_base.json') as f:
            data = json.load(f)
            for item in data:
                self.chinese_char_freq[item['char']] = freq_map.get(item['frequency'], 0)
        with open('CET4luan_2.json') as f: # tier 1
            for line in f.readlines():
                if len(line)>0:
                    item = json.loads(line)
                    self.english_word_freq[item['headWord']] = 0.1
        with open('CET6_2.json') as f: # tier 2
            for line in f.readlines():
                if len(line)>0:
                    item = json.loads(line)
                    self.english_word_freq[item['headWord']] = 0.7
        with open('Level4luan_2.json') as f: # tier 3
            for line in f.readlines():
                if len(line)>0:
                    item = json.loads(line)
                    self.english_word_freq[item['headWord']] = 0.9
        with open('Level8luan_2.json') as f: # tier 4
            for line in f.readlines():
                if len(line)>0:
                    item = json.loads(line)
                    self.english_word_freq[item['headWord']] = 1.0
        with open('THUOCL.txt', 'r') as f:
            for line in f.readlines():
                if len(line)>0:
                    if len(line.split())==2:
                        _word, _freq = line.split()
                        if int(_freq)>3491: # tier 1
                            self.chinese_word_freq[_word] = 0.1
                        elif int(_freq)>1201: # tier 2
                            self.chinese_word_freq[_word] = 0.7
                        elif int(_freq)>288: # tier 3
                            self.chinese_word_freq[_word] = 0.9
                        else: # tier 4
                            self.chinese_word_freq[_word] = 1.0

        self.chinese_parser = spacy.load("zh_core_web_sm")
        self.english_parser = spacy.load("en_core_web_sm")
        self.non_chinese_char_score = 0.0 # Used for Chinese character, None for not account, otherwise it's a score in [0, 1]
        self.non_english_word_score = None # Used for English word, None for not account, otherwise it's a score in [0, 1] 


    def score_word_freq(self, predictions: List, references: Optional[List] = None) -> Dict:
        """test word diveristy of generated text by looking the generated chinese characters, words, and english words."""
        freq_samples = []
        avg_len = int(np.mean([len(self.chinese_parser.tokenizer(x)) for x in predictions if len(x)>0]))
        for idx, pred in enumerate(predictions):
            _data = json.loads(references[idx])
            if ngram_repeat(pred, _data['prompt'], ngram=7)>0.7:
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
                            if len(_char_freqs)>0:
                                _freq.append(np.mean(_char_freqs))
            else:
                if non_chinese_ratio(pred) < 0.5:
                    pred = ""
                else:
                    for word in [x.text for x in self.english_parser.tokenizer(pred)]:
                        if word in self.word_freq:
                            _freq.append(self.word_freq[char])
                        else:
                            if self.non_english_word_score is not None:
                                _freq.append(self.non_english_word_score)

            if len(_freq)==0:
                _freq = [0.0] * avg_len
            freq_samples.append(_freq)
            
        details = [{'pred': '', 'answer': '', 'correct': x} for x in [(np.tanh(5*np.mean(x)) if len(x)>0 else 0.0) for x in freq_samples]]
        return {'score': np.tanh(5*np.mean(sum(freq_samples, []))), 'details': details}

evaluator = Evaluators()

if __name__ == "__main__":
    predictions = ["物质上富有而精神上贫穷。物质上贫穷而精神上富有，是“穷且益坚，不坠青云之志”。物质上富有而精神上贫穷，是“金玉其外，败絮其中”。物质上贫穷而精神上富有，是“穷且益坚，不坠青云之志”。物质上富有而精神上贫穷，是“金玉其外，败絮其中”。物质上贫穷而精神上富有，是“穷且益坚，不坠青云"]
    references = ["{\"task\": \"word_diversity\", \"prompt\": \"以上是两种较自然，让人较容易理解的状态。最让人玩味的，是另外两种。物质上贫穷而精神上富有。\"}"]
    ret = evaluator.score_word_freq(predictions, references)
    print(ret)

