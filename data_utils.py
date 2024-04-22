import random
from torch.utils.data import Dataset
import os
import copy
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
import re

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def split_string(line, split_symbol):
    """
    :param line: a string need be split
    :param split_symbol: a string: split symbol
    :return:
    """
    return list(filter(None, line.split(split_symbol)))


def clear_string(line, strip_symbol=None, replace_symbol=None):
    """
    :param line: a string
    :param strip_symbol:
    :param replace_symbol: a list of special symbol, need replace.
    :return:
    """
    if strip_symbol is not None:
        for sym in strip_symbol:
            line = line.strip(sym)

    if replace_symbol is not None:
        for sym in replace_symbol:
            line = line.replace(sym, "")

    return line
    
def read_line_examples_from_file(path):
    """
    :param path:
    :return: sent_col, sent_label_col and label_col
    """
    sent_col, sent_label_col, final_label_col = [], [], []
    last_sentence = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.rstrip('\n')

            # "[[" denote the begin of sequence label.
            if line[:2] == "[[":
                #  "[[16&&D80];[22&&D200];[];[4&&hard, 17&&ca 18&&n't 19&&do];[0]]"
                line = ''.join(line) 
                label_col.append(line)

            else:
                if last_sentence != "":
                    cur_sent, cur_sent_label = split_string(last_sentence, "\t")
                    sent_col.append(cur_sent)
                    sent_label_col.append(int(cur_sent_label))
                    final_label_col.append(label_col)

                last_sentence = clear_string(line, replace_symbol={u'\u3000': u""})
                label_col = []

        cur_sent, cur_sent_label = split_string(last_sentence, "\t")
        sent_col.append(cur_sent)
        sent_label_col.append(int(cur_sent_label))
        final_label_col.append(label_col)

    # if silence:
    #     print(f"Total examples = {len(sent_col)}")

        return sent_col, sent_label_col, final_label_col
        
def rewrite_final_label(final_label_col):
    new_final_labe_col = []
    for label_col in final_label_col:
        new_label_col = []
        for label in label_col:
            input_string = label
            ####ch:& en:&&
            output_string = re.sub(r'\d+&', '', input_string)
            label_pattern = r'\[\[?([^]]*)\]' 
            labels = re.findall(label_pattern, output_string)

            # 处理标签信息，将其转换为列表格式
            label_list = []
            for label in labels:
                # 将标签内容按逗号和空格分割，并去除空格
                processed_label = ' '.join([text.strip() for text in label.split(',')])
                label_list.append(processed_label)
            new_label_col.append(label_list)
        new_final_labe_col.append(new_label_col)
    return new_final_labe_col
    
pr2naturalword = {'0':'same','1':'better', '-1':'worse','2':'different'}
pr2naturalword_ch = {'0':'相同','1':'更好', '-1':'更差','2':'不同'}

def get_para_asqp_targets(sent_col, sent_label_col, final_label_col):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for i in range(len(sent_col)):
        all_quad_sentences = []
        if sent_label_col[i] == 1:
            for quad in final_label_col[i]:
                sub = quad[0]
                ob = quad[1]
                ap = quad[2]
                op = quad[3]
                pr = quad[4]
                
                man_pr = pr2naturalword[pr]
                
                if sub == '': sub = 'it'
                if ob == '': ob = 'others'
                if ap == '': ap = 'some aspect'

                
                one_quad_sentence = f"{sub} is {man_pr} to {ob} because {ap} is {op}"
                # one_quad_sentence = f"{sub} is {pr} to {ob} because {ap} is {op}"
                all_quad_sentences.append(one_quad_sentence)
                
            target = ' [SSEP] '.join(all_quad_sentences)
            targets.append(target)
        else:
            target = 'it is not a comparative sentence'
            targets.append(target)

    return targets
    
def get_para_asqp_targets_ch(sent_col, sent_label_col, final_label_col):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for i in range(len(sent_col)):
        all_quad_sentences = []
        if sent_label_col[i] == 1:
            for quad in final_label_col[i]:
                sub = quad[0]
                ob = quad[1]
                ap = quad[2]
                op = quad[3]
                pr = quad[4]
                
                man_pr = pr2naturalword_ch[pr]
                
                if sub == '': sub = '它'
                if ob == '': ob = '其他产品'
                if ap == '': ap = '在某方面'

                
                one_quad_sentence = f"{sub}相比{ob}是{man_pr}因为{ap}是{op}"
                # one_quad_sentence = f"{sub}相比{ob}是{pr}因为{ap}是{op}"
                all_quad_sentences.append(one_quad_sentence)
                
            target = ' [SSEP] '.join(all_quad_sentences)
            targets.append(target)
        else:
            target = '这不是对比句'
            targets.append(target)

    return targets

def get_transformed_io(data_path):
    """
    The main function to transform input & target according to the task
    """
    sents, sent_label_col, final_label_col = read_line_examples_from_file(data_path)
    new_final_label_col = rewrite_final_label(final_label_col)
    

    # the input is just the raw sentence
    # inputs = [list(s).copy() for s in sents]

    task = 'ch'
    if task == 'asqp':
        targets = get_para_asqp_targets(sents, sent_label_col, new_final_label_col)
    elif task =='ch':
        targets = get_para_asqp_targets_ch(sents,sent_label_col,new_final_label_col)
    else:
        raise NotImplementedError

    return sents, sent_label_col, targets, new_final_label_col

class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        # './data/rest16/train.txt'
        self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir

        self.inputs = []
        self.targets = []

        self.contrastive_labels = {}
        self.sentence_strings = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        preference_label = torch.tensor(self.contrastive_labels['preference'][index])
        aspect_label = torch.tensor(self.contrastive_labels['aspect'][index])
        opinion_label = torch.tensor(self.contrastive_labels['opinion'][index])
        subject_label = torch.tensor(self.contrastive_labels['subject'][index])
        object_label = torch.tensor(self.contrastive_labels['object'][index])
        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask,
                'preference_labels':preference_label, 'opinion_labels': opinion_label,
                'aspect_labels': aspect_label,'subject_labels':subject_label,
                'object_labels':object_label,}

    def _build_examples(self):

        inputs, sent_label_col, targets,new_final_label_col = get_transformed_io(self.data_path)
        self.sentence_strings = inputs
        # print('sent_label_col',sent_label_col)
        # print('new_final_label_col',new_final_label_col)

        for i in range(len(inputs)):
            # change input and target to two strings
            # input = ' '.join(inputs[i])
            target = targets[i]
            input = inputs[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

            def get_preference_labels(labels_in):
                preference_dict = {
                    '': 3, #####non-comparative
                    '0': 0,
                    '1': 1,
                    '-1': -1,
                    '2': 2
                }
                preference_labels = []
                for ex in labels_in:
                    label = list(set([quad[4] for quad in ex]))
                    if len(label) == 1:
                        label = preference_dict[label[0]]
                    else:
                        label = preference_dict['2']
                    assert label in [0,1,-1,2,3]
                    preference_labels.append(label)
                from collections import Counter
                # print("preference_labels distribution")
                # print(Counter(preference_labels))
                return preference_labels

            def get_opinion_labels(labels_in):
                opinion_dict = {
                    'NULL': 0,
                    'EXPLICIT': 1,
                    'BOTH': 2,
                }
                opinion_labels = []
                for ex in labels_in:
                    opinions = set([quad[3] for quad in ex])

                    if 'NULL' not in opinions:
                        label = opinion_dict['EXPLICIT']
                    else:
                        if len(opinions) == 1:
                            label = opinion_dict['NULL']
                        else:
                            label = opinion_dict['BOTH']

                    opinion_labels.append(label)
                return opinion_labels

            def get_aspect_labels(labels_in):
                aspect_dict = {
                    '': 0,
                    'EXPLICIT': 1,
                    'BOTH': 2,
                }
                aspect_labels = []
                for ex in labels_in:
                    aspects = set([quad[2] for quad in ex])

                    if 'NULL' not in aspects:
                        label = aspect_dict['EXPLICIT']
                    else:
                        if len(aspects) == 1:
                            label = aspect_dict['NULL']
                        else:
                            label = aspect_dict['BOTH']

                    aspect_labels.append(label)
                return aspect_labels

            def get_subject_labels(labels_in):
                subject_dict = {
                    '': 0,
                    'EXPLICIT': 1,
                    'BOTH': 2,
                }
                subject_labels = []
                for ex in labels_in:
                    subjects = set([quad[0] for quad in ex])

                    if 'NULL' not in subjects:
                        label = subject_dict['EXPLICIT']
                    else:
                        if len(subjects) == 1:
                            label = subject_dict['NULL']
                        else:
                            label = subject_dict['BOTH']

                    subject_labels.append(label)
                return subject_labels

            def get_object_labels(labels_in):
                object_dict = {
                    '': 0,
                    'EXPLICIT': 1,
                    'BOTH': 2,
                }
                object_labels = []
                for ex in labels_in:
                    objects = set([quad[1] for quad in ex])

                    if 'NULL' not in objects:
                        label = object_dict['EXPLICIT']
                    else:
                        if len(objects) == 1:
                            label = object_dict['NULL']
                        else:
                            label = object_dict['BOTH']

                    object_labels.append(label)
                return object_labels

            self.contrastive_labels['preference'] = get_preference_labels(new_final_label_col)
            self.contrastive_labels['opinion'] = get_opinion_labels(new_final_label_col)
            self.contrastive_labels['aspect'] = get_aspect_labels(new_final_label_col)
            self.contrastive_labels['subject'] = get_subject_labels(new_final_label_col)
            self.contrastive_labels['object'] = get_object_labels(new_final_label_col)