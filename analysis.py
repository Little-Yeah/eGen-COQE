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

        sub_implicit_num = 0
        ob_implicit_num = 0
        ap_implicit_num = 0
        total_num = 0
        op_implicit_num =0

        for i in range(len(sent_col)):
            all_quad_sentences = []

            if sent_label_col[i] == 1:
                for quad in final_label_col[i]:
                    input_string = quad
            ####ch:& en:&&
                    output_string = re.sub(r'\d+&', '', input_string)
                    label_pattern = r'\[\[?([^]]*)\]' 
                    labels = re.findall(label_pattern, output_string)
                    # print(labels)
                    sub = labels[0]
                    ob = labels[1]
                    ap = labels[2]
                    op = labels[3]
                    pr = labels[4]
                    total_num +=1

                    if sub == '': sub_implicit_num +=1
                    if ob == '': ob_implicit_num +=1
                    if ap == '': ap_implicit_num +=1
                    if op == '': op_implicit_num +=1
            else:
                print("44444444")
        print("total_num",total_num)
        print('sub_implicit_num',sub_implicit_num,sub_implicit_num/total_num)
        print('ob_implicit_num',ob_implicit_num,ob_implicit_num/total_num)
        print('ap_implicit_num',ap_implicit_num,ap_implicit_num/total_num)
        print('op_implicit_num',op_implicit_num,op_implicit_num/total_num)


        # total_examples = len(sent_col)
        # total_num = 0
        # for label_col in final_label_col:
        #     total_num +=len(label_col)
        #     new_label_col = []
        #     for label in label_col:
        #         input_string = label
        #         ####ch:& en:&&
        #         output_string = re.sub(r'\d+&&', '', input_string)
        #         label_pattern = r'\[\[?([^]]*)\]' 
        #         labels = re.findall(label_pattern, output_string)

        #         # 处理标签信息，将其转换为列表格式
        #         label_list = []
        #         for label in labels:
        #             # 将标签内容按逗号和空格分割，并去除空格
        #             processed_label = ' '.join([text.strip() for text in label.split(',')])
        #             label_list.append(processed_label)
        #             total_num +=1
        


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
            output_string = re.sub(r'\d+&&', '', input_string)
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
                total_num +=1

                
                if sub == '': sub_implicit_num +=1
                if ob == '': ob_implicit_num +=1
                if ap == '': ap_implicit_num +=1



if __name__ == '__main__':
    read_line_examples_from_file(f'data/Camera-COQE/total.txt')

