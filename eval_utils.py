# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re
naturalword2pr = {'same':'0','better':'1', 'worse':'-1','different':'2'}
naturalword2pr_ch = {'相同':'0','更好':'1', '更差':'-1','不同':'2'}

def extract_spans_para(task, seq,seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    
    if task == 'asqp':
        for s in sents:
            # Nikon camera bodies is better to one that Canon provides because quality is higher.
            # sub is pr to ob because ap is op
            # 
            try:
                if s == 'it is not a comparative sentence':
                    sub, ob, ap, op, pr = '', '', '', '',''
                else:
                    sub_pr_ob, ap_op = s.split(' because ')
                    sub, pr_ob = sub_pr_ob.split(' is ')
                    npr, ob = pr_ob.split(' to ')
                    ap , op = ap_op.split(' is ')
                    pr = naturalword2pr[npr]
                    # pr = npr

                    # if the aspect term is implicit
                    if sub == 'it': sub = ''
                    if ob == 'others': ob = ''
                    if ap == 'some aspect': ap = ''

            except (ValueError,KeyError):
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                sub, ob, ap, op,pr = '', '', '', '',''

            quads.append((sub, ob, ap, op,pr))
    elif task == 'ch':
        for s in sents:
            # Nikon camera bodies is better to one that Canon provides because quality is higher.
            # sub is pr to ob because ap is op
            # 
            try:
                if s == '这不是对比句':
                    sub, ob, ap, op, pr = '', '', '', '',''
                else:
                    sub_ob_pr, ap_op = s.split('因为')
                    sub, ob_pr = sub_ob_pr.split('相比')
                    ob, npr = ob_pr.split('是')
                    ap , op = ap_op.split('是')
                    pr = naturalword2pr_ch[npr]
                    # pr = npr

                    # if the aspect term is implicit
                    if sub == '它': sub = ''
                    if ob == '其他产品': ob = ''
                    if ap == '某方面': ap = ''

            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                sub, ob, ap, op,pr = '', '', '', '',''

            quads.append((sub, ob, ap, op,pr))
    else:
        raise NotImplementedError
    return quads


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para('asqp', gold_seqs[i], 'gold')
        pred_list = extract_spans_para('asqp', pred_seqs[i], 'pred')

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels)
    print(scores)

    return scores, all_labels, all_preds
