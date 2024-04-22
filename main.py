# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
import numpy as np

import json
from torch import nn
from torch.nn.functional import normalize
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.nn import CrossEntropyLoss
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from losses import SupConLoss
logger = logging.getLogger(__name__)
# DEVICE = f'cuda:{0}'
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='Camera-COQE', type=str, required=True,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--num_beams", type=int, required=True)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--cont_loss", type=float, default=0.0)
    parser.add_argument("--cont_temp", type=float, default=0.1)


    args = parser.parse_args()

    params = [['beams', str(args.num_beams)],
              ['wd', str(args.weight_decay)],
              ['max_epochs', str(args.num_train_epochs)],
              ['es', str(args.early_stopping)],
              ['acc', str(args.gradient_accumulation_steps)],
              ['lr', str(args.learning_rate)],
              ['cont_loss', str(args.cont_loss)],
              ['cont_temp', str(args.cont_temp)],# whether to truncate the category labels
              ['seed', str(args.seed)]]

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"outputs/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                       data_type=type_path, max_len=args.max_seq_length)


"""
Uncomment for tsne logging
tsne_dict = {
             'sentiment_vecs': [],
             'opinion_vecs': [],
             'aspect_vecs': [],
             'sentiment_labels': [],
             'opinion_labels': [],
             'aspect_labels': []
             }
"""

tsne_dict = {
             'preference_vecs': [],
             'subject_vecs': [],
             'object_vecs':[],
             'aspect_vecs': [],
             'preference_labels': [],
             'subject_labels': [],
             'object_labels': [],
             'aspect_labels':[]
             }

class LinearModel(nn.Module):
    """
    Linear models used for the aspect/opinion/sentiment-specific representations
    """
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        """
        Returns an encoding of input X and a simple dropout-perturbed version of X
        For use in the SupConLoss calculation
        """
        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features_summed = torch.sum(last_state, dim=1)
        dropped = self.dropout(features_summed)
        return torch.stack((self.layer_1(features_summed), self.layer_1(dropped)), 1)

class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, tfm_model, tokenizer,cont_model, op_model, as_model, sub_model, obj_model, cat_model):
        super(T5FineTuner, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = tfm_model
        self.cont_model = cont_model
        self.op_model = op_model
        self.as_model = as_model
        self.sub_model= sub_model
        self.obj_model=obj_model
        self.cat_model = cat_model
        self.tokenizer = tokenizer
        # self.save_hyperparameters(hparams)

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        main_pred = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        
        last_state = main_pred.encoder_last_hidden_state
        # print('last_state',last_state.size())

        # sentiment contrastive loss
        cont_pred = self.cont_model(last_state, attention_mask)
        # opinion contrastive loss
        op_pred = self.op_model(last_state, attention_mask)
        # aspect contrastive loss
        as_pred = self.as_model(last_state, attention_mask)
        sub_pred = self.sub_model(last_state, attention_mask)
        obj_pred = self.obj_model(last_state, attention_mask)

        masked_last_state = torch.mul(last_state, attention_mask.unsqueeze(-1))
        pooled_encoder_layer = torch.sum(masked_last_state, dim=1)
        pooled_encoder_layer = normalize(pooled_encoder_layer, p=2.0, dim=1)

        return main_pred, cont_pred, op_pred, as_pred, sub_pred, obj_pred, pooled_encoder_layer

    def _step(self, batch):
        lm_labels = torch.clone(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs,cont_pred, op_pred, as_pred, sub_pred, obj_pred, pooled_encoder_layer = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )

        # define loss with a temperature `temp`
        criterion = SupConLoss(loss_scaling_factor=self.hparams.cont_loss, temperature=self.hparams.cont_temp)
        preference_labels = batch['preference_labels']
        aspect_labels = batch['aspect_labels']
        opinion_labels = batch['opinion_labels']
        subject_labels = batch['subject_labels']
        object_labels = batch['object_labels']

        # Calculate the characteristic-specific losses
        cont_summed = cont_pred
        cont_normed = normalize(cont_summed, p=2.0, dim=2)  
        preference_contrastive_loss = criterion(cont_normed, preference_labels)
        #print('contr_loss:\t', sentiment_contrastive_loss)

        as_summed = as_pred
        as_normed = normalize(as_summed, p=2.0, dim=2)
        aspect_contrastive_loss = criterion(as_normed, aspect_labels)
        #print('as_loss:\t', aspect_contrastive_loss)

        op_summed = op_pred
        op_normed = normalize(op_summed, p=2.0, dim=2)
        opinion_contrastive_loss = criterion(op_normed, opinion_labels)
        #print('op_loss:\t', opinion_contrastive_loss)

        sub_summed = sub_pred
        sub_normed = normalize(sub_summed, p=2.0, dim=2)
        subject_contrastive_loss = criterion(sub_normed, subject_labels)
        #print('op_loss:\t', opinion_contrastive_loss)

        obj_summed = obj_pred
        obj_normed = normalize(obj_summed, p=2.0, dim=2)
        object_contrastive_loss = criterion(obj_normed, object_labels)
        #print('op_loss:\t', opinion_contrastive_loss)       
        # """
        # Uncomment this section to extract the tsne encodings/labels used for Figure 2 in paper

        # Use these for generating the 'w/ SCL' figures
        preference_encs = cont_normed.detach().cpu().numpy()[:,0].tolist()
        aspect_encs = as_normed.detach().cpu().numpy()[:,0].tolist()
        # opinion_encs = op_normed.detach().cpu().numpy()[:,0].tolist()
        subject_encs = sub_normed.detach().cpu().numpy()[:,0].tolist()
        object_encs = obj_normed.detach().cpu().numpy()[:,0].tolist()

        preference_labs = preference_labels.detach().tolist()
        aspect_labs = aspect_labels.detach().tolist()
        # opinion_labs = opinion_labels.detach().tolist()
        subject_labs = subject_labels.detach().tolist()
        object_labs = object_labels.detach().tolist()

        # Use these for the version without SCL (no characteristic-specific representations)
        # preference_encs = pooled_encoder_layer.detach().numpy().tolist()
        # aspect_encs = pooled_encoder_layer.detach().numpy().tolist()
        # opinion_encs = pooled_encoder_layer.detach().numpy().tolist()
        # subject_encs = pooled_encoder_layer.detach().numpy().tolist()
        # object_encs = pooled_encoder_layer.detach().numpy().tolist()
        # preference_labs = preference_labels.detach().tolist()
        # aspect_labs = aspect_labels.detach().tolist()
        # opinion_labs = opinion_labels.detach().tolist()
        # subject_labs = subject_labels.detach().tolist()
        # object_labs = object_labels.detach().tolist()

        tsne_dict['preference_vecs'] += preference_encs
        tsne_dict['aspect_vecs'] += aspect_encs
        tsne_dict['subject_vecs'] += subject_encs
        tsne_dict['object_vecs'] +=object_encs

        tsne_dict['preference_labels'] += preference_labs
        tsne_dict['aspect_labels'] += aspect_labs
        tsne_dict["subject_labels"] += subject_labs
        tsne_dict["object_labels"] +=object_labs
        # """
        # return original loss plus the characteristic-specific SCL losses
        loss = outputs[0] + opinion_contrastive_loss + preference_contrastive_loss + aspect_contrastive_loss + subject_contrastive_loss +object_contrastive_loss

        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss,_= self._step(batch)
        # self.log("train_loss", loss)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss,_ = self._step(batch)

        # self.log('val_batch_loss', loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss}
        # self.log('val_loss', avg_loss)
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        cont_model = self.cont_model
        op_model = self.op_model
        as_model = self.as_model
        sub_model = self.sub_model
        obj_model =self.obj_model

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in cont_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in cont_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in op_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in op_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in sub_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in sub_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },  
            {
                "params": [p for n, p in obj_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in obj_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },        
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, sents):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu}')
    model.model.to(device)

    model.eval()
    model.model.eval()

    outputs, targets = [], []

    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=args.max_seq_length*2,
                                    num_beams=args.num_beams)  # num_beams=8, early_stopping=True)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    '''
    print("\nPrint some results to check the sanity of generation method:", '\n', '-'*30)
    for i in [1, 5, 25, 42, 50]:
        try:
            print(f'>>Target    : {targets[i]}')
            print(f'>>Generation: {outputs[i]}')
        except UnicodeEncodeError:
            print('Unable to print due to the coding error')
    print()
    '''

    scores, all_labels, all_preds = compute_scores(outputs, targets, sents)
    results = {'scores': scores, 'labels': all_labels, 'preds': all_preds}
    pickle.dump(results, open(f"{args.output_dir}/results-{args.dataset}.pickle", 'wb'))
    # ex_list = []
    
    # for idx in range(len(all_preds)):
    #     new_dict = {}
    #     for key in results:
    #         new_dict[key] = results[key][idx]
    #     ex_list.append(new_dict)
    
    # results = {'performance_metrics': scores, 'examples': ex_list}

    # json.dump(results, open(f"{args.output_dir}/results-{args.dataset}.json", 'w'), indent=2, sort_keys=True)
    
    return scores



# initialization
args = init_args()
seed_everything(args.seed)
print("\n", "="*30, f"NEW EXP: ASQP on {args.dataset}", "="*30, "\n")

# sanity check
# show one sample to check the code and the expected output
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
tokenizer.add_tokens(['[SSEP]'])
print(f"Here is an example (from the dev set):")
dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                      data_type='dev', max_len=args.max_seq_length)
data_sample = dataset[35]  # a random data sample
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))


#training process
if args.do_train:
    print("\n****** Conduct Training ******")

    # initialize the T5 model
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tfm_model.resize_token_embeddings(len(tokenizer))

        
    cont_model = LinearModel()
    op_model = LinearModel()
    as_model = LinearModel()
    sub_model = LinearModel()
    obj_model = LinearModel()
    cat_model = LinearModel()

    model = T5FineTuner(args, tfm_model, tokenizer,cont_model, op_model, as_model, sub_model, obj_model, cat_model)

    if args.early_stopping:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
         dirpath=args.output_dir, monitor='val_loss', mode='min', save_top_k=1
        )
        callback_list = [checkpoint_callback, LoggingCallback(), EarlyStopping(monitor="val_loss", mode='min', patience=3)]
    else:
        callback_list = [LoggingCallback()]

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=3
    # )
    # if args.early_stopping:
    #     checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    #      dirpath=args.output_dir, monitor='val_loss', mode='min', save_top_k=1
    #     )
    #     callback_list = [checkpoint_callback, LoggingCallback(), EarlyStopping(monitor="val_loss", mode='min', patience=3)]
    # else:
    #     callback_list = [LoggingCallback()]

    # prepare for trainer
    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        callbacks=callback_list,
        auto_lr_find=False,
        deterministic=True,
        # accelerator='auto'
    )



    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    if args.early_stopping:
        ex_weights = torch.load(checkpoint_callback.best_model_path)['state_dict']
        model.load_state_dict(ex_weights)

    # save the final model
    model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Finish training and saving the model!")


# evaluation
if args.do_direct_eval:
    print("\n****** Conduct Evaluating with the last state ******")

    # model = T5FineTuner(args)

    # print("Reload the model")
    # model.model.from_pretrained(args.output_dir)            --do_train \
    # tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    # model = T5FineTuner(args, tfm_model, tokenizer)
    sents, _,_ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

    print()
    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                               data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, args.eval_batch_size, num_workers=4)
    # print(test_loader.device)

    # compute the performance scores
    scores = evaluate(test_loader, model, sents)

    # write to file
    log_file_path = f"results_log/{args.dataset}.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
    exp_results = f"F1 = {scores['f1']:.4f}"

    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    with open(log_file_path, "a+") as f:
        f.write(log_str)


if args.do_inference:
    print("\n****** Conduct inference on trained checkpoint ******")

    # initialize the T5 model from previous checkpoint
    print(f"Load trained model from {args.output_dir}")
    print('Note that a pretrained model is required and `do_true` should be False')
    tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)


    cont_model = LinearModel()
    op_model = LinearModel()
    as_model = LinearModel()
    sub_model = LinearModel()
    obj_model = LinearModel()
    cat_model = LinearModel()

    model = T5FineTuner(args, tfm_model, tokenizer,cont_model, op_model, as_model, sub_model, obj_model, cat_model)

    sents, _,_ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

    print()
    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                               data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, args.eval_batch_size, num_workers=4)
    # print(test_loader.device)

    # compute the performance scores
    scores = evaluate(test_loader, model, sents)

    # write to file
    log_file_path = f"results_log/{args.dataset}.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
    exp_results = f"F1 = {scores['f1']:.4f}"

    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    with open(log_file_path, "a+") as f:
        f.write(log_str)
