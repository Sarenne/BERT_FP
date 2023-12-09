
# For data
import sys
sys.path.append('/disk/scratch/swallbridge/discriminitive_turns_project/discriminative_turns/')

# Global imports
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Local imports 
from conversational_corpus import *
from discrimination_dataset import *

# Set torch device (if needed...)
import torch
print(torch.cuda.device_count())
print(torch.cuda.current_device())
    
# torch.cuda.set_device(3)

from FPT.swb_final import BERTDataset, InputExample, convert_example_to_features
from transformers import BertTokenizer,BertConfig, BertForPreTraining

sys.path.append('Fine-Tuning/')
from BERT_finetuning import NeuralNetwork as NeuralNetwork
from BERT_concat_finetuning import NeuralNetwork as NeuralNetwork_concat

"""Load BERT FP model"""
base_args = {
    'bert_model': 'bert-base-uncased', 
    'do_lower_case': True, 
    'gradient_accumulation_steps': 1, 
    'learning_rate': 1.5e-05, 
    'max_seq_length': 240, 
    'num_train_epochs': 25.0, 
    'output_dir': './FPT/PT_checkpoint/switchboard', 
    'train_batch_size': 8, 
    'train_file': './spoken_data/swb_train_small.pkl', 
    'warmup_proportion': 0.01
        }

tokenizer = BertTokenizer.from_pretrained(base_args['bert_model'], do_lower_case=base_args['do_lower_case'])
special_tokens_dict = {'eos_token': '[eos]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# Load model through Response_selection script
class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

args_dict = base_args.copy()
model_args_dict = {
    'task': 'switchboard', 
#     'is_training': , 
    'batch_size': 16, 
    'learning_rate': 1e-5, 
    'epochs': 5, 
    'save_path': './fine-tune/pretrained_swb_samples/', 
    'score_file_path': "./Fine-Tuning/scorefile.txt", 
    'do_lower_case': True, 
    'checkpoint_path': '',
    }
args_dict.update(model_args_dict)
args = Bunch(args_dict)

# Load the base BERT model
model = NeuralNetwork(args=args)   
# Load PTFT (Post-trained + Fine-tuned) checkpoint
checkpoint_path = './fine-tune/finetune_32_wts_nxt_20/switchboard_wts_nxt.0.pt'
model.load_model(checkpoint_path)


"""Load data"""
with open("/disk/scratch/swallbridge/pickles/wts_discrim_dataset_valtest.p", 'rb' ) as fp:
    wts_discrim_ds_valtest = pickle.load(fp)
    
with open("/disk/scratch/swallbridge/pickles/wts_discrim_dataset_train.p", 'rb' ) as fp:
    wts_discrim_ds_train = pickle.load(fp)

# Load the corpus data
with open("/disk/scratch/swallbridge/pickles/wts_corpus.p", 'rb' ) as fp:
    wts_corpus = pickle.load(fp)


"""Define sampling functions"""
def make_bertfp_stimulus(
    context_id, n_responses, dataset, model, 
    try_n_negs=500-1, idxs_to_ignore=[], sample_seed=123, batch_size=250,
    reverse=False, return_tokens=False,
                          ):
    """
    Modified from make_bertfp_stimuli in BERT_FP/stimuli_script.py for making perceptual stimuli.
    
    For a given context, return the highest scoring responses under a BERT_FP model.
    
    Args
        context_id (int): indx corresponding to an acceptable response 
        n_responses (int): number of responses to return 
        reverse (bool): (used for generating potential check questions)
        ...
    """
    # Make a (context, responses) BERT-FP sample
    tokenized_samples, ys, lines = dataset.make_cr_sample(context_id, try_n_negs)
    context_lines, response_lines = lines
    y_pred = model.predict({'cr':tokenized_samples, 'y':ys}, pred_batch_size=batch_size)

    # Get the true response
    target_response = response_lines[0]
    target_score = y_pred[0]
    target_tokens = tokenized_samples[0]

    # Return the top n_response negatives
    neg_response_lines = response_lines[1:]
    neg_response_tokens = tokenized_samples[1:]
    neg_y_pred = y_pred[1:]
    if reverse:
        sort_ids = np.argsort(neg_y_pred)
    else:
        sort_ids = np.argsort(neg_y_pred)[::-1]
    neg_lines = [neg_response_lines[x] for x in sort_ids[:n_responses - 1]]
    neg_scores = [neg_y_pred[x] for x in sort_ids[:n_responses - 1]]
    neg_tokens = [neg_response_tokens[x] for x in sort_ids[:n_responses - 1]]

    if return_tokens:
        return context_lines, neg_lines, target_response, neg_tokens, target_tokens
    else:
        return context_lines, neg_lines, target_response


def make_equiv_stimulus(
    context_id, dataset, model, 
    idxs_to_ignore=[], sample_seed=123, batch_size=250,
    reverse=False, return_tokens=False,
                          ):
    """
    Modified from make_bertfp_stimuli in BERT_FP/stimuli_script.py for making perceptual stimuli.
    
    For a given context, return the highest scoring responses under a BERT_FP model.
    
    Args
        context_id (int): indx corresponding to an acceptable response 
        n_responses (int): number of responses to return 
        reverse (bool): (used for generating potential check questions)
        ...
    """
    # Make a (context, responses) BERT-FP sample
    tokenized_samples, ys, lines = dataset.make_cr_sample(context_id, 0)
    context_lines, response_lines = lines

    # Get the true response
    target_response = response_lines[0]
    target_tokens = tokenized_samples[0]

    # Return equivalent responses if there are any
    turns = list(filter(lambda x: x != target_response, discrim_dataset.all_turns))
    turn_df = pd.DataFrame(turns)
    equiv_df = turn_df[turn_df["clean_text"] == target_response["clean_text"]]
    if len(equiv_df) > 0:
        equiv_lines = equiv_df.to_dict('records')
    else: equiv_lines = []

    if return_tokens:
        return context_lines, equiv_lines, target_response, target_tokens
    else:
        return context_lines, equiv_lines, target_response



"""Run"""
discrim_dataset = wts_discrim_ds_train
all_idxs = list(range(len(discrim_dataset.acceptable_context_turns)))
def main():
    # Check all stimuli for equivalent responses 
    equiv_dicts = []
    for cid in tqdm(all_idxs):
        context_lines, equiv_lines, target_response, target_tokens = make_equiv_stimulus(
            cid, 
            discrim_dataset, 
            model, 
            return_tokens=True
        )

        equiv_dicts.append({
            "context_id": cid,
            "context_lines": context_lines,
            "equiv_lines": equiv_lines,
            "equiv_num": len(equiv_lines),
            "target_response": target_response,
            "target_tokens": target_tokens,
        })
        
    equiv_df = pd.DataFrame(equiv_dicts)
    equiv_df.to_json('/disk/scratch2/swallbridge/equiv_responses.json')

    # Make some plausible ones
    neg_dicts = []
    for cid in tqdm(all_idxs[:1000]):
        context_lines, neg_lines, target_response, neg_tokens, target_tokens = make_bertfp_stimulus(
            cid, 
            5,
            discrim_dataset, 
            model, 
            return_tokens=True
        )

        neg_dicts.append({
            "context_id": cid,
            "context_lines": context_lines,
            "neg_lines": equiv_lines,
            "target_response": target_response,
            "neg_tokens": neg_tokens,
            "target_tokens": target_tokens,
        })
        
    neg_df = pd.DataFrame(neg_dicts)
    neg_df.to_json('/disk/scratch2/swallbridge/neg_responses.json')

if __name__ == "__main__":
    main()