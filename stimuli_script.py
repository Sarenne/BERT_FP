import sys
# For data
sys.path.append('/afs/inf.ed.ac.uk/user/s13/s1301730/Documents/discriminitive_turns_project/discriminative_turns/')

# Global imports
import pickle

import numpy as np
import pandas as pd

import torch

import matplotlib.pyplot as plt
# Local imports
from conversational_corpus import *
from discrimination_dataset import *

from FPT.swb_final import BERTDataset, InputExample, convert_example_to_features
from transformers import BertTokenizer,BertConfig, BertForPreTraining

sys.path.append('Fine-Tuning/')
from BERT_finetuning import NeuralNetwork as NeuralNetwork
from BERT_concat_finetuning import NeuralNetwork as NeuralNetwork_concat



def show_experiment(dataset, corpus, sample_id, model, n_negs=500-1, reverse=False, audio=False):
    # Make BERT-FP predictions
    tokenized_samples, ys, lines = dataset.make_cr_sample(sample_id, n_negs)
    context_lines, response_lines = lines
    y_pred = model.predict({'cr':tokenized_samples, 'y':ys}, pred_batch_size=100)

    top_scores, top_ids, top_lines, sort_ids = select_neighbours(y_pred, tokenized_samples,
                                                                 response_lines, 500,
                                                                 reverse=reverse
                                                                )

    target_id = np.argwhere(sort_ids == 0)[0][0]
    target_score = top_scores[target_id]

    # PRINT AND PLOT
    # Plot
    fig, axs = plt.subplots(2,1, figsize=(10,10))
    axs[0].hist(top_scores)
    axs[1].scatter(range(len(top_scores)), top_scores, )
    axs[1].axhline(target_score)
    axs[1].axvline(target_id, label=f'True response ({target_id})')
    axs[1].set_xlabel('Responses')
    axs[1].set_ylabel('CR score')
    axs[1].legend()
    plt.show()

    print(f'True turn: #{target_id}')
    print(f'CONTEXT: ({context_lines[-1]["conv_id"]}, {context_lines[-1]["turn_id"]})')
    print(f'A: {context_lines[0]["clean_text"]}')
    print(f'B: {context_lines[1]["clean_text"]}')
    print(f'A: {context_lines[2]["clean_text"]}')

    print('RESPONSES')
    for i, ss in enumerate(top_scores[:10]):
        if target_id == i:
            print(f' *** - ({i}) {ss:.3f}, {top_lines[i]["clean_text"]} ({top_lines[i]["conv_id"]}, {top_lines[i]["turn_id"]})')
        else:
            print(f' - ({i}) {ss:.3f}, {top_lines[i]["clean_text"]} ({top_lines[i]["conv_id"]}, {top_lines[i]["turn_id"]})')

    print(f' -  (TRUE RESPONSE) ({target_id}) {ss:.3f}, {top_lines[target_id]["clean_text"]} ({top_lines[target_id]["conv_id"]}, {top_lines[target_id]["turn_id"]})')
    print('\n----------\n')
    if audio:
        play_experiment(y_pred, tokenized_samples, context_lines, response_lines, corpus, reverse=reverse)


def print_context(turns):
    """Given a list of (3) context turns, return a list of texts and the printable text string"""
    turn_start = {0:' <div class="yours messages"><div class ="message last">',
                  1: '<div class="mine messages"><div class ="message last">',
                 }
    turn_end = "</div>"
    turn_dots = '<div class="dots messages"><div class="message"> .  .  . </div></div>'

    texts = [t['clean_text'] for t in turns]
    turn_string = ""
    for i, text in enumerate(texts):
        speaker = i % 2
        turn_string += turn_start[speaker]
        turn_string += text
        turn_string += turn_end
        turn_string += turn_end

    turn_string += turn_dots

    # The actual printing...
    text_print = '<div class="chat">' + turn_dots
    text_print += turn_string

    return texts, text_print

def print_response(turn, person='mine'):
    """Given a turn, return the text and printable string"""

    text = turn['clean_text']

    if person not in ['mine', 'yours']:
        print('ERROR in response_print()')
        return None

    text_print = f'<div class="response_chat"><div class="{person} messages"><div class="message last">{text}</div></div></div>'
#     <div class="response_chat"><div class="mine messages"><div class="message last">{<u><b>right that's what I thought too</b></u>}</div></div></div>
    return text, text_print


COLS = [
    'context_text', 'context_print', 'context_id', 'q_id',
    'response_text', 'response_print', 'response_id', 'target', 'response_score',
        ]

def make_bertfp_stimuli_df(n_samples, n_responses, reverse,
                           dataset, corpus, model, tokenizer, 
                           n_negs=500-1, indxs_to_ignore=[],
                           columns=COLS, sample_seed=123, q_id_str=''
                          ):
    """Fill grid_df with contexts and responses"""
    grid_df = pd.DataFrame(columns=columns)

    # Select some random context indices
    all_indxs = list(range(len(dataset.acceptable_context_turns)))
    if indxs_to_ignore:
	all_indxs = list(set(all_indxs).difference(set(indxs_to_ignore)))
    np.random.seed(sample_seed)
    idxs = np.random.choice(all_indxs, n_samples, replace=False)

    for sample_id in tqdm(idxs):
        # Make a (context, responses) BERT-FP sample
        tokenized_samples, ys, lines = dataset.make_cr_sample(sample_id, n_negs)
        context_lines, response_lines = lines
        y_pred = model.predict({'cr':tokenized_samples, 'y':ys}, pred_batch_size=100)

        sids = np.argsort(y_pred)
        top_scores, top_ids, top_lines, sort_ids = select_neighbours(y_pred, tokenized_samples,
                                                                 response_lines, n_negs+1,
                                                                 reverse=reverse
                                                                )

        # Get the true response
        target_id = np.argwhere(sort_ids == 0)[0][0]
        target_response = top_lines[target_id]
        target_score = top_scores[target_id]

        # Return the top(n_response) negatives (excluding the true response)
        num_responses = n_responses - 1
        if target_id < n_responses: # if the target is in top[n_responses], add an extra!
            num_responses += 1

        neg_lines = [response_lines[i] for i in sort_ids[:num_responses] if i != 0]
        neg_scores = [y_pred[i] for i in sort_ids[:num_responses] if i != 0]

        # Make rows in grid_df
        context_text, context_print = print_context(context_lines)
        context_id = f"{context_lines[-1]['conv_id']}_{context_lines[-1]['turn_id']}{q_id_str}"
        c_row = {
            'context_text': context_text,
            'context_print': context_print,
            'context_id': context_id,
            'q_id': context_id, # context_id = f"{context_turns[0]['conv_id']}_{context_turns[0]['turn_id']}"
                }

        # target response info
        response_text, response_print = print_response(target_response)
        response_id = f"{target_response['conv_id']}_{target_response['turn_id']}"
        t_row = {
            'response_text': response_text,
            'response_print': response_print,
            'response_id': response_id, # response_id = f"{neg['conv_id']}_{neg['turn_id']}"
            'target': 1,
            'response_score': target_score
                }

        row = {**c_row, **t_row,}
        grid_df = grid_df.append(row, ignore_index=True)

        for i, neg in enumerate(neg_lines):
            response_text, response_print = print_response(neg)
            response_id = f"{neg['conv_id']}_{neg['turn_id']}"
            r_row = {
                'response_text': response_text,
                'response_print': response_print,
                'response_id': response_id, # response_id = f"{neg['conv_id']}_{neg['turn_id']}"
                'target': 0,
                'response_score': neg_scores[i]
                    }
            row = {**c_row, **r_row,}
            grid_df = grid_df.append(row, ignore_index=True)

    return grid_df, indxs


if __name__ == "__main__":

    """Load model"""
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
    # checkpoint_path = './fine-tune/finetune_swb_samples/switchboard.0.pt'
    checkpoint_path = './fine-tune/finetune_32_wts_nxt_20/switchboard_wts_nxt.0.pt'
    model.load_model(checkpoint_path)


    """Load data"""
    # Load the corpus data
    with open( "/disk/scratch/swallbridge/pickles/wts_corpus.p", 'rb' ) as fp:
        wts_corpus = pickle.load(fp)

    with open("/disk/scratch/swallbridge/pickles/wts_dataset.p", 'rb' ) as fp:
        discrim_dataset = pickle.load(fp)

    """make 'check' grid"""
    check_df, check_indxs = make_bertfp_stimuli_df(
				     25, 5, True,
                                     dataset=discrim_dataset, corpus=wts_corpus, model=model, tokenizer=tokenizer,
                                     n_negs=1000-1, columns=COLS, sample_seed=123, q_id_str='_check',
                              			  )
    check_df.to_csv('tmp_check_grid.csv')

    """make "stimuli "grid"""
    grid_df, stimuli_indxs = make_bertfp_stimuli_df(
				     250, 5, False,
                                     dataset=discrim_dataset, corpus=wts_corpus, model=model, tokenizer=tokenizer,
                                     n_negs=1000-1, indxs_to_ignore=check_indxs, columns=COLS, sample_seed=123,
                               			     )
    grid_df.to_csv('tmp_bertfp_grid.csv')
    import IPython
    IPython.embed()

