import sys
# For data
sys.path.append('/afs/inf.ed.ac.uk/user/s13/s1301730/Documents/discriminitive_turns_project/discriminative_turns/')

# Global imports
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
from conversational_corpus import *
from FPT.swb_final import BERTDataset, InputExample, convert_example_to_features
from transformers import BertTokenizer,BertConfig


"""
Script for building the Fine-Tuning datasets for SWB (ie, in the format of `ubuntu_dataset_1M.pkl`)
"""

def make_ft_sample(self, item, num_negs, seed=123):
    """
    Method to build fine-tuning datasets (same format as stimuli in ubuntu_dataset_1M.py) from short-contexts.

    For a given sample, select num_neg negative turns from anywhere in the corpus.
    """
    sample = self.sample_to_doc[item]
    length = sample['end']

    # Get the context
    if length != 0:
        tokens_a = []
        for i in range(length - 1):
            tokens_a+=self.tokenizer.tokenize(self.all_docs[sample["doc_id"]][i])+[self.tokenizer.eos_token]
        tokens_a.pop()

        response_sample = {"doc_id": sample["doc_id"], "line": length - 1}
        response = self.all_docs[sample["doc_id"]][length - 1]

    else:
        t1 = self.all_docs[sample["doc_id"]][sample["line"]]
        t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
        t3 = self.all_docs[sample["doc_id"]][sample["line"] + 2]
        tokens_a = self.tokenizer.tokenize(t1)+[self.tokenizer.eos_token]+self.tokenizer.tokenize(t2)+[self.tokenizer.eos_token]+self.tokenizer.tokenize(t3)

        response_sample = {"doc_id": sample["doc_id"], "line": sample["line"] + 3}
        response = self.all_docs[sample["doc_id"]][sample["line"] + 3]

    # Get negative responses
    neg_responses = self.get_random_lines(response_sample, num_negs, seed=seed)
    tokens_bs = [self.tokenizer.tokenize(response)]
    for n in neg_responses:
        tokens_bs.append(self.tokenizer.tokenize(n))

    # Join samples and convert to ids
    tokenized_samples = [self.tokenizer.convert_tokens_to_ids(tokens_a + [self.tokenizer.eos_token] + [self.tokenizer.sep_token] + tokens_b) for tokens_b in tokens_bs]

    # Build labels
    ys = [1]
    ys.extend(list(np.zeros(num_negs, dtype=int)))

    return tokenized_samples, ys

def make_ft_doc(self, item, num_negs, seed=123):
    """
    Method to build fine-tuning datasets (same format as stimuli in ubuntu_dataset_1M.py) from full dialogues.

    For a given sample, select num_neg negative turns from anywhere in the corpus.
    """
    doc = self.all_docs[item]
    length = len(doc)

    # Get the context
    tokens_a = []
    for i in range(length - 1):
        tokens_a+=self.tokenizer.tokenize(doc[i])+[self.tokenizer.eos_token]
    tokens_a.pop()

    response_sample = {"doc_id": item, "line": length - 1} # TODO CHECK THIS WORKS
    response = doc[length - 1]

    # Get negative responses
    neg_responses = self.get_random_lines(response_sample, num_negs, seed=seed)
    tokens_bs = [self.tokenizer.tokenize(response)]
    for n in neg_responses:
        tokens_bs.append(self.tokenizer.tokenize(n))

    # Join samples and convert to ids
    tokenized_samples = [self.tokenizer.convert_tokens_to_ids(tokens_a + [self.tokenizer.eos_token] + [self.tokenizer.sep_token] + tokens_b) for tokens_b in tokens_bs]


    # Build labels
    ys = [1]
    ys.extend(list(np.zeros(num_negs, dtype=int)))

    return tokenized_samples, ys

args = {
    'bert_model': 'bert-base-uncased',
    'do_lower_case': False,
    'gradient_accumulation_steps': 1,
    'learning_rate': 1.5e-05,
    'max_seq_length': 240,
    'num_train_epochs': 25.0,
    'output_dir': './FPT/PT_checkpoint/switchboard',
    'train_batch_size': 8,
    'train_file': './spoken_data/swb_train_small.pkl',
    'warmup_proportion': 0.01
        }

def main():

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])
    special_tokens_dict = {'eos_token': '[eos]'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    # Load dataset
    train_small_dataset = BERTDataset(args['train_file'], tokenizer, seq_len=args['max_seq_length'],
                                corpus_lines=None)
    train_dataset = BERTDataset('./spoken_data/swb_train.pkl', tokenizer, seq_len=args['max_seq_length'],
                                corpus_lines=None)
    val_dataset = BERTDataset('./spoken_data/swb_val.pkl', tokenizer, seq_len=args['max_seq_length'],
                                corpus_lines=None)
    test_dataset = BERTDataset('./spoken_data/swb_test.pkl', tokenizer, seq_len=args['max_seq_length'],
                            corpus_lines=None)

    import IPython
    IPython.embed()

    # Make the sample-version of FT dataset
    swb_dset = []
    dset = train_dataset # val_dataset test_dataset
    cr = []
    y = []
    for i in tqdm(range(len(dset))):
        tokenized_samples, ys = make_ft_sample(dset, i, 1)
        cr.extend(tokenized_samples)
        y.extend(ys)
    swb_dset.append({'cr':cr, 'y':y})

    dset = val_dataset
    cr = []
    y = []
    for i in tqdm(range(len(dset))):
        tokenized_samples, ys = make_ft_sample(dset, i, 9)
        cr.extend(tokenized_samples)
        y.extend(ys)
    swb_dset.append({'cr':cr, 'y':y})

    dset = test_dataset
    cr = []
    y = []
    for i in tqdm(range(len(dset))):
        tokenized_samples, ys = make_ft_sample(dset, i, 9)
        cr.extend(tokenized_samples)
        y.extend(ys)
    swb_dset.append({'cr':cr, 'y':y})

    with open('switchboard_dataset_samples.pkl', 'wb') as fp:
        pickle.dump(swb_dset, fp)

    # Make the doc-version of FT dataset
    swb_doc_dset = []
    dset = train_dataset
    cr = []
    y = []
    for i in tqdm(range(len(dset.all_docs))):
        tokenized_samples, ys = make_ft_doc(dset, i, 1)
        cr.extend(tokenized_samples)
        y.extend(ys)
    swb_doc_dset.append({'cr':cr, 'y':y})

    dset = val_dataset
    cr = []
    y = []
    for i in tqdm(range(len(dset.all_docs))):
        tokenized_samples, ys = make_ft_doc(dset, i, 9)
        cr.extend(tokenized_samples)
        y.extend(ys)
    swb_doc_dset.append({'cr':cr, 'y':y})

    dset = test_dataset
    cr = []
    y = []
    for i in tqdm(range(len(dset.all_docs))):
        tokenized_samples, ys = make_ft_doc(dset, i, 9)
        cr.extend(tokenized_samples)
        y.extend(ys)
    swb_doc_dset.append({'cr':cr, 'y':y})

    with open('switchboard_dataset_docs.pkl', 'wb') as fp:
        pickle.dump(swb_doc_dset, fp)

    import IPython
    IPython.embed()

if __name__ == "__main__":
    main()
