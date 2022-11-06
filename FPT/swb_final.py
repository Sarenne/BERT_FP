from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from transformers import BertTokenizer,BertConfig
from transformers import BertForPreTraining
from transformers import AdamW
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import pickle

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import random
from setproctitle import setproctitle
setproctitle('jaden han')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger.info(f"CUDA info: device {torch.cuda.current_device()}, name: {torch.cuda.get_device_name(0)}")

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0



class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8-sig", corpus_lines=None):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.sample_to_doc = [] # map sample index to doc and line that have been filtered (empty/length)
        self.all_docs = [] # raw documents; list of lists (docs are lists of utterances)
        self.all_turns = []
        doc = [] # list of utterances in a document 

        crsets = pickle.load(file=open(corpus_path, 'rb'))
        cnt=0 # track the number of lines that aren't used  
        lcnt=0 # track the number of short documents ? 
        
        # Check if the loaded dataset contains meta info (each line is a dict) or not (each line is str)
        self.meta = False
        if type(crsets[0][0]) == dict:
            self.meta = True
        
        for crset in tqdm(crsets): # for each document in corpus, create a list of turns as strings + samples
            for line in crset:
                # Get the text from the line
                if self.meta:
                    l_text = line['clean_text']
                else:
                    l_text = line
              
                if len(l_text) == 0:
                    continue
                if len(l_text) < 10:
                    if len(self.tokenizer.tokenize(l_text)) == 0:
                        cnt += 1
                        continue
                sample = {"doc_id": len(self.all_docs),
                        "line": len(doc),
                        "end": 0 ,
                        "linenum":1
                        }
                self.sample_to_doc.append(sample)
                doc.append(line)
                
            if (len(doc) != 0):
                self.all_docs.append(doc)
            else:
                print("empty")

            # Record #turns if doc has < 4 (hardcoded for context length experiments); treated as special case
            if (len(doc) < 4):
                for i in range(len(doc) - 1):
                    self.sample_to_doc.pop()

                self.sample_to_doc[-1]['end'] = len(doc)
            
                lcnt+=1
            
            else:    
                self.sample_to_doc.pop()
                self.sample_to_doc.pop()
                self.sample_to_doc.pop()

            doc = []
            
        print(cnt,lcnt)
        
        for doc in self.all_docs:
            if len(doc) == 0:
                print("problem")

        self.all_turns = [t for conv in self.all_docs for t in conv]
        if self.meta:
            self.unique_turns = []
            self.unique_turns_text = []
            for t in self.all_turns:
                if t['clean_text'] not in self.unique_turns_text:
                    self.unique_turns.append(t)
                    self.unique_turns_text.append(t['clean_text'])
        else:
            self.unique_turns = list(set(self.all_turns))

    def __len__(self):
        return len(self.sample_to_doc)

    def __getitem__(self, item, tokenize_output=True):
        sample = self.sample_to_doc[item]
        length = sample['end']

        try:
            # Special case where dailogue session length < short context length k (sample will just contain less context)
            if length != 0: # essentially a boolean
                tokens_a = []
                context = []
                for i in range(length - 1):
                    line = self.all_docs[sample["doc_id"]][i]
                    if self.meta:
                        text = line['clean_text']
                    else:
                        text = line
                    tokens_a+=self.tokenizer.tokenize(line)+[self.tokenizer.eos_token]
                    context.append(line)
                tokens_a.pop()

                #response = self.all_docs[sample["doc_id"]][sample["line"] + length - 1]
                rand=random.random()
            
                if rand > 0.75:
                    # next correct response
                    response = self.all_docs[sample["doc_id"]][length - 1]
                    is_next_label = 2

                elif rand > 0.5:
                    # random utterance in the same dialogue session.
                    rand_idx = random.randint(0, length - 2)
                    response = self.all_docs[sample["doc_id"]][rand_idx]
                    is_next_label = 1

                else:
                    #random utterace
                    response = self.get_random_line(sample)
                    is_next_label = 0
                
                if self.meta:
                    r_text = response['clean_text']
                else:
                    r_text = response
                tokens_b = self.tokenizer.tokenize(r_text)
                # used later to avoid random nextSentence from same doc
                if not tokenize_output:
                    return context, response, is_next_label

            # Normal case (when dialogue session has long enough context)
            else:
                # short context length k= 3 in ubuntu corpus.
                t1, t2, t3, t4, is_next_label = self.random_sent(item)

                if not tokenize_output:
                    return [t1, t2, t3], t4, is_next_label

                # tokenize
                if self.meta:
                    t1t = t1['clean_text']
                    t2t = t2['clean_text']
                    t3t = t3['clean_text']
                    t4t = t4['clean_text']
                else:
                    t1t = t1
                    t2t = t2
                    t3t = t3
                    t4t = t4
                tokens_a = self.tokenizer.tokenize(t1t)+[self.tokenizer.eos_token]+self.tokenizer.tokenize(t2t)+[self.tokenizer.eos_token]+self.tokenizer.tokenize(t3t)
                tokens_b = self.tokenizer.tokenize(t4t)

            # combine to one sample
            cur_example = InputExample(tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)

            # transform sample to features
            cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

            cur_tensors = (torch.tensor(cur_features.input_ids),
                           torch.tensor(cur_features.input_mask),
                           torch.tensor(cur_features.segment_ids),
                           torch.tensor(cur_features.lm_label_ids),
                           torch.tensor(cur_features.is_next))

        except:
            import IPython
            IPython.embed()

        return cur_tensors

    def random_sent(self, index):
        """
        For a dialogue session with > 4 (hardcoded) context turns, build a (context, response, label) training sample where label is selected randomly between {0: random, 1: semantically-similar, 2: next}.

        Args
           index (int): index of document in self.samples_to_doc
        """
        sample = self.sample_to_doc[index]
        t1 = self.all_docs[sample["doc_id"]][sample["line"]]
        t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
        t3 = self.all_docs[sample["doc_id"]][sample["line"] + 2]
        self.current_doc = sample["doc_id"]
        rand = random.random()
        if rand > 0.75 :
            #next correct response
            label = 2
            t4 = self.all_docs[sample["doc_id"]][sample["line"] + 3]
           
        elif rand > 0.5:

            #random in the same dialogue session 
            samedoc = self.all_docs[sample["doc_id"]]
            linenum = random.randrange(len(samedoc))
            
            # not correct response.
            while linenum == sample["line"] + 3:
                linenum = random.randrange(len(samedoc))

            t4 = samedoc[linenum]
            label = 1

        else:
            #random utterance
            t4 = self.get_random_line(sample)
            label = 0

        assert len(t1) > 0
        assert len(t2) > 0
        assert len(t3) > 0
        assert len(t4) > 0
        return t1, t2, t3 ,t4, label


    def get_random_line(self,sample):

        #not in same dialogue session.
        while(True):
            rand_doc_idx = random.randint(0, len(self.all_docs)-1)
            if sample["doc_id"]!=rand_doc_idx:
                break

        rand_doc = self.all_docs[rand_doc_idx]
        line = rand_doc[random.randrange(len(rand_doc))]
            
        return line

    def get_random_lines(self, sample, num_negs, seed=123):
        """Randomly select num_neg turns (lines) from all turns in the corpus (ignoring the given sample turn) """
        # make the set of tuples (document, line) (ie, all lines in corpus)
        tups = []
        for i in range(len(self.all_docs)):
            for j in range(len(self.all_docs[i])):
                tups.append((i,j))

        # remove the line corresponding to sample
        indx = tups.index((sample["doc_id"], sample["line"]))
        tups.remove((sample["doc_id"], sample["line"]))

        # choose some random lines
        # np.random.seed(indx) # unseeded so that it selects different things!!
        line_ids = np.random.choice(range(len(tups)), size=num_negs, replace=True) # I think this should be replace=False
        lines = [self.all_docs[tups[i][0]][tups[i][1]] for i in line_ids]
        
        return lines

    def get_random_unique_lines(self, response_sample, num_negs, remove_context=True):
        """Randomly select num_neg turns (lines) from all UNIQUE turns in the corpus (ignoring the given sample turn) """
        # Turns to sample from
        sample_turns = self.unique_turns.copy()

        # Remove response sample text from sample_turns
        response_text = self.all_docs[response_sample["doc_id"]][response_sample["line"]]
        if response_text in sample_turns:
            sample_turns.remove(response_text)

        # Remove context turns from sample_turns
        if remove_context:
            t1_text = self.all_docs[response_sample["doc_id"]][response_sample["line"] - 3]
            t2_text = self.all_docs[response_sample["doc_id"]][response_sample["line"] - 2]
            t3_text = self.all_docs[response_sample["doc_id"]][response_sample["line"] - 1]

            for t in [t1_text, t2_text, t3_text]:
                if t in sample_turns:
                    sample_turns.remove(t)

        # Choose some random lines
        np.random.seed(int(str(response_sample["doc_id"])+str(response_sample["line"]))) # comment for unseeded, so that it selects different things!!
        lines = list(np.random.choice(sample_turns, size=num_negs, replace=False))

        return lines


class InputExample(object):
    def __init__(self, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids


def random_word(tokens, tokenizer):
  
    output_label = []

    for i, token in enumerate(tokens):
        if token=='[eos]':
            output_label.append(-1)
            continue
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return output_label


def convert_example_to_features(example, max_seq_length, tokenizer):
 
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    #dynamic mask
    t1_label = random_word(tokens_a, tokenizer)
    t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if len(tokens_b)==0:
        print(example.tokens_b)
    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    if False:
        logger.info("*** Example ***")
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #logger.info("LM label: %s " % (lm_label_ids))
        logger.info("Is next sentence label: %s " % (example.is_next))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next)
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default="./spoken_data/swb_train.pkl",
                        type=str,
                        help="The input train corpus.")
    parser.add_argument("--val_file",
                        default="./spoken_data/swb_val.pkl",
                        type=str,
                        help="The input validation corpus.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default="./FPT/PT_checkpoint/switchboard",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=240,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=50,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1.5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    args = parser.parse_args()

    device = torch.device("cuda")

    # For logging through tensorboard
    writer = SummaryWriter()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    special_tokens_dict = {'eos_token': '[eos]'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    bertconfig = BertConfig.from_pretrained(args.bert_model)
    model = BertForPreTraining.from_pretrained(args.bert_model, config=bertconfig)

    model.resize_token_embeddings(len(tokenizer))
    model.cls.seq_relationship = nn.Linear(bertconfig.hidden_size, 3)
    #load checkpoint here
    #model.bert.load_state_dict(state_dict=torch.load("ubuntu_final/checkpoint20-1637300/bert.pt"))
    model.to(device)

    num_train_steps = None
    print("Loading Train Dataset", args.train_file)
    train_dataset = BERTDataset(args.train_file, tokenizer, seq_len=args.max_seq_length,
                                corpus_lines=None)
    print("Loading Validation Dataset", args.val_file)
    val_dataset = BERTDataset(args.val_file, tokenizer, seq_len=args.max_seq_length,
                                corpus_lines=None)
    num_train_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate)

    global_step = 0
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)


    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=2)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.train_batch_size,num_workers=2)

    learning_rate=args.learning_rate
    before = 10

    import IPython
    IPython.embed()

    for epoch in trange(1, int(args.num_train_epochs) + 1, desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration",position=0)):
            with torch.no_grad():
                batch = (item.cuda(device=device) for item in batch)
            input_ids, input_mask, segment_ids,lm_label_ids, is_next = batch
            model.train()
            optimizer.zero_grad()
            prediction_scores, seq_relationship_score = model(input_ids=input_ids,attention_mask= input_mask, token_type_ids=segment_ids)
            # out = model(input_ids=input_ids,attention_mask= input_mask, token_type_ids=segment_ids)
            # prediction_scores = out['prediction_logits']
            # seq_relationship_score = out['seq_relationship_logits']           
            
            #logits = torch.sigmoid(output[0].squeeze())
            if lm_label_ids is not None and is_next is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size),
                                            lm_label_ids.view(-1))
                next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 3), is_next.view(-1))
                total_loss = masked_lm_loss + next_sentence_loss

            model.zero_grad()
            loss = total_loss

            if step%100==0:
                print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(step, loss.item(),args.train_batch_size) )
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            else:
                loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                if global_step / num_train_steps < args.warmup_proportion:
                    lr_this_step = learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        averloss=tr_loss/step
        writer.add_scalar("loss_train", averloss, epoch)
        print("epoch: %d\taverageloss: %f\tstep: %d "%(epoch,averloss,step))
        print("current learning_rate: ", learning_rate)
        if global_step/num_train_steps > args.warmup_proportion and averloss > before - 0.01:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
                learning_rate = param_group['lr']
            print("Decay learning rate to: ", learning_rate)

        before=averloss

        # Validation loop
        vl_loss = 0
        for step, batch in enumerate(tqdm(val_dataloader, desc="Iteration",position=0)):
            with torch.no_grad():
                batch = (item.cuda(device=device) for item in batch)
                input_ids, input_mask, segment_ids,lm_label_ids, is_next = batch
                model.eval()

                prediction_scores, seq_relationship_score = model(input_ids=input_ids,attention_mask= input_mask, token_type_ids=segment_ids)
                if lm_label_ids is not None and is_next is not None:
                    loss_fct = CrossEntropyLoss(ignore_index=-1)
                    masked_lm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size),
                                            lm_label_ids.view(-1))
                    next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 3), is_next.view(-1))
                    total_loss = masked_lm_loss + next_sentence_loss

                    vl_loss += total_loss

        averloss_val = vl_loss/step
        writer.add_scalar("loss_valid", averloss_val, epoch)
        print("epoch: %d\taverage_val_loss: %f"%(epoch,averloss_val))

        if True:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            checkpoint_prefix = 'checkpoint' + str(epoch)
            output_dir = os.path.join(args.output_dir, '{}-{}-{}-{}'.format(checkpoint_prefix, global_step, averloss, averloss_val))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir1 = output_dir + '/bert.pt'
            torch.save(model.bert.state_dict(), output_dir1)



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > 3*len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()

if __name__ == "__main__":
    main()
