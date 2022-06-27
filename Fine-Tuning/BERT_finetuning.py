import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from Metrics import Metrics
import logging
from torch.utils.data import  RandomSampler
from transformers import AdamW
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F

FT_model={
    'switchboard': 'bert-base-uncased',
    'ubuntu': 'bert-base-uncased',
    'douban': 'bert-base-chinese',
    'e_commerce': 'bert-base-chinese'
}

# checkpoint_bert_path = "/disk/scratch/swallbridge/BERT_FP/post-train/ubuntu25/bert.pt"
# checkpoint_bert_path = "/disk/scratch/swallbridge/BERT_FP/FPT/PT_checkpoint/switchboard/train_small/checkpoint25-825-1.958938054740429/bert.pt"
# checkpoint_bert_path = "/disk/scratch/swallbridge/BERT_FP/FPT/PT_checkpoint/switchboard/full_train_val/checkpoint8-66896-2.3206088173804336-2.4992520809173584/bert.pt"


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label, lenidx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.lenidx = lenidx


class BERTDataset(Dataset):
    def __init__(self, args, train,tokenizer):
        self.train = train
        self.args = args
        self.bert_tokenizer = tokenizer

    def __len__(self):
        return len(self.train['cr'])

    def __getitem__(self, item):
        cur_features = convert_examples_to_features(item, self.train, self.bert_tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.label, dtype=torch.float),
                       torch.tensor(cur_features.lenidx)
                       )

        return cur_tensors


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def convert_examples_to_features(item, train, bert_tokenizer):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    ex_index = item
    input_ids = train['cr'][item]


    sep = input_ids.index(bert_tokenizer.sep_token_id)
    context = input_ids[:sep]
    response = input_ids[sep + 1:]
    _truncate_seq_pair(context, response, 253)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    context_len = len(context)

    input_ids = [bert_tokenizer.cls_token_id] + context + [bert_tokenizer.sep_token_id] + response + [
        bert_tokenizer.sep_token_id]
    segment_ids = [0] * (context_len + 2)  # context
    segment_ids += [1] * (len(input_ids) - context_len - 2)  # #response
    
    lenidx = [1 + context_len, len(input_ids) - 1]

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = 256 - len(input_ids)

    if (padding_length > 0):
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length) 

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label=train['y'][item],
                             lenidx=lenidx)
    return features


class NeuralNetwork(nn.Module):

    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        self.args = args
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.optimizer = None
        self.best_result = [0, 0, 0, 0, 0, 0]
        self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

        self.bert_config = config_class.from_pretrained(FT_model[args.task],num_labels=1)
        self.bert_tokenizer = BertTokenizer.from_pretrained(FT_model[args.task],do_lower_case=args.do_lower_case)
        special_tokens_dict = {'eos_token': '[eos]'}
        num_added_toks = self.bert_tokenizer.add_special_tokens(special_tokens_dict)
        self.bert_model = model_class.from_pretrained(FT_model[args.task],config=self.bert_config)
        print(f'Loaded BERT from pre-trained ({FT_model[args.task]})') 

        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        """You can load the post-trained checkpoint here."""
        if args.checkpoint_path:
            self.bert_model.bert.load_state_dict(state_dict=torch.load(args.checkpoint_path))
            print(f'Loaded BERT from checkpoint ({checkpoint_bert_path})')
        # self.bert_model.bert.load_state_dict(state_dict=torch.load("./post-train/ubuntu25/bert.pt"))
        # self.bert_model.bert.load_state_dict(state_dict=torch.load("./FPT/PT_checkpoint/ubuntu25/bert.pt"))
        #self.bert_model.bert.load_state_dict(state_dict=torch.load("./FPT/PT_checkpoint/douban27/bert.pt"))
        #self.bert_model.bert.load_state_dict(state_dict=torch.load("./FPT/PT_checkpoint/e_commerce34/bert.pt"))
        
        self.bert_model = self.bert_model.cuda()

        """ Freeze layers here """
        unfreeze = ['classifier', 'pooler', '11', '10', '9', '8']
        for name, param in self.named_parameters():
            if any([ll in name for ll in unfreeze]): 
                param.requires_grad = True
            else:
                param.requires_grad = False


    def forward(self):
        raise NotImplementedError

    def train_step(self, i, data):
        with torch.no_grad():
            batch_ids, batch_mask, batch_seg, batch_y, batch_len = (item.cuda(device=self.device) for item in data)

        self.optimizer.zero_grad()
        output = self.bert_model(batch_ids, batch_mask, batch_seg)
        logits = torch.sigmoid(output[0])
        loss = self.loss_func(logits.squeeze(), target=batch_y)
        loss.backward()

        self.optimizer.step()

        if i % 100 == 0:
            print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(),
                                                                   batch_y.size(0)))  
        return loss

    def fit(self, train, dev):  

        if torch.cuda.is_available(): self.cuda()

        dataset = BERTDataset(self.args, train,self.bert_tokenizer)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler,num_workers=2)

        self.loss_func = nn.BCELoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate,correct_bias=True)  

        losses = []
        v_results = []

        for epoch in range(self.args.epochs):
            print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            avg_loss = 0

            self.train()
            for i, data in tqdm(enumerate(dataloader)): 
                if epoch >= 2 and self.patience >= 3:
                    print("Reload the best model...")
                    self.load_state_dict(torch.load(self.args.save_path))
                    self.adjust_learning_rate()
                    self.patience = 0

                loss = self.train_step(i, data)

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

                avg_loss += loss.item()
            cnt = len(train['y']) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss / cnt))
            losses.append(avg_loss / cnt)
            
            eval_result = self.evaluate(dev)
            v_results.append(eval_result)
        
        loss_list = {'train': losses, 'val': v_results}
        print(f'Losses: {loss_list}')
        with open(f'{self.args.save_path}-data_losses.json', 'w') as fp:
            json.dump(loss_list, fp)

    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)

    def evaluate(self, dev, is_test=False):
        y_pred = self.predict(dev)
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, dev['y']):
                output.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )
        if is_test == False and (self.args.task !='ubuntu' or self.args.task != 'switchboard'): # TODO, I think this is why the val scores are printing wrong -- might be setting to 2 instead of 10
            self.metrics.segment = 2
        else:
            self.metrics.segment = 10
        print(f'Number of samples: {self.metrics.segment}')
        result = self.metrics.evaluate_all_metrics()
        print("Evaluation Result: \n",
              "MAP:", result[0], "\t",
              "MRR:", result[1], "\t",
              "P@1:", result[2], "\t",
              "R1:", result[3], "\t",
              "R2:", result[4], "\t",
              "R5:", result[5])

        # if in training mode, and current results are best so far, save model 
        if not(is_test) and result[3] + result[4] + result[5] > self.best_result[3] + self.best_result[4] + \
                self.best_result[5]:
            self.best_result = result
            print("Best Result: \n",
                  "MAP:", self.best_result[0], "\t",
                  "MRR:", self.best_result[1], "\t",
                  "P@1:", self.best_result[2], "\t",
                  "R1:", self.best_result[3], "\t",
                  "R2:", self.best_result[4], "\t",
                  "R5:", self.best_result[5])
            self.patience = 0
            # self.best_result = result
            torch.save(self.state_dict(), self.args.save_path)
            print("save model!!!\n")
        else:
            self.patience += 1
        return result

    def predict(self, dev):
        self.eval()
        y_pred = []
        dataset = BERTDataset(self.args, dev,self.bert_tokenizer)
        dataloader = DataLoader(dataset, batch_size=400)

        for i, data in tqdm(enumerate(dataloader), desc="Prediction dataloader"):
            with torch.no_grad():
                batch_ids, batch_mask, batch_seg, batch_y, batch_len = (item.cuda() for item in data)
            with torch.no_grad():
                output = self.bert_model(batch_ids, batch_mask, batch_seg)
                logits = torch.sigmoid(output[0]).squeeze()

            if i % 1000 == 0:
                print('Batch[{}] batch_size:{}'.format(i, batch_ids.size(0))) 
            y_pred += logits.data.cpu().numpy().tolist()
        return y_pred

    def load_model(self, path):
        print(f"Loading model from {path}")
        self.load_state_dict(state_dict=torch.load(path))
        #self.bert_model.bert.load_state_dict(state_dict=torch.load(path)) # changed this to bert_model load state_dict
        if torch.cuda.is_available(): self.cuda()
    
    def describe_model(self):
        table = {}
        total_params=0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table[name] = params
            total_params+=params
        return table, total_params
