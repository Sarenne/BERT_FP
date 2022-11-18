from FPT.swb_final import BERTDataset
from tqdm import tqdm

# class Subclass(Superclass):
#     def __init__(self, subclass_arg1, *args, **kwargs):
#         super(Subclass, self).__init__(*args, **kwargs)

# wts_val_dataset = BERTDataset('./spoken_data/wts_nxt_info_val.pkl', tokenizer, seq_len=base_args['max_seq_length'],
#                             corpus_lines=None

# def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8-sig", corpus_lines=None):
        

class DiscriminationDataset(BERTDataset):
    def __init__(self, corpus_path, tokenizer, 
                 min_context_turns=3, max_tokens=50, min_context_s=2, max_context_s=10, min_context_turn=5, max_context_turn=5, 
                 min_response_s=0.25, max_response_s=10,
                ):
        super(DiscriminationDataset, self).__init__(corpus_path, tokenizer, seq_len=240)
        # NOTE seq_len is hard coded from base args in BERTFP
        self.acceptable_context_turns = self.make_acceptable_contexts()
        
        # Args for acceptable contexts + responses
        self.min_context_turns = min_context_turns
        self.max_context_turns = max_context_turns
        self.min_context_s = min_context_s
        self.max_context_s = max_context_s
        self.min_context_turn = min_context_turn
        self.max_context_turn = max_context_turn
        self.min_response_s = min_response_s
        self.max_response_s = max_response_s
        
      
    def acceptable_context_turn(self, turn, ):
#                                 min_tokens=3, max_tokens=50, 
#                                 min_s=2, max_s=10,
#                                 min_turn=5, max_turn=5, 
#                                 min_response_s=0.25, max_response_s=10,
#                                ): 
        
        # Filter for token length (ignore contexts that are too short/long to listen to)
        num_tokens = len(turn['clean_text'].split(' '))
        if num_tokens < self.min_context_turns or num_tokens > self.max_context_turns:
            return False

        # Filter for audio length (ignore contexts that are too short/long to listen to)
        len_s = turn['stop'] - turn['start']
        if len_s < self.min_context_s or len_s > self.max_context_s:
            return False

        # Filter for turn position (ignore contexts from beginning and end of conversations as these might be conventionalised)
        min_turn_id = self.min_context_turn
        num_conv_turns = max([c['turn_id'] for c in self.all_turns if c['conv_id'] == turn['conv_id']])
        max_turn_id = num_conv_turns - self.max_context_turn
        if turn['turn_id'] > max_turn_id or turn['turn_id'] < min_turn_id:
            return False

        # Filter for response length
        response_turns = [t for t in self.all_turns if (t['conv_id'] == turn['conv_id']) and (t['turn_id'] == turn['turn_id'] + 1)]
        if response_turns:
            response_len = response_turns[-1]['stop'] - response_turns[-1]['start']
            if response_len < self.min_response_s or response_len > self.max_response_s:
                return False
        else:
            return False
        # Filter for pause length, turn_audio ()
        # Filter for acceptable response?
        return True
      
    def make_acceptable_contexts(self):
        """Return (and set) acceptable context turns (ie, turns that can end a context)"""
        acceptable_turns = [t for t in tqdm(self.all_turns) if self.acceptable_context_turn(t)]
        return acceptable_turns 
      
    def acceptable_response(self, turn, context_turn):
        """
        Return true/false if given turn meets some acceptability criteria 
        - Must be longer than a certain length
        - Pause length: must to be shorter than the preceeding context turn (avoid total overlap)
        - Speaker ID: shouldn't be from the same speaker as context turn

        Args:
            self (BERTDataset): (needed to get the pause preceeding the turn)
            context turn (dict): the context utterance that turn could be joined with
        Return:
            (bool) : acceptability of turn as a response wrt context_turn
        """

        # Needs a preceeding turn to compute associated pause length
        if turn['turn_id'] == 0:
            return False

        # Check audio length
        turn_len = turn['stop'] - turn['start']
        if turn_len > self.max_response_s or turn_len < self.min_response_s:
            return False

        # Make sure it's not from the same speaker
        if (turn['conv_id'] == context_turn['conv_id']) and (turn['speaker'] == context_turn['speaker']):
            return False

        # Make sure previous pause is not longer than the context turn audio
        try:
            prev_turn = next(t for t in self.all_turns if (t['conv_id'] == turn['conv_id']) and (t['turn_id'] == turn['turn_id'] - 1))
            pause = turn['start'] - prev_turn['stop']
            if pause > (context_turn['stop'] - context_turn['start']):
                return False
        except Exception as e: 
            print(e)
            return False
        return True
    
    def sample_acceptable_responses(self, context_turn, n, utter=None):
        """
        Return n acceptable responses wrt a peice of context.  Acceptability is based on conditions:
        - Appropriate pause length (has to be shorter than the preceeding context turn (avoid total overlap))
        - shouldn't be from the same speaker in the conversation
        
        Args:
            context_turn ():
            n (int): number of responses to return
            utter (string): [NOT IMPLEMENTED] for sampling lexically equivilant responses
          
        Return:
            dfs
        """

        all_acceptable_responses = [t for t in self.all_turns if self.acceptable_response(t, context_turn)]

        # Sample responses with the same string (not implemented)
        # if utter:

        if self.meta:
            unique_acc_turns = []
            unique_acc_text = []
            for t in all_acceptable_responses:
                if t['clean_text'] not in unique_acc_text:
                    unique_acc_turns.append(t)
                    unique_acc_text.append(t['clean_text'])
        else:
            unique_acc_turns = list(set(unique_acc_turns))
            
        np.random.seed(int(context_turn["turn_id"]))
        lines = list(np.random.choice(unique_acc_turns, size=n, replace=False))
        return lines
      
    def make_cr_sample(self, item, num_negs, seed=123):
        """
        Create (context,responses) instance that can be passed to a scoring model. Use a given index (item) to select an acceptable context, then
        select num_neg negative, acceptable responses.
        
        Method taken from those for build fine-tuning datasets from short-contexts.
        
        Args:
            item (int): to select an acceptable context
            num_negs (int): number of negative (acceptable) responses to return
            seed (int):
        Return:
            tokenized_samples (list): token ids for true response and all n_neg (context, response) pairs in format for BERT-FP model
                (self.tokenizer.convert_tokens_to_ids(t1_tokens + EOS + t2_tokens + EOS + t3_tokens + EOS + SEP + response_tokens))
            ys (list): labels for tokenized samples ([1, 0, 0, ..., 0])
            context_lines (list): the 3 (hardcoded) lines of context
            lines (list): [response, n_negs_1, n_negs_2, ...]
        """
        # Get the context (an acceptable last turn)
        sample = self.acceptable_context_turns[item]
        doc_id = self.conv_to_doc[sample["conv_id"]]

        t1 = self.all_docs[doc_id][sample["turn_id"] - 2]
        t2 = self.all_docs[doc_id][sample["turn_id"] - 1]
        t3 = self.all_docs[doc_id][sample["turn_id"]]
        context_lines = [t1, t2, t3]

        t1_text = t1
        t2_text = t2
        t3_text = t3
        if self.meta:
            t1_text = t1['clean_text']
            t2_text = t2['clean_text']
            t3_text = t3['clean_text']

        tokens_a = self.tokenizer.tokenize(t1_text)+[self.tokenizer.eos_token]+self.tokenizer.tokenize(t2_text)+[self.tokenizer.eos_token]+self.tokenizer.tokenize(t3_text)

        response_sample = {"doc_id": doc_id, "line": sample["turn_id"] + 1}
        response = self.all_docs[doc_id][response_sample["line"]]
        response_text = response
        if self.meta:
            response_text = response_text['clean_text']

        # Store lines from all_docs that correspond with tokenized_samples
        lines = [response]

        # Get negative responses
        neg_responses = self.sample_acceptable_responses(t3, num_negs)
        tokens_bs = [self.tokenizer.tokenize(response_text)]    
        for n in neg_responses:
            lines.append(n)
            if self.meta:
                n = n['clean_text']
            tokens_bs.append(self.tokenizer.tokenize(n))

        # Join samples and convert to ids
        tokenized_samples = [self.tokenizer.convert_tokens_to_ids(tokens_a + [self.tokenizer.eos_token] + [self.tokenizer.sep_token] + tokens_b) for tokens_b in tokens_bs]

        # Build labels
        ys = [1]
        ys.extend(list(np.zeros(num_negs, dtype=int)))

        # Return lines from all_docs that correspond with tokenized_samples
        return tokenized_samples, ys, [context_lines, lines]
      

def select_neighbours(preds, ids, lines, n, subset=None, reverse=False):
    """
    Return the sorted scores and corresponding token_id sequences (+ the argsort order)
    
    Args
        preds (list of ints): model scores for each instance
        ids (list of lists): corresponding token ids 
        n (int): number of neighbours to return
        subset (int): set to only consider the first k preds/ids
    """
    
    if subset:
        preds = preds[:subset]
        ids = ids[:subset]
    
    sort_ids = np.argsort(preds)[::-1]
    if reverse:
        sort_ids = np.argsort(preds)
    
    return [preds[s] for s in sort_ids[:n]], [ids[s] for s in sort_ids[:n]], [lines[s] for s in sort_ids[:n]],  sort_ids
    
    
# def convert_to_string(ids, tokenizer=tokenizer):
#     """Clean up tokens for printing"""
#     string = ' '.join(tokenizer.convert_ids_to_tokens(ids))
#     string = string.replace(" ' ", "'")
#     string = string.replace(" ##", "")
#     return string


# def convert_to_context_response(ids, tokenizer=tokenizer):
#     """Clean up context, response for printing"""
#     string = convert_to_string(ids, tokenizer)
#     cr = string.split(' [SEP] ')
#     return cr[0], cr[1]
        
#     def show_experiment(dataset, corpus, sample_id, n_negs=500-1, model=model, reverse=False):

#     tokenized_samples, ys, lines = make_unique_ft_sample(dataset, sample_id, n_negs)
#     context_lines, response_lines = lines
#     y_pred = model.predict({'cr':tokenized_samples, 'y':ys}, pred_batch_size=100)

#     print(y_pred[0])
#     # Let's print some! (version 2; this one uses the stored lines and they're the same)
#     sids = np.argsort(y_pred)
#     top_scores, top_ids, top_lines, sort_ids = select_neighbours(y_pred, tokenized_samples, 
#                                                                  response_lines, 500, 
#                                                                  reverse=reverse
#                                                                 )

# def show_experiment(dataset, corpus, sample_id, n_negs=500-1, model=model, reverse=False):

#     tokenized_samples, ys, lines = make_unique_ft_sample(dataset, sample_id, n_negs)
#     context_lines, response_lines = lines
#     y_pred = model.predict({'cr':tokenized_samples, 'y':ys}, pred_batch_size=100)

#     print(y_pred[0])
#     # Let's print some! (version 2; this one uses the stored lines and they're the same)
#     sids = np.argsort(y_pred)
#     top_scores, top_ids, top_lines, sort_ids = select_neighbours(y_pred, tokenized_samples, 
#                                                                  response_lines, 500, 
#                                                                  reverse=reverse
#                                                                 )

#     target_id = np.argwhere(sort_ids == 0)[0][0]
#     target_score = top_scores[target_id]
       

  
