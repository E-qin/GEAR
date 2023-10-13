# QWC edit 2023/09/12 15:00
import torch
import torch.nn.functional as F
from base import BaseModel
from transformers.generation_logits_process import LogitsProcessor, LogitsProcessorList


def pad_or_truncate(tensor, size, padding_value=0):
    if tensor.shape[-1] < size:
        padding = torch.full((tensor.shape[0], size - tensor.shape[-1]), padding_value, 
                             device=tensor.device,
                             dtype=tensor.dtype)
        return torch.cat((tensor, padding), dim=-1)
    else:
        return tensor[:, :size]
def find_row(a, b):
    for i, row in enumerate(b):
        if torch.equal(a, row):
            return i
    return b.shape[0]



class DecodeTreeLogitsProcess(LogitsProcessor):
    def __init__(self, decode_tree, num_beams, batch_size):
        self.decode_tree = decode_tree
        self.num_beams = num_beams
        self.batch_size = batch_size


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        '''
        process banned token based on decode tree
        
        Parameters:
            input_ids: (batch_size * num_beams, cur_len)
            scores: (batch_size * num_beams, vocab_size)
        '''
        if self.decode_tree:
            mask = torch.ones_like(scores,requires_grad=False) * float("-inf")
            for i in range(self.num_beams * self.batch_size):
                previous_path = input_ids[i, :].tolist()
                cur = self.decode_tree
                for value in previous_path[1: ]: # ignore leading pad_token  !!!!!
                    if value not in cur.children:
                        # path not in decode tree, mostly caused by beam_size > len(node children)
                        next_candidate = [1] # eos_token
                        break
                    else:
                        cur = cur.children[value]
                else:
                    next_candidate = list(cur.children.keys())
                mask[i, next_candidate] = 0
            scores = scores + mask
        return scores



class RankinglossModel_v6(BaseModel):
    ############## YWJ... ###############
    def __init__(self, model, tokenizer, data_provider, config=None, dbg=False):
        super().__init__()
        self.data_provider = data_provider
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dbg = dbg
        self.labels_size = self.config[data_provider.dataset_name + "_data_prepare"]["labels_size"] 
        self.tree_root = data_provider.root
           
    def process_label_input(self, inputs, batch_size, input_ids_seq_len):
        decoder_input_ids = torch.cat([
                torch.zeros(inputs["label_list_batch"].size(0),1,dtype=torch.int64,device=inputs["label_list_batch"].device),
                inputs["label_list_batch"]
            ],
            dim=-1)[ : , :-1 ] # (labels_sizes' SUM or batch_size * labels_size , seq_len+1-1 )

        tgt_pad_mask = torch.logical_and(decoder_input_ids!=-100,  # no -100 mask
                                         decoder_input_ids!=self.tokenizer.pad_token_id)
        tgt_pad_mask[:,0] = True # not mask is true
        
        return {
            "input_ids" : self._expand_input_tensor_no_pad(inputs['input_ids'], inputs["effective_labels_lens"]),
            "attention_mask" : self._expand_input_tensor_no_pad(inputs['attention_mask'],  inputs["effective_labels_lens"]),
            "decoder_input_ids" : decoder_input_ids,
            "decoder_attention_mask" : tgt_pad_mask
        }
    
    def process_beams_input(self, inputs, beam_search_sequences, num_beams):
        # (batch_size * nb, 1+seq_len)    
        
        decoder_input_ids = beam_search_sequences[ : , :-1 ]
        tgt_pad_mask = torch.logical_and(decoder_input_ids!=-100,  # no -100 mask
                                         decoder_input_ids!=self.tokenizer.pad_token_id)
        tgt_pad_mask[:,0] = True # not mask is true
        
        return {
                "input_ids" : self._expand_input_tensor(inputs['input_ids'],  num_beams ),
                "attention_mask" : self._expand_input_tensor(inputs['attention_mask'],  num_beams ),
                "decoder_input_ids" : decoder_input_ids,
                # "decoder_attention_mask" : tgt_pad_mask
            }
        
    def forward(self, inputs, extra=False, eval_time=False):
        '''
        inputs:
            {
                "input_ids":input_ids, 
                "attention_mask":attention_mask, 
                "label_1_batch":label_1_batch,   # pt (batch_size , seq_len)                     
                "label_list_batch":label_list_batch,  # pt (labels_sizes' SUM or batch_size * labels_size , seq_len)   
                "effective_labels_lens":effective_labels_lens_tensor, # pt (batch_size,)  
                "rl_mask_batch": torch.tensor(mask_batch, dtype=torch.int64) # pt (batch_size,)
            } , 
        '''
        
        batch_size = inputs['input_ids'].size(0)
        input_ids_seq_len = inputs['input_ids'].size(-1)
        assert inputs['label_1_batch'].size(0) == batch_size 
        
        if 'label_list_batch' in inputs:
            label_big_batch_size = inputs['label_list_batch'].size(0)
            assert label_big_batch_size == batch_size * self.labels_size or label_big_batch_size == torch.sum(inputs['effective_labels_lens']).item()
        if 'rl_mask_batch' in inputs:
            rl_mask_batch = inputs['rl_mask_batch']
            assert rl_mask_batch.size(0) == batch_size
        ####################### original loss #######################
        
        if eval_time: 
            original_loss=0.0
        else:
            outputs = self.model(
                input_ids = inputs['input_ids'].contiguous(),
                attention_mask = inputs['attention_mask'].contiguous(),
                labels = inputs['label_1_batch'].contiguous(),
                use_cache=True
                )
            original_loss = outputs.loss
        

        if extra or eval_time:
            num_beams = self.config["generate_setup"]["num_beams"]
            logits_processor = LogitsProcessorList()
            if self.config["constrained_bs"]==True:  
                logits_processor.append(DecodeTreeLogitsProcess(self.tree_root, num_beams, batch_size))
            beam_search_sequences = self.model.generate(
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                logits_processor = logits_processor,
                max_length=self.config[self.data_provider.dataset_name + "_data_loaders"]["max_output_len"],
                num_beams=self.config["generate_setup"]["num_beams"],
                num_return_sequences=self.config["generate_setup"]["num_beams"],
                do_sample = self.config["generate_setup"]["do_sample"],
                top_k = self.config["generate_setup"]["top_k"],
                top_p = self.config["generate_setup"]["top_p"],
                early_stopping=self.config["generate_setup"]["early_stopping"],
            )
            ret =  (
            self.data_provider.decode_tree_decode_batch(beam_search_sequences,self.tokenizer.eos_token_id) \
                if self.data_provider.docid_ty.startswith("decode_tree") \
                else self.tokenizer.batch_decode(beam_search_sequences, skip_special_tokens=True),
            original_loss 
            )

            return ret


        ####################### labels scores  #######################
        label_input = self.process_label_input(inputs, batch_size, input_ids_seq_len)
        labels_big_batch_output_logits_origin = self.model(**label_input).logits # (label_big_batch_size, labels_seq_len, vocab_size)
        labels_big_batch_output_logits =  F.log_softmax(labels_big_batch_output_logits_origin, dim=-1)

        labels_scores = self._get_cumulative_log_prob_0(
            labels_big_batch_output_logits, inputs['label_list_batch'],
            self.tokenizer.eos_token_id, self.tokenizer.pad_token_id) # (label_big_batch_size, )


        ####################### beam search scores  ####################### 
        num_beams = self.config["generate_setup"]["num_beams"]
        logits_processor = LogitsProcessorList()
        if self.config["constrained_bs"]==True:  
            logits_processor.append(DecodeTreeLogitsProcess(self.tree_root, num_beams, batch_size))
        
        rl_lambda = self.config["loss_setup"]["rl_lambda"] 
        expert_lambda = self.config["loss_setup"]["expert_lambda"] 

        expert_loss = -torch.sum(labels_scores,dim=0) * expert_lambda
        
         
        # # else:
        beam_search_output = self.model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            logits_processor = logits_processor,
            max_length=self.config[self.data_provider.dataset_name + "_data_loaders"]["max_output_len"],
            num_beams=self.config["generate_setup"]["num_beams"],
            num_return_sequences=self.config["generate_setup"]["num_beams"],
            do_sample = self.config["generate_setup"]["do_sample"],
            top_k = self.config["generate_setup"]["top_k"],
            top_p = self.config["generate_setup"]["top_p"],
            early_stopping=self.config["generate_setup"]["early_stopping"],
            # return_dict_in_generate=True,
            # output_scores=True
        )
        # beam_search_sequences, sequences_scores = (
        #     beam_search_output.sequences,  # (batch_size * nb, seq_len)
        #     beam_search_output.sequences_scores  # (batch_size * nb, )
        # ) 
        beam_search_sequences = beam_search_output
        
        if "loss_speed_up" in self.config and self.config["loss_speed_up"]==True:
            ret =  (
                self.data_provider.decode_tree_decode_batch(beam_search_sequences,self.tokenizer.eos_token_id) \
                    if self.data_provider.docid_ty.startswith("decode_tree") \
                    else self.tokenizer.batch_decode(beam_search_sequences, skip_special_tokens=True),
                original_loss + expert_loss
            )
            return ret
            
        
        

        beams_input = self.process_beams_input( inputs, beam_search_sequences, self.config["generate_setup"]["num_beams"])
        
        beams_big_batch_output_logits = self.model(**beams_input).logits # (batch_size * nb, labels_seq_len, vocab_size)


        beams_big_batch_output_logits =  F.log_softmax(beams_big_batch_output_logits, dim=-1)
        beams_big_batch_scores_per_node = self._get_cumulative_log_prob_1(
            beams_big_batch_output_logits, beam_search_sequences[:,1:1+inputs["label_list_batch"].shape[-1]],
            self.tokenizer.eos_token_id, self.tokenizer.pad_token_id) # (label_big_batch_size, )
        # beams_big_batch_scores_per_node with shape [bs*nb, seq_len]

        beams_big_batch_scores_per_node = beams_big_batch_scores_per_node.reshape(batch_size, -1, beams_big_batch_scores_per_node.shape[-1])

       
        
        label_ids = inputs["label_list_batch"].reshape([batch_size, self.labels_size, -1]) # shape ([bs, labels_sizes, seq_len-1])
       
        pred_ids = beam_search_sequences.reshape([batch_size, -1, beam_search_sequences.shape[-1]])
        pred_ids = pred_ids[:,:label_ids.shape[1],1:1+label_ids.shape[-1]]   # shape ([bs, labels_sizes, seq_len-1])
        reward = -torch.abs(label_ids - pred_ids)

        weight = torch.pow(self.config["cost_lambda"], torch.arange(label_ids.shape[-1], requires_grad=False).float()).to(label_ids.device)  # 
        
        mu = 1.0
        #----#
        equal_to_zero = (reward == 0)
        reward = reward * weight.unsqueeze(0).unsqueeze(1)
        reward[equal_to_zero] = mu
        
        # if not torch.equal(reward, torch.zeros_like(reward)):
        #     reward = reward * weight.unsqueeze(0).unsqueeze(1)
        # else:
        #     reward = torch.full_like(reward, mu)
        #----#
        
        
        RL_loss = (reward * - beams_big_batch_scores_per_node[:,:reward.shape[1],:]).mean()  * rl_lambda
  
        ret =  (
            self.data_provider.decode_tree_decode_batch(beam_search_sequences,self.tokenizer.eos_token_id) \
                if self.data_provider.docid_ty.startswith("decode_tree") \
                else self.tokenizer.batch_decode(beam_search_sequences, skip_special_tokens=True),
            original_loss + RL_loss + expert_loss
        )

        return ret

    
    ####################### for REINFORCE #######################
    
    def _get_reward(self, beam_search_list_batch, label_list_batch, lambda_1=1., lambda_2=5e-2):
        '''
        both pred_list and label_list are list (with length batch_size) of tensors, 
        each of tensor is shaped as ([candidate_size, seq_len]), 
        candidate_size can be varying
        '''
        assert len(beam_search_list_batch) == len(label_list_batch), \
            "beam_search_list_batch and label_list_batch should have the same length"
            
        diff_list = []

        for beam_search, label in zip(beam_search_list_batch, label_list_batch):
            assert beam_search.ndim == 2 and label.ndim == 2, \
                    "The dimension of input tensors should be 2 (k, seq_len)"
            weight_decay = torch.ones_like(beam_search) * lambda_1
            weight_decay *= torch.arange(beam_search.shape[0], 0, -1).view(-1, 1)  
            weight_decay *= (lambda_2 ** torch.arange(0, beam_search.shape[1], 1))
            diff = -torch.abs(beam_search - label) 
            diff = diff * weight_decay
            diff_list.append(diff)
        
        ans = torch.cat(diff_list, dim=0)
        return ans
    
    
    def _get_reward_old(self, pred_list, label_list, lambda_1=1., lambda_2=1e-1):
        '''
        both pred_list and label_list are list (with length batch_size) of tensors, 
        each of tensor is shaped as ([candidate_size, seq_len]), candidate_size can be varying
        return a tensor with shape ([batch_size]), which means the reward per sample
        '''
        # get maximum candidate size
        max_candidate_size = max([tensor.shape[0] for tensor in pred_list])

        # padding tensors in pred_list and label_list to shape (max_candidate_size, seq_len)
        padded_pred_list, padded_label_list = [], []
        for pred, label in zip(pred_list, label_list):
            padded_pred_list.append(pad(pred, (0, 0, 0, max_candidate_size - pred.shape[0])))
            padded_label_list.append(pad(label, (0, 0, 0, max_candidate_size - label.shape[0])))

        # convert list of tensors into a new tensor
        preds, labels = torch.stack(padded_pred_list), torch.stack(padded_label_list)
        # generate weight for each candidate and sequence
        bs, seq_len = labels.shape[0], labels.shape[-1]
        
        candidate_weight = torch.ones(max_candidate_size, dtype=torch.float32)  * lambda_1
        seq_index = torch.arange(0, seq_len, 1)
        seq_len_weight = torch.pow(lambda_2, seq_index)  
        # candidate_weight is shaped ([max_candidate])
        # seq_len_weight is shaped ([seq_len])

        # generate the weight matrix
        weight = candidate_weight.unsqueeze(-1) @ seq_len_weight.unsqueeze(0)
        # weight is shaped ([max_candidate, seq_len])
        weight_batch = weight.expand((bs,weight.shape[0], weight.shape[1]))
        # weight_batch is shaped ([batch_size, max_candidate, seq_len])

        # compute difference
        diff = torch.abs(preds-labels)  # compute difference tensor by tensor

        diff = diff * weight_batch # multiply the weight
        return diff
            
            
    ####################### for labels scores #######################
    
    
    def _expand_input_tensor(self, input_tensor, extra_dim:int ):    
        return input_tensor.repeat_interleave(extra_dim, dim=0)
        # return input_tensor.unsqueeze(1)\
        #         .expand(batch_size, extra_dim, seq_len)\
        #         .contiguous().view(batch_size*extra_dim, seq_len)
                
    def _expand_input_tensor_no_pad(self, input_tensor, lens:torch.Tensor):    
        return input_tensor.repeat_interleave(lens, dim=0)
    
    def _get_cumulative_log_prob(self, prob_logits, tgt_ids ,eos_token_id:int, pad_token_id:int):
        '''
        prob_logits: pt (big_batch_size(2+3), labels_seq_len, vocab_size)
        tgt_ids: pt (big_batch_size(2+3), labels_seq_len)
        '''
        dim0,dim1,_ = prob_logits.size()
        assert dim0 == tgt_ids.size(0) and dim1 == tgt_ids.size(1)
        assert type(eos_token_id) == int and type(pad_token_id) == int
        
        tgt_ids = tgt_ids.contiguous().view(-1)
        prob_logits = prob_logits.view(-1, prob_logits.size(-1))
        tgt_prob_logits = prob_logits[torch.arange(prob_logits.size(0)), tgt_ids]
        tgt_prob_logits = tgt_prob_logits.view(dim0,dim1) # shape: (batch_size * labels_size, labels_seq_len)

        tgt_ids = tgt_ids.view(dim0,dim1) # shape: (batch_size * labels_size, labels_seq_len)
        tobe_0_mask = (tgt_ids == pad_token_id)
        tgt_prob_logits[tobe_0_mask] = 0.0
        div_len = torch.sum(~tobe_0_mask,dim=-1) + ((tgt_ids[:, -1] != eos_token_id) & (tgt_ids[:, -1] != pad_token_id)).long()  
        scores = torch.sum(tgt_prob_logits,dim=-1)/div_len # shape: (batch_size * labels_size, )
        
        return scores, tgt_prob_logits
    
    def _get_cumulative_log_prob_0(self, prob_logits, tgt_ids ,eos_token_id:int, pad_token_id:int):
        dim0,dim1,dim2 = prob_logits.size()
        assert dim0 == tgt_ids.size(0) and dim1 == tgt_ids.size(1)
        assert type(eos_token_id) == int and type(pad_token_id) == int
        
        tgt_ids = tgt_ids.contiguous().view(-1)
        prob_logits = prob_logits.view(-1, prob_logits.size(-1))
        tgt_prob_logits = prob_logits[torch.arange(prob_logits.size(0)), tgt_ids]
        tgt_prob_logits = tgt_prob_logits.view(dim0,dim1) # shape: (batch_size * labels_size, labels_seq_len)
        tgt_ids = tgt_ids.view(dim0,dim1) # shape: (batch_size * labels_size, labels_seq_len)
        tobe_0_mask = (tgt_ids == pad_token_id)
        tgt_prob_logits[tobe_0_mask] = 0.0
        div_len = torch.sum(~tobe_0_mask,dim=-1) + ((tgt_ids[:, -1] != eos_token_id) & (tgt_ids[:, -1] != pad_token_id)).long()  
        scores = torch.sum(tgt_prob_logits,dim=-1)/div_len # shape: (batch_size * labels_size, )
        return scores

    def _get_cumulative_log_prob_1(self, prob_logits, tgt_ids ,eos_token_id:int, pad_token_id:int):
        dim0,dim1,dim2 = prob_logits.size()
        assert dim0 == tgt_ids.size(0) and dim1 == tgt_ids.size(1)
        assert type(eos_token_id) == int and type(pad_token_id) == int
        
        tgt_ids = tgt_ids.contiguous().view(-1)
        prob_logits = prob_logits.view(-1, prob_logits.size(-1))
        tgt_prob_logits = prob_logits[torch.arange(prob_logits.size(0)), tgt_ids]
        tgt_prob_logits = tgt_prob_logits.view(dim0,dim1) # shape: (batch_size * labels_size, labels_seq_len)
        tgt_ids = tgt_ids.view(dim0,dim1) # shape: (batch_size * labels_size, labels_seq_len)
        tobe_0_mask = (tgt_ids == pad_token_id)
        tgt_prob_logits[tobe_0_mask] = 0.0
        return tgt_prob_logits
    
    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
    


   
   
   
   
   


