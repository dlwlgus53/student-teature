import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

import ontology
from utils.masked_loss import masked_cross_entropy


class ContextEncoder(nn.Module):
    def __init__(self, hidden_size, num_gates, num_domains, dropout, pad_idx, update_idx, domain_idx, slot_idx, assets_path):
        super(ContextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
        self.update_idx = update_idx
        self.domain_idx = domain_idx
        self.slot_idx = slot_idx
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear_gate = nn.Linear(hidden_size, num_gates)
        self.domain_state_tracker = nn.Linear(hidden_size, 2)

        # initialize embeddings of [hotel], ..., [police] to embeddings of hotel, ..., police
        domain_mapping = json.load(open(os.path.join(assets_path, "domain_mapping.json"), "r"))
        for key, value in domain_mapping.items():
            idx = int(key)
            self.bert.embeddings.word_embeddings.weight.data[idx] = self.bert.embeddings.word_embeddings.weight.data[value].clone()
    
    def forward(self, contexts, turn_segment, gates, max_num_updates, train):
        """
        contexts: [batch, time]
        turn_segment: [batch. time]
        gates: [batch, slots]
        """

        batch_size = contexts.size(0)
        pad_mask = (contexts != self.pad_idx)
        outputs, pooled_output = self.bert(contexts, attention_mask=pad_mask, token_type_ids=turn_segment)
            # [batch, time, hidden], [batch, hidden]

        domain_positions = (contexts == self.domain_idx)
        domain_outputs = outputs[domain_positions].view(batch_size, len(ontology.all_domains), self.hidden_size)  # [batch, domains, hidden]
        domain_state = self.domain_state_tracker(self.dropout(domain_outputs))  # [batch, domains, 2]

        slot_positions = (contexts == self.slot_idx)
        gate_outputs = outputs[slot_positions].view(batch_size, len(ontology.all_info_slots), self.hidden_size)  # [batch, slots, hidden]
        gate_scores = self.linear_gate(self.dropout(gate_outputs))  # [batch, slots, gates]

        if train:
            pred_gates = gates
        else:
            pred_gates = gate_scores.argmax(dim=2)  # [batch, slots]
            max_num_updates = (pred_gates == self.update_idx).sum(dim=1).max().item()

        # use BERT representations of [SLOT] as initial hidden state of GRUs
        updates = []
        for output, gate in zip(gate_outputs, pred_gates.eq(self.update_idx)):
            if gate.sum().item() != 0:  # update gate
                selected = output.masked_select(gate.unsqueeze(dim=1)).view(-1, self.hidden_size)  # [slots, hidden]
                num_updates = selected.size(0)
                pad_size = max_num_updates - num_updates
                if pad_size > 0:
                    pad = torch.zeros(pad_size, self.hidden_size).cuda()
                    padded = torch.cat([selected, pad], dim=0)
                else:
                    padded = selected
            else:  # other gate
                padded = torch.zeros(max_num_updates, self.hidden_size).cuda()
            updates.append(padded)  # [slots, hidden] * batch
        decoder_inputs = torch.stack(updates, dim=0)  # [batch, slots, hidden]

        return outputs, pooled_output, gate_scores, decoder_inputs, domain_state


class BeliefTracker(nn.Module):
    def __init__(self, vocab_size, shared_embedding, hidden_size, dropout, max_value_len, pad_idx, domain_idx, slot_idx, \
                attention_projection_layer, weight_tying):
        super(BeliefTracker, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = shared_embedding
        self.embedding_size = shared_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.embedding_size, hidden_size, batch_first=True)
        self.max_value_len = max_value_len
        self.pad_idx = pad_idx
        self.domain_idx = domain_idx
        self.slot_idx = slot_idx
        self.attention_projection_layer = attention_projection_layer
        self.weight_tying = weight_tying

        # use embedding matrix of encoder to decode words
        if weight_tying:
            self.linear_vocab = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.linear_vocab = nn.Linear(hidden_size * 2, vocab_size)
        
        # add projection layers for attention
        if attention_projection_layer:
            self.linear_query = nn.Linear(hidden_size, hidden_size)
            self.linear_key = nn.Linear(hidden_size, hidden_size)
            self.linear_value = nn.Linear(hidden_size, hidden_size)

    def forward(self, encoder_outputs, encoder_hidden, decoder_inputs, contexts, belief, train):
        """
        encoder_outputs: [batch, time, hidden]
        encoder_hidden: [batch, hidden]
        decoder_inputs: [batch, slots, hidden]
        contexts: [batch, time]
        belief: [batch, slots, time]
        """
        
        batch_size = contexts.size(0)
        max_value_len = belief.size(2) if train else self.max_value_len
        pad_mask = (contexts == self.pad_idx)  # [batch, time]
        num_updates = decoder_inputs.size(1)
        all_probs = torch.zeros(batch_size, num_updates, max_value_len, self.vocab_size).cuda()

        if num_updates == 0:
            return None

        for slot_idx in range(num_updates):
            if train:
                words = torch.cat([decoder_inputs[:, slot_idx, :].unsqueeze(dim=1), self.embedding(belief[:, slot_idx, :-1])], dim=1)
                    # [batch, time, hidden]
                words = self.dropout(words)
                outputs, hidden = self.gru(words, encoder_hidden.unsqueeze(dim=0))  # [batch, time, hidden], [1, batch, hidden]
                
                if self.attention_projection_layer:
                    query = self.linear_query(outputs)  # [batch, time, hidden]
                    key = self.linear_key(encoder_outputs)  # [batch, time, hidden]
                    value = self.linear_value(encoder_outputs)  # [batch, time, hidden]
                else:
                    query = outputs
                    key = encoder_outputs
                    value = encoder_outputs
                
                attention_context = torch.matmul(query, key.transpose(1, 2))  # [batch, time, time]
                attention_context = attention_context.masked_fill(pad_mask.unsqueeze(dim=1), value=-1e+9)
                attention_context = F.softmax(attention_context, dim=-1)
                context_vector = torch.matmul(attention_context, value)  # [batch, time, hidden]
                
                if self.weight_tying:
                    logits = self.linear_vocab(torch.cat([outputs, context_vector], dim=2))  # [batch, time, hidden]
                    logits = torch.matmul(logits, self.embedding.weight.transpose(0, 1))  # [batch, time, vocab]
                else:
                    logits = self.linear_vocab(torch.cat([outputs, context_vector], dim=2))  # [batch, time, vocab]
                probs = F.softmax(logits, dim=-1)
                
                # prevent to generate [DOMAIN] or [SLOT]
                domain_mask = (torch.arange(self.vocab_size) == self.domain_idx).cuda()
                slot_mask = (torch.arange(self.vocab_size) == self.slot_idx).cuda()
                probs = probs.masked_fill(domain_mask, value=-1e+9)
                probs = probs.masked_fill(slot_mask, value=-1e+9)

                all_probs[:, slot_idx, :, :] = probs
            else:
                word = decoder_inputs[:, slot_idx, :].unsqueeze(dim=1)  # [batch, 1, hidden]
                hidden = encoder_hidden.unsqueeze(dim=0)  # [1, batch, hidden]

                for i in range(max_value_len):
                    word = self.dropout(word)
                    outputs, hidden = self.gru(word, hidden)  # [batch, 1, hidden], [1, batch, hidden]

                    if self.attention_projection_layer:
                        query = self.linear_query(hidden.transpose(0, 1))  # [batch, 1, hidden]
                        key = self.linear_key(encoder_outputs)  # [batch, time, hidden]
                        value = self.linear_value(encoder_outputs)  # [batch, time, hidden]
                    else:
                        query = hidden.transpose(0, 1)
                        key = encoder_outputs
                        value = encoder_outputs
                    
                    attention_context = torch.matmul(query, key.transpose(1, 2))  # [batch, 1, time]
                    attention_context = attention_context.masked_fill(pad_mask.unsqueeze(dim=1), value=-1e+9)
                    attention_context = F.softmax(attention_context, dim=-1)
                    context_vector = torch.matmul(attention_context, value)  # [batch, 1, hidden]

                    if self.weight_tying:
                        logits = self.linear_vocab(torch.cat([hidden.squeeze(dim=0), context_vector.squeeze(dim=1)], dim=1))  # [batch, hidden]
                        logits = torch.matmul(logits, self.embedding.weight.transpose(0,1))  # [batch, vocab]
                    else:
                        logits = self.linear_vocab(torch.cat([hidden.squeeze(dim=0), context_vector.squeeze(dim=1)], dim=1))  # [batch, vocab]
                    probs = F.softmax(logits, dim=-1)
                    
                    domain_mask = (torch.arange(self.vocab_size) == self.domain_idx).cuda()
                    slot_mask = (torch.arange(self.vocab_size) == self.slot_idx).cuda()
                    probs = probs.masked_fill(domain_mask, value=-1e+9)
                    probs = probs.masked_fill(slot_mask, value=-1e+9)

                    all_probs[:, slot_idx, i, :] = probs
                    word = self.embedding(probs.argmax(dim=1)).unsqueeze(dim=1)  # [batch, 1, hidden]

        return all_probs


class DialoguePolicy(nn.Module):
    def __init__(self, shared_embedding, hidden_size, dropout, max_act_len, vocab_size, pad_idx, cls_idx, eos_idx, \
                attention_projection_layer, weight_tying, assets_path):
        super(DialoguePolicy, self).__init__()
        self.embedding = shared_embedding
        self.embedding_size = shared_embedding.embedding_dim
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        self.max_act_len = max_act_len
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx
        self.eos_idx = eos_idx
        
        """
        input of GRU is concat of previous word and [CLS] representation.
        this is jsut a trick in implementation, not reported in paper.
        """
        input_dim = self.embedding_size + hidden_size
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.attention_projection_layer = attention_projection_layer
        self.weight_tying = weight_tying

        if weight_tying:
            self.linear_vocab = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.linear_vocab = nn.Linear(hidden_size * 2, vocab_size)
        
        if attention_projection_layer:
            self.linear_query = nn.Linear(hidden_size, hidden_size)
            self.linear_key = nn.Linear(hidden_size, hidden_size)
            self.linear_value = nn.Linear(hidden_size, hidden_size)

    def forward(self, context_outputs, context_hidden, context, action, mode="train"):
        """
        context_outputs: [batch, time, hidden]
        context_hidden: [batch, hidden]
        context: [batch, time]
        action: [batch, time]
        """

        batch_size = context_outputs.size(0)
        max_act_len = action.size(1) if mode == "train" else self.max_act_len

        bos_batch = torch.ones(batch_size, 1, dtype=torch.int64).cuda() * self.cls_idx
        bos_batch = self.embedding(bos_batch)  # [batch, 1, hidden]

        hidden = context_hidden.unsqueeze(dim=0)
        pad_mask = (context == self.pad_idx)

        if mode == "train":
            inputs = torch.cat([bos_batch, self.embedding(action[:, :-1])], dim=1)  # [batch, time, hidden]
            inputs = torch.cat([inputs, context_outputs[:, 0, :].unsqueeze(dim=1).repeat(1, inputs.size(1), 1)], dim=2)
                # [batch, time, hidden * 2]
            inputs = self.dropout(inputs)
            outputs, hidden = self.gru(inputs, hidden)  # [batch, time, hidden], [1, batch, hidden]

            if self.attention_projection_layer:
                query = self.linear_query(outputs)  # [batch, time, hidden]
                key = self.linear_key(context_outputs)  # [batch, time, hidden]
                value = self.linear_value(context_outputs)  # [batch, time, hidden]
            else:
                query = outputs
                key = context_outputs
                value = context_outputs

            attention_context = torch.matmul(query, key.transpose(1, 2))  # [batch, time, time]
            attention_context = attention_context.masked_fill(pad_mask.unsqueeze(dim=1), value=-1e+9)
            attention_context = F.softmax(attention_context, dim=2)
            context_vector = torch.matmul(attention_context, value)  # [batch, time, hidden]

            if self.weight_tying:
                logits = self.linear_vocab(torch.cat([outputs, context_vector], dim=2))  # [batch, time, hidden]
                logits = torch.matmul(logits, self.embedding.weight.transpose(0,1))  # [batch, time, vocab]
            else:
                logits = self.linear_vocab(torch.cat([outputs, context_vector], dim=2))  # [batch, time, vocab]
            all_probs = F.softmax(logits, dim=-1)

            return all_probs
        elif mode == "test":
            all_probs = torch.zeros(batch_size, max_act_len, self.vocab_size).cuda()
            inputs = torch.cat([bos_batch, context_outputs[:, 0, :].unsqueeze(dim=1)], dim=2)

            for idx in range(max_act_len):
                inputs = self.dropout(inputs)
                output, hidden = self.gru(inputs, hidden)  # [batch, 1, hidden], [1, batch, hidden]

                if self.attention_projection_layer:
                    query = self.linear_query(hidden.transpose(0, 1))  # [batch, 1, hidden]
                    key = self.linear_key(context_outputs)  # [batch, time, hidden]
                    value = self.linear_value(context_outputs)  # [batch, time, hidden]
                else:
                    query = hidden.transpose(0, 1)
                    key = context_outputs
                    value = context_outputs
                
                attention_context = torch.matmul(query, key.transpose(1, 2))  # [batch, 1, time]
                attention_context = attention_context.masked_fill(pad_mask.unsqueeze(dim=1), value=-1e+9)
                attention_context = F.softmax(attention_context, dim=2)
                context_vector = torch.matmul(attention_context, value)  # [batch, 1, hidden]

                if self.weight_tying:
                    logits = self.linear_vocab(torch.cat([hidden.squeeze(dim=0), context_vector.squeeze(dim=1)], dim=1))  # [batch, hidden]
                    logits = torch.matmul(logits, self.embedding.weight.transpose(0, 1))  # [batch, vocab]
                else:
                    logits = self.linear_vocab(torch.cat([hidden.squeeze(dim=0), context_vector.squeeze(dim=1)], dim=1))  # [batch, vocab]
                probs = F.softmax(logits, dim=-1)

                all_probs[:, idx, :] = probs

                inputs = self.embedding(probs.argmax(dim=1)).unsqueeze(dim=1)  # [batch, 1, hidden]
                inputs = torch.cat([inputs, context_outputs[:, 0, :].unsqueeze(dim=1)], dim=2)
            
            return all_probs
        elif mode == "rl":
            all_probs = torch.zeros(batch_size, max_act_len, self.vocab_size).cuda()
            actions = []
            log_probs = []
            inputs = torch.cat([bos_batch, context_outputs[:, 0, :].unsqueeze(dim=1)], dim=2)
            eos_idx = torch.ones(batch_size).cuda() * max_act_len

            for idx in range(max_act_len):
                inputs = self.dropout(inputs)
                output, hidden = self.gru(inputs, hidden)  # [batch, 1, hidden], [1, batch, hidden]

                if self.attention_projection_layer:
                    query = self.linear_query(hidden.transpose(0, 1))  # [batch, 1, hidden]
                    key = self.linear_key(context_outputs)  # [batch, time, hidden]
                    value = self.linear_value(context_outputs)  # [batch, time, hidden]
                else:
                    query = hidden.transpose(0, 1)
                    key = context_outputs
                    value = context_outputs
                
                attention_context = torch.matmul(query, key.transpose(1, 2))  # [batch, 1, time]
                attention_context = attention_context.masked_fill(pad_mask.unsqueeze(dim=1), value=-1e+9)
                attention_context = F.softmax(attention_context, dim=2)
                context_vector = torch.matmul(attention_context, value)  # [batch, 1, hidden]

                if self.weight_tying:
                    logits = self.linear_vocab(torch.cat([hidden.squeeze(dim=0), context_vector.squeeze(dim=1)], dim=1))  # [batch, hidden]
                    logits = torch.matmul(logits, self.embedding.weight.transpose(0,1))  # [batch, vocab]
                else:
                    logits = self.linear_vocab(torch.cat([hidden.squeeze(dim=0), context_vector.squeeze(dim=1)], dim=1))  # [batch, vocab]
                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)  # [batch, vocab]

                all_probs[:, idx, :] = probs

                sampled_action = torch.multinomial(probs, 1)  # [batch, 1]
                sampled_log_prob = log_prob.gather(1, sampled_action).squeeze(dim=1)  # [batch]
                log_probs.append(sampled_log_prob)
                actions.append(sampled_action.squeeze(dim=1))

                inputs = self.embedding(sampled_action)  # [batch, 1, hidden]
                inputs = torch.cat([inputs, context_outputs[:, 0, :].unsqueeze(dim=1)], dim=2)

                # save the position of [EOS]
                for batch_idx in range(batch_size):
                    if sampled_action[batch_idx, 0] == self.eos_idx and eos_idx[batch_idx] == max_act_len:
                        eos_idx[batch_idx] = idx

            # mask the actions generated after [EOS]
            log_probs = torch.stack(log_probs, dim=1)  # [batch, time]
            actions = torch.stack(actions, dim=1)  # [batch, time]
            masked_log_probs = []  # [batch, time]
            for batch_idx in range(batch_size):
                after_eos_mask = torch.arange(0, max_act_len).cuda()  # [time]
                after_eos_mask = (after_eos_mask <= eos_idx[batch_idx])  # [time]
                masked_log_probs.append(torch.masked_select(log_probs[batch_idx], after_eos_mask))  # [time]

            return actions, masked_log_probs


class ResponseGenerator(nn.Module):
    def __init__(self, shared_embedding, hidden_size, dropout, max_sentence_len, vocab_size, cls_idx, pad_idx, eos_idx, \
                attention_projection_layer, weight_tying):
        super(ResponseGenerator, self).__init__()
        self.embedding = shared_embedding
        self.embedding_size = self.embedding.embedding_dim
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.max_sentence_len = max_sentence_len
        self.vocab_size = vocab_size
        self.cls_idx = cls_idx
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        input_dim = self.embedding_size + hidden_size
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.attention_projection_layer = attention_projection_layer
        self.weight_tying = weight_tying

        if weight_tying:
            self.linear_vocab = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.linear_vocab = nn.Linear(hidden_size * 2, vocab_size)
        
        if attention_projection_layer:
            self.linear_query = nn.Linear(hidden_size, hidden_size)
            self.linear_key = nn.Linear(hidden_size, hidden_size)
            self.linear_value = nn.Linear(hidden_size, hidden_size)

    def forward(self, context_outputs, context_hidden, context, response, train):
        """
        context_outputs: [batch, time, hidden]
        context_hidden: [batch, hidden]
        context: [batch, time]
        response: [batch, time]
        """

        batch_size = context_outputs.size(0)
        max_sentence_len = response.size(1) if train else self.max_sentence_len
        
        hidden = context_hidden.unsqueeze(dim=0)
        pad_mask = (context == self.pad_idx)

        if train:
            inputs = torch.cat([self.embedding(response), context_outputs[:, 0, :].unsqueeze(dim=1).repeat(1, response.size(1), 1)], dim=2)
                # [batch, time, hidden * 2]
            inputs = self.dropout(inputs)
            outputs, hidden = self.gru(inputs, hidden)  # [batch, time, hidden], [1, batch, hidden]

            if self.attention_projection_layer:
                query = self.linear_query(outputs)  # [batch, time, hidden]
                key = self.linear_key(context_outputs)  # [batch, time, hidden]
                value = self.linear_value(context_outputs)  # [batch, time, hidden]
            else:
                query = outputs
                key = context_outputs
                value = context_outputs

            attention_context = torch.matmul(query, key.transpose(1,2))  # [batch, time, time]
            attention_context = attention_context.masked_fill(pad_mask.unsqueeze(dim=1), value=-float("inf"))
            attention_context = F.softmax(attention_context, dim=2)
            context_vector = torch.matmul(attention_context, value)  # [batch, time, hidden]

            if self.weight_tying:
                logits = self.linear_vocab(torch.cat([outputs, context_vector], dim=2))  # [batch, time, hidden]
                logits = torch.matmul(logits, self.embedding.weight.transpose(0,1))  # [batch, time, vocab]
            else:
                logits = self.linear_vocab(torch.cat([outputs, context_vector], dim=2))  # [batch, time, vocab]
            all_probs = F.softmax(logits, dim=-1)
        else:
            bos_batch = torch.ones(batch_size, 1, dtype=torch.int64).cuda() * self.cls_idx
            bos_batch = self.embedding(bos_batch)  # [batch, 1, hidden]
            inputs = torch.cat([bos_batch, context_outputs[:, 0, :].unsqueeze(dim=1)], dim=2)  # [batch, 1, hidden * 2]
            all_probs = torch.zeros(batch_size, max_sentence_len, self.vocab_size).cuda()

            for idx in range(max_sentence_len):
                inputs = self.dropout(inputs)
                output, hidden = self.gru(inputs, hidden)  # [batch, 1, hidden], [1, batch, hidden]

                if self.attention_projection_layer:
                    query = self.linear_query(hidden.transpose(0,1))  # [batch, 1, hidden]
                    key = self.linear_key(context_outputs)  # [batch, time, hidden]
                    value = self.linear_value(context_outputs)  # [batch, time, hidden]
                else:
                    query = hidden.transpose(0,1)
                    key = context_outputs
                    value = context_outputs

                attention_context = torch.matmul(query, key.transpose(1,2))  # [batch, 1, time]
                attention_context = attention_context.masked_fill(pad_mask.unsqueeze(dim=1), value=-float("inf"))
                attention_context = F.softmax(attention_context, dim=2)
                context_vector = torch.matmul(attention_context, value)  # [batch, 1, hidden]

                if self.weight_tying:
                    logits = self.linear_vocab(torch.cat([hidden.squeeze(dim=0), context_vector.squeeze(dim=1)], dim=1))  # [batch, hidden]
                    logits = torch.matmul(logits, self.embedding.weight.transpose(0,1))  # [batch, vocab]
                else:
                    logits = self.linear_vocab(torch.cat([hidden.squeeze(dim=0), context_vector.squeeze(dim=1)], dim=1))  # [batch, vocab]
                probs = F.softmax(logits, dim=-1)

                all_probs[:, idx, :] = probs

                inputs = self.embedding(probs.argmax(dim=1)).unsqueeze(dim=1)
                inputs = torch.cat([inputs, context_outputs[:, 0, :].unsqueeze(dim=1)], dim=2)

        return all_probs


class DORA(nn.Module):
    def __init__(self, db, config):
        super(DORA, self).__init__()
        self.vocab_size = config.vocab_size
        with open(os.path.join(config.assets_path, "never_split.txt"), "r") as f:
            never_split = f.read().split("\n")
        self.tokenizer = BertTokenizer(os.path.join(config.assets_path, "vocab.txt"), never_split=never_split)
        self.db = db
        self.cls_idx = config.cls_idx
        self.sep_idx = config.sep_idx
        self.pad_idx = config.pad_idx
        self.eos_idx = config.eos_idx
        self.update_idx = config.update_idx
        self.delete_idx = config.delete_idx
        self.copy_idx = config.copy_idx
        self.dontcare_idx = config.dontcare_idx
        self.slot_idx = config.slot_idx
        self.domain_idx = config.domain_idx
        self.action_idx = config.action_idx
        self.max_context_len = config.max_context_len
        self.max_value_len = config.max_value_len
        self.max_belief_len = config.max_belief_len
        self.max_act_len = config.max_act_len
        self.num_domains = config.num_domains
        self.context_encoder = ContextEncoder(config.hidden_size, config.num_gates, config.num_domains, config.dropout, config.pad_idx, \
            config.update_idx, config.domain_idx, config.slot_idx, config.assets_path)
        self.belief_tracker = BeliefTracker(config.vocab_size, self.context_encoder.bert.embeddings.word_embeddings, config.hidden_size, \
            config.dropout, config.max_value_len, config.pad_idx, config.domain_idx, config.slot_idx, config.attention_projection_layer, \
            config.weight_tying)
        self.dialogue_policy = DialoguePolicy(self.context_encoder.bert.embeddings.word_embeddings, config.hidden_size, config.dropout, \
            config.max_act_len, config.vocab_size, config.pad_idx, config.cls_idx, config.eos_idx, config.attention_projection_layer, \
            config.weight_tying, config.assets_path) 
        self.response_generator = ResponseGenerator(self.context_encoder.bert.embeddings.word_embeddings, config.hidden_size, config.dropout, \
            config.max_sentence_len, config.vocab_size, config.cls_idx, config.pad_idx, config.eos_idx, config.attention_projection_layer, \
            config.weight_tying)

    def forward(self, turn_inputs, turn_contexts, turn_segment, mode="train", postprocessig=False):
        """
        turn_inputs: {
            "user": [batch, time]
            "response": [batch, time]
            "belief": [batch, slots, time]
            "action": [batch, time]
            "usr": [batch] => string list
            "context: [batch, turns, 2] => string list
            "prev_belief": [batch, slots, time]
            "gate": [batch, slots]
            "slot_order": [batch, slots]
            "domain_state": [batch]
        }
        turn_contexts: [batch, time]
        turn_segment: [batch, time]
        """

        if mode == "train":
            return self.train_forward(turn_inputs, turn_contexts, turn_segment)
        elif mode == "val":
            return self.val_forward(turn_inputs, turn_contexts, turn_segment, postprocessig)
        elif mode == "rl":
            return self.rl_forward(turn_inputs, turn_contexts, turn_segment)

    def train_forward(self, turn_inputs, turn_contexts, turn_segment):
        batch_size = turn_contexts.size(0)

        # make belief of update slots
        gates = turn_inputs["gate"]
        max_num_updates = 0
        max_value_len = 0
        updates = []
        for batch_idx, slots in enumerate(turn_inputs["gate"]):
            updates_ = []
            for slot_idx, gate in enumerate(slots):
                slot_idx = turn_inputs["slot_order"][batch_idx, slot_idx].item()
                if gate == self.update_idx:
                    value = turn_inputs["belief"][batch_idx, slot_idx, :].tolist()  # including [EOS]
                    max_value_len = max(max_value_len, len(value))
                    updates_.append(value)
            num_updates = len(updates_)
            max_num_updates = max(max_num_updates, num_updates)
            updates.append(updates_)
        updates_label = torch.zeros(batch_size, max_num_updates, max_value_len, dtype=torch.int64).cuda()
        for batch_idx, slots in enumerate(updates):
            for slot_idx, value in enumerate(slots):
                updates_label[batch_idx, slot_idx, :] = torch.tensor(value)

        if max_num_updates > 0:
            encoder_outputs, pooled_output, gate_scores, decoder_inputs, domain_state = \
                self.context_encoder(turn_contexts, turn_segment, gates, max_num_updates, train=True)
                # [batch, time, hidden], [batch, hidden], [batch, slots, gates], [batch, slots, hidden], [batch, domains]
            belief_probs = self.belief_tracker(encoder_outputs, pooled_output, decoder_inputs, turn_contexts, updates_label, train=True)
                # [batch, slots, time, vocab]
        else:  # dummy output to prevent empty output
            updates_label = torch.ones(batch_size, 1, 10, dtype=torch.int64).cuda()
            encoder_outputs, pooled_output, gate_scores, decoder_inputs, domain_state = \
                self.context_encoder(turn_contexts, turn_segment, gates, 1, train=True)
            belief_probs = self.belief_tracker(encoder_outputs, pooled_output, decoder_inputs, turn_contexts, updates_label, train=True)

        gate_label = turn_inputs["gate"]
        gate_loss = F.cross_entropy(gate_scores.view(-1, gate_scores.size(2)), gate_label.view(-1))
        gate_pred = gate_scores.detach().argmax(dim=2)  # [batch, slots]
        gate_acc = (gate_pred == gate_label).float().mean()

        domain_label = turn_inputs["domain_state"]
        domain_loss = F.cross_entropy(domain_state.view(-1, domain_state.size(2)), domain_label.view(-1))
        domain_pred = domain_state.detach().argmax(dim=2)  # [batch, domains]
        domain_acc = (domain_pred == domain_label).float().mean()

        pad_mask = (updates_label == self.pad_idx)
        if max_num_updates > 0:
            value_loss = masked_cross_entropy(belief_probs, updates_label, pad_mask)
        else:  # dummy loss
            value_loss = belief_probs.sum() * 0

        value_label = turn_inputs["belief"]  # [batch, slots, time]
        belief_acc = torch.ones(batch_size, len(ontology.all_info_slots)).cuda()
        
        all_pred_words = belief_probs.argmax(dim=3)  # [batch, slots, time]
        belief_gen = []  # [batch, slots, time]
        for batch_idx, batch in enumerate(gate_label):
            belief_gen_ = []  # [slots, time]
            update_idx = 0
            for idx, gate in enumerate(batch):
                slot_idx = turn_inputs["slot_order"][batch_idx, idx].item()
                if gate == self.update_idx:
                    pred_words = all_pred_words[batch_idx, update_idx, :]
                    update_idx += 1
                elif gate == self.delete_idx:
                    pred_words = torch.tensor(self.tokenizer.encode("none", add_special_tokens=False) + \
                        self.tokenizer.convert_tokens_to_ids(["[EOS]"]))
                elif gate == self.dontcare_idx:
                    pred_words = torch.tensor(self.tokenizer.encode("dontcare", add_special_tokens=False) + \
                        self.tokenizer.convert_tokens_to_ids(["[EOS]"]))
                else:
                    pred_words = turn_inputs["prev_belief"][batch_idx, slot_idx, :]
                for word_idx, word in enumerate(pred_words):
                    if word == self.eos_idx:
                        break
                word_idx += 1
                pred = pred_words.tolist()[:word_idx]  # include [EOS]
                belief_gen_.append(pred)
                label = value_label[batch_idx, slot_idx, :][value_label[batch_idx, slot_idx, :] != self.pad_idx].tolist()
                if pred != label:
                    belief_acc[batch_idx, slot_idx] = 0
            belief_gen.append(belief_gen_)

        # convert belief state to sentence form
        belief_flat = []
        belief_state = value_label.tolist()
        for batch_idx, belief in enumerate(belief_state):
            belief_ = []
            for slot_idx, value in enumerate(belief):
                slot = ontology.all_info_slots[slot_idx]
                domain, slot = slot.split("-")
                slot = "[{}] - {}".format(domain, slot)
                for word_idx, word in enumerate(value):
                    if word == self.eos_idx:
                        break
                belief_ += [self.slot_idx] + self.tokenizer.encode(slot, add_special_tokens=False) + \
                    self.tokenizer.convert_tokens_to_ids(["-"]) + value[:word_idx]  # [SLOT] [domain] - slot - value
            belief_flat.append(belief_)
        
        # make belief context
        # [[CLS], user_t, [SEP], domain_state_t, belief_t, db_t, [SEP]]
        belief_contexts = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
        belief_contexts_list = []
        max_context_len = 0
        for batch_idx in range(batch_size):
            
            # [CLS], user_t, [SEP]
            user_t = turn_inputs["user"][batch_idx]
            belief_contexts_ = user_t[user_t != self.pad_idx].tolist()

            # domain_state_t
            domain_state_ = []
            for domain_idx, domain in enumerate(ontology.all_domains):
                domain_state_.append("[DOMAIN]")
                domain_state_.append("[{}]".format(domain))
                if domain_label[batch_idx, domain_idx] == 1:
                    domain_state_.append("[ON]")
                else:
                    domain_state_.append("[OFF]")
            domain_state_ = " ".join(domain_state_)
            belief_contexts_ += self.tokenizer.encode(domain_state_, add_special_tokens=False)

            # belief_t
            belief_contexts_ += belief_flat[batch_idx]  # belief_t

            # db_t
            db_result = ["[DB]"]
            for domain_idx, domain in enumerate(ontology.all_domains):
                if domain_label[batch_idx, domain_idx] == 1:
                    db_result.append("[{}]".format(domain))
                    db_result.append("{}".format(turn_inputs["db_results"][batch_idx, domain_idx].item()))
            db_result = " ".join(db_result)
            belief_contexts_ += self.tokenizer.encode(db_result, add_special_tokens=False)

            belief_contexts_.append(self.sep_idx) # [SEP]
            len_ = len(belief_contexts_)
            max_context_len = max(max_context_len, len_)
            belief_contexts_list.append(belief_contexts_)
            belief_contexts[batch_idx, :len_] = torch.tensor(belief_contexts_)
        belief_contexts = belief_contexts[:, :max_context_len]

        # make belief context vector representation
        pad_mask = (belief_contexts != self.pad_idx)
        segments = torch.ones_like(belief_contexts).cuda()
        belief_outputs, belief_hidden = self.context_encoder.bert(belief_contexts, attention_mask=pad_mask, token_type_ids=segments)
            # [batch, time, hidden], [batch, hidden]

        # generate system actions
        action_probs = self.dialogue_policy(belief_outputs, belief_hidden, belief_contexts, turn_inputs["action"], mode="train")
            # [batch, time, vocab]

        action_label = turn_inputs["action"]
        pad_mask = (action_label == self.pad_idx)
        action_loss = masked_cross_entropy(action_probs, action_label, pad_mask)

        action_pred = action_probs.argmax(dim=2).long()
        action_gen = []
        for batch_idx, action in enumerate(action_pred):
            for word_idx, word in enumerate(action):
                if word == self.eos_idx:
                    break
            action_gen.append(action.tolist()[:word_idx+1])  # including [EOS]

        # make action context
        # [[CLS], user_t, [SEP], domain_state_t, belief_t, db_t, action_t, [SEP]]
        action_contexts = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
        max_context_len = 0
        for batch_idx in range(batch_size):

            # [CLS], user_t, [SEP], domain_state_t, belief_t, db_t
            action_contexts_ = belief_contexts_list[batch_idx][:-1] 

            # action_t
            for word_idx, word in enumerate(action_label[batch_idx]):
                if word == self.eos_idx:
                    break
            action_contexts_ += action_label[batch_idx, :word_idx].tolist()

            action_contexts_.append(self.sep_idx)  # [SEP]
            len_ = len(action_contexts_)
            max_context_len = max(max_context_len, len_)
            action_contexts[batch_idx, :len_] = torch.tensor(action_contexts_)
        action_contexts = action_contexts[:, :max_context_len]

        # make action context vector representation
        pad_mask = (action_contexts != self.pad_idx)
        segments = torch.ones_like(action_contexts).cuda()
        action_outputs, action_hidden = self.context_encoder.bert(action_contexts, attention_mask=pad_mask, token_type_ids=segments)

        # generate system response
        response_probs = self.response_generator(action_outputs, action_hidden, action_contexts, turn_inputs["response_delex"][:, :-1], \
            train=True)

        response_label = turn_inputs["response_delex"][:, 1:].contiguous()
        pad_mask = (response_label == self.pad_idx)
        response_loss = masked_cross_entropy(response_probs, response_label, pad_mask)

        response_pred = response_probs.argmax(dim=2).long()
        response_gen = []
        for batch_idx, response in enumerate(response_pred):
            for word_idx, word in enumerate(response):
                if word == self.sep_idx:
                    break
            response_gen.append(response.tolist()[:word_idx+1])  # including [EOS]
        
        return domain_loss, domain_acc, gate_loss, gate_acc, value_loss, belief_acc, action_loss, response_loss
    
    def val_forward(self, turn_inputs, turn_contexts, turn_segment, postprocessing):
        batch_size = turn_contexts.size(0)

        # make belief of update slots
        gates = turn_inputs["gate"]
        max_num_updates = 0
        max_value_len = 0
        updates = []
        for batch_idx, slots in enumerate(turn_inputs["gate"]):
            updates_ = []
            for slot_idx, gate in enumerate(slots):
                slot_idx = turn_inputs["slot_order"][batch_idx, slot_idx].item()
                if gate == self.update_idx:
                    value = turn_inputs["belief"][batch_idx, slot_idx, :].tolist()  # including [EOS]
                    max_value_len = max(max_value_len, len(value))
                    updates_.append(value)
            num_updates = len(updates_)
            max_num_updates = max(max_num_updates, num_updates)
            updates.append(updates_)
        updates_label = torch.zeros((batch_size, max_num_updates, max_value_len), dtype=torch.int64).cuda()
        for batch_idx, slots in enumerate(updates):
            for slot_idx, value in enumerate(slots):
                updates_label[batch_idx, slot_idx, :] = torch.tensor(value)
                
        encoder_outputs, pooled_output, gate_scores, decoder_inputs, domain_state = \
            self.context_encoder(turn_contexts, turn_segment, gates=None, max_num_updates=None, train=False)
            # [batch, time, hidden], [batch, hidden], [batch, slots, gates], [batch, slots, hidden], [batch, domains]
        all_probs = self.belief_tracker(encoder_outputs, pooled_output, decoder_inputs, turn_contexts, updates_label, 0)
            # [batch, slots, time, vocab]

        gate_label = turn_inputs["gate"]
        gate_pred = gate_scores.detach().argmax(dim=2)  # [batch, slots]
        gate_acc = (gate_pred == gate_label).float().mean()

        domain_label = turn_inputs["domain_state"]
        domain_pred = domain_state.detach().argmax(dim=2)  # [batch, domains]
        domain_acc = (domain_pred == domain_label).float().mean()

        value_label = turn_inputs["belief"]  # [batch, slots, time]
        belief_acc = torch.ones(batch_size, len(ontology.all_info_slots)).cuda()
        if all_probs is not None:
            all_pred_words = all_probs.argmax(dim=3)  # [batch, slots, time]
        belief_gen = []  # [batch, slots, time]
        for batch_idx, batch in enumerate(gate_pred):
            belief_gen_ = []  # [slots, time]
            update_idx = 0
            for idx, gate in enumerate(batch):
                if gate == self.update_idx:
                    pred_words = all_pred_words[batch_idx, update_idx, :]
                    update_idx += 1
                elif gate == self.delete_idx:
                    pred_words = torch.tensor(self.tokenizer.encode("none", add_special_tokens=False) + \
                        self.tokenizer.convert_tokens_to_ids(["[EOS]"]))
                elif gate == self.dontcare_idx:
                    pred_words = torch.tensor(self.tokenizer.encode("dontcare", add_special_tokens=False) + \
                        self.tokenizer.convert_tokens_to_ids(["[EOS]"]))
                else:
                    pred_words = torch.tensor(turn_inputs["prev_belief"][batch_idx][idx])
                for word_idx, word in enumerate(pred_words):
                    if word == self.eos_idx:
                        break
                word_idx += 1
                pred = pred_words.tolist()[:word_idx]  # include [EOS]
                belief_gen_.append(pred)
                label = value_label[batch_idx, idx, :][value_label[batch_idx, idx, :] != self.pad_idx].tolist()
                if pred != label:
                    belief_acc[batch_idx, idx] = 0
            belief_gen.append(belief_gen_)

        # flat belief state
        belief_flat = []
        for batch_idx, belief in enumerate(belief_gen):
            belief_ = []
            for slot_idx, value in enumerate(belief):
                slot = ontology.all_info_slots[slot_idx]
                domain, slot = slot.split("-")
                slot = "[{}] - {}".format(domain, slot)
                belief_ += [self.slot_idx] + self.tokenizer.encode(slot, add_special_tokens=False) + \
                    self.tokenizer.convert_tokens_to_ids(["-"]) + value[:-1]  # [SLOT] [domain] - slot - value
            belief_flat.append(belief_)
        
        # make belief context
        # [[CLS], user_t, [SEP], domain_state_t, belief_t, db_t, [SEP]]
        belief_contexts = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
        belief_contexts_list = []
        max_context_len = 0
        for batch_idx in range(batch_size):
            user_t = turn_inputs["user"][batch_idx]
            belief_contexts_ = user_t[user_t != self.pad_idx].tolist()  # [CLS], user_t, [SEP]
            domain_state_ = []
            for domain_idx, domain in enumerate(ontology.all_domains):
                domain_state_.append("[DOMAIN]")
                domain_state_.append("[{}]".format(domain))
                if domain_pred[batch_idx, domain_idx] == 1:
                    domain_state_.append("[ON]")
                else:
                    domain_state_.append("[OFF]")
            domain_state_ = " ".join(domain_state_)
            belief_contexts_ += self.tokenizer.encode(domain_state_, add_special_tokens=False)  # domain_state_t
            belief_contexts_ += belief_flat[batch_idx]  # belief_t
            db_result = ["[DB]"]
            for domain_idx, domain in enumerate(ontology.all_domains):
                if domain_pred[batch_idx, domain_idx] == 1:
                    db_result.append("[{}]".format(domain))
                    belief_decoded = [self.tokenizer.decode(value[:-1]) for value in belief_gen[batch_idx]]
                    db_entries = self.db.get_match(belief_decoded, domain)
                    db_result.append("{}".format(len(db_entries)))
            db_result = " ".join(db_result)
            belief_contexts_ += self.tokenizer.encode(db_result, add_special_tokens=False)  # db_t
            belief_contexts_.append(self.sep_idx)  # [SEP]
            len_ = len(belief_contexts_)
            max_context_len = max(max_context_len, len_)
            belief_contexts_list.append(belief_contexts_)
            belief_contexts[batch_idx, :len_] = torch.tensor(belief_contexts_)
        belief_contexts = belief_contexts[:, :max_context_len]

        # make belief context vector
        pad_mask = (belief_contexts != self.pad_idx)
        segments = torch.ones_like(belief_contexts).cuda()
        belief_outputs, belief_hidden = self.context_encoder.bert(belief_contexts, attention_mask=pad_mask, token_type_ids=segments)
            # [batch, time, hidden], [batch, hidden]

        action_probs = self.dialogue_policy(belief_outputs, belief_hidden, belief_contexts, turn_inputs["action"], mode="test")
            # [batch, time, vocab]

        action_pred = action_probs.argmax(dim=2).long()
        action_gen = []
        for batch_idx, action in enumerate(action_pred):
            for word_idx, word in enumerate(action):
                if word == self.eos_idx:
                    break
            action = action.tolist()[:word_idx+1]

            # system action control
            if postprocessing:
                decoded_action = self.tokenizer.decode(action[:-1])
                if "[ACTION] [attraction] - [inform]" in decoded_action:
                    if "[ACTION] [attraction] - [inform] - postcode" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [attraction] - [inform] - postcode", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [attraction] - [inform] - address" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [attraction] - [inform] - address", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [attraction] - [inform] - phone" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [attraction] - [inform] - phone", add_special_tokens=False) + action[-1:]
                if "[ACTION] [hotel] - [inform]" in decoded_action:
                    if "[ACTION] [hotel] - [inform] - postcode" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [hotel] - [inform] - postcode", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [hotel] - [inform] - address" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [hotel] - [inform] - address", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [hotel] - [inform] - phone" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [hotel] - [inform] - phone", add_special_tokens=False) + action[-1:]
                if "[ACTION] [restaurant] - [inform]" in decoded_action:
                    if "[ACTION] [restaurant] - [inform] - postcode" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [restaurant] - [inform] - postcode", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [restaurant] - [inform] - address" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [restaurant] - [inform] - address", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [restaurant] - [inform] - phone" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [restaurant] - [inform] - phone", add_special_tokens=False) + action[-1:]

            action_gen.append(action)  # including [EOS]

        # make action context
        # [[CLS], user_t, [SEP], domain_state_t, belief_t, db_t, action_t, [SEP]]
        action_contexts = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
        max_context_len = 0
        for batch_idx in range(batch_size):
            action_contexts_ = belief_contexts_list[batch_idx][:-1]  # [CLS], user_t, [SEP], domain_state_t, belief_t, db_t
            action_contexts_ += action_gen[batch_idx][:-1]  # action_t
            action_contexts_.append(self.sep_idx)  # [SEP]
            len_ = len(action_contexts_)
            max_context_len = max(max_context_len, len_)
            action_contexts[batch_idx, :len_] = torch.tensor(action_contexts_)
        action_contexts = action_contexts[:, :max_context_len]

        # make action context vector
        pad_mask = (action_contexts != self.pad_idx)
        segments = torch.ones_like(action_contexts).cuda()
        action_outputs, action_hidden = self.context_encoder.bert(action_contexts, attention_mask=pad_mask, token_type_ids=segments)

        # make response
        response_probs = self.response_generator(action_outputs, action_hidden, action_contexts, turn_inputs["response_delex"][:, 1:], 0)

        response_pred = response_probs.argmax(dim=2).long()
        response_gen = []
        for batch_idx, response in enumerate(response_pred):
            for word_idx, word in enumerate(response):
                if word == self.sep_idx:
                    break
            response_gen.append(response.tolist()[:word_idx+1])  # including [EOS]

        return domain_acc, gate_acc, belief_acc, domain_pred, belief_gen, action_gen, response_gen

    def test_forward(self, context, segment, user, prev_belief, postprocessing):
        batch_size = 1

        # make belief of update slots
        encoder_outputs, pooled_output, gate_scores, decoder_inputs, domain_state = \
            self.context_encoder(context, segment, gates=None, max_num_updates=None, train=False)
            # [batch, time, hidden], [batch, hidden], [batch, slots, gates], [batch, slots, hidden], [batch, domains]
        all_probs = self.belief_tracker(encoder_outputs, pooled_output, decoder_inputs, context, None, 0)  # [batch, slots, time, vocab]

        gate_pred = gate_scores.detach().argmax(dim=2)  # [batch, slots]
        domain_pred = domain_state.detach().argmax(dim=2)  # [batch, domains]
        
        if all_probs is not None:
            all_pred_words = all_probs.argmax(dim=3)  # [batch, slots, time]
        belief_gen = []  # [batch, slots, time]
        for batch_idx, batch in enumerate(gate_pred):
            belief_gen_ = []  # [slots, time]
            update_idx = 0
            for idx, gate in enumerate(batch):
                if gate == self.update_idx:
                    pred_words = all_pred_words[batch_idx, update_idx, :]
                    update_idx += 1
                elif gate == self.delete_idx:
                    pred_words = torch.tensor(self.tokenizer.encode("none", add_special_tokens=False) + \
                        self.tokenizer.convert_tokens_to_ids(["[EOS]"]))
                elif gate == self.dontcare_idx:
                    pred_words = torch.tensor(self.tokenizer.encode("dontcare", add_special_tokens=False) + \
                        self.tokenizer.convert_tokens_to_ids(["[EOS]"]))
                else:
                    pred_words = torch.tensor(prev_belief[0][idx])
                for word_idx, word in enumerate(pred_words):
                    if word == self.eos_idx:
                        break
                word_idx += 1
                pred = pred_words.tolist()[:word_idx]  # include [EOS]
                belief_gen_.append(pred)
            belief_gen.append(belief_gen_)

        # flat belief state
        belief_flat = []
        for batch_idx, belief in enumerate(belief_gen):
            belief_ = []
            for slot_idx, value in enumerate(belief):
                slot = ontology.all_info_slots[slot_idx]
                domain, slot = slot.split("-")
                slot = "[{}] - {}".format(domain, slot)
                belief_ += [self.slot_idx] + self.tokenizer.encode(slot, add_special_tokens=False) + \
                    self.tokenizer.convert_tokens_to_ids(["-"]) + value[:-1]  # [SLOT] [domain] - slot - value
            belief_flat.append(belief_)
        
        # make belief context
        # [[CLS], user_t, [SEP], domain_state_t, belief_t, db_t, [SEP]]
        belief_contexts = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
        belief_contexts_list = []
        max_context_len = 0
        for batch_idx in range(batch_size):
            belief_contexts_ = user  # [CLS], user_t, [SEP]
            domain_state_ = []
            for domain_idx, domain in enumerate(ontology.all_domains):
                domain_state_.append("[DOMAIN]")
                domain_state_.append("[{}]".format(domain))
                if domain_pred[batch_idx, domain_idx] == 1:
                    domain_state_.append("[ON]")
                else:
                    domain_state_.append("[OFF]")
            domain_state_ = " ".join(domain_state_)
            belief_contexts_ += self.tokenizer.encode(domain_state_, add_special_tokens=False)  # domain_state_t
            belief_contexts_ += belief_flat[batch_idx]  # belief_t
            db_result = ["[DB]"]
            for domain_idx, domain in enumerate(ontology.all_domains):
                if domain_pred[batch_idx, domain_idx] == 1:
                    db_result.append("[{}]".format(domain))
                    belief_decoded = [self.tokenizer.decode(value[:-1]) for value in belief_gen[batch_idx]]
                    db_entries = self.db.get_match(belief_decoded, domain)
                    db_result.append("{}".format(len(db_entries)))
            db_result = " ".join(db_result)
            belief_contexts_ += self.tokenizer.encode(db_result, add_special_tokens=False)  # db_t
            belief_contexts_.append(self.sep_idx)  # [SEP]
            len_ = len(belief_contexts_)
            max_context_len = max(max_context_len, len_)
            belief_contexts_list.append(belief_contexts_)
            belief_contexts[batch_idx, :len_] = torch.tensor(belief_contexts_)
        belief_contexts = belief_contexts[:, :max_context_len]

        # make belief context vector
        pad_mask = (belief_contexts != self.pad_idx)
        segments = torch.ones_like(belief_contexts).cuda()
        belief_outputs, belief_hidden = self.context_encoder.bert(belief_contexts, attention_mask=pad_mask, token_type_ids=segments)
            # [batch, time, hidden], [batch, hidden]

        action_probs = self.dialogue_policy(belief_outputs, belief_hidden, belief_contexts, None, mode="test")  # [batch, time, vocab]

        action_pred = action_probs.argmax(dim=2).long()
        action_gen = []
        for batch_idx, action in enumerate(action_pred):
            for word_idx, word in enumerate(action):
                if word == self.eos_idx:
                    break
            action = action.tolist()[:word_idx+1]

            if postprocessing:
                decoded_action = self.tokenizer.decode(action[:-1])
                if "[ACTION] [attraction] - [inform]" in decoded_action:
                    if "[ACTION] [attraction] - [inform] - postcode" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [attraction] - [inform] - postcode", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [attraction] - [inform] - address" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [attraction] - [inform] - address", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [attraction] - [inform] - phone" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [attraction] - [inform] - phone", add_special_tokens=False) + action[-1:]
                if "[ACTION] [hotel] - [inform]" in decoded_action:
                    if "[ACTION] [hotel] - [inform] - postcode" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [hotel] - [inform] - postcode", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [hotel] - [inform] - address" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [hotel] - [inform] - address", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [hotel] - [inform] - phone" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [hotel] - [inform] - phone", add_special_tokens=False) + action[-1:]
                if "[ACTION] [restaurant] - [inform]" in decoded_action:
                    if "[ACTION] [restaurant] - [inform] - postcode" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [restaurant] - [inform] - postcode", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [restaurant] - [inform] - address" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [restaurant] - [inform] - address", add_special_tokens=False) + action[-1:]
                    if "[ACTION] [restaurant] - [inform] - phone" not in decoded_action:
                        action = action[:-1] + self.tokenizer.encode("[ACTION] [restaurant] - [inform] - phone", add_special_tokens=False) + action[-1:]

            action_gen.append(action)  # including [EOS]

        # make action context
        # [[CLS], user_t, [SEP], domain_state_t, belief_t, db_t, action_t, [SEP]]
        action_contexts = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
        max_context_len = 0
        for batch_idx in range(batch_size):
            action_contexts_ = belief_contexts_list[batch_idx][:-1]  # [CLS], user_t, [SEP], domain_state_t, belief_t, db_t
            action_contexts_ += action_gen[batch_idx][:-1]  # action_t
            action_contexts_.append(self.sep_idx)  # [SEP]
            len_ = len(action_contexts_)
            max_context_len = max(max_context_len, len_)
            action_contexts[batch_idx, :len_] = torch.tensor(action_contexts_)
        action_contexts = action_contexts[:, :max_context_len]

        # make action context vector
        pad_mask = (action_contexts != self.pad_idx)
        segments = torch.ones_like(action_contexts).cuda()
        action_outputs, action_hidden = self.context_encoder.bert(action_contexts, attention_mask=pad_mask, token_type_ids=segments)

        # make response
        response_probs = self.response_generator(action_outputs, action_hidden, action_contexts, None, 0)

        response_pred = response_probs.argmax(dim=2).long()
        response_gen = []
        for batch_idx, response in enumerate(response_pred):
            for word_idx, word in enumerate(response):
                if word == self.sep_idx:
                    break
            response_gen.append(response.tolist()[:word_idx+1])  # including [EOS]

        return domain_pred, belief_gen, action_gen, response_gen

    def rl_forward(self, turn_inputs, turn_contexts, turn_segment):
        batch_size = turn_contexts.size(0)

        value_label = turn_inputs["belief"]  # [batch, slots, time]
        domain_label = turn_inputs["domain_state"]
        gate_label = turn_inputs["gate"]  # [batch, slots]

        # flat belief state
        belief_flat = []
        belief_state = value_label.clone().tolist()
        for batch_idx, belief in enumerate(belief_state):
            belief_ = []
            for slot_idx, value in enumerate(belief):
                slot = ontology.all_info_slots[slot_idx]
                domain, slot = slot.split("-")
                slot = "[{}] - {}".format(domain, slot)
                for word_idx, word in enumerate(value):
                    if word == self.eos_idx:
                        break
                belief_ += [self.slot_idx] + self.tokenizer.encode(slot, add_special_tokens=False) + \
                    self.tokenizer.convert_tokens_to_ids(["-"]) + value[:word_idx]  # [SLOT] [domain] - slot - value
            belief_flat.append(belief_)

        belief_gen = []  # [batch, slots, time]
        for batch_idx, batch in enumerate(gate_label):
            belief_gen_ = []  # [slots, time]
            update_idx = 0
            for idx, gate in enumerate(batch):
                if gate == self.update_idx:
                    pred_words = value_label[batch_idx, update_idx, :]
                    update_idx += 1
                elif gate == self.delete_idx:
                    pred_words = torch.tensor(self.tokenizer.encode("none", add_special_tokens=False) + \
                        self.tokenizer.convert_tokens_to_ids(["[EOS]"]))
                elif gate == self.dontcare_idx:
                    pred_words = torch.tensor(self.tokenizer.encode("dontcare", add_special_tokens=False) + \
                        self.tokenizer.convert_tokens_to_ids(["[EOS]"]))
                else:
                    pred_words = turn_inputs["prev_belief"][batch_idx][idx]
                for word_idx, word in enumerate(pred_words):
                    if word == self.eos_idx:
                        break
                word_idx += 1
                pred = pred_words.tolist()[:word_idx]  # include [EOS]
                belief_gen_.append(pred)
            belief_gen.append(belief_gen_)

        # make belief context
        # [[CLS], user_t, [SEP], domain_state_t, belief_t, db_t, [SEP]]
        belief_contexts = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
        belief_contexts_list = []
        max_context_len = 0
        for batch_idx in range(batch_size):
            user_t = turn_inputs["user"][batch_idx]
            belief_contexts_ = user_t[user_t != self.pad_idx].tolist()  # [CLS], user_t, [SEP]
            domain_state_ = []
            for domain_idx, domain in enumerate(ontology.all_domains):
                domain_state_.append("[DOMAIN]")
                domain_state_.append("[{}]".format(domain))
                if domain_label[batch_idx, domain_idx] == 1:
                    domain_state_.append("[ON]")
                else:
                    domain_state_.append("[OFF]")
            domain_state_ = " ".join(domain_state_)
            belief_contexts_ += self.tokenizer.encode(domain_state_, add_special_tokens=False)  # domain_state_t
            belief_contexts_ += belief_flat[batch_idx]  # belief_t
            db_result = ["[DB]"]
            for domain_idx, domain in enumerate(ontology.all_domains):
                if domain_label[batch_idx, domain_idx] == 1:
                    db_result.append("[{}]".format(domain))
                    db_result.append("{}".format(turn_inputs["db_results"][batch_idx, domain_idx].item()))
            db_result = " ".join(db_result)
            belief_contexts_ += self.tokenizer.encode(db_result, add_special_tokens=False)  # db_t
            belief_contexts_.append(self.sep_idx)  # [SEP]
            len_ = len(belief_contexts_)
            max_context_len = max(max_context_len, len_)
            belief_contexts_list.append(belief_contexts_)
            belief_contexts[batch_idx, :len_] = torch.tensor(belief_contexts_)
        belief_contexts = belief_contexts[:, :max_context_len]

        # make belief context vector
        pad_mask = (belief_contexts != self.pad_idx)
        segments = torch.ones_like(belief_contexts).cuda()
        belief_outputs, belief_hidden = self.context_encoder.bert(belief_contexts, attention_mask=pad_mask, token_type_ids=segments)

        action_pred, log_probs = self.dialogue_policy(belief_outputs, belief_hidden, belief_contexts, turn_inputs["action"], mode="rl")
            # [batch, time], [batch, time]

        action_gen = []
        for batch_idx, action in enumerate(action_pred):
            for word_idx, word in enumerate(action):
                if word == self.eos_idx:
                    break
            action = action.tolist()[:word_idx+1]
            action_gen.append(action)  # including [EOS]

        # make action context
        # [[CLS], user_t, [SEP], domain_state_t, belief_t, db_t, action_t, [SEP]]
        action_contexts = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
        max_context_len = 0
        for batch_idx in range(batch_size):
            action_contexts_ = belief_contexts_list[batch_idx][:-1]  # [CLS], user_t, [SEP], domain_state_t, belief_t, db_t
            action_contexts_ += action_gen[batch_idx][:-1]  # action_t
            action_contexts_.append(self.sep_idx)  # [SEP]
            len_ = len(action_contexts_)
            max_context_len = max(max_context_len, len_)
            action_contexts[batch_idx, :len_] = torch.tensor(action_contexts_)
        action_contexts = action_contexts[:, :max_context_len]

        # make action context vector
        pad_mask = (action_contexts != self.pad_idx)
        segments = torch.ones_like(action_contexts).cuda()
        action_outputs, action_hidden = self.context_encoder.bert(action_contexts, attention_mask=pad_mask, token_type_ids=segments)

        # make response
        response_probs = self.response_generator(action_outputs, action_hidden, action_contexts, turn_inputs["response_delex"][:, 1:], 0)

        response_pred = response_probs.argmax(dim=2).long()
        response_gen = []
        for batch_idx, response in enumerate(response_pred):
            for word_idx, word in enumerate(response):
                if word == self.sep_idx:
                    break
            response_gen.append(response.tolist()[:word_idx+1])  # including [EOS]
            
        return log_probs, domain_label, belief_gen, action_gen, response_gen