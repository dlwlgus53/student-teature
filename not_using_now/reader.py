import os
import json
import logging
import random
import numpy as np

import torch
# from transformers import BertTokenizer

import ontology


class Dataset(torch.utils.data.Dataset):
    def __init__(self, BertTokenizer,  type, db):
        with open(os.path.join(config.assets_path, "never_split.txt"), "r") as f:
            never_split = f.read().split("\n")
        self.tokenizer = BertTokenizer(os.path.join(config.assets_path, "vocab.txt"), never_split=never_split)
        self.db = db
        self.encoded = {}
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.max_value_len = config.max_value_len
        self.max_act_len = config.max_act_len
        self.max_sentence_len = config.max_sentence_len
        self.max_context_len = config.max_context_len
        self.eos_idx = config.eos_idx
        self.pad_idx = config.pad_idx
        self.type
        
    def _encodin_all(self, data):
        processed = {}
        for dial_id, dial in data.items():
            turns = len(dial["log"])  # number of turns
            if not self.train.get(turns):
                processed[turns] = {}
            processed[turns][dial_id] = []
            # 모든것을 인코딩해서 넣는다.
            for turn_dial in dial["log"]:
                usr = turn_dial["user"]
                sys = turn_dial["response"]
                turn_dial["user"] = self.tokenizer.encode(turn_dial["user"])
                turn_dial["response"] = self.tokenizer.encode(turn_dial["response"])
                turn_dial["response_delex"] = self.tokenizer.encode(turn_dial["response_delex"])
                turn_dial["belief"], turn_dial["db_results"] = self._encode_belief(turn_dial["belief"])
                turn_dial["action"] = self.encode_action(turn_dial["action"])
                turn_dial["usr"] = usr
                processed[turns][dial_id].append(turn_dial)
                
        return processed
                
    def load_data(self, mode="train"):
        """Load train/dev/test data.
        Divide data by number of turns for batch.
        Encode user utterance & system response."""

        if mode == "train":
            train_data = json.load(open(os.path.join(self.data_path, "train_data.json"), "r"))
            dev_data = json.load(open(os.path.join(self.data_path, "dev_data.json"), "r"))
            self.train = self._encodin_all(self, train_data)
            self.dev = self._encodin_all(self, dev_data)

        else: # when the mode is test
            test_data = json.load(open(os.path.join(self.data_path, "test_data.json"), "r"))
            self.test = self._encodin_all(self, test_data)


    def _encode_belief(self, belief):
        """Encode belief and DB results.
        
        Outputs: encoded_belief, db_result
            encoded_belief: List of encoded belief(values of domain-slot pairs)
            db_results: Number of DB search results for every domains

        Shapes:
            encoded_belief: [slots, time]
            db_results: [domains]
        """

        encoded_belief = []
        decoded_belief = []
        for domain_slot in ontology.all_info_slots:
            if belief.get(domain_slot):
                value = belief[domain_slot]
                encoded_belief.append(self.tokenizer.encode(belief[domain_slot], add_special_tokens=False) + [self.eos_idx])
            else:
                value = "none"
                encoded_belief.append(self.tokenizer.encode("none", add_special_tokens=False) + [self.eos_idx])
            decoded_belief.append(value)
        db_results = []
        for domain in ontology.all_domains:
            db_results.append(len(self.db.get_match(decoded_belief, domain)))
        
        return encoded_belief, db_results

    def encode_action(self, action):
        """Encode system action.
        
        Outputs: encoded_action
            encoded_action: Encoded concat of actions(domain-action-slot)
        """

        actions = []
        dialogue_acts_slots = ontology.dialogue_acts_slots
        for domain_act in dialogue_acts_slots.keys():
            if action.get(domain_act):
                for slot_value in action[domain_act]:
                    domain_, act_ = domain_act.split("-")
                    action_ = self.tokenizer.encode("[ACTION] [{}] - [{}]".format(domain_, act_), add_special_tokens=False)
                    if slot_value != "none":  # act has no slot
                        action_ += self.tokenizer.encode("- {}".format(slot_value.split("-")[0]), add_special_tokens=False)
                    actions.append(action_)
        encoded_action = []
        for action in actions:
            encoded_action += action
        encoded_action.append(self.eos_idx)

        return encoded_action

    def make_batch(self, data):
        """Make batches and return iterator.

        Outputs: batch
            batch: Dictionary of batches
        
        Example:
            batch = {
                "dial_id" = [...]
                0: {
                    "user": [[...], ..., [...]],
                    "response": [[...], ..., [...]],
                    "response_delex": [[...], ..., [...]],
                    "belief": [[...], ..., [...]],
                    "action": [[...], ..., [...]],
                    "usr": [~, ..., ~] => string
                },
                1: {
                    "user": [[...], ..., [...]],
                    "response": [[...], ..., [...]],
                    "response_delex": [[...], ..., [...]],
                    "belief": [[...], ..., [...]],
                    "action": [[...], ..., [...]]
                    "usr": [~, ..., ~] => string
                }
            }
        """

        all_batches = []
        for turn_num, dials in data.items():
            batch = {"dial_id": []}
            for dial_id, turns in dials.items():
                if len(batch["dial_id"]) == self.batch_size:  # current batch is full
                    all_batches.append(batch)
                    batch = {"dial_id": []}
                batch["dial_id"].append(dial_id)
                for turn in turns:
                    cur_turn = turn["turn_num"]
                    if not batch.get(cur_turn):
                        batch[cur_turn] = {
                            "user": [],
                            "response": [],
                            "response_delex": [],
                            "belief": [],
                            "action": [],
                            "usr": [],
                            "domain_state": [],
                            "db_results": []
                        }
                    for key in batch[cur_turn].keys():
                        batch[cur_turn][key].append(turn[key])
            all_batches.append(batch)
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def make_input(self, batch, mode):
        """Make input of torch tensors.

        Outputs: inputs, contexts, segments, dial_ids
            inputs: List of tensors.
            contexts: List of history including user utterance and system response.
            segments: Segment ids for BERT encoder. 
                1 for current (user_t, domain_state_t, belief_t, ...) and 0 for previous (domain_state_t-1, belief_t-1).
            dial_ids: List of dialogue id.

        Example:
            inputs = [
                {
                    "user": tensor,  # [batch, time]
                    "response": tensor,  # [batch, time]
                    "response_delex": tensor,  # [batch, time]
                    "belief": tensor,  # [batch, slots, time]
                    "action": tensor,  # [batch, time]
                    "usr": list,  # [batch] => string
                    "prev_belief": tensor,  # [batch, slots, time]
                    "gate": tensor,  # [batch, slots]
                    "slot_order": tensor,  # [batch, slots]
                    "domain_state": tensor,  # [batch, domains]
                    "prev_domain_state": tensor,  # [batch, domains]
                    "db_result": tensor,  # [batch, domains]
                },
                {
                    ...
                }
            ]

        Shapes:
            contexts: [batch, time] * turns
            segments: [batch, time] * turns
        """

        inputs = []
        batch_size = len(batch["dial_id"])
        dial_ids = batch.pop("dial_id")  # dialog ids for evaluation
        turns = list(batch.keys())
        turns.sort()
        contexts = []
        segments = []
        prev_resp = []
        turn_context = []

        # for belief tracking
        # all domains are OFF & all slot-values are none
        prev_domain_state = torch.zeros(batch_size, len(ontology.all_domains), dtype=torch.int64).cuda()  # [batch, domains]
        prev_belief = torch.stack([\
            torch.ones(batch_size, len(ontology.all_info_slots), dtype=torch.int64).cuda() * self.tokenizer.convert_tokens_to_ids(["none"])[0], \
            torch.ones(batch_size, len(ontology.all_info_slots), dtype=torch.int64).cuda() * self.eos_idx], dim=2)  # [batch, slots, 2]

        for turn in turns:
            turn_input = {}

            # shuffle the order of slots in context to train the semantic of each slot independently of position
            slot_orders = torch.arange(0, len(ontology.all_info_slots)).repeat(batch_size, 1).cuda()  # [batch, slots]
            
            # make tensor of user utterance
            user_ = torch.zeros(batch_size, self.max_sentence_len, dtype=torch.int64)
            max_len = 0
            for idx, user in enumerate(batch[turn]["user"]):
                len_ = len(user)
                user_[idx, :len_] = torch.tensor(user[:self.max_sentence_len])
                if len_ > max_len:
                    max_len = len_
            turn_input["user"] = user_[:, :max_len]

            # make tensor of system response
            resp_ = torch.zeros(batch_size, self.max_sentence_len, dtype=torch.int64)
            max_len = 0
            for idx, resp in enumerate(batch[turn]["response"]):
                len_ = len(resp)
                resp_[idx, :len_] = torch.tensor(resp[:self.max_sentence_len])
                if len_ > max_len:
                    max_len = len_
            turn_input["response"] = resp_[:, :max_len]

            # make tensor of delexicalized system response
            resp_delex_ = torch.zeros(batch_size, self.max_sentence_len, dtype=torch.int64)
            max_len = 0
            for idx, resp in enumerate(batch[turn]["response_delex"]):
                len_ = len(resp)
                resp_delex_[idx, :len_] = torch.tensor(resp[:self.max_sentence_len])
                if len_ > max_len:
                    max_len = len_
            turn_input["response_delex"] = resp_delex_[:, :max_len]

            # make tensor of belief
            belief_ = torch.zeros(batch_size, len(ontology.all_info_slots), self.max_value_len, dtype=torch.int64)
            max_len = 0
            for idx, belief in enumerate(batch[turn]["belief"]):
                for s_idx, value in enumerate(belief):
                    len_ = len(value)
                    belief_[idx, s_idx, :len_] = torch.tensor(value[:self.max_value_len])
                    if len_ > max_len:
                        max_len = len_
            turn_input["belief"] = belief_[:, :, :max_len]
            
            # for belief tracking
            turn_input["prev_belief"] = prev_belief
            prev_belief = turn_input["belief"]
            gates = []
            none_token = self.tokenizer.encode("none", add_special_tokens=False) + [self.eos_idx]
            dontcare_token = self.tokenizer.encode("dontcare", add_special_tokens=False) + [self.eos_idx]
            for batch_idx, belief in enumerate(turn_input["belief"]):
                gate = []
                for slot_idx, value in enumerate(belief):
                    value = value[value != self.tokenizer.pad_token_id].tolist()
                    prev_value = turn_input["prev_belief"][batch_idx, slot_idx, :]
                    prev_value = prev_value[prev_value != self.tokenizer.pad_token_id].tolist()
                    if value == prev_value:
                        gate.append(ontology.gate_idx["copy"])
                    elif value == none_token:
                        gate.append(ontology.gate_idx["delete"])
                    elif value == dontcare_token:
                        gate.append(ontology.gate_idx["dontcare"])
                    else:
                        gate.append(ontology.gate_idx["update"])
                gates.append(gate)
            turn_input["gate"] = torch.tensor(gates)  # [batch, slots]

            # make tensor of action
            act_ = torch.zeros(batch_size, self.max_act_len, dtype=torch.int64)
            max_len = 0
            for idx, act in enumerate(batch[turn]["action"]):
                len_ = len(act)
                act_[idx, :len_] = torch.tensor(act[:self.max_act_len])
                if len_ > max_len:
                    max_len = len_
            turn_input["action"] = act_[:, :max_len]

            turn_input["usr"] = batch[turn]["usr"]

            db_results = batch[turn]["db_results"]
            turn_input["db_results"] = torch.tensor(db_results)

            # make initial context => [[CLS], usr_t, [SEP], domain_state_t-1, belief_t-1, [SEP]]
            turn_context = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
            turn_segment = torch.zeros(batch_size, self.max_context_len, dtype=torch.int64).cuda()
            max_context_length = 0
            for batch_idx in range(batch_size):
                turn_context_ = self.tokenizer.encode(turn_input["usr"][batch_idx])  # [CLS], usr_t, [SEP]
                len_ = len(turn_context_)
                turn_segment[batch_idx, :len_] = 1

                # in training, use true domain_state_t-1 and belief_t-1
                if mode != "test":
                    # shuffle slots order
                    if mode == "train":
                        slot_order = slot_orders[batch_idx] if np.random.random() > 0.5 \
                            else slot_orders[batch_idx][torch.randperm(len(ontology.all_info_slots))]
                    elif mode == "rl":
                        slot_order = slot_orders[batch_idx]
                    slot_orders[batch_idx, :] = slot_order

                    # add prev domain state to initial context
                    domain_state = []
                    for domain_idx, domain in enumerate(ontology.all_domains):
                        domain_state.append("[DOMAIN]")
                        domain_state.append("[{}]".format(domain))
                        if prev_domain_state[batch_idx, domain_idx] == 1:
                            domain_state.append("[ON]")
                        else:
                            domain_state.append("[OFF]")
                    domain_state = " ".join(domain_state)
                    turn_context_ += self.tokenizer.encode(domain_state, add_special_tokens=False)

                    # add prev belief to initial context
                    for slot_idx in slot_order.tolist():
                        slot = ontology.all_info_slots[slot_idx]
                        domain, slot = slot.split("-")
                        slot = "[{}] - {}".format(domain, slot)
                        value = turn_input["prev_belief"][batch_idx, slot_idx, :]
                        value = value[value != self.pad_idx].tolist()
                        turn_context_ += self.tokenizer.convert_tokens_to_ids(["[SLOT]"])
                        turn_context_ += self.tokenizer.encode(slot, add_special_tokens=False)
                        turn_context_ += self.tokenizer.convert_tokens_to_ids(["-"])
                        turn_context_ += value[:-1]  # except [EOS]
                    turn_context_.append(self.tokenizer.sep_token_id)  # [SEP]

                # if context is longer than max context length (512), cut the context
                context_length = len(turn_context_)
                turn_segment_ = turn_segment[batch_idx, :context_length].tolist()
                max_context_length = max(max_context_length, context_length)
                turn_context[batch_idx, :context_length] = torch.tensor(turn_context_[:1] + \
                    turn_context_[-(min(context_length, self.max_context_len)-1):])
                turn_segment[batch_idx, :context_length] = torch.tensor(turn_segment_[:1] + \
                    turn_segment_[-(min(context_length, self.max_context_len)-1):])

                # permute gate's order
                if mode == "train":
                    turn_input["gate"][batch_idx] = turn_input["gate"][batch_idx][slot_order]
            
            turn_context = turn_context[:, :max_context_length]
            turn_segment = turn_segment[:, :max_context_length]
            contexts.append(turn_context)
            segments.append(turn_segment)

            turn_input["slot_order"] = slot_orders
            turn_input["prev_domain_state"] = prev_domain_state
            turn_input["domain_state"] = torch.tensor(batch[turn]["domain_state"])
            prev_domain_state = turn_input["domain_state"]

            for key, value in turn_input.items():
                if type(turn_input[key]) != list:
                    turn_input[key] = value.cuda()

            inputs.append(turn_input)

        return inputs, contexts, segments, dial_ids



if __name__ == "__main__":
    from config import Config
    import time

    config = Config()
    parser = config.parser
    config = parser.parse_args()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    
    reader = Reader(config)
    
    start = time.time()
    logger.info("Loading data...")
    reader.load_data()
    end = time.time()
    logger.info("Loaded. {} secs".format(end-start))
    start = time.time()
    logger.info("Making batches...")
    iterator = reader.make_batch(reader.train)
    end = time.time()
    logger.info("Making batch finished. {} secs".format(end-start))
    
    for batch in iterator:
        inputs, contexts, segments, dial_ids = reader.make_input(batch)