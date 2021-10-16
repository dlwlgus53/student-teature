import os
import logging
import time
import random
import json
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from dora import DORA
from config import Config
from reader import Reader
import ontology
from db import DB
from evaluate import MultiWozEvaluator


def test(config):
    logger = logging.getLogger("DORA")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    
    torch.cuda.set_device(config.cuda_device)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    db = DB(config.db_path)

    reader = Reader(db, config)
    start = time.time()
    logger.info("Loading data...")
    reader.load_data("test")
    end = time.time()
    logger.info("Loaded. {} secs".format(end-start))

    evaluator = MultiWozEvaluator("test", config.data_path, config.db_path, config.assets_path)

    model = DORA(db, config).cuda()
    model.eval()

    # load saved model, optimizer
    assert config.save_path is not None
    load(model, config.save_path, config.cuda_device)

    max_iter = len(list(reader.make_batch(reader.test)))

    slot_acc = 0
    joint_acc = 0
    batch_count = 0
    gate_acc = 0
    domain_acc = 0

    with open(os.path.join(config.assets_path, "never_split.txt"), "r") as f:
        never_split = f.read().split("\n")
    tokenizer = BertTokenizer(os.path.join(config.assets_path, "vocab.txt"), never_split=never_split)

    test_dial_gens = {}
    test_dial_gens_decoded = {}

    with torch.no_grad():
        iterator = reader.make_batch(reader.test)

        t = tqdm(enumerate(iterator), total=max_iter, ncols=150, position=0, leave=True)

        for batch_idx, batch in t:
            # don't shuffle slot order nor use true previous domain state and belief
            inputs, contexts, segments, dial_ids = reader.make_input(batch, mode="test")
            batch_size = len(contexts[0])

            turns = len(inputs)

            inputs[0]["prev_belief"] = inputs[0]["prev_belief"].tolist()

            dial_gens = [[] for i in range(batch_size)]
            dial_gens_decoded = [[] for i in range(batch_size)]
            belief_gens = [[] for i in range(batch_size)]
            action_gens = [[] for i in range(batch_size)]

            for turn_idx in range(turns):
                turn_context = torch.zeros(batch_size, config.max_context_len, dtype=torch.int64).cuda()
                turn_segment = torch.zeros(batch_size, config.max_context_len, dtype=torch.int64).cuda()
                max_context_len = 0
                for idx in range(len(contexts[turn_idx])):
                    turn_context_ = contexts[turn_idx][idx].tolist()
                    turn_segment_ = segments[turn_idx][idx].tolist()
                    try:
                        turn_context_ = turn_context_[:turn_context_.index(config.pad_idx)]
                    except:
                        turn_context_ = turn_context_
                    turn_segment_ = turn_segment_[:len(turn_context_)]

                    # add previous domain state to context
                    domain_state = []
                    prev_domain_state = inputs[turn_idx]["prev_domain_state"]
                    for domain_idx, domain in enumerate(ontology.all_domains):
                        domain_state.append("[DOMAIN]")
                        domain_state.append("[{}]".format(domain))
                        if prev_domain_state[idx, domain_idx] == 1:
                            domain_state.append("[ON]")
                        else:
                            domain_state.append("[OFF]")
                    domain_state = " ".join(domain_state)
                    turn_context_ += tokenizer.encode(domain_state, add_special_tokens=False)

                    # add previous belief state to context
                    for slot_idx in range(len(ontology.all_info_slots)):
                        slot = ontology.all_info_slots[slot_idx]
                        domain, slot = slot.split("-")
                        slot = "[{}] - {}".format(domain, slot)
                        value = inputs[turn_idx]["prev_belief"][idx][slot_idx]
                        if config.slot_idx in value:
                            value = tokenizer.encode("none")[1:]
                        turn_context_ += tokenizer.convert_tokens_to_ids(["[SLOT]"])
                        turn_context_ += tokenizer.encode(slot, add_special_tokens=False)
                        turn_context_ += tokenizer.convert_tokens_to_ids(["-"])
                        turn_context_ += value[:-1]  # except [EOS]
                    turn_context_.append(tokenizer.sep_token_id)  # [SEP]
                    context_len = len(turn_context_)
                    max_context_len = max(max_context_len, context_len)
                    turn_context[idx, :context_len] = torch.tensor(turn_context_[:1] + turn_context_[-(min(context_len, config.max_context_len)-1):])
                    turn_segment[idx, :len(turn_segment_)] = torch.tensor(turn_segment_)
                turn_context = turn_context[:, :max_context_len]
                turn_segment = turn_segment[:, :max_context_len]

                domain_acc_, gate_acc_, belief_acc, domain_state, belief_gen, action_gen, response_gen = \
                    model.forward(inputs[turn_idx], turn_context, turn_segment, "val", config.postprocessing)

                if turn_idx < turns-1:
                    inputs[turn_idx+1]["prev_belief"] = deepcopy(belief_gen)  # generated belief, not ground truth
                    inputs[turn_idx+1]["prev_domain_state"] = domain_state

                domain_acc += domain_acc_ * batch_size
                gate_acc += gate_acc_ * batch_size
                slot_acc += belief_acc.sum(dim=1).sum(dim=0)
                joint_acc += (belief_acc.mean(dim=1) == 1).sum(dim=0).float()
                batch_count += batch_size

                torch.cuda.empty_cache()

                # for evaluation
                response_gens = [response[:-1] for response in response_gen]
                response_gens_decoded = [tokenizer.decode(response[:-1]) for response in response_gen]
                action_gen_decoded = [tokenizer.decode(action[:-1]) for action in action_gen]
                for b_idx, belief in enumerate(belief_gen):
                    belief_gen[b_idx] = [tokenizer.decode(value[:-1]) for value in belief]
                for b_idx in range(batch_size):
                    dial_gens[b_idx].append(response_gens[b_idx])
                    dial_gens_decoded[b_idx].append(response_gens_decoded[b_idx])
                    belief = {}
                    for slot_idx, slot in enumerate(ontology.all_info_slots):
                        belief[slot] = belief_gen[b_idx][slot_idx]
                    belief_gens[b_idx].append(belief)
                    action_gens[b_idx].append(action_gen_decoded[b_idx])

            t.set_description("iter: {}".format(batch_idx+1))
            time.sleep(1)

            for b_idx in range(batch_size):
                dial_id = dial_ids[b_idx]
                dial_id = "{}.json".format(dial_id)
                test_dial_gens[dial_id] = dial_gens[b_idx]
                test_dial_gens_decoded[dial_id] = {}
                test_dial_gens_decoded[dial_id]["response"] = dial_gens_decoded[b_idx]
                test_dial_gens_decoded[dial_id]["belief_state"] = belief_gens[b_idx]
                test_dial_gens_decoded[dial_id]["action"] = action_gens[b_idx]

    gate_acc = gate_acc.item() / batch_count * 100
    domain_acc = domain_acc.item() / batch_count * 100
    slot_acc = slot_acc.item() / batch_count / len(ontology.all_info_slots) * 100
    joint_acc = joint_acc.item() / batch_count * 100

    test_dial = json.load(open(os.path.join(config.data_path, "test_data.json"), "r"))

    _, inform_rate, success_rate, bleu_score = evaluator.evaluateModel(test_dial_gens_decoded, test_dial_gens, test_dial, mode='test', \
        save_path=config.save_path.split("/")[1].split(".")[0], make_report=config.make_report)

    logger.info("accuracy(domain/gate/joint/slot): {:.2f}, {:.2f}, {:.2f}, {:.2f}, inform: {:.2f}, success: {:.2f}, bleu: {:.2f}"\
        .format(domain_acc, gate_acc, joint_acc, slot_acc, inform_rate, success_rate, bleu_score))

def load(model, save_path, cuda_device):
    checkpoint = torch.load(save_path, map_location = lambda storage, loc: storage.cuda(cuda_device))
    model.load_state_dict(checkpoint["model"])

if __name__ == "__main__":
    config = Config()
    parser = config.parser
    config = parser.parse_args()

    test(config)
