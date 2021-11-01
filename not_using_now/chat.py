import os
import logging
import random
from copy import deepcopy

import numpy as np
import torch
from transformers import BertTokenizer

from dora import DORA
from config import Config
import ontology
from db import DB


NONE_IDX = 3904
EOS_IDX = 3

def init_session():
    domain_state = torch.zeros(1, len(ontology.all_domains), dtype=torch.int64).cuda()  # [1, domains]
    belief_state = torch.stack([torch.ones(1, len(ontology.all_info_slots), dtype=torch.int64).cuda() * NONE_IDX, \
        torch.ones(1, len(ontology.all_info_slots), dtype=torch.int64).cuda() * EOS_IDX], dim=2).tolist()  # [1, slots, 2]

    return domain_state, belief_state

def chat(config):
    logger = logging.getLogger("DST")
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

    model = DORA(db, config).cuda()
    model.eval()

    # load saved model, optimizer
    assert config.save_path is not None
    load(model, config.save_path)

    with open(os.path.join(config.assets_path, "never_split.txt"), "r") as f:
        never_split = f.read().split("\n")
    tokenizer = BertTokenizer(os.path.join(config.assets_path, "vocab.txt"), never_split=never_split)

    with torch.no_grad():
        domain_state, belief_state = init_session()
        while True:
            batch_size = 1
            user = input("User: ")
            if user == "re":
                domain_state, belief_state = init_session()
                continue
            user = tokenizer.encode(user)
            context = deepcopy(user)

            prev_domain_state = []
            for domain_idx, domain in enumerate(ontology.all_domains):
                prev_domain_state.append("[DOMAIN]")
                prev_domain_state.append("[{}]".format(domain))
                if domain_state[0, domain_idx] == 1:
                    prev_domain_state.append("[ON]")
                else:
                    prev_domain_state.append("[OFF]")
            prev_domain_state = " ".join(prev_domain_state)
            context += tokenizer.encode(prev_domain_state, add_special_tokens=False)

            for slot_idx, slot in enumerate(ontology.all_info_slots):
                domain, slot = slot.split("-")
                slot = "[{}] - {}".format(domain, slot)
                value = belief_state[0][slot_idx]
                context += tokenizer.convert_tokens_to_ids(["[SLOT]"])
                context += tokenizer.encode(slot, add_special_tokens=False)
                context += tokenizer.convert_tokens_to_ids(["-"])
                context += value[:-1]
            context.append(tokenizer.sep_token_id)
            
            context = torch.tensor(context, dtype=torch.int64).cuda().unsqueeze(dim=0)
            segment = torch.zeros_like(context)
            segment[0, :len(user)] = 1

            domain_pred, belief_gen, action_gen, response_gen = model.test_forward(
                context, segment, user, belief_state, postprocessing=config.postprocessing)

            belief_state = deepcopy(belief_gen)
            domain_state = deepcopy(domain_pred)
            
            response = tokenizer.decode(response_gen[0][:-1])
            logger.info("Belief: ")
            for slot_idx, slot in enumerate(ontology.all_info_slots):
                value = tokenizer.decode(belief_gen[0][slot_idx][:-1])
                if value != "none":
                    logger.info("\t {} : {}".format(slot, value))
            action = tokenizer.decode(action_gen[0][:-1])
            logger.info("Action: {}".format(action))
            logger.info("System: {}".format(response))
            logger.info("="*20)

def load(model, save_path):
    checkpoint = torch.load(save_path, map_location = lambda storage, loc: storage.cuda(0))
    model.load_state_dict(checkpoint["model"])

if __name__ == "__main__":
    config = Config()
    parser = config.parser
    config = parser.parse_args()

    chat(config)
