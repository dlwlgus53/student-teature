import os
import logging
import time
import random
import re
import json
from copy import deepcopy

import numpy as np
import torch
from torch.optim import SGD
from tqdm import tqdm
from transformers import BertTokenizer

from dora import DORA
from config import Config
from reader import Reader
import ontology
from db import DB
from evaluate import MultiWozEvaluator


def init_process(config):
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
    
    if not os.path.exists("save"):
        os.mkdir("save")
    save_path = "save/model_{}.pt".format(re.sub("\s+", "_", time.asctime()))

    db = DB(config.db_path)

    reader = Reader(db, config)
    start = time.time()
    logger.info("Loading data...")
    reader.load_data("train")
    end = time.time()
    logger.info("Loaded. {} secs".format(end-start))

    evaluator = MultiWozEvaluator("valid", config.data_path, config.db_path, config.assets_path)

    lr = config.lr

    model = DORA(db, config).cuda()

    # only update gru of dialogue policy
    model.requires_grad_(False)
    model.dialogue_policy.gru.requires_grad_(True)
    optimizer = SGD(model.parameters(), lr=lr)

    global_epoch = 0
    max_score = 0
    early_stop_count = config.early_stop_count

    # load saved model, optimizer
    if config.save_path is not None:
        global_epoch, max_score = load(model, optimizer, config.save_path, config.cuda_device, mode="sgd")

    train.max_iter = len(list(reader.make_batch(reader.train)))
    validate.max_iter = len(list(reader.make_batch(reader.dev)))

    # gate_acc, joint_acc, slot_acc, domain_acc, inform_rate, success_rate, bleu_score = validate(model, reader, evaluator, config)
    # logger.info("accuracy(domain/gate/joint/slot): {:.2f}, {:.2f}, {:.2f}, {:.2f}, inform: {:.2f}, success: {:.2f}, bleu: {:.2f}"\
    #     .format(domain_acc, gate_acc, joint_acc, slot_acc, inform_rate, success_rate, bleu_score))

    # standardize success rate reward and action rate reward separately
    train.success_rewards = []
    train.action_rewards = []

    for epoch in range(global_epoch, global_epoch + config.max_epochs):
        logger.info("Train...")
        start = time.time()

        train(model, reader, optimizer, evaluator, config)
        
        end = time.time()
        logger.info("epoch: {}, {:.4f} secs".format(epoch+1, end-start))

        logger.info("Validate...")
        gate_acc, joint_acc, slot_acc, domain_acc, inform_rate, success_rate, bleu_score = validate(model, reader, evaluator, config)
        logger.info("accuracy(domain/gate/joint/slot): {:.2f}, {:.2f}, {:.2f}, {:.2f}, inform: {:.2f}, success: {:.2f}, bleu: {:.2f}"\
            .format(domain_acc, gate_acc, joint_acc, slot_acc, inform_rate, success_rate, bleu_score))
        
        score = inform_rate + success_rate + bleu_score
        if score > max_score:  # save model
            save(model, optimizer, save_path, epoch, score)
            logger.info("Saved to {}.".format(os.path.abspath(save_path)))
            
            max_score = score
            early_stop_count = config.early_stop_count
        else:  # ealry stopping
            if early_stop_count == 0:
                if epoch < config.min_epochs:
                    early_stop_count += 1
                    logger.info("Too early to stop training.")
                    logger.info("early stop count: {}".format(early_stop_count))
                else:
                    logger.info("Early stopped.")
                    break
            elif early_stop_count == 2:
                lr = lr / 2
                logger.info("learning rate schedule: {}".format(lr))
                for param in optimizer.param_groups:
                    param["lr"] = lr
            early_stop_count -= 1
            logger.info("early stop count: {}".format(early_stop_count))
    logger.info("Training finished.")

def train(model, reader, optimizer, evaluator, config):
    iterator = reader.make_batch(reader.train)
    t = tqdm(enumerate(iterator), total=train.max_iter, ncols=250, position=0, leave=True)

    with open(os.path.join(config.assets_path, "never_split.txt"), "r") as f:
        never_split = f.read().split("\n")
    tokenizer = BertTokenizer(os.path.join(config.assets_path, "vocab.txt"), never_split=never_split)

    for batch_idx, batch in t:
        inputs, contexts, segments, dial_ids = reader.make_input(batch, mode="rl")

        batch_size = len(contexts[0])
        turns = len(inputs)

        val_dial_gens = {}
        val_dial_gens_decoded = {}

        log_probs = [[] for i in range(batch_size)]  # [batch, turns * time]
        rewards = [[] for i in range(batch_size)]  # [batch, turns * time]

        # for evaluation
        dial_gens = [[] for i in range(batch_size)]
        dial_gens_decoded = [[] for i in range(batch_size)]
        belief_gens = [[] for i in range(batch_size)]

        for turn_idx in range(turns):
            model.zero_grad()
            log_prob, turn_domain, belief_gen, action_gen, response_gen = model(inputs[turn_idx], contexts[turn_idx], segments[turn_idx], mode="rl")
            for b_idx in range(batch_size):
                log_probs[b_idx].append(log_prob[b_idx])
                rewards[b_idx].append(torch.zeros_like(log_prob[b_idx]))

            # calculate action rate reward
            if config.use_action_rate:
                for b_idx in range(batch_size):
                    action_gen_parsed = parse_action(tokenizer.decode(action_gen[b_idx][:-1]))
                    action = inputs[turn_idx]["action"][b_idx]
                    action = action[action != config.pad_idx].tolist()
                    action_parsed = parse_action(tokenizer.decode(action[:-1]))
                    generated = 0
                    correct = 0
                    for act in action_gen_parsed:
                        generated += 1
                        if act in action_parsed:
                            correct += 1
                            if config.weighted_action_reward and act in ontology.important_actions:
                                # weighted action method
                                correct += 1
                            action_parsed.remove(act)
                        elif config.negative_action_reward:
                            # negative action method
                            correct -= 1
                    reward = correct / generated if generated != 0 else 0
                    train.action_rewards.append(reward)
                    reward = torch.tensor(reward, dtype=torch.float32) - np.mean(train.action_rewards)
                    reward = reward / max(1e-4, np.std(train.action_rewards))
                    reward = reward.cuda()
                    reward *= config.beta
                    for idx in reversed(range(len(rewards[b_idx][-1]))):
                        rewards[b_idx][-1][idx] = reward
                        reward *= config.discount_factor

            # for evaluation
            response_gens = [response[:-1] for response in response_gen]
            response_gens_decoded = [tokenizer.decode(response[:-1]) for response in response_gen]
            for b_idx, belief in enumerate(belief_gen):
                belief_gen[b_idx] = [tokenizer.decode(value[:-1]) for value in belief]
            for b_idx in range(batch_size):
                dial_gens[b_idx].append(response_gens[b_idx])
                dial_gens_decoded[b_idx].append(response_gens_decoded[b_idx])
                belief = {}
                for slot_idx, slot in enumerate(ontology.all_info_slots):
                    belief[slot] = belief_gen[b_idx][slot_idx]
                belief_gens[b_idx].append(belief)

        for b_idx in range(batch_size):
            dial_id = dial_ids[b_idx]
            dial_id = "{}.json".format(dial_id)
            val_dial_gens[dial_id] = dial_gens[b_idx]
            val_dial_gens_decoded[dial_id] = {}
            val_dial_gens_decoded[dial_id]["response"] = dial_gens_decoded[b_idx]
            val_dial_gens_decoded[dial_id]["belief_state"] = belief_gens[b_idx]

        # obtain success rate
        successes = evaluator.evaluateModel(val_dial_gens_decoded, val_dial_gens, mode='rl')
        train.success_rewards += successes

        returns = torch.tensor(successes, dtype=torch.float32) - np.mean(train.success_rewards)
        returns = returns / max(1e-4, np.std(train.success_rewards))
        returns = returns.cuda()  # [batch]
        loss = 0

        for b_idx in range(batch_size):
            log_probs[b_idx] = torch.cat(log_probs[b_idx], dim=0)
            
            # make success rate turn-level
            for turn_idx in range(turns):
                reward = returns[b_idx]
                for idx in reversed(range(len(rewards[b_idx][turn_idx]))):
                    rewards[b_idx][turn_idx][idx] += reward
                    reward *= config.discount_factor
            rewards[b_idx] = torch.cat(rewards[b_idx], dim=0)

            for log_prob, reward in zip(log_probs[b_idx], rewards[b_idx]):
                loss -= log_prob * reward

        loss = loss / batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        optimizer.step()

        t.set_description("iter: {}, loss: {:.4f}".format(batch_idx+1, loss))
        time.sleep(1)
        
        del loss
        torch.cuda.empty_cache()

def validate(model, reader, evaluator, config):
    model.eval()
    val_loss = 0
    slot_acc = 0
    joint_acc = 0
    batch_count = 0
    gate_acc = 0
    domain_acc = 0

    with open(os.path.join(config.assets_path, "never_split.txt"), "r") as f:
        never_split = f.read().split("\n")
    tokenizer = BertTokenizer(os.path.join(config.assets_path, "vocab.txt"), never_split=never_split)

    val_dial_gens = {}
    val_dial_gens_decoded = {}

    with torch.no_grad():
        iterator = reader.make_batch(reader.dev)
        t = tqdm(enumerate(iterator), total=validate.max_iter, ncols=150, position=0, leave=True, dynamic_ncols=config.dynamic_tqdm)

        for batch_idx, batch in t:
            # don't shuffle slot order nor use true previous domain state and belief
            inputs, contexts, segments, dial_ids = reader.make_input(batch, mode="test")
            batch_size = len(contexts[0])

            turns = len(inputs)

            # all slot-values are none
            inputs[0]["prev_belief"] = inputs[0]["prev_belief"].tolist()

            # for evaluation
            dial_gens = [[] for i in range(batch_size)]
            dial_gens_decoded = [[] for i in range(batch_size)]
            belief_gens = [[] for i in range(batch_size)]

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
                for b_idx, belief in enumerate(belief_gen):
                    belief_gen[b_idx] = [tokenizer.decode(value[:-1]) for value in belief]
                for b_idx in range(batch_size):
                    dial_gens[b_idx].append(response_gens[b_idx])
                    dial_gens_decoded[b_idx].append(response_gens_decoded[b_idx])
                    belief = {}
                    for slot_idx, slot in enumerate(ontology.all_info_slots):
                        belief[slot] = belief_gen[b_idx][slot_idx]
                    belief_gens[b_idx].append(belief)

            t.set_description("iter: {}".format(batch_idx+1))
            time.sleep(1)

            for b_idx in range(batch_size):
                dial_id = dial_ids[b_idx]
                dial_id = "{}.json".format(dial_id)
                val_dial_gens[dial_id] = dial_gens[b_idx]
                val_dial_gens_decoded[dial_id] = {}
                val_dial_gens_decoded[dial_id]["response"] = dial_gens_decoded[b_idx]
                val_dial_gens_decoded[dial_id]["belief_state"] = belief_gens[b_idx]

    model.train()
    gate_acc = gate_acc / batch_count * 100
    domain_acc = domain_acc / batch_count * 100
    slot_acc = slot_acc / batch_count / len(ontology.all_info_slots) * 100
    joint_acc = joint_acc / batch_count * 100

    del gate_acc, domain_acc, slot_acc, joint_acc
    torch.cuda.empty_cache()

    val_dial = json.load(open(os.path.join(config.data_path, "dev_data.json"), "r"))

    _, inform_rate, success_rate, bleu = evaluator.evaluateModel(val_dial_gens_decoded, val_dial_gens, val_dial, mode='valid')

    return gate_acc.item(), joint_acc.item(), slot_acc.item(), domain_acc.item(), inform_rate, success_rate, bleu

def parse_action(decoded_action):
    decoded_action = re.sub("\s+", "", decoded_action)
    if not re.fullmatch("(\[ACTION\]\[\w+\]-\[\w+\](-\w+)?)+", decoded_action):  # not valid action
        return []
    all_actions = []
    actions = decoded_action.split("[ACTION]")[1:]
    for action in actions:
        action = action.split("-")
        if len(action) == 2:  # no slot
            all_actions.append("-".join(action))
        elif len(action) == 3:  # one slot
            all_actions.append("-".join(action))
    
    return all_actions

def save(model, optimizer, save_path, epoch, score):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "score": score
    }
    torch.save(checkpoint, save_path)

def load(model, optimizer, save_path, cuda_device, mode="adam"):
    checkpoint = torch.load(save_path, map_location = lambda storage, loc: storage.cuda(cuda_device))
    model.load_state_dict(checkpoint["model"])
    if mode == "adam":
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        pass

    return checkpoint["epoch"], checkpoint["score"]

if __name__ == "__main__":
    config = Config()
    parser = config.parser
    config = parser.parse_args()
    init_process(config)
