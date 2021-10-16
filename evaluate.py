# This code referenced
# https://github.com/budzianowski/multiwoz

import math
from collections import Counter
from nltk.util import ngrams
import json
import sqlite3
import os
from os.path import dirname, abspath, join
import random
import logging
import re
from transformers import BertTokenizer

import ontology
from db import DB


class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            # if type(hyps[0]) is list:
            #    hyps = [hyp.split() for hyp in hyps[0]]
            # else:
            #    hyps = [hyp.split() for hyp in hyps]

            # refs = [ref.split() for ref in refs]
            hyps = [hyps]
            # Shawn's evaluation
            # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class MultiWozEvaluator(BaseEvaluator):
    def __init__(self, data_name, data_path, db_path, assets_path):
        self.data_name = data_name
        self.delex_dialogues = json.load(open(os.path.join(data_path, "delex.json"), "r"))
        self.db = DB(db_path)
        self.labels = list()
        self.hyps = list()
        with open(os.path.join(assets_path, "never_split.txt"), "r") as f:
            never_split = f.read().split("\n")
        self.tokenizer = BertTokenizer(os.path.join(assets_path, "vocab.txt"), never_split=never_split)

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def _parseGoal(self, goal, d, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
        if 'info' in d['goal'][domain]:
        # if d['goal'][domain].has_key('info'):
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    if 'trainID' in d['goal'][domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    for s in d['goal'][domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append("reference")

            goal[domain]["informable"] = d['goal'][domain]['info']
            if 'book' in d['goal'][domain]:
            # if d['goal'][domain].has_key('book'):
                goal[domain]["booking"] = d['goal'][domain]['book']

        return goal

    def _evaluateGeneratedDialogue(self, dialog, goal, realDialogue, real_requestables, belief_gen, filename, soft_acc=False):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""
        # for computing corpus success
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, sent_t in enumerate(dialog):
            for domain in goal.keys():
                # for computing success
                if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION

                        belief_state = ["none" for i in range(len(ontology.all_info_slots))]
                        for slot, value in belief_gen[t].items():
                            belief_state[ontology.all_info_slots.index(slot)] = value
                        venues = self.db.get_match(belief_state, domain)
                        
                        # if the generated belief state matches the entry that was offered in dataset, select the entry as venue_offered,
                        # not just sample randomly to remove randomness in evaulation
                        if self.delex_dialogues[filename]["offered_entries"][domain] != {} and self.delex_dialogues[filename]["offered_entries"][domain] in venues:
                            venues = [self.delex_dialogues[filename]["offered_entries"][domain]]

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                            # venue_offered[domain] = [venues[0]]
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                                # venue_offered[domain] = [venues[0]]
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'train_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            # if realDialogue['goal'][domain].has_key('info'):
            if 'info' in realDialogue['goal'][domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in realDialogue['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # the original method
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         # if realDialogue['goal'][domain].has_key('reqt') and 'id' not in realDialogue['goal'][domain]['reqt']:
            #         if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # Wrong one in HDSA
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         if goal[domain]['requestable'] and 'id' not in goal[domain]['requestable']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        match_domains = []
        success_domains = []
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_belief = ["none" for i in range(len(ontology.all_info_slots))]
                for slot, value in goal[domain]["informable"].items():
                    goal_belief[ontology.all_info_slots.index(ontology.belief_state_mapping[domain][slot])] = value
                goal_venues = self.db.get_match(goal_belief, domain)

                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_domains.append(domain)
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_domains.append(domain)
                    match_stat = 1
            else:
                if domain + '_name]' in venue_offered[domain]:
                    match += 1
                    match_domains.append(domain)
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_domains.append(domain)
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_domains.append(domain)
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        # rint requests, 'DIFF', requests_real, 'SUCC', success
        return success, match, stats, match_domains, success_domains

    def _evaluateRealDialogue(self, dialog, filename):
        """Evaluation of the real dialogue.
        First we loads the user goal and then go through the dialogue history.
        Similar to evaluateGeneratedDialogue above."""
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # get the list of domains in the goal
        domains_in_goal = []
        goal = {}
        for domain in domains:
            if dialog['goal'][domain]:
                goal = self._parseGoal(goal, dialog, domain)
                domains_in_goal.append(domain)

        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in goal.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = goal[domain]['requestable']

        return goal, real_requestables, 

    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities

    def evaluateModel(self, dialogues, dialogues_encoded, real_dialogues=False, mode='valid', save_path=None, make_report=False):
        """Gathers statistics for the whole sets."""
        delex_dialogues = self.delex_dialogues
        if mode == "rl":
            successes = []
            matches = []
        else:
            successes, matches = 0, 0
        total = 0

        gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                         'taxi': [0, 0, 0], 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        inform_fail_report = {}
        success_fail_report = {}
        reports = {}

        for filename, dial in dialogues.items():
            data = delex_dialogues[filename]
            goal, requestables = self._evaluateRealDialogue(data, filename)
            success, match, stats, match_domains, success_domains = \
                self._evaluateGeneratedDialogue(dial["response"], goal, data, requestables, dial["belief_state"], filename, soft_acc=mode=='rl')

            if mode == "rl":
                successes.append(success)
                matches.append(match)
            else:
                successes += success
                matches += match
            total += 1

            # failure report
            if save_path is not None:
                if match == 0:
                    inform_fail_report[filename] = {"log": []}
                    for turn_idx in range(len(dial["response"])):
                        inform_fail_report[filename]["log"].append(data["log"][turn_idx*2]["text"])
                        inform_fail_report[filename]["log"].append(dial["response"][turn_idx])
                    inform_fail_report[filename]["match_domains"] = match_domains
                    inform_fail_report[filename]["success_domains"] = success_domains
                    goal_belief = ["" for i in range(len(ontology.all_info_slots))]
                    for domain in goal.keys():
                        if domain in ["hotel", "restaurant", "attraction", "train"]:
                            for slot, value in goal[domain]["informable"].items():
                                goal_belief[ontology.all_info_slots.index(ontology.belief_state_mapping[domain][slot])] = value
                    inform_fail_report[filename]["goal"] = goal_belief
                    inform_fail_report[filename]["belief_gen"] = dial["belief_state"][-1]
                elif success == 0:
                    success_fail_report[filename] = {"log": []}
                    for turn_idx in range(len(dial["response"])):
                        success_fail_report[filename]["log"].append(data["log"][turn_idx*2]["text"])
                        success_fail_report[filename]["log"].append(dial["response"][turn_idx])
                    success_fail_report[filename]["match_domains"] = match_domains
                    success_fail_report[filename]["success_domains"] = success_domains
                    goal_belief = ["" for i in range(len(ontology.all_info_slots))]
                    goal_requests = {}
                    for domain in goal.keys():
                        if domain in ["hotel", "restaurant", "attraction", "train"]:
                            for slot, value in goal[domain]["informable"].items():
                                goal_belief[ontology.all_info_slots.index(ontology.belief_state_mapping[domain][slot])] = value
                        goal_requests[domain] = goal[domain]["requestable"]
                    success_fail_report[filename]["goal"] = goal_belief
                    success_fail_report[filename]["goal_request"] = goal_requests
                    success_fail_report[filename]["belief_gen"] = dial["belief_state"][-1]

            if make_report:
                reports[filename] = {"log": []}
                for turn_idx in range(len(dial["response"])):
                    log = {}
                    log["user"] = data["log"][turn_idx*2]["text"]
                    log["response"] = dial["response"][turn_idx]
                    log["belief"] = dial["belief_state"][turn_idx]
                    log["action"] = dial["action"][turn_idx]
                    reports[filename]["log"].append(log)

            for domain in gen_stats.keys():
                gen_stats[domain][0] += stats[domain][0]
                gen_stats[domain][1] += stats[domain][1]
                gen_stats[domain][2] += stats[domain][2]

            if 'SNG' in filename:
                for domain in gen_stats.keys():
                    sng_gen_stats[domain][0] += stats[domain][0]
                    sng_gen_stats[domain][1] += stats[domain][1]
                    sng_gen_stats[domain][2] += stats[domain][2]

        if real_dialogues:
            # BLUE SCORE
            corpus = []
            model_corpus = []
            bscorer = BLEUScorer()

            for dialogue in dialogues:
                dial_id = dialogue.split(".")[0]
                data = real_dialogues[dial_id]
                model_turns, corpus_turns = [], []
                for idx, turn in enumerate(data["log"]):
                    corpus_turns.append([self.tokenizer.encode(turn["response_delex"])])
                for turn in dialogues_encoded[dialogue]:
                    model_turns.append(turn)

                if len(model_turns) == len(corpus_turns):
                    corpus.extend(corpus_turns)
                    model_corpus.extend(model_turns)
                else:
                    raise('Wrong amount of turns')

            blue_score = bscorer.score(model_corpus, corpus)
        else:
            blue_score = 0.

        if mode != "rl":
            report = ""
            report += '{} Corpus Matches : {:2.2f}%'.format(mode, (matches / float(total) * 100)) + "\n"
            report += '{} Corpus Success : {:2.2f}%'.format(mode, (successes / float(total) * 100)) + "\n"
            report += '{} Corpus BLEU : {:2.2f}%'.format(mode, blue_score * 100) + "\n"
            report += 'Total number of dialogues: %s ' % total

            if save_path is not None:
                if not os.path.exists("results"):
                    os.mkdir("results")
                if not os.path.exists("results/{}".format(save_path)):
                    os.mkdir("results/{}".format(save_path))
                with open("results/{}/inform_failure_report.json".format(save_path), "w") as f:
                    json.dump(inform_fail_report, f, indent=2)
                with open("results/{}/success_failure_report.json".format(save_path), "w") as f:
                    json.dump(success_fail_report, f, indent=2)
                if make_report:
                    with open("results/{}/report.json".format(save_path), "w") as f:
                        json.dump(reports, f, indent=2)

            return report, matches/float(total) * 100, successes/float(total) * 100, blue_score * 100
        else:
            return successes


if __name__ == '__main__':
    mode = "test"
    evaluator = MultiWozEvaluator(mode, "data/MultiWOZ_2.1", "db")

    with open("data/test_dials.json", "r") as f:
        human_raw_data = json.load(f)
    human_proc_data = {}
    for key, value in human_raw_data.items():
        human_proc_data[key] = value['sys']

    # PROVIDE HERE YOUR GENERATED DIALOGUES INSTEAD
    generated_data = human_proc_data

    evaluator.evaluateModel(generated_data, False, mode=mode)
