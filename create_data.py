import json
import logging
import os
import re

import numpy as np
from tqdm import tqdm

import ontology

from config import Config
from clean_data import clean_text, clean_slot_values
from delexicalize import get_delex_dict, get_addresses_and_names
import dbPointer
from db import DB
import pdb



def addBookingPointer(task, turn, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if task['goal']['restaurant']:
        if turn['metadata']['restaurant'].get("book"):
            if turn['metadata']['restaurant']['book'].get("booked"):
                if turn['metadata']['restaurant']['book']["booked"]:
                    if "reference" in turn['metadata']['restaurant']['book']["booked"][0]:
                        rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if task['goal']['hotel']:
        if turn['metadata']['hotel'].get("book"):
            if turn['metadata']['hotel']['book'].get("booked"):
                if turn['metadata']['hotel']['book']["booked"]:
                    if "reference" in turn['metadata']['hotel']['book']["booked"][0]:
                        hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if task['goal']['train']:
        if turn['metadata']['train'].get("book"):
            if turn['metadata']['train']['book'].get("booked"):
                if turn['metadata']['train']['book']["booked"]:
                    if "reference" in turn['metadata']['train']['book']["booked"][0]:
                        train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    return pointer_vector

def addDBPointer(turn):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    for domain in domains:
        num_entities = dbPointer.queryResult(domain, turn)
        pointer_vector = dbPointer.oneHotVector(num_entities, domain, pointer_vector)

    return pointer_vector


class DataCreator(object):
    def __init__(self, data_path, db_path, config):
        self.data_path = data_path
        self.db_path = db_path
        
        self.train_list = open(os.path.join(data_path, "trainListFile.txt"), "r").read().split("\n")[:-1]
        self.dev_list = open(os.path.join(data_path, "valListFile.txt"), "r").read().split("\n")[:-1]
        self.test_list = open(os.path.join(data_path, "testListFile.txt"), "r").read().split("\n")[:-1]
        
        self.data = json.load(open(os.path.join(data_path, "data.json"), "r"))
        self.acts = json.load(open(os.path.join(data_path, "system_acts.json"), "r"))
        self.delex_dict, self.train_time_dict = get_delex_dict(data_path, db_path)
        self.addresses, self.names = get_addresses_and_names(db_path)
        self.db = DB(db_path)
        
        self.priceranges = ["cheap", "cheaper", "inexpensive", "inexpensively", "low cost", "lower end", "pricey", "less costly", \
            "cheapest", "quite low", "chear", "espensive", "budget- friendly" \
            "moderate", "moderately", "reasonable", "affordable", "boderate", \
            "expensive", "high class", "high-end"]
        self.areas = ["east", "eastside", "eastern part"\
            "west", "western part", "wet part", "wet end", "westside", \
            "south", "southside", "southern", "southend", \
            "north", "northe part", "northern", "northside", "northend", \
            "centre", "centrally located", "centry area", "central district", "centrally", "central zone", "centra", "centrem", "central cambridge", \
            "central region", "cetnre", "cetre", "cenre", "centreof", "central area"]
        
    def part1(self, text):
        text_ = text.lower()
        for domain, addresses in self.addresses.items():
            for address in addresses:
                text_ = text_.replace(address, "[{}_address]".format(domain))
        for domain, names in self.names.items():
            for name in names:
                text_ = text_.replace(name, "[{}_name]".format(domain))
        text_ = clean_text(text_)
        text_ = " " + text_ + " " # add a whitespace for delexicalization
        for k, v in self.delex_dict.items():
            text_ = text_.replace(k, v)  # delexicalize values
        text_ = text_.strip()

        # delexicalize other cases of price
        text_ = re.sub(r"\d+\.?\d* pounds?", "[train_price]", text_)

        is_train = False
        for token in ["train", "arrive", "arrives", "arrived", "arriving", "arrival", "destination", "reach",
            "leave", "leaves", "leaving", "leaved", "depart", "departing", "departs", "departure", "[train_"]:
            if token in text_:
                is_train = True
                break
        if is_train:
            for k, v in self.train_time_dict.items():
                text_ = text_.replace(k, v)  # delexicalize train times

        text_ = re.sub("(\d\s?){7,11}", "[taxi_phone]", text_)  # delexicalize phone number
        
        while text_.find("[train_time]") != -1:  # replace [train_time] to [train_arrive] or [train_leave] by rule
            text_split = text_.split()
            idx = text_split.index("[train_time]")
            replaced = False
            for token in text_split[:idx][::-1]:
                if token in ["arrive", "arrives", "arrived", "arriving", "arrival", "destination", "reach", "by", "before", "have", "to"]:
                    text_split[idx] = "[train_arrive]"
                    replaced = True
                    break
                elif token in ["leave", "leaves", "leaving", "leaved", "depart", "departing", "departs", "departure", "from", "after", "earlier", "there"]:
                    text_split[idx] = "[train_leave]"
                    replaced = True
                    break
            if not replaced:
                text_split[idx] = "[train_leave]"
            text_ = " ".join(text_split)
        return text_
        
    def delexation(self,turn,text_):                  
        if turn["domain_state"][ontology.all_domains.index("restaurant")] == 1:
            for pricerange in self.priceranges:
                text_ = text_.replace(pricerange, "[restaurant_pricerange]")
            for area in self.areas:
                text_ = text_.replace(area, "[restaurant_area]")
            text_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                r"\1 [restaurant_choice] ", text_)
            text_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|place|restaurant)", \
                r"[restaurant_choice] \2", text_)
        elif turn["domain_state"][ontology.all_domains.index("hotel")] == 1:
            for pricerange in self.priceranges:
                text_ = text_.replace(pricerange, "[hotel_pricerange]")
            for area in self.areas:
                text_ = text_.replace(area, "[hotel_area]")
            text_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                r"\1 [hotel_choice] ", text_)
            text_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|hotel|guest|place)", \
                r"[hotel_choice] \2", text_)
        elif turn["domain_state"][ontology.all_domains.index("attraction")] == 1:
            for area in self.areas:
                text_ = text_.replace(area, "[attraction_area]")
            text_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                r"\1 [attraction_choice] ", text_)
            text_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|attraction|place)", \
                r"[attraction_choice] \2", text_)
        elif turn["domain_state"][ontology.all_domains.index("train")] == 1:
            text_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                r"\1 [train_choice] ", text_)
            text_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|train)", \
                r"[train_choice] \2", text_)

        return text_


    def add_qa_answer(self, domain, slot, answer):
        try:
            if f'{domain}-{slot}' in ontology.QA['multichoice-domain']:
                answer_list = ontology.QA[f'{domain}-{slot}']['values']
                if answer in answer_list:
                    index = answer_list.index(answer)
                else:
                    answer = re.split('\||>',answer)[0] # cheap|moderate, cheap>moderate
                    answer = re.split('<',answer)[-1] # monday<thursday
                    index = answer_list.index(answer)
                return [answer, index]
            else:
                return answer
        except:
            print(f'{domain}-{slot}-{answer}')
            return -1
        
        
    def process_goal(self,raw_goal):
        goal = {}
        dial_domains = []
        
        for key, value in raw_goal.items():  # process user's goal
            if key in ontology.all_domains and value != {}:
                if value.get("reqt"):  # normalize requestable slot names
                    for idx, slot in enumerate(value["reqt"]):
                        if ontology.normlize_slot_names.get(slot):
                            value["reqt"][idx] = ontology.normlize_slot_names[slot]
                goal[key] = value
                dial_domains.append(key)
        
        return goal, dial_domains
    

    def user_side(self, turn_num, turn_dial, turn, prev_domain, raw_dial, domain_state, user_history):
        turn["turn_num"] = int(turn_num/2)
        turn["user"] = clean_text(turn_dial["text"])
        
        user_  = self.part1(turn_dial["text"].lower())
        
        user_history += turn["user"]  # TODO

        # check activated domains
        domains = []
        if turn_dial.get("dialog_act"):
            for domain_act in turn_dial["dialog_act"].keys():
                domains.append(domain_act.split("-")[0].lower().replace("bus", "train"))
            
        if turn_dial.get("span_info"):
            for info in turn_dial["span_info"]:
                domains.append(info[0].split("-")[0].lower().replace("bus", "train"))
        
        for domain in domains:        
            if domain not in ["booking","general"]:
                domain_state[ontology.all_domains.index(domain)] = 1
            
                    
        # make domain state
        if domain_state == [0 for i in range(len(ontology.all_domains))]:
            turn["domain_state"] = prev_domain
        else:
            turn["domain_state"] = domain_state
            prev_domain = turn["domain_state"]

        user_ = self.delexation(turn,user_)
        user_ = clean_text(user_)
        raw_dial["log"][turn_num]["text"] = user_
        
        return turn, user_history, domain_state, raw_dial, prev_domain
        
        
    def _searching_DB(self,domain,key, value):
        for entry in self.db.db[domain]:
            if entry[key] == value:
                return entry
        
    def make_belief(self, dial_domains,turn_dial, user_history,raw_dial):
        belief = {}
        
        for domain in dial_domains:  # active domains of dialogue
            for slot, value in turn_dial["metadata"][domain]["book"].items():  # book
                if slot == "booked":
                    try:
                        if len(turn_dial["metadata"][domain]["book"]["booked"]) != 0:
                            if domain == "train":
                                value = turn_dial["metadata"][domain]["book"]["booked"][0]["trainID"]
                                entry = self._searching_DB(domain,"id",value)
                                if entry: raw_dial["offered_entries"]["train"] = entry
                                        
                            elif domain in ["hotel", "restaurant"]:
                                value = turn_dial["metadata"][domain]["book"]["booked"][0]["name"]
                                entry = self._searching_DB(domain,"name",value)
                                if entry: raw_dial["offered_entries"]["train"] = entry
                    except:
                        pass
                else:
                    slot, value = clean_slot_values(domain, slot, value)
                    if value != "":
                        if slot == "name" and value not in user_history:
                            continue # DORA doesn't use Name value, from system's recommendation but never mentioned by user, as label of belief tracking
                        belief["{}-{}".format(domain, slot)] = self.add_qa_answer(domain, slot, value)
                        if self.add_qa_answer(domain, slot, value) == -1: print('wrong')
                    
                    
            for slot, value in turn_dial["metadata"][domain]["semi"].items():  # semi
                try:
                    if slot == "name":
                        entry = self._searching_DB(domain,"name",value)
                        if entry: raw_dial["offered_entries"][domain] = entry
                except:
                    pass
                    
                slot, value = clean_slot_values(domain, slot, value)
                if value != "":
                    if slot == "name" and value not in user_history:
                        continue
                    belief["{}-{}".format(domain, slot)] = self.add_qa_answer( domain, slot, value)
                    if self.add_qa_answer(domain, slot, value) == -1: print('wrong')
                    
        return belief, raw_dial

        
    def system_side(self, turn_num, turn_dial, turn, prev_domain, raw_dial, turn_acts,domain_state, user_history, dial_domains):
        turn["response"] = clean_text(turn_dial["text"])
        
        response_ = self.part1(turn_dial["text"].lower())
        
        act = {}
        belief, raw_dial = self.make_belief(dial_domains,turn_dial,user_history, raw_dial)
        turn["belief"] = belief
        
        if turn_acts.get(str(turn["turn_num"]+1)) and type(turn_acts.get(str(turn["turn_num"]+1))) != str:  # mapping system action
            for domain_act, slots in turn_acts[str(turn["turn_num"]+1)].items():
                act_temp = []
                for slot in slots:  # slot: [slot, value]
                    domain, act_ = domain_act.split("-")
                    slot_, value_ = clean_slot_values(domain, slot[0], slot[1])
                    if slot_ == "price" and domain.lower() in ["hotel", "restaurant"]:
                        # mapping hotel-price & restaurant-price to hotel-pricerange & restaurant-pricerange
                        slot_ = "pricerange"
                    if slot_ in ["open"]:  # ignore slots
                        continue
                    if slot_ == "none" or value_ in ["?", "none"]:  # general domain or request slot or parking
                        act_temp.append(slot_)
                    else:
                        act_temp.append("{}-{}".format(slot_, value_))

                    # make offered entry
                    try:
                        if domain.lower() == "train" and act_.lower() in ["recommend", "inform", "offerbook", "offerbooked"] and slot_ == "id":
                            for entry in self.db.db["train"]:
                                if entry["id"] == value_:
                                    raw_dial["offered_entries"]["train"] = entry
                        elif domain.lower() in ["restaurant", "hotel", "attraction"] and act_.lower() in ["recommend", "inform"] and slot_ == "name":
                            for entry in self.db.db[domain.lower()]:
                                if entry["name"] == value_:
                                    raw_dial["offered_entries"][domain.lower()] = entry
                    except:
                        pass

                act[domain_act.lower()] = act_temp
        turn["action"] = act
        

        # check activated domains
        domains = []
        if turn_dial.get("dialog_act"):
            for domain_act in turn_dial["dialog_act"].keys():
                domains.append(domain_act.split("-")[0].lower().replace("bus", "train"))
            
        if turn_dial.get("span_info"):
            for info in turn_dial["span_info"]:
                domains.append(info[0].split("-")[0].lower().replace("bus", "train"))
        
        for domain in domains:        
            if domain not in ["booking","general"]:
                domain_state[ontology.all_domains.index(domain)] = 1
                
                
        if domain_state == [0 for i in range(len(ontology.all_domains))]:
            turn["domain_state"] = prev_domain
        else:
            turn["domain_state"] = domain_state
            prev_domain = turn["domain_state"]

        response_ = self.delexation(turn, response_)
        response_ = clean_text(response_)
        turn["response_delex"] = response_

        # for evaluation
        pointer_vector = addDBPointer(turn_dial)
        pointer_vector = addBookingPointer(raw_dial, turn_dial, pointer_vector)
        raw_dial["log"][turn_num-1]["db_pointer"] = pointer_vector.tolist()
        raw_dial["log"][turn_num]["text"] = response_

        return turn, raw_dial
    
    def create_data(self):
        data = {}
        train = {}
        dev = {}
        test = {}
        ignore_list = ["SNG1213", "PMUL0382", "PMUL0237", "SNG0225", "SSNG0388", "MUL2677"]
        logger.info("Processing data...")
        
        for dial_id, raw_dial in tqdm(self.data.items(), ncols=150): # 대화를 하나 가져와서
            dial_id = dial_id.split(".")[0]
            if dial_id in ignore_list: continue
            dialogue = {}
            dialogue["goal"] = goal            
            dialogue["log"] = []
            raw_dial["offered_entries"] = {"hotel": {}, "restaurant": {}, "attraction": {}, "train": {}} # offered by system
            
            
            goal, dial_domains = self.process_goal(raw_dial["goal"])
            if len(dial_domains) == 0:  continue

            turn = {}
            turn_acts = self.acts[dial_id] # TODO
            prev_domain, domain_state = [0 for i in range(len(ontology.all_domains))], []
            user_history = ""
            
            for turn_num, turn_dial in enumerate(raw_dial["log"]):
                if turn_dial["metadata"] == {}:  # user turn
                    domain_state = [0 for i in range(len(ontology.all_domains))] # TODO
                    turn, user_history, domain_state, raw_dial, prev_domain  = \
                    self.user_side( turn_num, turn_dial, turn, prev_domain,raw_dial,domain_state, user_history)
                else:  # system turn
                    # raw dial을 다시 받을 필요가 있나..?
                    turn, raw_dial = self.system_side(turn_num, turn_dial, turn, prev_domain,raw_dial,turn_acts, domain_state,user_history, dial_domains)
                    dialogue["log"].append(turn)
                    turn = {}  # clear turn
                    
            data[dial_id] = dialogue

        logger.info("Processing finished.")
        logger.info("Dividing data to train/dev/test...")
        for dial_id in self.train_list:
            dial_id = dial_id.split(".")[0]
            if dial_id not in ignore_list:
                train[dial_id] = data[dial_id]
        for dial_id in self.dev_list:
            dial_id = dial_id.split(".")[0]
            if dial_id not in ignore_list:
                dev[dial_id] = data[dial_id]
        for dial_id in self.test_list:
            dial_id = dial_id.split(".")[0]
            if dial_id not in ignore_list:
                test[dial_id] = data[dial_id]
        logger.info("Dividing finished.")

        logger.info("Save delex.json for evaluation...")
        with open(os.path.join(data_path, "delex.json"), "w") as f:
            json.dump(self.data, f, indent=2)
        logger.info("Saved.")

        return train, dev, test


if __name__=='__main__':
    config = Config()
    parser = config.parser
    hparams = parser.parse_args()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    data_path = hparams.data_path
    db_path = hparams.db_path
    creator = DataCreator(data_path, db_path, hparams)
    train, dev, test = creator.create_data()
    
    logger.info("Saving data...")
    with open(os.path.join(data_path, "train_data_.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(data_path, "dev_data_.json"), "w") as f:
        json.dump(dev, f, indent=2)
    with open(os.path.join(data_path, "test_data_.json"), "w") as f:
        json.dump(test, f, indent=2)
    logger.info("Saved.")