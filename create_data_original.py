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

    def create_data(self):
        data = {}
        train = {}
        dev = {}
        test = {}
        ignore_list = ["SNG1213", "PMUL0382", "PMUL0237", "SNG0225"]
        logger.info("Processing data...")
        for dial_id, dial in tqdm(self.data.items(), ncols=150):
            dial_id = dial_id.split(".")[0]
            if dial_id in ignore_list:
                continue
            dialogue = {}
            goal = {}
            dial_domains = []
            for key, value in dial["goal"].items():  # process user's goal
                if key in ontology.all_domains and value != {}:
                    if value.get("reqt"):  # normalize requestable slot names
                        for idx, slot in enumerate(value["reqt"]):
                            if ontology.normlize_slot_names.get(slot):
                                value["reqt"][idx] = ontology.normlize_slot_names[slot]
                    goal[key] = value
                    dial_domains.append(key)
            if len(dial_domains) == 0:
                ignore_list.append(dial_id)
                continue

            # entries offered by the system in dataset
            dial["offered_entries"] = {"hotel": {}, "restaurant": {}, "attraction": {}, "train": {}}

            dialogue["goal"] = goal
            dialogue["log"] = []
            acts = self.acts[dial_id]
            turn = {}
            prev_domain = [0 for i in range(len(ontology.all_domains))]
            user_history = ""
            for turn_num, turn_dial in enumerate(dial["log"]):
                meta_data = turn_dial["metadata"]
                if meta_data == {}:  # user turn
                    turn["turn_num"] = int(turn_num/2)
                    user_ = turn_dial["text"].lower()
                    turn["user"] = clean_text(turn_dial["text"])
                    
                    for domain, addresses in self.addresses.items():
                        for address in addresses:
                            user_ = user_.replace(address, "[{}_address]".format(domain))
                    for domain, names in self.names.items():
                        for name in names:
                            user_ = user_.replace(name, "[{}_name]".format(domain))
                    user_ = clean_text(user_)
                    user_ = " " + user_ + " " # add a whitespace for delexicalization
                    for k, v in self.delex_dict.items():
                        user_ = user_.replace(k, v)  # delexicalize values
                    user_ = user_.strip()

                    user_ = re.sub(r"\d+\.?\d* pounds?", "[train_price]", user_)

                    is_train = False
                    for token in ["train", "arrive", "arrives", "arrived", "arriving", "arrival", "destination", "reach",
                        "leave", "leaves", "leaving", "leaved", "depart", "departing", "departs", "departure", "[train_"]:
                        if token in user_:
                            is_train = True
                            break
                    if is_train:
                        for k, v in self.train_time_dict.items():
                            user_ = user_.replace(k, v)  # delexicalize train times

                    user_ = re.sub("(\d\s?){7,11}", "[taxi_phone]", user_)  # delexicalize phone number
                    
                    while user_.find("[train_time]") != -1:  # replace [train_time] to [train_arrive] or [train_leave] by rule
                        user_split = user_.split()
                        idx = user_split.index("[train_time]")
                        replaced = False
                        for token in user_split[:idx][::-1]:
                            if token in ["arrive", "arrives", "arrived", "arriving", "arrival", "destination", "reach", "by", "before", "have", "to"]:
                                user_split[idx] = "[train_arrive]"
                                replaced = True
                                break
                            elif token in ["leave", "leaves", "leaving", "leaved", "depart", "departing", "departs", "departure", "from", "after", "earlier", "there"]:
                                user_split[idx] = "[train_leave]"
                                replaced = True
                                break
                        if not replaced:
                            user_split[idx] = "[train_leave]"
                        user_ = " ".join(user_split)

                    user_history += turn["user"]

                    # check activated domains
                    domain_state = [0 for i in range(len(ontology.all_domains))]
                    if turn_dial.get("dialog_act"):
                        for domain_act in turn_dial["dialog_act"].keys():
                            domain = domain_act.split("-")[0].lower().replace("bus", "train")
                            if domain == "booking" or domain == "general":
                                continue
                            else:
                                domain_idx = ontology.all_domains.index(domain)
                            domain_state[domain_idx] = 1
                    if turn_dial.get("span_info"):
                        for info in turn_dial["span_info"]:
                            domain_act = info[0]
                            domain = domain_act.split("-")[0].lower().replace("bus", "train")
                            if domain == "booking" or domain == "general":
                                continue
                            else:
                                domain_idx = ontology.all_domains.index(domain)
                            domain_state[domain_idx] = 1

                    # make domain state
                    if domain_state == [0 for i in range(len(ontology.all_domains))]:
                        turn["domain_state"] = prev_domain
                    else:
                        turn["domain_state"] = domain_state
                        prev_domain = turn["domain_state"]

                    # delexicalize user utterance for evaluation
                    if turn["domain_state"][ontology.all_domains.index("restaurant")] == 1:
                        for pricerange in self.priceranges:
                            user_ = user_.replace(pricerange, "[restaurant_pricerange]")
                        for area in self.areas:
                            user_ = user_.replace(area, "[restaurant_area]")
                        user_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                            r"\1 [restaurant_choice] ", user_)
                        user_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|place|restaurant)", \
                            r"[restaurant_choice] \2", user_)
                    elif turn["domain_state"][ontology.all_domains.index("hotel")] == 1:
                        for pricerange in self.priceranges:
                            user_ = user_.replace(pricerange, "[hotel_pricerange]")
                        for area in self.areas:
                            user_ = user_.replace(area, "[hotel_area]")
                        user_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                            r"\1 [hotel_choice] ", user_)
                        user_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|hotel|guest|place)", \
                            r"[hotel_choice] \2", user_)
                    elif turn["domain_state"][ontology.all_domains.index("attraction")] == 1:
                        for area in self.areas:
                            user_ = user_.replace(area, "[attraction_area]")
                        user_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                            r"\1 [attraction_choice] ", user_)
                        user_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|attraction|place)", \
                            r"[attraction_choice] \2", user_)
                    elif turn["domain_state"][ontology.all_domains.index("train")] == 1:
                        user_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                            r"\1 [train_choice] ", user_)
                        user_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|train)", \
                            r"[train_choice] \2", user_)

                    user_ = clean_text(user_)
                    dial["log"][turn_num]["text"] = user_

                else:  # system turn
                    turn["response"] = clean_text(turn_dial["text"])

                    # delexicalize address & name before other preprocessing
                    response_ = turn_dial["text"].lower()
                    for domain, addresses in self.addresses.items():
                        for address in addresses:
                            response_ = response_.replace(address, "[{}_address]".format(domain))
                    for domain, names in self.names.items():
                        for name in names:
                            response_ = response_.replace(name, "[{}_name]".format(domain))
                    response_ = clean_text(response_)
                    response_ = " " + response_ + " " # add a whitespace for delexicalization
                    for k, v in self.delex_dict.items():
                        response_ = response_.replace(k, v)  # delexicalize values
                    response_ = response_.strip()

                    # delexicalize other cases of price
                    response_ = re.sub(r"\d+\.?\d* pounds?", "[train_price]", response_)

                    is_train = False
                    for token in ["train", "arrive", "arrives", "arrived", "arriving", "arrival", "destination", "reach",
                        "leave", "leaves", "leaving", "leaved", "depart", "departing", "departs", "departure", "[train_"]:
                        if token in response_:
                            is_train = True
                            break
                    if is_train:
                        for k, v in self.train_time_dict.items():
                            response_ = response_.replace(k, v)  # delexicalize train times

                    response_ = re.sub("(\d\s?){7,11}", "[taxi_phone]", response_)  # delexicalize phone number
                    
                    while response_.find("[train_time]") != -1:  # replace [train_time] to [train_arrive] or [train_leave] by rule
                        response_split = response_.split()
                        idx = response_split.index("[train_time]")
                        replaced = False
                        for token in response_split[:idx][::-1]:
                            if token in ["arrive", "arrives", "arrived", "arriving", "arrival", "destination", "reach", "by", "before", "have", "to"]:
                                response_split[idx] = "[train_arrive]"
                                replaced = True
                                break
                            elif token in ["leave", "leaves", "leaving", "leaved", "depart", "departing", "departs", "departure", "from", "after", "earlier", "there"]:
                                response_split[idx] = "[train_leave]"
                                replaced = True
                                break
                        if not replaced:
                            response_split[idx] = "[train_leave]"
                        response_ = " ".join(response_split)

                    belief = {}
                    act = {}

                    for domain in dial_domains:  # active domains of dialogue
                        for slot, value in meta_data[domain]["book"].items():  # book
                            if slot == "booked":

                                # make offered entry
                                try:
                                    if len(meta_data[domain]["book"]["booked"]) != 0:
                                        if domain == "train":
                                            value = meta_data[domain]["book"]["booked"][0]["trainID"]
                                            for entry in self.db.db["train"]:
                                                if entry["id"] == value:
                                                    dial["offered_entries"]["train"] = entry
                                        elif domain in ["hotel", "restaurant"]:
                                            value = meta_data[domain]["book"]["booked"][0]["name"]
                                            for entry in self.db.db[domain]:
                                                if entry["name"] == value:
                                                    dial["offered_entries"][domain] = entry
                                except:
                                    pass
                                continue
                            slot, value = clean_slot_values(domain, slot, value)
                            if value != "":
                                # DORA doesn't use Name value, from system's recommendation but never mentioned by user, as label of belief tracking
                                if slot == "name" and value not in user_history:
                                    continue
                                belief["{}-{}".format(domain, slot)] = value
                        for slot, value in meta_data[domain]["semi"].items():  # semi
                            slot, value = clean_slot_values(domain, slot, value)
                            if value != "":
                                if slot == "name" and value not in user_history:
                                    continue
                                belief["{}-{}".format(domain, slot)] = value

                                # make offered entry
                                try:
                                    if slot == "name":
                                        for entry in self.db.db[domain]:
                                            if entry["name"] == value:
                                                dial["offered_entries"][domain] = entry
                                except:
                                    pass

                    turn["belief"] = belief

                    if acts.get(str(turn["turn_num"]+1)) and type(acts.get(str(turn["turn_num"]+1))) != str:  # mapping system action
                        for domain_act, slots in acts[str(turn["turn_num"]+1)].items():
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
                                                dial["offered_entries"]["train"] = entry
                                    elif domain.lower() in ["restaurant", "hotel", "attraction"] and act_.lower() in ["recommend", "inform"] and slot_ == "name":
                                        for entry in self.db.db[domain.lower()]:
                                            if entry["name"] == value_:
                                                dial["offered_entries"][domain.lower()] = entry
                                except:
                                    pass

                            act[domain_act.lower()] = act_temp
                    turn["action"] = act

                    if turn_dial.get("dialog_act"):
                        for domain_act in turn_dial["dialog_act"].keys():
                            domain = domain_act.split("-")[0].lower().replace("bus", "train")
                            if domain == "booking" or domain == "general":
                                continue
                            else:
                                domain_idx = ontology.all_domains.index(domain)
                            domain_state[domain_idx] = 1
                    if turn_dial.get("span_info"):
                        for info in turn_dial["span_info"]:
                            domain_act = info[0]
                            domain = domain_act.split("-")[0].lower().replace("bus", "train")
                            if domain == "booking" or domain == "general":
                                continue
                            else:
                                domain_idx = ontology.all_domains.index(domain)
                            domain_state[domain_idx] = 1

                    if domain_state == [0 for i in range(len(ontology.all_domains))]:
                        turn["domain_state"] = prev_domain
                    else:
                        turn["domain_state"] = domain_state
                        prev_domain = turn["domain_state"]

                    # delexicalize system response
                    if turn["domain_state"][ontology.all_domains.index("restaurant")] == 1:
                        for pricerange in self.priceranges:
                            response_ = response_.replace(pricerange, "[restaurant_pricerange]")
                        for area in self.areas:
                            response_ = response_.replace(area, "[restaurant_area]")
                        response_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                            r"\1 [restaurant_choice] ", response_)
                        response_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|place|restaurant)", \
                            r"[restaurant_choice] \2", response_)
                    elif turn["domain_state"][ontology.all_domains.index("hotel")] == 1:
                        for pricerange in self.priceranges:
                            response_ = response_.replace(pricerange, "[hotel_pricerange]")
                        for area in self.areas:
                            response_ = response_.replace(area, "[hotel_area]")
                        response_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                            r"\1 [hotel_choice] ", response_)
                        response_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|hotel|guest|place)", \
                            r"[hotel_choice] \2", response_)
                    elif turn["domain_state"][ontology.all_domains.index("attraction")] == 1:
                        for area in self.areas:
                            response_ = response_.replace(area, "[attraction_area]")
                        response_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                            r"\1 [attraction_choice] ", response_)
                        response_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|attraction|place)", \
                            r"[attraction_choice] \2", response_)
                    elif turn["domain_state"][ontology.all_domains.index("train")] == 1:
                        response_ = re.sub(r"(are|have|has|find|found|see|saw|get|got|show|about|over) ([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) ", \
                            r"\1 [train_choice] ", response_)
                        response_ = re.sub(r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten) (different|options|train)", \
                            r"[train_choice] \2", response_)

                    response_ = clean_text(response_)
                    turn["response_delex"] = response_

                    # for evaluation
                    pointer_vector = addDBPointer(turn_dial)
                    pointer_vector = addBookingPointer(dial, turn_dial, pointer_vector)
                    dial["log"][turn_num-1]["db_pointer"] = pointer_vector.tolist()
                    dial["log"][turn_num]["text"] = response_

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
    with open(os.path.join(data_path, "train_data.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(data_path, "dev_data.json"), "w") as f:
        json.dump(dev, f, indent=2)
    with open(os.path.join(data_path, "test_data.json"), "w") as f:
        json.dump(test, f, indent=2)
    logger.info("Saved.")