import os
import json
import random
import re

import numpy as np
from nltk import edit_distance

import ontology


class DB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db = {}
        for domain in ontology.all_domains:
            self.db[domain] = json.load(open(os.path.join(db_path, "{}_db.json".format(domain)), "r"))
            for idx, entry in enumerate(self.db[domain]):
                temp = {}
                for i in range(len(entry.keys())):
                    (slot, value) = entry.popitem()
                    slot = slot.lower()
                    slot = ontology.normlize_slot_names[slot] if ontology.normlize_slot_names.get(slot) else slot
                    temp[slot] = value
                self.db[domain][idx] = temp
        self.name_list = {}
        for domain in ["restaurant", "hotel", "attraction"]:
            self.name_list[domain] = [entry["name"] for entry in self.db[domain]]

    def postprocessing(self, belief):
        """Postprocess the decoded values

        belief: [slots] => list(decoded)
        """
        for slot_idx, value in enumerate(belief):
            if ontology.all_info_slots[slot_idx].split("-")[1] in ["leave", "arrive"]:
                belief[slot_idx] = re.sub(r"([0-9]{2}) \: ([0-9]{2})", r"\1:\2", value)
            elif ontology.all_info_slots[slot_idx].split("-")[1] == "type":
                value = value.replace("guest house", "guesthouse")
                value = value.replace("swimming pool", "swimmingpool")
                value = value.replace("concert hall", "concerthall")
                value = value.replace("night club", "nightclub")
                value = value.replace("multiple sports", "mutliple sports")  # typo in dataset
                belief[slot_idx] = value
            elif ontology.all_info_slots[slot_idx].split("-")[1] == "food":
                value = value.replace("gastro pub", "gastropub")
                belief[slot_idx] = value
            elif ontology.all_info_slots[slot_idx].split("-")[1] in ["destination", "departure"]:
                value = value.replace("bishop stortford", "bishops stortford")
                belief[slot_idx] = value

        return belief

    def check_name_typo(self, domain, name):
        if ontology.name_typo.get(name):
            return ontology.name_typo[name]

        name_list = self.name_list[domain]
        if name in name_list:
            return name

        max_same_words = 0
        for candidate in name_list:
            num_same_words = 0
            words = candidate.split()
            for word in name.split():
                if word in words:
                    num_same_words += 1
            if num_same_words > max_same_words:
                selected = candidate
                max_same_words = num_same_words
        if max_same_words == 0:
            min_distance = 1000
            for candidate in name_list:
                distance = edit_distance(name, candidate)
                if distance < min_distance:
                    min_distance = distance
                    selected = candidate
        
        return selected

    def time_chcek(self, request_time, entity_time, slot):
        try:
            req_h, req_m = request_time.split(":")
            req_h = int(req_h)
            req_m = int(req_m)
            ent_h, ent_m = entity_time.split(":")
            ent_h = int(ent_h)
            ent_m = int(ent_m)
        except:
            return False
        if slot == "leave":  # check if entity time is later than request time
            if req_h < ent_h:
                return True
            elif req_h == ent_h and req_m <= ent_m:
                return True
            else:
                return False
        elif slot == "arrive":  # check if entity time is ealrier than request time
            if req_h > ent_h:
                return True
            elif req_h == ent_h and req_m >= ent_m:
                return True
            else:
                return False

    def get_match(self, belief, turn_domain):
        """Find entries matched to belief in DB.

        belief: [slots] => list(decoded)
        turn_domain: string
        """

        belief = self.postprocessing(belief)
        entries = []
        constraint = {}
        if turn_domain == "police":
            entries.append(self.db["police"][0])
        elif turn_domain == "taxi":
            entry = {}
            entry["car"] = random.choice(["toyota", "skoda", "bmw", "honda", "ford", "audi", "lexus", "volvo", "volkswagen", "tesla"])
            entry["color"] = random.choice(["black", "white", "red", "yellow", "blue", "grey"])
            entry["phone"] = "".join(np.array2string(np.random.randint(low=0, high=10, size=10))[1:-1].split())
            entries.append(entry)
        else:
            if turn_domain in ["hotel", "restaurant", "attraction"]:
                slot = "{}-name".format(turn_domain)
                slot_idx = ontology.all_info_slots.index(slot)
                if belief[slot_idx] != "none":
                    name = belief[slot_idx]
                    name = self.check_name_typo(turn_domain, name)
                    for entry_idx, entry in enumerate(self.db[turn_domain]):
                        if entry["name"] == name:
                            entries.append(entry)
                            return entries
            for idx, belief_ in enumerate(belief):
                domain, slot = ontology.all_info_slots[idx].split("-")
                if domain != turn_domain:
                    continue
                value = belief_
                if value not in ["none", "dontcare"]:
                    constraint[slot] = value
            for entry in self.db[turn_domain]:
                matched = True
                for slot, value in constraint.items():
                    if slot in ["leave", "arrive"]:
                        if not self.time_chcek(value, entry[slot], slot):
                            matched = False
                            break
                    elif slot in ontology.entry_slots[turn_domain] and entry[slot] != value:
                        matched = False
                        break
                if matched:
                    entries.append(entry)
        
        return entries

    def restore_delex(self, response_gen, entry):
        """
        response_gen: string
        entry: DB entry
        """

        restored = response_gen
        reference_nums = [str(i) for i in range(10)] + [chr(i) for i in range(65,91)]

        if entry is None:
            return response_gen

        for domain, slots in ontology.delex_slots.items():
            for slot in slots:
                if domain == "common":
                    if type(entry.get(slot, slot)) == dict:
                        continue
                    if "[{}]".format(slot) in restored:
                        restored = restored.replace("[{}]".format(slot), entry.get(slot, slot))
                elif slot == "reference":
                    if "[{}_{}]".format(domain, slot) in restored:
                        reference_num = random.sample(reference_nums[1:], 1)[0]
                        for i in range(7):
                            reference_num += random.sample(reference_nums, 1)[0]
                        restored = restored.replace("[{}_{}]".format(domain, slot), reference_num)
                else:
                    if "[{}_{}]".format(domain, slot) in restored:
                        restored = restored.replace("[{}_{}]".format(domain, slot), entry.get(slot, slot))
                        
        
        return restored

if __name__ == "__main__":
    db = DB("data/MultiWOZ_2.1")
    print("")