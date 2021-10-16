import json
import os

import ontology
import clean_data


def get_addresses_and_names(db_path):
    domains = ["hotel", "restaurant", "attraction", "police"]
    
    addresses = {}
    names = {}
    db = {}
    for domain in domains:
        db[domain] = json.load(open(os.path.join(db_path, "{}_db.json".format(domain)), "r"))
        addresses[domain] = []
        names[domain] = []
    for domain, entries in db.items():
        for entry in entries:
            addresses[domain].append(entry["address"].lower())
            names[domain].append(entry["name"].lower())
        addresses[domain] = list(set(addresses[domain]))
        names[domain] = list(set(names[domain]))
    names["hospital"] = ["addenbrookes hospital"]
    addresses["hospital"] = ["hills rd, cambridge", "hills rd., cambridge", "hills rd in cambridge", "hills rd, in cambridge", \
        "hills rd., in cambridge", "hills rd"]

    return addresses, names

def get_references(data_path):
    data = json.load(open(os.path.join(data_path, "data.json"), "r"))

    reference_nums = []
    for dial_id, dial in data.items():
        for turn in dial["log"]:
            belief = turn["metadata"]
            for domain, categories in belief.items():
                if domain in ["hotel", "restaurant", "train"]:
                    books = categories["book"]["booked"]
                    for book in books:
                        if book.get("reference"):
                            reference_nums.append(domain + "-" + book["reference"].lower())
    
    return list(set(reference_nums))

def get_delex_dict(data_path, db_path):
    delex_dict = {}
    train_time_dict = {}
    for domain in ontology.all_domains:
        if domain == "taxi":
            for color in ["black", "white", "red", "yellow", "blue", "grey"]:
                for car in ["toyota", "skoda", "bmw", "honda", "ford", "audi", "lexus", "volvo", "volkswagen", "tesla"]:
                    delex_dict["{} {}".format(color, car)] = " [taxi_car] "
        else:
            db = json.load(open(os.path.join(db_path, "{}_db.json".format(domain)), "r"))
            for idx, entry in enumerate(db):
                for slot, value in entry.items():
                    if slot == "location" or slot == "price" and domain == "hotel":
                        continue
                    slot, value = clean_data.clean_slot_values(domain, slot, value)
                    if slot == "price":
                        if value in ["?", "free"]:
                            continue
                        delex_dict[" {} ".format(value)] = " [{}_price] ".format(domain)
                    elif slot in ["food", "postcode", "duration", "name", "phone"]:
                        delex_dict[" {} ".format(value)] = " [{}_{}] ".format(domain, slot)
                    elif slot == "stars":
                        rules = ["{} star", "{}-star", "rating of {}", "star of {}"]
                        for rule in rules:
                            delex_dict[rule.format(value)] = rule.format("[hotel_stars]")
                    elif slot == "id" and domain == "train":
                        delex_dict[" {} ".format(value)] = " [train_id] "
                    elif slot in ["arrive", "leave"]:
                        train_time_dict[" {} ".format(value)] = " [train_time] "  # replace to arrive or leave after delexicalization during creating data
                    elif slot == "type" and domain in ["hotel", "attraction"]:
                        delex_dict[" {} ".format(value)] = " [{}_type] ".format(domain)
                        delex_dict[" {}s ".format(value)] = " [{}_type] ".format(domain)
                        delex_dict[" {}es ".format(value)] = " [{}_type] ".format(domain)
                    elif slot == "department":
                        delex_dict[" {} ".format(value)] = " [hospital_department] "
            delex_dict["cb11jg"] = "[police_postcode]"
            delex_dict["cb20qq"] = "[hospital_postcode]"
            delex_dict["01223245151"] = "[hospital_general_phone]"

    reference_nums = get_references(data_path)
    for value in reference_nums:
        domain, num = value.split("-")
        delex_dict[" {} ".format(num)] = " [{}_reference] ".format(domain)

    return delex_dict, train_time_dict

if __name__ == "__main__":
    delex_dict = get_delex_dict("data/MultiWOZ_2.1")