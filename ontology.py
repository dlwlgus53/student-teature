all_domains = ["restaurant", "hotel", "attraction", "train", "taxi", "hospital", "police"]

# normalize slot names
normlize_slot_names = {
    "car type": "car",
    "entrance fee": "price",
    "leaveat": "leave",
    "arriveby": "arrive",
    "price range": "pricerange",
    "trainid": "id",
    "addr": "address",
    "fee": "price",
    "post": "postcode",
    "ref": "reference",
    "ticket": "price",
    "depart": "departure",
    "dest": "destination"
}

req_slots = {
    "taxi": ["car", "phone"],
    "train": ["duration", "leave", "price", "arrive", "id"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area"],
    "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area"],
    "hospital": ["address", "postcode", "phone"],
    "police": ["address", "postcode", "phone"]
}

all_req_slots = [
    "taxi-car",
    "taxi-phone",
    "train-duration",
    "train-leave",
    "train-price",
    "train-arrive",
    "train-id",
    "restaurant-phone",
    "restaurant-postcode",
    "restaurant-address",
    "restaurant-pricerange",
    "restaurant-food",
    "restaurant-area",
    "hotel-address",
    "hotel-postcode",
    "hotel-internet",
    "hotel-phone",
    "hotel-parking",
    "hotel-type",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-area",
    "attraction-price",
    "attraction-type",
    "attraction-address",
    "attraction-postcode",
    "attraction-phone",
    "attraction-area",
    "hospital-address",
    "hospital-postcode",
    "hospital-phone",
    "police-address",
    "police-postcode",
    "police-phone"
]

info_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "train": ["people", "leave", "destination", "day", "arrive", "departure"],
    "restaurant": ["time", "day", "people", "food", "pricerange", "name", "area"],
    "hotel": ["stay", "day", "people", "name", "area", "parking", "pricerange", "stars", "internet", "type"],
    "attraction": ["type", "name", "area"],
    "hospital": ["department"]
}

all_info_slots = [
    "taxi-leave",
    "taxi-destination",
    "taxi-departure",
    "taxi-arrive",
    "train-people",
    "train-leave",
    "train-destination",
    "train-day",
    "train-arrive",
    "train-departure",
    "restaurant-time",
    "restaurant-day",
    "restaurant-people",
    "restaurant-food",
    "restaurant-pricerange",
    "restaurant-name",
    "restaurant-area",
    "hotel-stay",
    "hotel-day",
    "hotel-people",
    "hotel-name",
    "hotel-area",
    "hotel-parking",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-internet",
    "hotel-type",
    "attraction-type",
    "attraction-name",
    "attraction-area",
    "hospital-department"
]

belief_state_mapping = {
    "taxi": {
        "leaveAt": "taxi-leave",
        "destination": "taxi-destination",
        "departure": "taxi-departure",
        "arriveBy": "taxi-arrive"
    },
    "train": {
        "leaveAt": "train-leave",
        "destination": "train-destination",
        "day": "train-day",
        "arriveBy": "train-arrive",
        "departure": "train-departure"
    },
    "restaurant": {
        "food": "restaurant-food",
        "pricerange": "restaurant-pricerange",
        "name": "restaurant-name",
        "area": "restaurant-area"
    },
    "hotel": {
        "name": "hotel-name",
        "area": "hotel-area",
        "parking": "hotel-parking",
        "pricerange": "hotel-pricerange",
        "stars": "hotel-stars",
        "internet": "hotel-internet",
        "type": "hotel-type"
    },
    "attraction": {
        "type": "attraction-type",
        "name": "attraction-name",
        "area": "attraction-area"
    },
    "hospital": {
        "department": "hospital-department"
    }
}

belief_state_mapping_reversed = {
    "taxi": {
        "taxi-leave": "leaveAt",
        "taxi-destination": "destination",
        "taxi-departure": "departure",
        "taxi-arrive": "arriveBy"
    },
    "train": {
        "train-leave": "leaveAt",
        "train-destination": "destination",
        "train-day": "day",
        "train-arrive": "arriveBy",
        "train-departure": "departure"
    },
    "restaurant": {
        "restaurant-food": "food",
        "restaurant-pricerange": "pricerange",
        "restaurant-name": "name",
        "restaurant-area": "area"
    },
    "hotel": {
        "hotel-name": "name",
        "hotel-area": "area",
        "hotel-parking": "parking",
        "hotel-pricerange": "pricerange",
        "hotel-stars": "stars",
        "hotel-internet": "internet",
        "hotel-type": "type"
    },
    "attraction": {
        "attraction-type": "type",
        "attraction-name": "name",
        "attraction-area": "area"
    },
    "hospital": {
        "hospital-department": "department"
    }
}

entry_slots = {
    "train": ["leave", "destination", "day", "arrive", "departure"],
    "restaurant": ["food", "pricerange", "name", "area"],
    "hotel": ["name", "area", "parking", "pricerange", "stars", "internet", "type"],
    "attraction": ["type", "name", "area"],
    "hospital": ["department"]
}

delex_slots = {
    "taxi": ["car", "phone"],
    "train": ["duration", "leave", "arrive", "id", "reference", "price", "choice"],
    "restaurant": ["postcode", "address", "food", "reference", "name", "phone", "pricerange", "area", "choice"],
    "hotel": ["address", "postcode", "type", "stars", "reference", "name", "phone", "pricerange", "area", "choice"],
    "attraction": ["type", "address", "postcode", "name", "phone", "price", "area", "choice"],
    "hospital": ["department", "phone", "postcode", "address", "name"],
    "police": ["name", "address", "phone", "postcode"]
}
# information not in DB
# hospital_name: "Addenbrookes Hospital"
# hospital_address: "Hills Rd, Cambridge"
# hospital_postcode: "CB20QQ"
# police_phone: "CB11JG"

dialog_acts = {
    "attraction": ["inform", "nooffer", "recommend", "request", "select"],
    "booking": ["book", "inform", "nobook", "request"],
    "hotel": ["inform", "nooffer", "recommend", "request", "select"],
    "restaurant": ["inform", "nooffer", "recommend", "request", "select"],
    "taxi": ["inform", "request"],
    "train": ["inform", "nooffer", "offerbook", "offerbooked", "request", "select"],
    "hospital": ["inform", "request"],
    "police": ["inform"],
    "general": ["bye", "greet", "reqmore", "welcome"]
}

dialogue_acts_slots = {
    "attraction-inform": ["area", "type", "choice", "postcode", "name", "phone", "address", "price", "open"],
    "attraction-nooffer": ["area", "type", "name", "choice", "address", "price"],
    "attraction-recommend": ["postcode", "name", "area", "phone", "address", "type", "choice", "price", "open"],
    "attraction-request": ["area", "type", "price", "name"],
    "attraction-select": ["type", "area", "name", "choice", "phone", "price", "address"],
    "booking-book": ["reference", "name", "people", "time", "day", "stay"],
    "booking-inform": ["stay", "name", "day", "people", "reference", "time"],
    "booking-nobook": ["reference", "stay", "day", "people", "time", "name"],
    "booking-request": ["time", "day", "people", "stay"],
    "general-bye": [],
    "general-greet": [],
    "general-reqmore": [],
    "general-welcome": [],
    "hotel-inform": ["name", "reference", "type", "choice", "address", "postcode", "area", "internet", "parking", "stars", "pricerange", "phone"],
    "hotel-nooffer": ["type", "pricerange", "area", "stars", "internet", "parking", "name", "choice"],
    "hotel-recommend": ["name", "pricerange", "area", "stars", "parking", "internet", "address", "type", "phone", "postcode", "choice"],
    "hotel-request": ["area", "pricerange", "stars", "type", "parking", "internet", "name"],
    "hotel-select": ["pricerange", "name", "choice", "type", "area", "stars", "parking", "internet", "address", "phone"],
    "restaurant-inform": ["postcode", "food", "name", "pricerange", "address", "phone", "area", "choice", "reference"],
    "restaurant-nooffer": ["food", "area", "pricerange", "choice", "name"],
    "restaurant-recommend": ["area", "pricerange", "name", "food", "address", "phone", "choice", "postcode"],
    "restaurant-request": ["area", "name", "pricerange", "food"],
    "restaurant-select": ["food", "area", "pricerange", "name", "choice", "address"],
    "taxi-inform": ["phone", "car", "departure", "destination", "leave", "arrive"],
    "taxi-request": ["destination", "leave", "departure", "arrive"],
    "train-inform": ["arrive", "id", "leave", "duration", "destination", "price", "departure", "day", "choice", "reference", "people"],
    "train-nooffer": ["departure", "destination", "leave", "day", "arrive", "id", "choice"],
    "train-offerbook": ["leave", "id", "people", "arrive", "destination", "departure", "day", "duration", "price", "choice", "reference"],
    "train-offerbooked": ["reference", "price", "people", "id", "leave", "departure", "destination", "duration", "arrive", "day", "choice"],
    "train-request": ["leave", "day", "departure", "destination", "arrive", "people"],
    "train-select": ["arrive", "leave", "id", "day", "price", "choice", "departure", "destination", "people"],
    "hospital-inform": ["address", "department", "phone", "postcode"],
    "hospital-request": ["department"],
    "police-inform": ["address", "name", "phone", "postcode"]
}

gate_idx = {
    "delete": 0,
    "update": 1,
    "dontcare": 2,
    "copy": 3
}

name_typo = {
    "the peking": "peking retaurant",
    "the gonvile hotel": "gonville hotel",
    "saint johns college": "saint john's college",
    "yipee noodle bar": "yippee noodle bar",
    "pizza hut fenditton": "pizza hut fen ditton",
    "queens college": "queens' college",
    "ABC Theatre": "abc theatre",
    "churchills college": "churchill college",
    "saint johns college": "saint john's college",
    "el shaddia guesthouse": "el shaddai",
    "college": "not mentioned",
    "cambridge": "not mentioned",
    "kings college": "king's college",
    "cafe uno": "caffe uno",
    "alpha milton guest house": "alpha-milton guest house",
    "NOT(hamilton lodge)": "hamilton lodge",
    "rosas bed and breakfast": "rosa's bed and breakfast"
}

important_actions = [
    "[hotel]-[inform]-name",
    "[hotel]-[inform]-phone",
    "[hotel]-[inform]-postcode",
    "[hotel]-[inform]-address",
    "[hotel]-[inform]-reference",
    "[restaurant]-[inform]-name",
    "[restaurant]-[inform]-phone",
    "[restaurant]-[inform]-postcode",
    "[restaurant]-[inform]-address",
    "[restaurant]-[inform]-reference",
    "[attraction]-[inform]-name",
    "[attraction]-[inform]-phone",
    "[attraction]-[inform]-postcode",
    "[attraction]-[inform]-address",
    "[train]-[inform]-id",
    "[train]-[inform]-reference",
    "[taxi]-[inform]-phone",
    "[booking]-[book]-reference",
    "[booking]-[book]-name",
    "[booking]-[inform]-name",
    "[train]-[offerbooked]-reference",
    "[train]-[offerbooked]-id",
    "[train]-[offerbook]-id"
]