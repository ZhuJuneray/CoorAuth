import json
import numpy as np
def read_data_name_from_json(filepath = "src/data.json"):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_list = [f"{item['studytype']}-{item['names']}-{item['date']}" for item in data['data']]
    latter_auth_list = [f"{item['studytype']}-{item['names']}-{item['date']}" for item in data['latter_auth']]
    return data_list, latter_auth_list
def map_names_to_numbers(names):
    name_to_number = {}
    number_list = []
    counter = 1
    for name in names:
        if name not in name_to_number:
            name_to_number[name] = counter
            counter += 1
        number_list.append(name_to_number[name])
    return number_list
import matplotlib.pyplot as plt
import numpy as np

