import json
import numpy as np
def read_data_name_from_json(filepath = "src/data.json"):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    result_list = [f"{item['studytype']}-{item['names']}-{item['date']}" for item in data['data']]
    return result_list
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


# print(read_data_name_from_json("src/data.json"))
# print([x.split('-')[0] for x in ['study1-7','study1-8']])


# positive_label = [user_names[0],user_names[1],user_names[9]]
authentications_per_person = 3
# labels = np.array([1 if user in positive_label else 0 for user in user_names for _ in range(authentications_per_person)])
user_names = ['1','2','3','4','5','6','7','8','7','9']
num_people=len(user_names)
print(np.repeat(map_names_to_numbers(user_names), authentications_per_person))
# labels = np.repeat(np.arange(num_people), authentications_per_person)
# print(labels)
# print(np.arange(num_people))
