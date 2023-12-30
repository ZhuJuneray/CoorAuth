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
import itertools

# a=[1,2,1,4,1,6,7,8,10]
# b=[1,2,3,3,2,4]
# # print(a[b])
# x = np.random.choice(np.where(np.array(a)==1)[0], size=2)
# print(x)
# print(np.array(b)[x])
# print([x for x in range(1,6)])
# print(len([x for x in itertools.product([1,2,3],[4,5,6],[7,8,9])]))
# a={key1:{key2:[] for key2 in [1,2,3]} for key1 in [1,2,3]}
# a[1][2].extend([1,2,3])
# print(a)
# a[1][2].extend([4,5,6])
# print(a)



print(np.max([-0.15553798 -0.15283738 -0.14982002]))