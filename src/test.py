import json
import itertools

import numpy as np

def read_data_name_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    train_set=[]
    train_set_positive_label = []
    train_set_negative_label = []
    test_set = []

    # 处理 train_set
    if data["scene"] and data["train_set_scene"]:
            raise Exception("scene and train_set_scene can't be both set")
    for item in data['train_set']['positive_label']:
        names = [item['names']] if isinstance(item['names'], str) else item['names']
        dates = [item['date']] if isinstance(item['date'], str) else item['date']
        range_ = item['range'] if item['range'] else 'all'
        namess=[]
        if data["scene"]:
            for name in names:
                namess.append(data["scene"]+name)
        elif data["train_set_scene"]:
            for name in names:
                namess.append(data["train_set_scene"]+name)
        else:
            namess=names
        for name, date in itertools.product(namess, dates):
            train_set.append([name, date, range_])
            train_set_positive_label.append([name, date, range_])
    
    for item in data['train_set']['negative_label']:
        names = [item['names']] if isinstance(item['names'], str) else item['names']
        dates = [item['date']] if isinstance(item['date'], str) else item['date']
        range_ = item['range'] if item['range'] else 'all'
        namess=[]
        if data["scene"]:
            for name in names:
                namess.append(data["scene"]+name)
        elif data["train_set_scene"]:
            for name in names:
                namess.append(data["train_set_scene"]+name)
        else:
            namess=names
        for name, date in itertools.product(namess, dates):
            train_set.append([name, date, range_])
            train_set_negative_label.append([name, date, range_])

    # 处理 test_set
    if data["scene"] and data["test_set_scene"]:
            raise Exception("scene and test_set_scene can't be both set")
    for item in data['test_set']:
        names = [item['names']] if isinstance(item['names'], str) else item['names']
        dates = [item['date']] if isinstance(item['date'], str) else item['date']
        range_ = item['range'] if item['range'] else 'all'
        namess=[]
        if data["scene"]:
            for name in names:
                namess.append(data["scene"]+name)
        elif data["test_set_scene"]:
            for name in names:
                namess.append(data["test_set_scene"]+name)
        else:
            namess=names
        for name, date in itertools.product(namess, dates):
            test_set.append([name, date, range_])

    return train_set, train_set_positive_label, train_set_negative_label, test_set

# 处理并输出数据
# train_set, train_set_positive, train_set_negative, test_set= read_data_name_from_json("src/data4ml.json") # return 2 list consists of [name, date, range]

# positive_label = [x[0] for x in train_set_positive]
# print(positive_label)

def range_to_int_value(range_str):
        def range_to_int_start_end(range_str, value='start'):
            values = list(map(int, range_str.split('-')))
            return values[0] if value == 'start' else values[1]
        return range_to_int_start_end(range_str, 'end')-range_to_int_start_end(range_str, 'start')

# print(range_to_int_value('2-6'))

# labels = np.array([1 if user in [x[0] for x in train_set_positive] else 0 for user in [x[0] for x in train_set_positive] for _ in range(range_to_int_value(x[2]))]  for x in train_set_positive)
# 首先获取所有独特的用户
# unique_users = set([x[0] for x in train_set_positive])

# # 为每个用户创建标签
# labels = np.array([1 if user in unique_users else 0 
#                    for user, _, range_ in train_set
#                    for _ in range(range_to_int_value(range_))])
# # print(labels)xs


def google_sheet_to_json(studytype = "study1", credential_path = "src/credentials.json", google_sheet_name = "被试招募", json_save_path= "src/data.json"):
    import gspread
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

    client = gspread.service_account(filename=credential_path)
    spreadsheet = client.open(google_sheet_name)
    sheet = spreadsheet.sheet1
    # Fetch the first column values
    first_column = sheet.col_values(1)
    # for participant in range(count_study):
    first_occurrence = None
    last_occurrence = None

    for i, value in enumerate(first_column, start=1):  # start=1 to start counting from row 1
        if value == studytype:
            last_occurrence = i
            if first_occurrence is None:
                first_occurrence = i

    data_range = f"D{first_occurrence}:D{last_occurrence}"
    column_data = sheet.range(data_range)
    names = [cell.value for cell in column_data if cell.value.strip()]
    numbered_list = map_names_to_numbers(names)

    data_list = []

    for i in range(last_occurrence-first_occurrence+1):  # Adjust the range as needed
        # Generate or collect your data
        data_item = {"studytype" : studytype, "names": numbered_list[i], "date": sheet.col_values(2)[i+first_occurrence-1]}
        # Append the data item to the list
        data_list.append(data_item)

    # Wrap the list in a dictionary under the key 'data'
    data_to_write = {"data": data_list}

    # Write the data to a JSON file
    with open(json_save_path, 'w', encoding='utf-8') as file:
        json.dump(data_to_write, file, ensure_ascii=False, indent=4)


google_sheet_to_json()
