import os
import re


def find_matches(main_string, pattern):
    matches = re.finditer(f'(?=({pattern}))', main_string)
    return [match.group(1) for match in matches]


def longest_substring_matching(main_str, sub_str):
    main_len = len(main_str)
    sub_len = len(sub_str)

    # 构建一个二维数组来保存子问题的解
    dp = [[0] * (sub_len + 1) for _ in range(main_len + 1)]

    max_length = 0  # 最长匹配子串的长度
    end_index = 0  # 最长匹配子串在主串中的结束位置

    for i in range(1, main_len + 1):
        for j in range(1, sub_len + 1):
            if main_str[i - 1] == sub_str[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                # 更新最长匹配子串的信息
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i - 1
            else:
                dp[i][j] = 0

    if max_length == 0:
        return "没有找到匹配的子串"
    else:
        return main_str[end_index - max_length + 1: end_index + 1]


def longest_subsequence_ordered(main_str, sub_str):
    m, n = len(main_str), len(sub_str)

    # dp[i][j]表示主串前i个字符和子串前j个字符的最长公共子序列长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 填充动态规划表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if main_str[i - 1] == sub_str[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 根据动态规划表构造最长子序列
    i, j = m, n
    longest_subseq = []
    while i > 0 and j > 0:
        if main_str[i - 1] == sub_str[j - 1]:
            longest_subseq.append(main_str[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(longest_subseq))


# 指定文件夹路径
folder_path = r"E:\\Desktop\\data\\VRAuth1"

# 指定正则表达式模式
pattern = r'^PINEntry_study1-\d{1,2}-\d{4}-(\d+)-(\d+)-\d+.txt$'

# 创建字典，用于存储文件内容
file_dict = {}

# 遍历文件夹中的文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 构造文件的完整路径
        file_path = os.path.join(root, file)

        # 使用正则表达式匹配文件名
        match = re.match(pattern, file)

        if match:
            # 提取匹配的信息
            third_digit = int(match.group(1))
            second_digit = int(match.group(2))

            # 构建字典键
            key = (third_digit, second_digit)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 将文件内容添加到字典中
            if key not in file_dict:
                file_dict[key] = []
            file_dict[key].append(content)

# 正确的pin
pin1 = '210367458'
pin2 = '876452103'
pin3 = '036785241'
pin4 = '678501243'

find_dict = {1: pin1, 2: pin2, 3: pin3, 4: pin4}

new_dict_pin = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

# 打印字典内容
for key, contents in file_dict.items():
    for content in contents:
        ordered_input = content.replace('-', '')
        main_s = find_dict[key[1]]
        longest_match_ordered = longest_subsequence_ordered(main_s, ordered_input)
        num = len(longest_match_ordered)
        new_dict_pin[key[0]].append(num)

for key, item in new_dict_pin.items():
    print(f"size: {key}, accuracy_ordered: {sum(item)/len(item)/9}")