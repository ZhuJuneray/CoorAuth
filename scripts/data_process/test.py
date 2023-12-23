def longest_subsequence(main_str, sub_str):
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

# 示例
main_str = "0123456789"
sub_str = "0123546798"
result = longest_subsequence(main_str, sub_str)
print(result)
