import numpy as np


data_1 = np.random.randint(0, 8, 8)
data_2 = np.random.randint(0, 8, 8)
data_check = np.random.randint(0, 8, 8)

print(data_1)
print()
print(data_2)

print(data_1 > data_2)
print()
print(data_check)
check_index = data_1 > data_2
print(check_index)
data_check[check_index] = data_2[check_index]
print(data_check)
# print(np.minimum([1, 59, 3, 2], [2, 4, 66, 1]))

# print(min_arr)

# print(5 in data)
# print(np.linalg.norm(np.array([1, 2, 2]) - np.array([0, 0, 0])))
# print(data - [1, 2, 3, 4])
# print(data[[1, 4]])
# print(data.min(axis=0))