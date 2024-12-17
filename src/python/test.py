import numpy as np

vec = range(10)
print(vec)
split = np.array_split(vec, 3)
print(split)

for arr in split:
    number = arr[0]
    print(number)

print(np.array(vec))
