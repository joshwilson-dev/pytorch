import itertools
 
# initializing lists
test_list1 = [1, 4, 5]
test_list2 = [3, 8, 9]

# to interleave lists
res = list(itertools.chain(*zip(test_list1, test_list2)))
print(res)