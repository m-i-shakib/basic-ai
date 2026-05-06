def find(lst):
    lst.sort()
    small = lst[0]
    large = lst[-1]
    i = len(lst) - 2
    while i >= 0 and lst[i] == large:
        i -= 1
    if i >= 0:
        s_large = lst[i]
    else:
        s_large = None
    print(small)
    print(s_large)
    print(large)
n = int(input())
lst = []
for _ in range(n):
    lst.append(int(input()))
find(lst)
