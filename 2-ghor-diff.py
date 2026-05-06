n = int (input("ENter n:"))
total = 0;
i = 1;
while i<=n:
    total += i
    i+=3
print(total)
"""""
def sum_series(n):
    total = 0
    i = 1
    while i<=n:
        total += i
        i+=3
    return total
n = sum_series(7)
print(n)
"""""