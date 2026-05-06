n = int(input("How many numbers will you enter? "))
sum_even = 0
for i in range(n):
    num = int(input(f"Enter number {i+1}: "))
    if num % 2 == 0:
        sum_even += num
print(f"The sum of even numbers is: {sum_even}")