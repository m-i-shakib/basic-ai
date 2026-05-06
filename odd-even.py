def check_odd_even(num):
    if num % 2 == 0:
        return "Even"
    else:
        return "Odd"
number = int(input("Enter a number: "))
result = check_odd_even(number)
print(f"The number {number} is {result}.")