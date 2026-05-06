start = int(input("Enter the start of the range: "))
end = int(input("Enter the end of the range: "))
even_numbers = [num for num in range(start, end + 1) if num % 2 == 0]
print(f"The even numbers in the range from {start} to {end} are: {even_numbers}")
"""""
start = int(input("Enter the starting number of the range: "))
end = int(input("Enter the ending number of the range: "))
divisor = int(input("Enter the divisor: "))

sum_divisible = 0
for num in range(start, end+1):
    if num % divisor == 0:  # যদি সংখ্যা divisor দিয়ে ভাগ করা যায়
        sum_divisible += num

print(f"The sum of numbers divisible by {divisor} in the range {start} to {end} is: {sum_divisible}")
"""