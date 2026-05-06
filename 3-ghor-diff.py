start = int(input("Enter the starting number of the range: "))
end = int(input("Enter the ending number of the range: "))
sum_of_numbers = 0
for num in range(start, end+1, 3):
    sum_of_numbers += num
print(f"The sum of every third number from {start} to {end} is: {sum_of_numbers}")