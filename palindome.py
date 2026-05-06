def is_palindrome(word):
    word = word.lower()
    return word == word[::-1]
input_word = input("Enter a word: ")

if is_palindrome(input_word):
    print(f"'{input_word}' is a palindrome.")
else:
    print(f"'{input_word}' is not a palindrome.")