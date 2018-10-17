import time
import math

start_time = time.time()


def letters(string):
    return "".join([character for character in string if character.isalpha()])


english_chars = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]


def caesar_crypt(text, shift):
    result_text = ''
    for character in text:
        if character.isalpha():
            result_text += caesar_char_shift(character, shift)
        else:
            result_text += character
    return result_text


def caesar_char_shift(char, shift):
    return english_chars[(english_chars.index(char.lower()) + shift) % 26]


print(caesar_crypt("ifmmp xpsme!", 25))
