import time

start_time = time.time()

def letters(string):
    return "".join([character for character in string if character.isalpha()])

print(letters("87u54oqn   sdfa8o2"))