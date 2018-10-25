import time
import math
import collections
import statistics
import sympy
import scipy
from scipy import stats
import itertools
from sys import getsizeof
import cipher_texts

start_time = time.time()

# -----------------------
# -----------------------
# ---Utility functions---
# -----------------------
# -----------------------


def match(original, formatted):
    formatted = list(formatted)
    for index, value in enumerate(formatted):
        if not original[index].isalpha() and formatted[index].isalpha():
            formatted.insert(index, original[index])
        elif original[index].isupper() and formatted[index].isalpha():
            formatted[index] = formatted[index].upper()
    if not original[-1].isalpha():  # Above loop does miss last character
        formatted.append(original[-1])
    return "".join(formatted)


def letters(string):
    return "".join(character for character in string if character.isalpha())

# -----------------------
# -----------------------
# --Frequency analysis---
# -----------------------
# -----------------------


ENGLISH_LANG_LEN = 26
ENGLISH_LOWER_CODEX = 0.06

english_chars = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# Source: Wikipedia
english_1gram_expected_dict = {
    'e': 12.49, 't': 9.28, 'a': 8.04, 'o': 7.64,
    'i': 7.57, 'n': 7.23, 's': 6.51, 'r': 6.28,
    'h': 5.05, 'l': 4.07, 'd': 3.82, 'c': 3.34,
    'u': 2.73, 'm': 2.51, 'f': 2.40, 'p': 2.14,
    'g': 1.87, 'w': 1.68, 'y': 1.66, 'b': 1.48,
    'v': 1.05, 'k': 0.54, 'x': 0.23, 'j': 0.16,
    'q': 0.12, 'z': 0.09
}

CharFreq = collections.namedtuple(
    'CharacterFrequency', ['character', 'frequency'])


def auto_freq_analyser(text):
    local_alphabet_freq = collections.defaultdict(int)
    text = letters(text).lower()
    for character in text:
        local_alphabet_freq[character] += 1
    freq_table = [
        CharFreq(character=key, frequency=value*100/len(text))
        for key, value in local_alphabet_freq.items()
    ]
    return sorted(
        list(freq_table), key=lambda elem: elem.frequency, reverse=True
    )


def english_1gram_chi(text):
    counts = {char: 0 for char in english_chars}
    text = letters(text).lower()
    for char in text:
        counts[char] += 1
    observed = [
        count[1] for count in sorted(counts.items())
    ]
    expected = [
        math.ceil(english_1gram_expected_dict[char] * (
            len(text) / 100)
        ) for char in english_chars
    ]
    return sum((o - e)**2 / e for o, e in zip(observed, expected))


def codex(text):
    text = letters(text).lower()
    length = len(text)
    return sum(
        count * (count - 1) for count in collections.Counter(text).values())/(
        length * (length - 1))


english_4gram_expected_dict = dict()
"""
with open("english_quadgrams.txt") as f:
    total = 0
    for line in f:
        line = line.split(" ")
        english_4gram_expected_dict[line[0]] = int(line[1])
        total += int(line[1])
    for key, count in english_4gram_expected_dict.items():
        english_4gram_expected_dict[key] = math.log10(count/total)
"""


def english_quadgram_fitness(text):
    if not english_4gram_expected_dict:
        with open("english_quadgrams.txt") as f:
            total = 0
            for line in f:
                line = line.split(" ")
                english_4gram_expected_dict[line[0]] = int(line[1])
                total += int(line[1])
            for key, count in english_4gram_expected_dict.items():
                english_4gram_expected_dict[key] = math.log10(count/total)
    fitness = 0
    text = letters(text).upper()
    for index in range(len(text) - 3):
        quadgram = text[index:index + 4]
        if quadgram in english_4gram_expected_dict:
            fitness += english_4gram_expected_dict[quadgram]
        else:
            fitness -= 10
    return -1*fitness

# -----------------------
# -----------------------
# ------Decryption-------
# -----------------------
# -----------------------


class Caesar:
    def __init__(self, text, shift=0, forced=False):
        self.text = text
        self.shift = shift
        self.auto = not (bool(self.shift) or forced)

    @staticmethod
    def char_shift(char, shift):
        return english_chars[
            (
                shift + english_chars.index(char.lower())
            ) % ENGLISH_LANG_LEN
        ]

    def encipher(self):
        if self.auto:
            modal_char = auto_freq_analyser(self.text)[0].character
            self.shift = (
                english_chars.index("e") - english_chars.index(modal_char)
            ) % ENGLISH_LANG_LEN
        enciphered = "".join(
            self.char_shift(char, self.shift) if char.isalpha()
            else char for char in self.text
        )
        return match(self.text, enciphered)


class Affine:

    Key = collections.namedtuple('AffineKey', ['a', 'b'])
    TextChi = collections.namedtuple('TextChi', ['text', 'chi'])

    def __init__(self, text, switch=(1, 0)):
        self.text = text
        self.switch = Affine.Key(*switch)
        self.auto = bool(sum(switch) < 2)

    def modal_pairs(self):
        freq_chars = [
            freq.character for freq in auto_freq_analyser(self.text)[0:5]
        ]
        return list(itertools.combinations(freq_chars, 2))

    def key_generator(self):
        """
        a*c1 + b = p1
        a*c2 + b = p2
        a*(c1 - c2) = p1 - p2
        a = (p1 - p2)(c1 - c2)^-1
        b = p1 - a*c1
        """
        possible_keys = list()
        for pair in self.modal_pairs():
            try:
                cipher1 = english_chars.index(pair[0])
                plain1 = english_chars.index("e")
                cipher2 = english_chars.index(pair[1])
                plain2 = english_chars.index("t")
                a = (plain1 - plain2)*sympy.mod_inverse(
                    (cipher1 - cipher2), ENGLISH_LANG_LEN) % ENGLISH_LANG_LEN
                b = (plain1 - a*cipher1) % ENGLISH_LANG_LEN
            except ValueError:
                continue
            else:
                possible_keys.append(Affine.Key(a, b))
        return possible_keys

    @staticmethod
    def char_shift(char, key):
        return english_chars[
            (english_chars.index(char.lower())*key.a + key.b)
            % ENGLISH_LANG_LEN
        ]

    def encipher(self, key=None):
        if key is None:
            key = self.switch
        enciphered = "".join(
            self.char_shift(char, key) if char.isalpha()
            else char for char in self.text)
        return match(self.text, enciphered)

    def auto_decipher(self):
        possible_texts = []
        for key in self.key_generator():
            deciphered = self.encipher(key)
            possible_texts.append(Affine.TextChi(
                deciphered, english_1gram_chi(deciphered))
            )
        return sorted(
            possible_texts, key=lambda elem: elem[1])


class Viginere:

    ChiShift = collections.namedtuple("ChiShift", ['chi', 'shift'])

    def __init__(self, text, key="", beaufort=False):
        self.text = text
        self.key = key
        self.auto = not bool(key)
        self.beaufort = beaufort

    @staticmethod
    def beaufort_char_shift(char, shift):
        return english_chars[
            (
                shift - english_chars.index(char.lower())
            ) % ENGLISH_LANG_LEN
        ]

    @property
    def prob_key_length(self):
        text = letters(self.text).lower()
        for possible_length in range(1, 1000):
            split_text = [
                "".join(text[offset::possible_length])
                for offset in range(possible_length)
            ]
            average_codex = statistics.mean(
                codex(split) for split in split_text)
            if average_codex > ENGLISH_LOWER_CODEX:
                return possible_length
        else:
            return 1

    @property
    def prob_key(self):
        text = letters(self.text).lower()
        split_text = [
            "".join(text[offset::self.prob_key_length])
            for offset in range(self.prob_key_length)
        ]
        shifts = []
        for split in split_text:
            split_shifts = list()
            for possible_shift in range(26):
                shifted_text = Caesar(
                    split, shift=possible_shift, forced=True).encipher()
                split_shifts.append(
                    Viginere.ChiShift(english_1gram_chi(shifted_text), possible_shift))
            split_shift = sorted(split_shifts)[0].shift
            shifts.append(-1*split_shift % ENGLISH_LANG_LEN)
        return "".join(english_chars[shift] for shift in shifts)

    def encipher(self):
        if self.auto:
            self.key = self.prob_key
        text = letters(self.text).lower()
        split_text = [
            "".join(text[offset::self.prob_key_length])
            for offset in range(self.prob_key_length)
        ]
        shifted_split = list()
        for index, split in enumerate(split_text):
            if self.beaufort:
                split = "".join(self.beaufort_char_shift(
                    char, english_chars.index(self.key[index])) for char in list(split))
            else:
                split = Caesar(
                    split,
                    shift=ENGLISH_LANG_LEN-english_chars.index(self.key[index]),
                    # Above added since vigenere keys are the complement, usually
                ).encipher()
            shifted_split.append(split)
        enciphered = "".join(
            "".join(chunk)
            for chunk in itertools.zip_longest(*shifted_split, fillvalue=" ")
        )
        return match(self.text, enciphered.rstrip())


class AffineViginere:
    def __init__(self, text, key="", switch=(1, 0)):
        self.text = text
        self.key = key
        self.switch = Affine.Key(*switch)
        self.auto = bool(sum(switch) < 2)

    def encipher(self):
        if self.auto:
            possible_switches = [
                (switch, 0) for switch in range(26)
                if math.gcd(switch, 26) == 1
            ]
            aff_texts = [Affine(self.text, switch=possible_switch).encipher()
                         for possible_switch in possible_switches]
            vig_texts = [Viginere(aff_text).encipher()
                         for aff_text in aff_texts]
            possible_texts = [Affine.TextChi(
                text=vig_text, chi=english_1gram_chi(vig_text)) for vig_text in vig_texts]
            enciphered = sorted(
                possible_texts, key=lambda text_chi: text_chi.chi)[0].text
        else:
            aff_text = Affine(self.text, switch=self.switch).encipher()
            enciphered = Viginere(aff_text, key=self.key).encipher()
        return match(self.text, enciphered)


class Scytale:
    TextFit = collections.namedtuple("TextFitness", ['text', 'fitness'])

    def __init__(self, text, key=1, auto=True):
        self.text = text
        self.key = key
        self.auto = auto

    def encipher(self):
        text = letters(self.text).lower()
        if self.auto:
            possible_texts = []
            for length in range(1, 10):
                possible_text = "".join(
                    text[i::len(text)//length] for i in range(len(text)//length))
                possible_texts.append(
                    Scytale.TextFit(
                        text=possible_text,
                        fitness=english_quadgram_fitness(possible_text)
                    )
                )
            enciphered = sorted(
                possible_texts, key=lambda text_fit: text_fit.fitness)[0].text
        else:
            enciphered = "".join(text[i::len(text)//self.key]
                                 for i in range(len(text)//self.key))
        return enciphered


if __name__ == "__main__":
    text_1A = Caesar(cipher_texts.Challenge2018.encrypted_text_1A)
    print(text_1A.encipher())
    pass
