from time import time as time
from math import gcd, log10, ceil
from collections import namedtuple, defaultdict, Counter, OrderedDict
from itertools import combinations, zip_longest
from sys import getsizeof
import cipher_texts

start_time = time()
# -----------------------
# -----------------------
# ---Utility functions---
# -----------------------
# -----------------------


def match(original: str, formatted: str) -> str:
    formatted = list(formatted)
    for index, value in enumerate(formatted):
        if not original[index].isalpha() and formatted[index].isalpha():
            formatted.insert(index, original[index])
        elif original[index].isupper() and formatted[index].isalpha():
            formatted[index] = formatted[index].upper()
    return "".join(formatted)


def letters(string: str, keep=[]) -> str:
    return "".join(
        character for character in string
        if character.isalpha() or character in keep)


def mod_inverse(num: int, mod: int) -> int:
    num = num % mod
    for possible_inverse in range(mod):
        if num * possible_inverse % mod == 1:
            return possible_inverse
    raise ValueError

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

CharFreq = namedtuple(
    'CharacterFrequency', ['character', 'frequency']
)
TextFit = namedtuple(
    "TextFitness", ['text', 'fitness']
)

TextKey = namedtuple("TextKey", ['text', 'key'])


def auto_freq_analyser(text: str) -> list:
    local_alphabet_freq = defaultdict(int)
    text = letters(text).lower()
    for character in text:
        local_alphabet_freq[character] += 1
    freq_table = (
        CharFreq(character=key, frequency=value*100/len(text))
        for key, value in local_alphabet_freq.items()
    )
    return sorted(
        freq_table, key=lambda elem: elem.frequency, reverse=True
    )


def english_1gram_chi(text: str) -> float:
    counts = {char: 0 for char in english_chars}
    text = letters(text).lower()
    for char in text:
        counts[char] += 1
    observed = (
        count[1] for count in sorted(counts.items())
    )
    expected = (
        ceil(english_1gram_expected_dict[char] * (
            len(text) / 100)
        ) for char in english_chars
    )
    return sum((o - e)**2 / e for o, e in zip(observed, expected))


def codex(text: str) -> float:
    text = letters(text).lower()
    length = len(text)
    return sum(
        count * (count - 1)
        for count in Counter(text).values()
    )/(
        length * (length - 1)
    )


english_4gram_expected_dict = dict()


def english_quadgram_fitness(text: str) -> float:
    if not english_4gram_expected_dict:
        with open("english_quadgrams.txt") as f:
            total = 0
            for line in f:
                line = line.split(" ")
                english_4gram_expected_dict[line[0]] = int(line[1])
                total += int(line[1])
            for key, count in english_4gram_expected_dict.items():
                english_4gram_expected_dict[key] = log10(count/total)
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

    def __init__(self, text: str, shift: int=0, forced: bool=False):
        self.text = text
        self.shift = shift
        self.auto = not (bool(self.shift) or forced)

    @staticmethod
    def char_shift(char: str, shift: int) -> str:
        return english_chars[
            (
                shift + english_chars.index(char.lower())
            ) % ENGLISH_LANG_LEN
        ]

    def encipher(self, give_key=False) -> str:
        if self.auto:
            modal_char = auto_freq_analyser(self.text)[0].character
            self.shift = (
                english_chars.index("e") - english_chars.index(modal_char)
            ) % ENGLISH_LANG_LEN
        enciphered = "".join(
            self.char_shift(char, self.shift) if char.isalpha()
            else char for char in self.text
        )
        if give_key:
            return TextKey(match(self.text, enciphered), self.shift)
        else:
            return match(self.text, enciphered)


class Affine:

    Key = namedtuple('AffineKey', ['a', 'b'])
    TextChiKey = namedtuple('TextChiKey', ['text', 'chi', 'key'])

    def __init__(self, text: str, switch: tuple=(1, 0)):
        self.text = text
        self.key = Affine.Key(*switch)
        self.auto = bool(sum(switch) < 2)

    @property
    def modal_pairs(self):
        freq_chars = (
            freq.character for freq in auto_freq_analyser(self.text)[0:5]
        )
        return combinations(freq_chars, 2)

    @property
    def prob_keys(self) -> list:
        """
        a*c1 + b = p1
        a*c2 + b = p2
        a*(c1 - c2) = p1 - p2
        a = (p1 - p2)(c1 - c2) ^ -1
        b = p1 - a*c1
        """
        possible_keys = list()
        for pair in self.modal_pairs:
            try:
                cipher1 = english_chars.index(pair[0])
                plain1 = english_chars.index("e")
                cipher2 = english_chars.index(pair[1])
                plain2 = english_chars.index("t")
                a = (plain1 - plain2)*mod_inverse(
                    num=(cipher1 - cipher2),
                    mod=ENGLISH_LANG_LEN) % ENGLISH_LANG_LEN
                b = (plain1 - a*cipher1) % ENGLISH_LANG_LEN
            except ValueError:
                continue
            else:
                possible_keys.append(Affine.Key(a, b))
        return possible_keys

    @staticmethod
    def char_shift(char: str, key) -> str:
        return english_chars[
            (english_chars.index(char.lower())*key.a + key.b)
            % ENGLISH_LANG_LEN
        ]

    def encipher(self, give_key=False) -> str:
        if self.auto:
            possible_texts = list()
            for key in self.prob_keys:
                deciphered = "".join(
                    self.char_shift(char, key) if char.isalpha()
                    else char for char in self.text
                )
                possible_texts.append(
                    Affine.TextChiKey(
                        text=deciphered,
                        chi=english_1gram_chi(deciphered),
                        key=key
                    )
                )
            best = sorted(
                possible_texts, key=lambda elem: elem.chi)[0]
            enciphered = best.text
            self.key = best.key
        else:
            enciphered = "".join(
                self.char_shift(char, self.key) if char.isalpha()
                else char for char in self.text
            )
        if give_key:
            return TextKey(match(self.text, enciphered), self.key)
        else:
            return match(self.text, enciphered)


class Viginere:

    ChiShift = namedtuple("ChiShift", ['chi', 'shift'])

    def __init__(self, text: str, key: str=""):
        self.text = text
        self.key = key
        self.auto = not bool(key)

    @property
    def prob_key_length(self) -> int:
        text = letters(self.text).lower()
        for possible_length in range(1, 20):
            split_text = list(
                "".join(text[offset::possible_length])
                for offset in range(possible_length)
            )
            average_codex = sum(
                codex(split) for split in split_text
            ) / len(split_text)
            # average_codex = sum(
            #    codex(split) for split in split_text
            # ) / len(split_text)
            if average_codex > ENGLISH_LOWER_CODEX:
                return possible_length
        else:
            return 1

    @property
    def split_text(self):
        text = letters(self.text).lower()
        return (
            "".join(text[offset::self.prob_key_length])
            for offset in range(self.prob_key_length)
        )

    @property
    def prob_key(self) -> str:
        shifts = list()
        for split in self.split_text:
            split_shifts = list()
            for possible_shift in range(26):
                shifted_text = Caesar(
                    split, shift=possible_shift, forced=True).encipher()
                split_shifts.append(
                    Viginere.ChiShift(
                        english_1gram_chi(shifted_text),
                        possible_shift
                    )
                )
            split_shift = sorted(split_shifts)[0].shift
            shifts.append(-1*split_shift % ENGLISH_LANG_LEN)
        return "".join(english_chars[shift] for shift in shifts)

    def encipher(self, give_key=False) -> str:
        if self.auto:
            self.key = self.prob_key
            split_text = self.split_text
        else:
            text = letters(self.text).lower()
            split_text = (
                "".join(text[offset::len(self.key)])
                for offset in range(len(self.key))
            )
        shifted_split = list()
        for index, split in enumerate(split_text):
            split = Caesar(
                split,
                shift=ENGLISH_LANG_LEN-english_chars.index(self.key[index]),
                # Above: Complement of key for decryption
            ).encipher()
            shifted_split.append(split)
        enciphered = "".join(
            "".join(chunk)
            for chunk in zip_longest(*shifted_split, fillvalue="")
        )
        if give_key:
            return TextKey(match(self.text, enciphered), self.key)
        else:
            return match(self.text, enciphered)


class AffineViginere:
    def __init__(self, text: str, key: str="", switch: tuple=(1, 0)):
        self.text = text
        self.key = key
        self.switch = Affine.Key(*switch)
        self.auto = bool(sum(switch) < 2)

    def encipher(self) -> str:
        if self.auto:
            possible_switches = (
                (switch, 0) for switch in range(26)
                if gcd(switch, 26) == 1
            )
            aff_texts = (Affine(self.text, switch=possible_switch).encipher()
                         for possible_switch in possible_switches)
            vig_texts = (Viginere(aff_text).encipher()
                         for aff_text in aff_texts)
            possible_texts = (
                Affine.TextChiKey(
                    text=vig_text, chi=english_1gram_chi(vig_text), key=None
                )
                for vig_text in vig_texts)
            enciphered = sorted(
                possible_texts, key=lambda text_chi: text_chi.chi)[0].text
        else:
            aff_text = Affine(self.text, switch=self.switch).encipher()
            enciphered = Viginere(aff_text, key=self.key).encipher()
        return match(self.text, enciphered)


class Scytale:
    TextFitLen = namedtuple("TextFitnessLength", ['text', 'fitness', 'length'])

    def __init__(self, text: str, key: int=1, auto: bool=True, keep=[]):
        self.text = text
        self.key = key
        self.auto = not bool(key > 1)
        self.keep = keep

    def encipher(self, give_key=False) -> str:
        text = letters(self.text, keep=self.keep).lower()
        if self.auto:
            possible_texts = list()
            for length in range(1, 10):
                """
                if self.ceil:
                    possible_text = "".join(
                        text[i::ceil(len(text)/length)]
                        for i in range(ceil(len(text)/length))
                    )
                else:
                    possible_text = "".join(
                        text[i::len(text)//length]
                        for i in range(len(text)//length)
                    )
                """
                possible_text = "".join(
                    text[i::round(len(text)/length)]
                    for i in range(round(len(text)/length))
                )
                possible_texts.append(
                    Scytale.TextFitLen(
                        text=possible_text,
                        fitness=english_quadgram_fitness(possible_text),
                        length=length
                    )
                )
            best = sorted(
                possible_texts, key=lambda text_fit: text_fit.fitness)[0]
            enciphered = best.text
            self.key = best.length
        else:
            """
            if self.ceil:
                enciphered = "".join(
                    text[i::ceil(len(text)/self.key)]
                    for i in range(ceil(len(text)/self.key))
                )
            else:
                enciphered = "".join(
                    text[i::len(text)//self.key]
                    for i in range(len(text)//self.key)
                )
            """
            enciphered = "".join(
                text[i::round(len(text)/self.key)]
                for i in range(round(len(text)/self.key))
            )
        if give_key:
            return TextKey(enciphered, self.key)
        else:
            return enciphered


class ScytaleViginere:
    def __init__(self, text: str, length: int=1, key: str="", keep=[]):
        self.text = text
        self.length = length
        self.key = key
        self.auto = not bool(key)
        self.keep = keep

    def encipher(self):
        text = letters(self.text, keep=self.keep).lower()
        if self.auto:
            for length in range(2, 10):
                possible_text = Scytale(
                    text, key=length, keep=self.keep).encipher()
                if Viginere(possible_text).prob_key_length != 1:
                    break
            enciphered = Viginere(possible_text).encipher()
        else:
            enciphered = Scytale(
                text,
                key=self.length,
                keep=self.keep
            ).encipher()
            enciphered = Viginere(enciphered, key=self.key).encipher()
        return match(self.text, enciphered)


class MonoSub:

    KeyFit = namedtuple('KeyFitness', ['key', 'fitness'])
    CharSwap = namedtuple('CharSwap', ['char', 'swap_char'])

    def __init__(self, text: str, key=None, keyword=False):
        self.text = text
        self.key = key
        self.auto = not bool(key)
        if keyword:
            self.key = self.keyword_to_key(key)

    @staticmethod
    def keyword_to_key(key):
        new_key = "".join(OrderedDict.fromkeys(letters(key).lower()))
        start = max(english_chars.index(char) for char in new_key)
        characters = english_chars[start:] + english_chars[:start]
        for char in characters:
            if char not in new_key:
                new_key += char
        final_key = {
            key_char: eng_char.upper()
            for key_char, eng_char in zip(new_key, english_chars)
        }
        return final_key

    @property
    def prob_key(self):
        analysed = auto_freq_analyser(self.text)
        for char in english_1gram_expected_dict:
            if all(char not in char_freq.character for char_freq in analysed):
                analysed.append(CharFreq(character=char, frequency=0))
        current_key = {
            observed.character: expected
            for observed, expected in zip(
                analysed,
                english_1gram_expected_dict
            )
        }
        return current_key

    @staticmethod
    def new_key(key, swap):
        new_key = list(MonoSub.CharSwap(*item) for item in key.items())
        pair = [new_key[swap[0]], new_key[swap[1]]]
        a = pair[0].swap_char
        b = pair[1].swap_char
        pair[0] = MonoSub.CharSwap(char=pair[0].char, swap_char=b)
        pair[1] = MonoSub.CharSwap(char=pair[1].char, swap_char=a)
        (new_key[swap[0]],
         new_key[swap[1]]) = pair[0], pair[1]
        new_key = dict(new_key)
        return new_key

    @property
    def best_key(self):
        current_key = self.prob_key
        for count in range(1000):
            parent_text = self.encipher(key=current_key)
            parent_fit = english_quadgram_fitness(parent_text)
            possible_keys = [MonoSub.KeyFit(current_key, parent_fit)]
            for swap in combinations(range(26), 2):
                new_key = self.new_key(current_key, swap)
                child_text = self.encipher(key=new_key)
                child_fit = english_quadgram_fitness(child_text)
                possible_keys.append(MonoSub.KeyFit(new_key, child_fit))
            best_new_key = sorted(
                possible_keys,
                key=lambda x: x.fitness
            )[0].key
            if current_key == best_new_key:
                return current_key
            else:
                current_key = best_new_key
        return current_key

    def encipher(self, key: dict={}, give_key=False):
        if not key and self.auto:
            key = self.best_key
        elif self.key:
            key = self.key
        enciphered = "".join(
            key[char] if char in key
            else char for char in self.text.lower()
        )
        if give_key:
            return TextKey(match(self.text, enciphered), key)
        else:
            return match(self.text, enciphered)


class DuoSub:
    def __init__(self, text, key_square: list=[]):
        self.text = text
        if key_square:
            self.key = self.create_substitution_dict(key_square)
        self.auto = not bool(key_square)

    @staticmethod
    def duo_to_mono(text):
        new_text = letters(text).lower()
        split_text = list(
            new_text[i:i + 2] for i in range(0, len(new_text), 2)
        )
        substitutions = {}
        eng_index = 0
        for bigram in split_text:
            if bigram not in substitutions:
                substitutions[bigram] = english_chars[eng_index]
                eng_index += 1
            if eng_index == 24:
                break
        return "".join(substitutions[bigram] for bigram in split_text)

    @staticmethod
    def create_substitution_dict(key):
        substitutions = dict()
        eng_index = 0
        for row in key[0]:
            for col in key[1]:
                substitutions[row + col] = english_chars[eng_index].upper()
                eng_index += 1
                if eng_index == english_chars.index("j"):
                    eng_index += 1
        return substitutions

    def encipher(self, give_key=False):
        if self.auto:
            new_text = self.duo_to_mono(self.text)
            enciphered = MonoSub(new_text).encipher(give_key=give_key)
            if give_key:
                self.key = enciphered.key
                enciphered = enciphered.text
        else:
            new_text = letters(self.text).lower()
            split_text = (
                new_text[i: i + 2] for i in range(0, len(new_text), 2)
            )
            enciphered = "".join(self.key[bigram] for bigram in split_text)
        if give_key:
            return TextKey(match(self.text, enciphered), self.key)
        else:
            return match(self.text, enciphered)


class Challenge2017:
    solution_1A = Caesar(
        cipher_texts.Challenge2017.encrypted_text_1A,
        shift=5
    ).encipher()
    solution_1B = Caesar(
        cipher_texts.Challenge2017.encrypted_text_1B,
        shift=17
    ).encipher()
    solution_2A = MonoSub(
        cipher_texts.Challenge2017.encrypted_text_2A,
        key="cairo",
        keyword=True
    ).encipher()
    solution_2B = Scytale(
        cipher_texts.Challenge2017.encrypted_text_2B,
        key=6
    ).encipher()
    solution_3A = Affine(
        cipher_texts.Challenge2017.encrypted_text_3A,
        switch=(15, 22)
    ).encipher()
    solution_3B = DuoSub(
        cipher_texts.Challenge2017.encrypted_text_3B,
        key_square=[('x', 'l', 'c', 'd', 'm'), ('x', 'l', 'c', 'd', 'm')]
    ).encipher()
    solution_4A = MonoSub(
        cipher_texts.Challenge2017.encrypted_text_4A,
        key="gaza frequens Libycum: duxit Karthago triumphum!",
        keyword=True
    ).encipher()
    solution_4B = Viginere(
        cipher_texts.Challenge2017.encrypted_text_4B,
        key="arcanaimperii"
    ).encipher()
    solution_5A = MonoSub(
        cipher_texts.Challenge2017.encrypted_text_5A,
        key="decoy",
        keyword=True
    ).encipher()
    solution_5B = Affine(
        cipher_texts.Challenge2017.encrypted_text_5B,
        switch=(25, 0)
    ).encipher()
    solution_5B = Viginere(
        solution_5B,
        key=Affine("arcanaimperii", switch=(25, 0)).encipher()
    ).encipher()
    solution_6A = Viginere(
        cipher_texts.Challenge2017.encrypted_text_6A,
        key="zeus"
    ).encipher()
    solution_6B = Viginere(
        cipher_texts.Challenge2017.encrypted_text_6B,
        key="agricolaemortem"
    ).encipher()
    solution_6B = Affine(
        solution_6B,
        switch=(15, 0)
    ).encipher()
    solution_7A = Viginere(
        cipher_texts.Challenge2017.encrypted_text_7A,
        key="hanginggardens"
    ).encipher()
    solution_7B = Scytale(
        cipher_texts.Challenge2017.encrypted_text_7B,
        key=6,
        keep=["_"]
    ).encipher()
    solution_7B = Viginere(
        solution_7B,
        key="scytale"
    ).encipher()
    solution_7B = ScytaleViginere(
        cipher_texts.Challenge2017.encrypted_text_7B,
        keep=["_"]
    ).encipher()
    solution_8A = Viginere(
        "".join(reversed(cipher_texts.Challenge2017.encrypted_text_8A)),
        key="nijmegen"
    ).encipher()


if __name__ == "__main__":
    text_1A = Caesar(
        cipher_texts.Challenge2018.encrypted_text_1A,
        shift=19
    )
    solution_1A = text_1A.encipher()
    text_1B = Caesar(
        cipher_texts.Challenge2018.encrypted_text_1B,
        shift=15
    )
    solution_1B = text_1B.encipher()
    text_2A = Caesar(
        cipher_texts.Challenge2018.encrypted_text_2A,
        shift=19
    )
    solution_2A = text_2A.encipher()
    text_2B = Affine(
        cipher_texts.Challenge2018.encrypted_text_2B,
        switch=(11, 4)
    )
    solution_2B = text_2B.encipher()
    text_3A = Affine(
        cipher_texts.Challenge2018.encrypted_text_3A,
        switch=(9, 25)
    )
    solution_3A = text_3A.encipher()
    text_3B = MonoSub(
        cipher_texts.Challenge2018.encrypted_text_3B,
        key="loyalot",
        keyword=True
    )
    solution_3B = text_3B.encipher()
    #print("1A: ", solution_1A)
    #print("1B: ", solution_1B)
    #print("2A: ", solution_2A)
    #print("2B: ", solution_2B)
    #print("3A: ", solution_3A)
    #print("3B: ", solution_3B)
    """
    text_2_1A = MonoSub(cipher_texts.Challenge2017.encrypted_text_1A)
    text_2_3B = MonoSub(cipher_texts.Challenge2018.encrypted_text_3B)
    print(text_2_1A.encipher())
    """
    # print(solution_3B)
    # for item in combinations(range(26), 2):
    # print(item)
    # print(MonoSub.keyword_to_key("loyalot"))
    print(Challenge2017.solution_8A)
    print("--- %s seconds ---" % (time() - start_time))
