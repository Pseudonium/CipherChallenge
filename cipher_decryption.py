import time
import math
import collections
import itertools
import functools
import cipher_texts
import pdb
import random
import numpy as np
start_time = time.time()
# -----------------------
# -----------------------
# ---Utility functions---
# -----------------------
# -----------------------


def match(original: str, formatted: str) -> str:
    formatted = list(letters(formatted).lower())
    original = list(original)
    for index, value in enumerate(formatted):
        if not original[index].isalpha() and formatted[index].isalpha():
            formatted.insert(index, original[index])
        elif original[index].isupper() and formatted[index].isalpha():
            formatted[index] = formatted[index].upper()
    result = "".join(formatted)
    return result


def letters(string: str, keep: list=[]) -> str:
    """Return only alphabetic letters or those in keep."""
    return "".join(
        character for character in string
        if character.isalpha() or character in keep)


def mod_inverse(num: int, mod: int) -> int:
    """Return the modular inverse of num modulo mod, if it exists."""
    num = num % mod
    for possible_inverse in range(mod):
        if num * possible_inverse % mod == 1:
            return possible_inverse
    raise ValueError


def word_reverse(text: str) -> str:
    """Reverse all the words in a text."""
    return " ".join(
        "".join(reversed(word))
        for word in text.split(" ")
    )


def pad_to_length(string: str, length: int, fillvalue=" "):
    new_string = string
    if len(new_string) >= length:
        return new_string
    else:
        new_string += fillvalue
        return pad_to_length(new_string, length, fillvalue=fillvalue)


def chunked(iterable, chunk_length):
    return (
        iterable[i: i + chunk_length]
        for i in range(0, len(iterable), chunk_length)
    )


def hill_climbing(
    initial_key,
    fitness,
    neighbors,
    count=1000
):
    current_key = initial_key
    for c in range(count):
        possible_keys = list()
        parent_fitness = fitness(current_key)
        print(parent_fitness)
        possible_keys.append(KeyFit(key=current_key, fitness=parent_fitness))
        for child_key in neighbors(current_key):
            child_fitness = fitness(child_key)
            possible_keys.append(KeyFit(key=child_key, fitness=child_fitness))
        best_key = sorted(
            possible_keys,
            key=lambda elem: elem.fitness,
            reverse=True
        )[0].key
        if current_key == best_key:
            break
        else:
            current_key = best_key
    return current_key


def simulated_annealing(
    initial_key,
    fitness,
    new_key,
    initial_temp=50,
    count=10000,
    max_length=1000,
    stale=100000,
    stale_fitness=-100000,
    threshold=-100000
):
    temp_step = initial_temp / count
    temp = initial_temp
    current_key = initial_key
    same_key = 0
    best_fitness = -100000
    for c in range(count):
        if same_key == max_length:
            break
        print(c)
        if c == stale and best_fitness < stale_fitness:
            print("Stale! Restarting.")
            return simulated_annealing(
                initial_key=best_key,
                fitness=fitness,
                new_key=new_key,
                initial_temp=initial_temp,
                count=count,
                max_length=max_length,
                stale=stale,
                stale_fitness=stale_fitness,
                threshold=threshold
            )
        parent_fitness = fitness(current_key)
        print("Fitness: ", parent_fitness)
        child_key = new_key(current_key)
        child_fitness = fitness(child_key)
        dF = child_fitness - parent_fitness
        if dF > 0:
            current_key = child_key
            same_key = 0
            print("New key!")
        elif dF < 0 and math.e ** (dF/temp) >= random.random():
            current_key = child_key
            same_key = 0
            print("New key!")
        else:
            same_key += 1
        temp -= temp_step
        if child_fitness > best_fitness:
            best_key = child_key
            best_fitness = child_fitness
    if best_fitness < threshold:
        print("Stale! Restarting.")
        return simulated_annealing(
            initial_key=best_key,
            fitness=fitness,
            new_key=new_key,
            initial_temp=initial_temp,
            count=count,
            max_length=max_length,
            stale=stale,
            stale_fitness=stale_fitness,
            threshold=threshold
        )
    return current_key
# -----------------------
# -----------------------
# --Frequency analysis---
# -----------------------
# -----------------------


ENGLISH_LANG_LEN = 26
ENGLISH_LOWER_CODEX = 0.06
ENGLISH_UPPER_CODEX = 0.071

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
    'CharacterFrequency', ['character', 'frequency']
)
TextFit = collections.namedtuple(
    "TextFitness", ['text', 'fitness']
)
TextKey = collections.namedtuple(
    "TextKey", ['text', 'key']
)
KeyFit = collections.namedtuple(
    "KeyFitness", ['key', 'fitness']
)


def auto_freq_analyser(text: str, keep: list=[]) -> list:
    """Analyse the frequency of characters in a text."""
    local_alphabet_freq = collections.defaultdict(int)
    text = letters(text, keep=keep).lower()
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
    """
        Return the chi-squared stat
        between a text and the normal 1gram distribution for english.
    """
    counts = {char: 0 for char in english_chars}
    text = letters(text).lower()
    for char in text:
        counts[char] += 1
    observed = (
        count[1] for count in sorted(counts.items())
    )
    expected = (
        math.ceil(english_1gram_expected_dict[char] * (
            len(text) / 100)
        ) for char in english_chars
    )
    return sum((o - e)**2 / e for o, e in zip(observed, expected))


def codex(text: str) -> float:
    """Return the index of coincidence of a text."""
    text = letters(text).lower()
    length = len(text)
    return sum(
        count * (count - 1)
        for count in collections.Counter(text).values()
    )/(
        length * (length - 1)
    )


english_4gram_expected_dict = dict()


@functools.lru_cache(maxsize=128)
def english_quadgram_fitness(text: str) -> float:
    """Return the fitness of a text, based on quadgram count."""
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
    return fitness

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
        """Shift a character by shift."""
        return english_chars[
            (
                shift + english_chars.index(char.lower())
            ) % ENGLISH_LANG_LEN
        ]

    def encipher(self, give_key=False) -> str:
        """Encipher the text."""
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

    Key = collections.namedtuple('AffineKey', ['a', 'b'])
    TextChiKey = collections.namedtuple('TextChiKey', ['text', 'chi', 'key'])
    MAX_SEARCH = 5

    def __init__(self, text: str, switch: tuple=(1, 0)):
        self.text = text
        self.key = Affine.Key(*switch)
        self.auto = bool(sum(switch) < 2)

    @property
    def modal_pairs(self):
        """Find the possible pairs of 'e' and 't' in the text."""
        freq_chars = (
            freq.character for freq in auto_freq_analyser(
                self.text
            )[0:Affine.MAX_SEARCH]
        )
        return itertools.combinations(freq_chars, 2)

    @property
    def prob_keys(self) -> list:
        """
        Finds the possible affine keys according to this formula:
        a*c1 + b = p1
        a*c2 + b = p2
        a*(c1 - c2) = p1 - p2
        a = (p1 - p2)(c1 - c2) ^ -1
        b = p1 - a*c1
        """
        #possible_keys = list()
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
                yield Affine.Key(a, b)
                #possible_keys.append(Affine.Key(a, b))
        # return possible_keys

    @staticmethod
    def char_shift(char: str, key) -> str:
        """Shift a char using the affine key."""
        return english_chars[
            (english_chars.index(char.lower())*key.a + key.b)
            % ENGLISH_LANG_LEN
        ]

    def encipher(self, give_key=False) -> str:
        """Encrypt the given text."""
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

    ChiShift = collections.namedtuple("ChiShift", ['chi', 'shift'])
    MAX_SEARCH = 20

    def __init__(self, text: str, key: str=""):
        self.text = text
        self.key = key
        self.auto = not bool(key)

    @property
    def prob_key_length(self) -> int:
        text = letters(self.text).lower()
        for possible_length in range(2, Viginere.MAX_SEARCH):
            split_text = list(
                "".join(text[offset::possible_length])
                for offset in range(possible_length)
            )
            average_codex = sum(
                codex(split) for split in split_text
            ) / len(split_text)
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
            for possible_shift in range(ENGLISH_LANG_LEN):
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
            for chunk in itertools.zip_longest(*shifted_split, fillvalue="")
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
                (switch, 0) for switch in range(ENGLISH_LANG_LEN)
                if math.gcd(switch, ENGLISH_LANG_LEN) == 1
            )
            for switch in possible_switches:
                possible_text = Affine(self.text, switch=switch).encipher()
                if Viginere(possible_text).prob_key_length != 1:
                    break
            enciphered = Viginere(possible_text).encipher()
        else:
            aff_text = Affine(self.text, switch=self.switch).encipher()
            enciphered = Viginere(aff_text, key=self.key).encipher()
        return match(self.text, enciphered)


class Scytale:
    TextFitLen = collections.namedtuple(
        "TextFitnessLength", ['text', 'fitness', 'length'])
    MAX_SEARCH = 10

    def __init__(self, text: str, key: int=1, auto: bool=True, keep=[]):
        self.text = text
        self.key = key
        self.auto = not bool(key > 1)
        self.keep = keep

    def encipher(self, give_key=False, pretty=False) -> str:
        text = letters(self.text, keep=self.keep).lower()
        if self.auto:
            possible_texts = list()
            for length in range(1, Scytale.MAX_SEARCH):
                skip = round(len(text) / length)
                possible_text = "".join(
                    text[i::skip]
                    for i in range(skip)
                )
                possible_texts.append(
                    Scytale.TextFitLen(
                        text=possible_text,
                        fitness=english_quadgram_fitness(possible_text),
                        length=length
                    )
                )
            best = sorted(
                possible_texts,
                key=lambda text_fit: text_fit.fitness,
                reverse=True
            )[0]
            enciphered = best.text
            self.key = best.length
        else:
            skip = round(len(text) / self.key)
            enciphered = "".join(
                text[i::skip]
                for i in range(skip)
            )
        if pretty:
            enciphered = match(self.text, enciphered)
        if give_key:
            return TextKey(enciphered, self.key)
        else:
            return enciphered


class ScytaleViginere:

    MAX_SEARCH = 10

    def __init__(self, text: str, length: int=1, key: str="", keep=[]):
        self.text = text
        self.length = length
        self.key = key
        self.auto = not bool(key)
        self.keep = keep

    def encipher(self):
        text = letters(self.text, keep=self.keep).lower()
        if self.auto:
            for length in range(2, ScytaleViginere.MAX_SEARCH):
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

    KeyFit = collections.namedtuple('KeyFitness', ['key', 'fitness'])
    CharSwap = collections.namedtuple('CharSwap', ['char', 'swap_char'])
    MAX_SEARCH = 1000

    def __init__(self, text: str, key=None, keyword=False):
        self.text = text
        self.key = key
        self.auto = not bool(key)
        if keyword:
            self.key = self.keyword_to_key(key)

    @staticmethod
    def keyword_to_key(key: str) -> dict:
        """Converts a keyword into a substituiton dict."""
        new_key = "".join(
            collections.OrderedDict.fromkeys(letters(key).lower()))
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
    def prob_key(self) -> dict:
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
    def gen_neigbors_key(key):
        for swap1, swap2 in itertools.combinations(range(ENGLISH_LANG_LEN), 2):
            new_key = list(list(item) for item in key.items())
            pair1, pair2 = new_key[swap1], new_key[swap2]
            pair1[1], pair2[1] = pair2[1], pair1[1]
            yield {char: swap_char for char, swap_char in new_key}

    @property
    def text_fitness(self):
        def key_fitness(key):
            return english_quadgram_fitness(
                self.encipher(key=key)
            )
        return key_fitness

    @property
    def best_key(self) -> dict:
        return hill_climbing(
            initial_key=self.prob_key,
            fitness=self.text_fitness,
            neighbors=MonoSub.gen_neigbors_key
        )

    def encipher(self, key: dict={}, give_key=False) -> str:
        if not key:
            if self.key:
                key = self.key
            else:
                key = self.best_key
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
                if eng_index == english_chars.index("j"):  # Skip j
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


class MultiSub:
    def __init__(self, text, size, keep=[]):
        self.text = text
        self.size = size
        self.keep = keep

    @staticmethod
    def multi_to_mono(text, size, keep=[]):
        new_text = letters(text, keep=keep)
        split_text = list(
            new_text[i:i + size] for i in range(0, len(new_text), size)
        )
        print(split_text)
        substitutions = {}
        eng_index = 0
        for ngram in split_text:
            if ngram not in substitutions:
                substitutions[ngram] = english_chars[eng_index]
                eng_index += 1
        return "".join(substitutions[ngram] for ngram in split_text)

    def encipher(self):
        best = MonoSub(
            self.multi_to_mono(self.text, self.size, self.keep)
        ).encipher(give_key=True)
        return TextKey(best.text, best.key)


class Straddle:
    TextKeyCodex = collections.namedtuple(
        'TextKeyCodex', ['text', 'key', 'codex'])

    def __init__(self, text):
        self.text = text

    def convert_to_eng(self, blanks):
        row_num = 0
        final_ls = list()
        for index, char in enumerate(self.text):
            if row_num != 0:
                row_num = 0
                continue
            int_c = int(char)
            if int_c not in blanks:
                final_ls.append(char)
            else:
                row_num = 1
                final_ls.append(char + self.text[index + 1])
        substitutions = {}
        eng_index = 0
        final_string = ""
        for item in final_ls:
            if item not in substitutions:
                substitutions[item] = english_chars[eng_index]
                eng_index += 1
            final_string += substitutions[item]
        return final_string

    def encipher(self, give_key=False):
        possible_keys = list()
        for blanks in itertools.combinations(range(10), 2):
            try:
                possible_text = y.convert_to_eng(blanks)
            except IndexError:
                continue
            else:
                if (
                    ENGLISH_UPPER_CODEX > codex(possible_text)
                    and codex(possible_text) > ENGLISH_LOWER_CODEX
                ):
                    possible_keys.append(
                        Straddle.TextKeyCodex(
                            text=possible_text,
                            key=blanks,
                            codex=codex(possible_text)
                        )
                    )
                continue
        print(possible_keys)
        best = sorted(
            possible_keys,
            key=lambda elem: elem.codex,
            reverse=True
        )[0]
        enciphered = MonoSub(
            best.text
        ).encipher()
        if give_key:
            return TextKey(enciphered, best.key)
        else:
            return enciphered


class AutoKey:
    def __init__(self, text: str, size, key: str="", reset: int=None):
        self.text = text
        self.key = key
        self.auto = not bool(key)
        self.size = size
        if reset:
            self.reset = reset

    @staticmethod
    def char_shift(cipher_char, plain_or_key_char):
        return english_chars[
            (english_chars.index(
                cipher_char
            ) - english_chars.index(
                plain_or_key_char
            )) % ENGLISH_LANG_LEN
        ]

    @property
    def text_fitness(self):
        def key_fitness(key):
            return english_quadgram_fitness(
                self.encipher(key=key)
            )
        return key_fitness

    @staticmethod
    def gen_new_key(key):
        choice = random.randrange(len(key))
        new_letter = english_chars[random.randrange(26)]
        new_key = list(key)
        new_key[choice] = new_letter
        return "".join(new_key)

    @property
    def best_key(self):
        initial = "".join(
            random.choice(english_chars)
            for i in range(self.size)
        )
        return simulated_annealing(
            initial_key=initial,
            fitness=self.text_fitness,
            new_key=AutoKey.gen_new_key,
            initial_temp=30,
            count=10000,
            max_length=1000,
            stale=5000,
            stale_fitness=-20000,
            threshold=-19000
        )

    def encipher(self, key="", give_key=False, pretty=False):
        if not key:
            if self.key:
                key = self.key
            else:
                key = self.best_key
        text = letters(self.text).lower()
        if hasattr(self, "reset"):
            split_text = list(
                text[i:i + self.reset]
                for i in range(0, len(text), self.reset)
            )
        else:
            split_text = [text]
        final_plain = ""
        for split in split_text:
            initial_plain = "".join(
                self.char_shift(cipher_char, key_char)
                for cipher_char, key_char
                in zip(split, key)
            )
            start_index = len(initial_plain)
            split = split[start_index:]
            plain_index = 0
            for char in split:
                initial_plain += self.char_shift(
                    char, initial_plain[plain_index]
                )
                plain_index += 1
            final_plain += initial_plain
        enciphered = final_plain
        if pretty:
            enciphered = match(self.text, enciphered)
        if give_key:
            return TextKey(enciphered, key)
        else:
            return enciphered


class ColTrans:

    MAX_SEARCH = 10

    TextFitPerm = collections.namedtuple(
        'TextFitnessPermutation',
        ['text', 'fitness', 'perm']
    )

    def __init__(self, text, key: tuple=(), guessed_length: int=1, keep=[]):
        self.text = text
        self.key = key
        self.guessed_length = guessed_length
        self.auto_length = guessed_length == 1
        self.keep = keep
        self.auto = not bool(key)

    @staticmethod
    def permute(block: str, key: tuple):
        if len(block) > len(key):
            return block
        elif len(block) < len(key):
            block = pad_to_length(block, len(key))
        else:
            pass
        return "".join(
            block[perm_index]
            for perm_index in key
        )

    @property
    def text_fitness(self):
        def key_fitness(key):
            return english_quadgram_fitness(self.encipher(key=key))
        return key_fitness

    @staticmethod
    def swap_two_pos(key):
        swap1, swap2 = tuple(random.choices(range(len(key)), k=2))
        new_key = list(key)
        new_key[swap1], new_key[swap2] = new_key[swap2], new_key[swap1]
        return tuple(new_key)

    @staticmethod
    def segment_slide(key):
        seg_1, seg_2 = random.sample(range(len(key)), k=2)
        if seg_1 > seg_2:
            seg_1, seg_2 = seg_2, seg_1
        segment = key[seg_1:seg_2]
        key_no_seg = key[:seg_1] + key[seg_2:]
        shift = random.randrange(len(segment))
        return key_no_seg[:shift] + segment + key_no_seg[shift:]

    @staticmethod
    def gen_new_key(key):
        choice = random.randrange(2)
        transformations = {
            0: ColTrans.swap_two_pos,
            1: ColTrans.segment_slide
        }
        return transformations[choice](key)

    @property
    def best_key(self):
        initial = tuple(
            random.sample(
                range(self.guessed_length),
                k=self.guessed_length
            )
        )
        print(initial)
        return simulated_annealing(
            initial_key=initial,
            fitness=self.text_fitness,
            new_key=ColTrans.gen_new_key,
            # stale_fitness=-10000,
            # threshold=-10000
        )
        pass

    def encipher(self, key: tuple=(), give_key=False, keep=[], pretty=False):
        text = letters(self.text, keep=self.keep)
        if not key:
            if self.key:
                key = self.key
            else:
                if self.auto_length:
                    raise NotImplementedError
                else:
                    key = self.best_key
        split_text = (
            text[i: i + len(key)]
            for i in range(0, len(text), len(key))
        )
        enciphered = "".join(
            self.permute(split, key)
            for split in split_text
        )
        if pretty:
            enciphered = match(self.text, enciphered)
        if give_key:
            return TextKey(enciphered, key)
        else:
            return enciphered


class ScyColTrans:
    Key = collections.namedtuple('Key', ['scytale', 'columnar'])
    TextScyColFit = collections.namedtuple(
        'TextScytaleColumnar',
        ['text', 'scytale', 'columnar', 'fitness']
    )
    MAX_SEARCH = 7

    def __init__(self, text, scy_key: int=1, col_key: tuple=()):
        self.text = text
        self.scy_key = scy_key
        self.col_key = col_key
        self.auto_scy = scy_key == 1
        self.auto_col = not bool(col_key)
        self.key = ScyColTrans.Key(self.scy_key, self.col_key)

    def encipher(self, give_key=False, pretty=False):
        if self.auto_scy and self.auto_col:
            possible_texts = list()
            for pos_scy_key in range(2, ScyColTrans.MAX_SEARCH):
                pos_text = Scytale(self.text, key=pos_scy_key).encipher()
                best_from_key = ColTrans(pos_text).encipher(give_key=True)
                possible_texts.append(
                    ScyColTrans.TextScyColFit(
                        text=best_from_key.text,
                        scytale=pos_scy_key,
                        columnar=best_from_key.key,
                        fitness=english_quadgram_fitness(best_from_key.text)
                    )
                )
            best = sorted(
                possible_texts,
                key=lambda elem: elem.fitness,
                reverse=True
            )[0]
            enciphered = best.text
            self.key = ScyColTrans.Key(
                scytale=best.scytale,
                columnar=best.columnar
            )
        elif self.auto_col:
            text = Scytale(self.text, key=self.scy_key).encipher()
            best = ColTrans(text).encipher(give_key=True)
            enciphered = best.text
            self.key = ScyColTrans.Key(
                scytale=self.scy_key,
                columnar=best.key
            )
        elif self.auto_scy:
            possible_texts = list()
            for pos_scy_key in range(2, ScyColTrans.MAX_SEARCH):
                pos_text = Scytale(self.text, key=pos_scy_key).encipher()
                column_text = ColTrans(pos_text, key=self.col_key).encipher()
                possible_texts.append(
                    ScyColTrans.TextScyColFit(
                        text=column_text,
                        scytale=pos_scy_key,
                        columnar=self.col_key,
                        fitness=english_quadgram_fitness(column_text)
                    )
                )
            best = sorted(
                possible_texts,
                key=lambda elem: elem.fitness,
                reverse=True
            )[0]
            enciphered = best.text
            self.key = ScyColTrans.Key(
                scytale=best.scytale,
                columnar=self.col_key
            )
        else:
            scytale = Scytale(self.text, key=self.scy_key).encipher()
            enciphered = ColTrans(scytale, key=self.col_key).encipher()
        if pretty:
            enciphered = match(self.text, enciphered)
        if give_key:
            return TextKey(enciphered, self.key)
        else:
            return enciphered


class Bifid:
    RowCol = collections.namedtuple('RowColumn', ['row', 'col'])
    ALPHABET_NO_J = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    TextFitKeyPer = collections.namedtuple(
        'TextFitnessKeyPeriod',
        ['text', 'fitness', 'key', 'period']
    )
    MAX_SEARCH = 7

    def __init__(self, text, period: int=1, key: str=""):
        self.text = text
        self.period = period
        self.key = key
        self.auto_period = period < 2
        self.auto_key = not bool(key)

    @staticmethod
    def key_to_square(key):
        split_key = list(
            key[i: i + 5]
            for i in range(0, 25, 5)
        )
        square = list(
            list(split)
            for split in split_key
        )
        return square

    @staticmethod
    def split_shift(split, key):
        low_key = key.lower()
        bifid_coords = list()
        for char in split:
            position = low_key.index(char)
            row = position // 5
            col = position % 5
            bifid_coords.extend((row, col))
        half_split = len(bifid_coords)//2
        bifid_row_coords = itertools.islice(
            bifid_coords, 0, half_split
        )
        bifid_col_coords = itertools.islice(
            bifid_coords, half_split, half_split*2
        )
        plain_coords = (
            row*5 + col
            for row, col in zip(
                bifid_row_coords,
                bifid_col_coords
            )
        )
        return "".join(
            low_key[index]
            for index in plain_coords
        )

    @property
    def text_fitness(self):
        def key_fitness(key):
            return english_quadgram_fitness(
                self.encipher(key=key)
            )
        return key_fitness

    @staticmethod
    def gen_new_key(key):
        (swap_1, swap_2) = tuple(random.choices(range(25), k=2))
        new_key = list(key)
        new_key[swap_1], new_key[swap_2] = new_key[swap_2], new_key[swap_1]
        return "".join(new_key)

    def best_key(self):
        return simulated_annealing(
            initial_key=Bifid.ALPHABET_NO_J,
            fitness=self.text_fitness,
            new_key=self.gen_new_key,
            count=10000,
            initial_temp=70
        )

    def encipher(self, key="", give_key=False, pretty=False):
        text = letters(self.text).lower()
        if not key and self.auto_key:
            # First of all, see if we also have to search for a period
            if self.auto_period:
                possible_texts = list()
                for possible_period in range(2, Bifid.MAX_SEARCH):
                    print("Testing period: ", possible_period)
                    time.sleep(3)
                    possible_text = Bifid(
                        self.text,
                        period=possible_period
                    ).encipher(give_key=True)
                    possible_texts.append(
                        Bifid.TextFitKeyPer(
                            text=possible_text.text,
                            fitness=english_quadgram_fitness(
                                possible_text.text),
                            key=possible_text.key,
                            period=possible_period
                        )
                    )
                best = sorted(
                    possible_texts,
                    key=lambda elem: elem.fitness,
                    reverse=True
                )[0]
                self.period = best.period
                self.key = best.key
            else:
                self.key = self.best_key()
        elif key:
            self.key = key
        split_text = (
            text[i: i + self.period]
            for i in range(0, len(text), self.period)
        )
        enciphered = "".join(
            self.split_shift(split, self.key)
            for split in split_text
        )
        if pretty:
            enciphered = match(self.text, enciphered)
        if give_key:
            return TextKey(enciphered, self.key)
        else:
            return enciphered


class Playfair:
    ALPHABET_NO_J = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

    def __init__(self, text: str, key: str=""):
        self.text = text
        self.key = key

    @staticmethod
    def bigram_crypt(bigram, key):
        low_key = key.lower()
        bigram = bigram.lower()
        if bigram[0] == bigram[1]:
            raise ValueError("Can't encrypt with same.")
        pos_0 = low_key.index(bigram[0])
        pos_1 = low_key.index(bigram[1])

        row_0 = pos_0 // 5
        col_0 = pos_0 % 5
        row_1 = pos_1 // 5
        col_1 = pos_1 % 5

        same_rows = row_0 == row_1
        same_cols = col_0 == col_1

        if same_rows:
            row_2 = row_0
            row_3 = row_0
            col_2 = (col_0 - 1) % 5
            col_3 = (col_1 - 1) % 5
        elif same_cols:
            row_2 = (row_0 - 1) % 5
            row_3 = (row_1 - 1) % 5
            col_2 = col_0
            col_3 = col_0
        else:
            row_2 = row_0
            row_3 = row_1
            col_2 = col_1
            col_3 = col_0
        pos_2 = row_2 * 5 + col_2
        pos_3 = row_3 * 5 + col_3
        return key[pos_2] + key[pos_3]

    @staticmethod
    def normalise(text):
        final_text = ""
        for index, char in enumerate(text):
            if not (
                char.lower() == "x" and text[index - 1] == text[index + 1]
            ):
                final_text += char
        return final_text

    @property
    def text_fitness(self):
        def key_fitness(key):
            return english_quadgram_fitness(
                self.encipher(key=key)
            )
        return key_fitness

    @staticmethod
    def exchange_letters(key):
        (swap_1, swap_2) = tuple(random.choices(range(25), k=2))
        new_key = list(key)
        new_key[swap_1], new_key[swap_2] = new_key[swap_2], new_key[swap_1]
        return "".join(new_key)

    @staticmethod
    def exchange_rows(key):
        (swap_1, swap_2) = tuple(random.choices(range(5), k=2))
        new_key = list(chunked(key, 5))
        new_key[swap_1], new_key[swap_2] = new_key[swap_2], new_key[swap_1]
        return "".join(new_key)

    @staticmethod
    def flip_top_bottom(key):
        new_key = list(chunked(key, 5))
        return "".join(reversed(new_key))

    @staticmethod
    def exchange_cols(key):
        (swap_1, swap_2) = tuple(random.choices(range(5), k=2))
        new_key = list(
            key[offset::5]
            for offset in range(5)
        )
        new_key[swap_1], new_key[swap_2] = new_key[swap_2], new_key[swap_1]
        combined = "".join(new_key)
        return "".join(
            combined[offset::5]
            for offset in range(5)
        )

    @staticmethod
    def flip_left_right(key):
        new_key = list(
            key[offset::5]
            for offset in range(5)
        )
        combined = "".join(reversed(new_key))
        return "".join(
            combined[offset::5]
            for offset in range(5)
        )

    @staticmethod
    def key_reverse(key):
        return "".join(reversed(key))

    @staticmethod
    def gen_new_key(key):
        choice = random.randint(0, 8)
        if choice == 8:
            choice = random.randint(1, 5)
        else:
            choice = 0
        transformations = {
            0: Playfair.exchange_letters,
            1: Playfair.exchange_rows,
            2: Playfair.exchange_cols,
            3: Playfair.flip_left_right,
            4: Playfair.flip_top_bottom,
            5: Playfair.key_reverse
        }
        return transformations[choice](key)

    @property
    def best_key(self):
        return simulated_annealing(
            initial_key="".join(random.sample(Playfair.ALPHABET_NO_J, k=25)),
            fitness=self.text_fitness,
            new_key=Playfair.gen_new_key,
            initial_temp=30,
            count=20000,
            max_length=10000,
            stale=10000,
            stale_fitness=-7600
        )

    def encipher(self, key: str="", give_key=False, pretty=False):
        if not key:
            if self.key:
                key = self.key
            else:
                key = self.best_key
        text = letters(self.text).lower()
        split_text = chunked(text, 2)
        enciphered = "".join(
            self.bigram_crypt(bigram, key)
            for bigram in split_text
        )
        if pretty:
            enciphered = Playfair.normalise(enciphered)
        if give_key:
            return TextKey(match(self.text, enciphered), key)
        else:
            return match(self.text, enciphered)


class Foursquare:
    ALPHABET_NO_J = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    alphabet = ALPHABET_NO_J.lower()

    def __init__(self, text, key1="", key2=""):
        self.text = text
        self.key1 = key1
        self.key2 = key2

    @staticmethod
    def bigram_crypt(bigram, key1, key2):
        bigram = bigram.lower()
        key1 = key1.lower()
        key2 = key2.lower()
        alphabet = Foursquare.alphabet
        a, b = bigram[0], bigram[1]

        pos_a = key1.index(a)
        pos_b = key2.index(b)

        row_a = pos_a // 5
        row_b = pos_b // 5
        col_a = pos_a % 5
        col_b = pos_b % 5

        row_c = row_a
        row_d = row_b
        col_c = col_b
        col_d = col_a

        pos_c = row_c * 5 + col_c
        pos_d = row_d * 5 + col_d
        return alphabet[pos_c] + alphabet[pos_d]

    @property
    def text_fitness(self):
        def key_fitness(key):
            return english_quadgram_fitness(
                self.encipher(
                    key1=key[0],
                    key2=key[1]
                )
            )
        return key_fitness

    @staticmethod
    def gen_new_key(key):
        key_to_change = random.randint(0, 1)
        new_key = list(key)
        new_key[key_to_change] = Playfair.gen_new_key(new_key[key_to_change])
        return new_key

    @property
    def best_key(self):
        initial = [
            "".join(random.sample(Foursquare.ALPHABET_NO_J, k=25)),
            "".join(random.sample(Foursquare.ALPHABET_NO_J, k=25))
        ]
        return simulated_annealing(
            initial_key=initial,
            fitness=self.text_fitness,
            new_key=Foursquare.gen_new_key,
            initial_temp=30,
            count=20000,
            max_length=10000,
            stale=10000,
            stale_fitness=-11000,
            threshold=-11000
        )

    def encipher(self, key1="", key2="", give_key=False, pretty=False):
        if not key1:
            if self.key1:
                key1 = self.key1
                key2 = self.key2
            else:
                best = self.best_key
                key1, key2 = best[0], best[1]
        text = letters(self.text).lower()
        split_text = chunked(text, 2)
        enciphered = "".join(
            self.bigram_crypt(bigram, key1, key2)
            for bigram in split_text
        )
        if pretty:
            enciphered = match(self.text, enciphered)
        if give_key:
            return TextKey(enciphered, [key1, key2])
        else:
            return enciphered
        pass
    pass


class Hill:
    MAX_SEARCH = 5
    ChiMat = collections.namedtuple('ChiMatrix', ['chi', 'matrix'])
    FitMat = collections.namedtuple('FitnessMatrix', ['fitness', 'matrix'])
    TextFitKey = collections.namedtuple(
        'TextFitnessKey',
        ['text', 'fitness', 'key'])

    def __init__(self, text, size: int=1, key: list=[]):
        self.text = text
        if key:
            top, bottom = key
            self.key = np.matrix(key)
        if size != 1:
            self.size = size
            self.auto = False
        else:
            self.auto = True

    @property
    def matrix_text(self):
        text = letters(self.text).lower()
        split_text = (
            text[i: i + self.size]
            for i in range(0, len(text), self.size)
        )
        return np.matrix(
            list(
                list(
                    english_chars.index(char)
                    for char in split
                )
                for split in split_text
            )
        ).transpose()

    @property
    def best_rows(self):
        text = letters(self.text).lower()
        possible_matrices = list()
        for row in itertools.product(
            range(ENGLISH_LANG_LEN),
            repeat=self.size
        ):
            # common = math.gcd(*row)
            row = list(row)
            common = functools.reduce(math.gcd, row)
            if math.gcd(common, ENGLISH_LANG_LEN) != 1:
                continue
            possible_matrix = np.matrix([*row])
            possible_text = "".join(
                english_chars[value % ENGLISH_LANG_LEN]
                for value in (
                    possible_matrix * self.matrix_text
                ).tolist()[0]
            )
            possible_matrices.append(
                Hill.ChiMat(
                    chi=english_1gram_chi(possible_text),
                    matrix=possible_matrix
                )
            )
        return list(
            best.matrix
            for best in sorted(
                possible_matrices,
                key=lambda elem: elem.chi
            )[:self.size]
        )

    @property
    def best_matrix(self):
        best_rows = (
            matrix.tolist()[0]
            for matrix in self.best_rows
        )
        possible_texts = list()
        for item in itertools.permutations(best_rows, self.size):
            possible_matrix = np.matrix(list(item))
            possible_text = self.encipher(key=possible_matrix)
            possible_texts.append(
                Hill.FitMat(
                    fitness=english_quadgram_fitness(possible_text),
                    matrix=possible_matrix
                )
            )
        return sorted(
            possible_texts,
            key=lambda elem: elem.fitness,
            reverse=True
        )[0].matrix

    def encipher(self, key=None, give_key=False):
        if key is None:
            if hasattr(self, 'key'):
                key = self.key
            elif self.auto:
                possible_texts = list()
                for possible_size in range(2, Hill.MAX_SEARCH):
                    if len(letters(self.text).lower()) % possible_size != 0:
                        continue
                    possible = Hill(
                        self.text,
                        size=possible_size
                    ).encipher(give_key=True)
                    possible_texts.append(
                        Hill.TextFitKey(
                            text=possible.text,
                            fitness=english_quadgram_fitness(possible.text),
                            key=possible.key
                        )
                    )
                key = sorted(
                    possible_texts,
                    key=lambda elem: elem.fitness,
                    reverse=True
                )[0].key
                self.size = len(key)
            else:
                key = self.best_matrix
        encoded = key * self.matrix_text
        enciphered = "".join(
            "".join(
                english_chars[i % ENGLISH_LANG_LEN]
                for i in row.tolist()[0]
            )
            for row in encoded.transpose()
        )
        if give_key:
            return TextKey(match(self.text, enciphered), key)
        else:
            return match(self.text, enciphered)


class Challenge2004:
    solution_1A = Caesar(
        cipher_texts.Challenge2004.encrypted_text_1A,
        shift=13
    ).encipher()
    solution_1B = Affine(
        cipher_texts.Challenge2004.encrypted_text_1B,
        switch=(15, 4)
    ).encipher()
    solution_2A = Caesar(
        cipher_texts.Challenge2004.encrypted_text_2A,
        shift=20
    ).encipher()
    solution_2B = Affine(
        cipher_texts.Challenge2004.encrypted_text_2B,
        switch=(25, 0)
    ).encipher()
    solution_3A = Caesar(
        cipher_texts.Challenge2004.encrypted_text_3A,
        shift=7
    ).encipher()
    solution_3B = ColTrans(
        cipher_texts.Challenge2004.encrypted_text_3B,
        key=(8, 6, 0, 3, 7, 1, 2, 5, 4),
        keep=["0", "1"]
    ).encipher()
    solution_4A = Affine(
        cipher_texts.Challenge2004.encrypted_text_4A,
        switch=(19, 7)
    ).encipher()
    solution_4B = MonoSub(
        cipher_texts.Challenge2004.encrypted_text_4B,
        key={
            'n': 'A',
            'i': 'B',
            'c': 'C',
            'o': 'D',
            'l': 'E',
            'a': 'F',
            's': 'G',
            'f': 'H',
            'm': 'I',
            'e': 'J',
            'g': 'K',
            'h': 'L',
            'j': 'M',
            'k': 'N',
            'p': 'O',
            'q': 'P',
            'r': 'Q',
            't': 'R',
            'u': 'S',
            'v': 'T',
            'w': 'U',
            'x': 'V',
            'y': 'W',
            'z': 'X',
            'b': 'Y',
            'd': 'Z'
        }
    ).encipher()
    solution_5A = MonoSub(
        cipher_texts.Challenge2004.encrypted_text_5A,
        key={
            'i': 'A',
            's': 'B',
            'a': 'C',
            'c': 'D',
            'n': 'E',
            'e': 'F',
            'w': 'G',
            't': 'H',
            'o': 'I',
            'b': 'J',
            'd': 'K',
            'f': 'L',
            'g': 'M',
            'h': 'N',
            'j': 'O',
            'k': 'P',
            'x': 'Q',
            'm': 'R',
            'p': 'S',
            'q': 'T',
            'r': 'U',
            'u': 'V',
            'v': 'W',
            'l': 'X',
            'y': 'Y',
            'z': 'Z'
        }
    ).encipher()


class Challenge2016:
    solution_1A = Caesar(
        cipher_texts.Challenge2016.encrypted_text_1A,
        shift=18
    ).encipher()
    solution_1B = match(
        cipher_texts.Challenge2016.encrypted_text_1B,
        Caesar(
            "".join(reversed(cipher_texts.Challenge2016.encrypted_text_1B)),
            shift=15
        ).encipher()
    )
    solution_2A = Caesar(
        cipher_texts.Challenge2016.encrypted_text_2A,
        shift=20
    ).encipher()
    solution_2B = Affine(
        cipher_texts.Challenge2016.encrypted_text_2B,
        switch=(21, 3)
    ).encipher()
    solution_3A = MonoSub(
        cipher_texts.Challenge2016.encrypted_text_3A,
        key="weston",
        keyword=True
    ).encipher()
    solution_3B = MonoSub(
        cipher_texts.Challenge2016.encrypted_text_3B,
        key="neural",
        keyword=True
    ).encipher()
    solution_4A = MonoSub(
        cipher_texts.Challenge2016.encrypted_text_4A,
        key="waveform",
        keyword=True
    ).encipher()
    solution_4B = ColTrans(
        "".join(reversed(cipher_texts.Challenge2016.encrypted_text_4B)),
        key=(2, 3, 1, 0, 4)
    ).encipher()
    solution_5A = MonoSub(
        cipher_texts.Challenge2016.encrypted_text_5A,
        key="charlie",
        keyword=True
    ).encipher()
    solution_5B = ScyColTrans(
        cipher_texts.Challenge2016.encrypted_text_5B,
        scy_key=5,
        col_key=(4, 0, 1, 3, 2)
    ).encipher()
    solution_6A = Viginere(
        cipher_texts.Challenge2016.encrypted_text_6A,
        key="nsa"
    ).encipher()
    solution_6B = Viginere(
        cipher_texts.Challenge2016.encrypted_text_6B,
        key="trainer"
    ).encipher()
    solution_7A = Viginere(
        "".join(reversed(cipher_texts.Challenge2016.encrypted_text_7A)),
        key="usehill"
    ).encipher()
    solution_7B = Bifid(
        cipher_texts.Challenge2016.encrypted_text_7B,
        period=4,
        key="LIGOABCDEFHKMNPQRSTUVWXYZ".lower()
    ).encipher(pretty=True)
    solution_8A = Hill(
        cipher_texts.Challenge2016.encrypted_text_8A,
        size=2,
        key=[[25, 22], [1, 23]]
    ).encipher()
    temp_1_8B = letters(
        cipher_texts.Challenge2016.encrypted_text_8B,
        keep=["0", "1", "2"]
    ).split("2")
    solution_8B = MonoSub(
        cipher_texts.Challenge2016.encrypted_trans_8B,
        key={
            'f': 'A',
            'p': 'B',
            't': 'C',
            'i': 'D',
            'h': 'E',
            'b': 'F',
            'k': 'G',
            'm': 'H',
            'a': 'I',
            'x': 'J',
            'q': 'K',
            'r': 'L',
            'o': 'M',
            'j': 'N',
            'd': 'O',
            'u': 'P',
            'w': 'Q',
            'g': 'R',
            'n': 'S',
            'l': 'T',
            'e': 'U',
            'v': 'V',
            's': 'W',
            'y': 'X',
            'c': 'Y',
            'z': 'Z'
        }
    ).encipher()


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
    solution_8B = match(
        cipher_texts.Challenge2017.encrypted_text_8B,
        letters(AutoKey(
            word_reverse(cipher_texts.Challenge2017.encrypted_text_8B),
            size=1,
            key="a",
            reset=12
        ).encipher())
    )
    solutions = [
        solution_1A,
        solution_1B,
        solution_2A,
        solution_2B,
        solution_3A,
        solution_3B,
        solution_4A,
        solution_4B,
        solution_5A,
        solution_5B,
        solution_6A,
        solution_6B,
        solution_7A,
        solution_7B,
        solution_8A,
        solution_8B
    ]


class Challenge2018:
    solution_1A = Caesar(
        cipher_texts.Challenge2018.encrypted_text_1A,
        shift=19
    ).encipher()
    solution_1B = Caesar(
        cipher_texts.Challenge2018.encrypted_text_1B,
        shift=15
    ).encipher()
    solution_2A = Caesar(
        cipher_texts.Challenge2018.encrypted_text_2A,
        shift=19
    ).encipher()
    solution_2B = Affine(
        cipher_texts.Challenge2018.encrypted_text_2B,
        switch=(11, 4)
    ).encipher()
    solution_3A = Affine(
        cipher_texts.Challenge2018.encrypted_text_3A,
        switch=(9, 25)
    ).encipher()
    solution_3B = MonoSub(
        cipher_texts.Challenge2018.encrypted_text_3B,
        key="loyalty",
        keyword=True
    ).encipher()
    solution_4A = MonoSub(
        cipher_texts.Challenge2018.encrypted_text_4A,
        key="lidar",
        keyword=True
    ).encipher()
    solution_4B = MonoSub(
        cipher_texts.Challenge2018.encrypted_text_4B,
        key="realpolitik",
        keyword=True
    ).encipher()
    solution_5A = MonoSub(
        cipher_texts.Challenge2018.encrypted_text_5A,
        key="ariadne",
        keyword=True
    ).encipher()
    solution_5B = MonoSub(
        "".join(reversed(cipher_texts.Challenge2018.encrypted_text_5B)),
        key="reichstad",
        keyword=True
    ).encipher()
    solution_6A = MonoSub(
        cipher_texts.Challenge2018.encrypted_text_6A,
        key="nautilus",
        keyword=True
    ).encipher()
    solution_6B = ColTrans(
        cipher_texts.Challenge2018.encrypted_text_6B,
        key=(1, 0, 4, 3, 2)
    ).encipher()
    solution_7A = MonoSub(
        cipher_texts.Challenge2018.encrypted_text_7A,
        key="danger",
        keyword=True
    ).encipher()
    solution_7B = ScyColTrans(
        cipher_texts.Challenge2018.encrypted_text_7B,
        scy_key=7,
        col_key=(1, 6, 4, 2, 0, 5, 3)
    ).encipher()
    solution_8A = ScyColTrans(
        cipher_texts.Challenge2018.encrypted_text_8A,
        scy_key=8,
        col_key=(2, 5, 7, 3, 4, 0, 6, 1)
    ).encipher()
    solution_8B = Viginere(
        cipher_texts.Challenge2018.encrypted_text_8B,
        key="shadow"
    ).encipher()


if __name__ == "__main__":
    x = cipher_texts.Challenge2018.encrypted_text_8B
    w = Viginere(x, key="shadow")
    print(Challenge2018.solution_8B)
    print("--- %s seconds ---" % (time.time() - start_time))
