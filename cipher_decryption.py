import time
import math
import collections
import itertools
import functools
import cipher_texts
import pdb
import random

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


def simulated_annealing(
    initial_key,
    fitness,
    new_key,
    initial_temp=50,
    count=10000,
    max_length=1000
):
    temp_step = initial_temp / count
    temp = initial_temp
    current_key = initial_key
    same_key = 0
    for c in range(count):
        if same_key == max_length:
            break
        print(c)
        parent_fitness = fitness(current_key)
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
    return current_key
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
    'CharacterFrequency', ['character', 'frequency']
)
TextFit = collections.namedtuple(
    "TextFitness", ['text', 'fitness']
)
TextKey = collections.namedtuple(
    "TextKey", ['text', 'key']
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
        print("Needed to make it.")
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

    def encipher(self, give_key=False) -> str:
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
        if give_key:
            return TextKey(match(self.text, enciphered), self.key)
        else:
            return match(self.text, enciphered)


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
    def new_key(key: dict, swap: tuple) -> dict:
        """Swaps two substitutions in a key."""
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
    def best_key(self) -> dict:
        current_key = self.prob_key
        for count in range(MonoSub.MAX_SEARCH):
            parent_text = self.encipher(key=current_key)
            parent_fit = english_quadgram_fitness(parent_text)
            possible_keys = [MonoSub.KeyFit(current_key, parent_fit)]
            for swap in itertools.combinations(range(ENGLISH_LANG_LEN), 2):
                new_key = self.new_key(current_key, swap)
                child_text = self.encipher(key=new_key)
                child_fit = english_quadgram_fitness(child_text)
                possible_keys.append(MonoSub.KeyFit(new_key, child_fit))
            best_new_key = sorted(
                possible_keys,
                key=lambda x: x.fitness,
                reverse=True
            )[0].key
            if current_key == best_new_key:
                return current_key
            else:
                current_key = best_new_key
        return current_key

    def encipher(self, key: dict={}, give_key=False) -> str:
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


class AutoKey:
    def __init__(self, text: str, key: str="", reset: int=None):
        self.text = text
        self.key = key
        self.auto = not bool(key)
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

    def encipher(self, give_key=False):
        if self.auto:
            raise NotImplementedError
        else:
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
                    in zip(split, self.key)
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
        return match(self.text, final_plain)


class ColTrans:

    MAX_SEARCH = 7

    TextFitPerm = collections.namedtuple(
        'TextFitnessPermutation',
        ['text', 'fitness', 'perm']
    )

    def __init__(self, text, key: tuple=()):
        self.text = text
        self.key = key
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

    def encipher(self, key: tuple=(), give_key=False):
        text = letters(self.text).lower()
        if self.auto:
            possible_texts = list()
            for key_length in range(2, ColTrans.MAX_SEARCH):
                split_text = list(
                    text[i: i + key_length]
                    for i in range(0, len(text), key_length)
                )
                for perm in itertools.permutations(range(key_length)):
                    enciphered = "".join(
                        self.permute(split, perm)
                        for split in split_text
                    )
                    possible_texts.append(
                        ColTrans.TextFitPerm(
                            text=enciphered,
                            fitness=english_quadgram_fitness(enciphered),
                            perm=perm
                        )
                    )
            best = sorted(
                possible_texts,
                key=lambda elem: elem.fitness,
                reverse=True
            )[0]
            enciphered = best.text
            self.key = best.perm
        else:
            split_text = (
                text[i: i + len(self.key)]
                for i in range(0, len(text), len(self.key))
            )
            enciphered = "".join(
                self.permute(split, self.key)
                for split in split_text
            )
        if give_key:
            return TextKey(match(self.text, enciphered), self.key)
        else:
            return match(self.text, enciphered)


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

    def encipher(self, give_key=False):
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
        if give_key:
            return TextKey(match(self.text, enciphered), self.key)
        else:
            return match(self.text, enciphered)


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


class Hill:
    EXPECT_QUAD = "THAT"
    Top = collections.namedtuple('TopRow', ['a', 'b'])
    Bottom = collections.namedtuple('BottomRow', ['c', 'd'])
    Matrix = collections.namedtuple('Matrix', ['top', 'bottom'])
    Vector = collections.namedtuple('Vector', ['top', 'bottom'])

    def __init__(self, text, key: tuple=()):
        self.text = text
        if key:
            top, bottom = key
            self.key = Hill.Matrix(
                top=Hill.Top(a=top[0], b=top[1]),
                bottom=Hill.Bottom(c=bottom[0], d=bottom[1])
            )
            self.auto = False
        else:
            self.auto = True

    @staticmethod
    def bigram_crypt(bigram, key):
        start = Hill.Vector(
            top=english_chars.index(bigram[0]),
            bottom=english_chars.index(bigram[1])
        )
        crypted = Hill.Vector(
            top=(
                key.top.a * start.top + key.top.b * start.bottom
            ) % ENGLISH_LANG_LEN,
            bottom=(
                key.bottom.c * start.top + key.bottom.d * start.bottom
            ) % ENGLISH_LANG_LEN
        )
        return english_chars[crypted.top] + english_chars[crypted.bottom]

    @staticmethod
    def find_matrix(quadgram):
        """
        Trying to solve this matrix equation:
        ((a, b),  (e   (19
        (c, d)) * f) = 7)
        ((a, b),  (g   (0
        (c, d)) * h) = 19)
        with 19, 7, 0, 19 being the indices of "THAT".
        Replace 19, 7, 0, 19 with i, j, k, l
        Then, have to solve these equations:
        ae + bf = i
        ag + bh = k
        ce + df = j
        cg + dh = l
        j being zero is the default case, so it's tricky for inverses.
        However, that is solvable, and gives the following formulas:
        (only work for "THAT", for now)
        a = i * inv(e - f*g*inv(h))
        b = (i - ae)*inv(f)
        c = l*inv(g + j*h*inv(f) - e*h*inv(f))
        d = (j - ec)*inv(f)
        """
        ell = ENGLISH_LANG_LEN
        (i, j, k, l) = tuple(
            english_chars.index(char)
            for char in Hill.EXPECT_QUAD.lower()
        )
        (e, f, g, h) = tuple(english_chars.index(char) for char in quadgram)
        a = (
            i * mod_inverse(
                e - f*g*mod_inverse(h, ell), ell
            )
        ) % ENGLISH_LANG_LEN
        b = (
            (i - a*e)*mod_inverse(f, ell)
        ) % ENGLISH_LANG_LEN
        c = (
            l * mod_inverse(
                g + j*h*mod_inverse(f, ell) - e*h*mod_inverse(f, ell), ell
            )
        ) % ENGLISH_LANG_LEN
        d = (
            (j - e*c)*mod_inverse(f, ell)
        )
        top = Hill.Top(a, b)
        bottom = Hill.Bottom(c, d)
        return Hill.Matrix(top, bottom)

    def encipher(self, give_key=False):
        text = letters(self.text).lower()
        if self.auto:
            quad_split = (
                text[i: i + 4]
                for i in range(0, len(text), 4)
            )
        split_text = (
            text[i: i + 2]
            for i in range(0, len(text), 2)
        )
        enciphered = "".join(
            self.bigram_crypt(split, self.key)
            for split in split_text
        )
        if give_key:
            return TextKey(match(self.text, enciphered), self.key)
        else:
            return match(self.text, enciphered)


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
            key="a",
            reset=12
        ).encipher())
    )


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
        key="loyalot",
        keyword=True
    ).encipher()
    solution_4A = MonoSub(
        cipher_texts.Challenge2018.encrypted_text_4A,
        key="lidar",
        keyword=True
    ).encipher()


if __name__ == "__main__":
    x = cipher_texts.Challenge2018.encrypted_text_4B
    y = MonoSub(x, key="realpoitk", keyword=True)
    print(y.encipher())
    key = {
        'r': 'A',
        'e': 'B',
        'a': 'C',
        'l': 'D',
        'p': 'E',
        'o': 'F',
        'i': 'G',
        't': 'H',
        'k': 'I',
        'u': 'J',
        'v': 'K',
        'w': 'L',
        'x': 'M',
        'y': 'N',
        'z': 'O',
        'b': 'P',
        's': 'Q',
        'd': 'R',
        'f': 'S',
        'g': 'T',
        'h': 'U',
        'j': 'V',
        'm': 'W',
        'n': 'X',
        'q': 'Y',
        'c': 'Z'
    }
    print("".join(iter(key)))
    print("--- %s seconds ---" % (time.time() - start_time))
