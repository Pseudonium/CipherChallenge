import time
import math
import collections
import statistics
import sympy
import scipy
from scipy import stats
import itertools

start_time = time.time()

# -----------------------
# -----------------------
# ---Utility functions---
# -----------------------
# -----------------------


def letters(string):
    return "".join([character for character in string if character.isalpha()])

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
    freq_table = list()
    for key, value in local_alphabet_freq.items():
        value *= 100 / len(text)
        freq_table.append(CharFreq(character=key, frequency=value))
    return sorted(
        list(freq_table), key=lambda elem: elem.frequency, reverse=True
    )
    return [tuple(elem) for elem in sorted(
        list(freq_table), key=lambda elem: elem.frequency, reverse=True
    )]


def english_1gram_chi(text):
    observed_freq = auto_freq_analyser(text)
    observed = [
        elem.frequency for elem in sorted(
            observed_freq, key=lambda elem: elem.character)
    ]
    observed = (observed + ENGLISH_LANG_LEN * [0])[:ENGLISH_LANG_LEN]
    expected = [english_1gram_expected_dict[char] for char in english_chars]
    return scipy.stats.chisquare(
        observed,
        f_exp=expected
    )


def codex(text):
    text = letters(text).lower()
    length = len(text)
    return sum(
        count * (count - 1) for count in collections.Counter(text).values())/(
        length * (length - 1))


# -----------------------
# -----------------------
# ------Decryption-------
# -----------------------
# -----------------------


class Caesar:
    def __init__(self, text, shift=0):
        self.text = text
        self.shift = shift
        self.auto = not bool(self.shift)

    @staticmethod
    def char_shift(char, shift):
        return english_chars[
            (english_chars.index(char.lower()) + shift) % ENGLISH_LANG_LEN
        ]

    def encipher(self):
        if self.auto:
            modal_char = auto_freq_analyser(self.text)[0].character
            self.shift = (
                english_chars.index("e") - english_chars.index(modal_char)
            ) % ENGLISH_LANG_LEN
        return "".join(
            self.char_shift(char, self.shift) if char.isalpha()
            else char for char in self.text
        )


class Affine:

    Key = collections.namedtuple('AffineKey', ['a', 'b'])
    TextChi = collections.namedtuple('TextChi', ['text', 'chi'])

    def __init__(self, text, switch=(1, 0)):
        self.text = text
        self.switch = switch
        self.auto = bool(switch[0] + switch[1] < 2)

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

    def encipher(self, key):
        return "".join(
            self.char_shift(char, key) if char.isalpha()
            else char for char in self.text)

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
    def __init__(self, text, key=""):
        self.text = text
        self.key = key
        self.auto = bool(key)

    @property
    def prob_key_length(self):
        text = letters(self.text).lower()
        for possible_length in range(1, 100):
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


if __name__ == "__main__":

    encrypted_text_1A = """
    HVMTVH,
    DO DN BMZVO OJ CZVM AMJH TJP. RZ YDY KDXF PK NJHZ XCVOOZM V XJPKGZ JA HJIOCN VBJ VIY E RVN HZIODJIZY OCZMZ OJJ, NJ RZ VGMZVYT CVQZ V ADGZ JI CZM. CZM IVHZ DN EJYDZ VIY NCZ RJMFN VN GDVDNJI WZORZZI OCZ WMDODNC GDWMVMT VIY OCZ WMDODNC HPNZPH, MZNZVMXCDIB GDIFN WZORZZI VMOZAVXON VIY DHKZMDVG MJHVI OZSON, NJ OCVO ODZN DI RDOC OCZ DIOZGGDBZIXZ TJP CVQZ WZZI MZXZDQDIB. IJOCDIB NPBBZNON OCVO NCZ CVN WZZI DIQJGQZY DI VITOCDIB NCVYT VIY NCZ CVN CZGKZY RDOC NZQZMVG DINPMVIXZ AMVPY XVNZN. NCZ CVN VI DIOZMZNODIB WVXFBMJPIY. NCZ YDY V KCY JI CPHVI HDBMVODJI NOPYDZN, HVDIGT HVOCZHVODXVG HJYZGGDIB, OCZI HJQZY JI OJ NOPYT FIJRGZYBZ HDBMVODJI RCDXC BJO CZM DIOJ OCZ WDWGDJKCDGZ XDMXPDO. VAOZM BMVYPVODIB NCZ NKZIO NJHZ ODHZ RDOC JIZ JA OCZ GJIYJI VPXODJI CJPNZN RJMFDIB JI KMJQZIVIXZ WZAJMZ OVFDIB CZM XPMMZIO KJNDODJI RDOC OCZ GDWMVMT. OCZMZ MZVGGT DN IJOCDIB NPNKDXDJPN DI CZM WVXFBMJPIY VIY D RVN DIXGDIZY OJ RMDOZ CZM JAA VN V GZVY, WPO RCZI D BJO TJPM HZNNVBZ D YZXDYZY D RVIOZY OJ HZZO CZM. D OMDZY OJ NZO OCVO PK JIGT OJ WZ OJGY OCVO NCZ DN JPO JA XJPIOMT AJM V RCDGZ. DI XVDMJ.
    D RDGG NZZ TJP OCZMZ.
    CVMMT
    """

    text_1A = Affine(encrypted_text_1A)
    print(text_1A.key_generator())
    solved = text_1A.encipher(Affine.Key(1, 5))
    print(codex(text_1A.text))
    print(codex(solved))
    print(collections.Counter(letters(text_1A.text)))
    text_1A_3 = Viginere(encrypted_text_1A)
