import time
import math
import collections
import sympy
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

english_chars = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

LetFreq = collections.namedtuple('LetterFrequency', ['character', 'frequency'])


def auto_freq_analyser(text):
    local_alphabet_freq = collections.defaultdict(int)
    text = letters(text).lower()
    for character in text:
        local_alphabet_freq[character] += 1
    freq_table = list()
    for key, value in local_alphabet_freq.items():
        value *= 100 / len(text)
        freq_table.append(LetFreq(character=key, frequency=value))
    return sorted(
        freq_table[:], key=lambda elem: elem.frequency, reverse=True
    )
    return [tuple(elem) for elem in sorted(
        freq_table[:], key=lambda elem: elem.frequency, reverse=True
    )]

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

    AffineKey = collections.namedtuple('AffineKey', ['a', 'b'])

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
        # a*p1 + b = c1
        # a*p2 + b = c2
        # p1 = index of e, p2 = index of t
        # a*p1 - a*p2 = c1 - c2
        # a(p1 - p2) = c1 - c2
        # a = (c1 - c2)(p1 - p2)^-1
        # b = c1 - a*p1
        """
        possible_keys = list()
        for pair in self.modal_pairs():
            try:
                cipher1 = english_chars.index(pair[0])
                plain1 = english_chars.index("e")
                cipher2 = english_chars.index(pair[1])
                plain2 = english_chars.index("t")
                a = (cipher1 - cipher2)*sympy.mod_inverse(
                    (plain1 - plain2), ENGLISH_LANG_LEN) % ENGLISH_LANG_LEN
                b = (cipher1 - a*plain1) % ENGLISH_LANG_LEN
            except ValueError:
                continue
            else:
                possible_keys.append(Affine.AffineKey(a, b))
        return possible_keys


encrypted_text_1A = """
HVMTVH,
DO DN BMZVO OJ CZVM AMJH TJP. RZ YDY KDXF PK NJHZ XCVOOZM V XJPKGZ JA HJIOCN VBJ VIY E RVN HZIODJIZY OCZMZ OJJ, NJ RZ VGMZVYT CVQZ V ADGZ JI CZM. CZM IVHZ DN EJYDZ VIY NCZ RJMFN VN GDVDNJI WZORZZI OCZ WMDODNC GDWMVMT VIY OCZ WMDODNC HPNZPH, MZNZVMXCDIB GDIFN WZORZZI VMOZAVXON VIY DHKZMDVG MJHVI OZSON, NJ OCVO ODZN DI RDOC OCZ DIOZGGDBZIXZ TJP CVQZ WZZI MZXZDQDIB. IJOCDIB NPBBZNON OCVO NCZ CVN WZZI DIQJGQZY DI VITOCDIB NCVYT VIY NCZ CVN CZGKZY RDOC NZQZMVG DINPMVIXZ AMVPY XVNZN. NCZ CVN VI DIOZMZNODIB WVXFBMJPIY. NCZ YDY V KCY JI CPHVI HDBMVODJI NOPYDZN, HVDIGT HVOCZHVODXVG HJYZGGDIB, OCZI HJQZY JI OJ NOPYT FIJRGZYBZ HDBMVODJI RCDXC BJO CZM DIOJ OCZ WDWGDJKCDGZ XDMXPDO. VAOZM BMVYPVODIB NCZ NKZIO NJHZ ODHZ RDOC JIZ JA OCZ GJIYJI VPXODJI CJPNZN RJMFDIB JI KMJQZIVIXZ WZAJMZ OVFDIB CZM XPMMZIO KJNDODJI RDOC OCZ GDWMVMT. OCZMZ MZVGGT DN IJOCDIB NPNKDXDJPN DI CZM WVXFBMJPIY VIY D RVN DIXGDIZY OJ RMDOZ CZM JAA VN V GZVY, WPO RCZI D BJO TJPM HZNNVBZ D YZXDYZY D RVIOZY OJ HZZO CZM. D OMDZY OJ NZO OCVO PK JIGT OJ WZ OJGY OCVO NCZ DN JPO JA XJPIOMT AJM V RCDGZ. DI XVDMJ.
D RDGG NZZ TJP OCZMZ.
CVMMT
"""

text_1A = Affine(encrypted_text_1A)
print(text_1A.key_generator())
text_2_1A = Caesar(encrypted_text_1A, shift=5)
print(text_2_1A.encipher())
