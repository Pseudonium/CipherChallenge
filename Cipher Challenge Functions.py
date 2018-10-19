import time
import math
import collections
import sympy

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


def caesar_char_shift(char, shift):
    return english_chars[
        (english_chars.index(char.lower()) + shift) % ENGLISH_LANG_LEN
    ]


def caesar_crypt(text, shift):
    result_text = ''
    for character in text:
        if character.isalpha():
            result_text += caesar_char_shift(character, shift)
        else:
            result_text += character
    return result_text


def auto_caesar_crypt(text):
    modal_char = auto_freq_analyser(text)[0].character
    shift = (
        english_chars.index("e") - english_chars.index(modal_char)
    ) % ENGLISH_LANG_LEN
    return caesar_crypt(text, shift)


encrypted_text_1A = """
HVMTVH,
DO DN BMZVO OJ CZVM AMJH TJP. RZ YDY KDXF PK NJHZ XCVOOZM V XJPKGZ JA HJIOCN VBJ VIY E RVN HZIODJIZY OCZMZ OJJ, NJ RZ VGMZVYT CVQZ V ADGZ JI CZM. CZM IVHZ DN EJYDZ VIY NCZ RJMFN VN GDVDNJI WZORZZI OCZ WMDODNC GDWMVMT VIY OCZ WMDODNC HPNZPH, MZNZVMXCDIB GDIFN WZORZZI VMOZAVXON VIY DHKZMDVG MJHVI OZSON, NJ OCVO ODZN DI RDOC OCZ DIOZGGDBZIXZ TJP CVQZ WZZI MZXZDQDIB. IJOCDIB NPBBZNON OCVO NCZ CVN WZZI DIQJGQZY DI VITOCDIB NCVYT VIY NCZ CVN CZGKZY RDOC NZQZMVG DINPMVIXZ AMVPY XVNZN. NCZ CVN VI DIOZMZNODIB WVXFBMJPIY. NCZ YDY V KCY JI CPHVI HDBMVODJI NOPYDZN, HVDIGT HVOCZHVODXVG HJYZGGDIB, OCZI HJQZY JI OJ NOPYT FIJRGZYBZ HDBMVODJI RCDXC BJO CZM DIOJ OCZ WDWGDJKCDGZ XDMXPDO. VAOZM BMVYPVODIB NCZ NKZIO NJHZ ODHZ RDOC JIZ JA OCZ GJIYJI VPXODJI CJPNZN RJMFDIB JI KMJQZIVIXZ WZAJMZ OVFDIB CZM XPMMZIO KJNDODJI RDOC OCZ GDWMVMT. OCZMZ MZVGGT DN IJOCDIB NPNKDXDJPN DI CZM WVXFBMJPIY VIY D RVN DIXGDIZY OJ RMDOZ CZM JAA VN V GZVY, WPO RCZI D BJO TJPM HZNNVBZ D YZXDYZY D RVIOZY OJ HZZO CZM. D OMDZY OJ NZO OCVO PK JIGT OJ WZ OJGY OCVO NCZ DN JPO JA XJPIOMT AJM V RCDGZ. DI XVDMJ.
D RDGG NZZ TJP OCZMZ.
CVMMT
"""

#print(caesar_crypt("ifmmp xpsme!", 25))
# print(auto_caesar_crypt(encrypted_text_1A))

print(sympy.mod_inverse(3, ENGLISH_LANG_LEN))
