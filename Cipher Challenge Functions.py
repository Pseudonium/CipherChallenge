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
        def __init__(self, text, key=1, auto=True):
            self.text = text
            self.key = key
            self.auto = auto

        def encipher(self):
            pass
        pass


if __name__ == "__main__":

    encrypted_text_1A = """
    HVMTVH,
    DO DN BMZVO OJ CZVM AMJH TJP. RZ YDY KDXF PK NJHZ XCVOOZM V XJPKGZ JA HJIOCN VBJ VIY E RVN HZIODJIZY OCZMZ OJJ, NJ RZ VGMZVYT CVQZ V ADGZ JI CZM. CZM IVHZ DN EJYDZ VIY NCZ RJMFN VN GDVDNJI WZORZZI OCZ WMDODNC GDWMVMT VIY OCZ WMDODNC HPNZPH, MZNZVMXCDIB GDIFN WZORZZI VMOZAVXON VIY DHKZMDVG MJHVI OZSON, NJ OCVO ODZN DI RDOC OCZ DIOZGGDBZIXZ TJP CVQZ WZZI MZXZDQDIB. IJOCDIB NPBBZNON OCVO NCZ CVN WZZI DIQJGQZY DI VITOCDIB NCVYT VIY NCZ CVN CZGKZY RDOC NZQZMVG DINPMVIXZ AMVPY XVNZN. NCZ CVN VI DIOZMZNODIB WVXFBMJPIY. NCZ YDY V KCY JI CPHVI HDBMVODJI NOPYDZN, HVDIGT HVOCZHVODXVG HJYZGGDIB, OCZI HJQZY JI OJ NOPYT FIJRGZYBZ HDBMVODJI RCDXC BJO CZM DIOJ OCZ WDWGDJKCDGZ XDMXPDO. VAOZM BMVYPVODIB NCZ NKZIO NJHZ ODHZ RDOC JIZ JA OCZ GJIYJI VPXODJI CJPNZN RJMFDIB JI KMJQZIVIXZ WZAJMZ OVFDIB CZM XPMMZIO KJNDODJI RDOC OCZ GDWMVMT. OCZMZ MZVGGT DN IJOCDIB NPNKDXDJPN DI CZM WVXFBMJPIY VIY D RVN DIXGDIZY OJ RMDOZ CZM JAA VN V GZVY, WPO RCZI D BJO TJPM HZNNVBZ D YZXDYZY D RVIOZY OJ HZZO CZM. D OMDZY OJ NZO OCVO PK JIGT OJ WZ OJGY OCVO NCZ DN JPO JA XJPIOMT AJM V RCDGZ. DI XVDMJ.
    D RDGG NZZ TJP OCZMZ.
    CVMMT
    """

    encrypted_text_4B = """OMGR GHM KTEIA BHV JEEOQO GSCM WF RIRVCWXP EK EITCKNT SBDTIK KIMV VO GHM ZDXZKM OW TOZE IZS XNW LETCDRS TMIII PM WRU ACPWUCXVL XRF EOASCX PRU OWVVTNBR WR QVZBINEKA OY MYEIIWZ VVUPNSQMC. E WME YVCRF ANFTV YQA AGROVNBYTRK DMSGCSVAV’E HSE, LWMZVINN, JQREDM MMGGRBR, IZS SEM WF YKS SIZEI ETBA WRU TB IAEJI KPM OIFEE TW UBTCMUEEV TUE VQL MDXMRZCL PIXTTVJ QVVVPTRD JK ILV KQPYGR FCPADP.

ZV BHV KNGEZHTRZVO YVCRF TPQ CMEBP HRF BREV XTJK BW FZIHG LWZTPP ASIIOIFHME PRU JITKNEF FZAB XYMQR SCSR AB QQSIIKUD. YIGH NAGXZBCDV CNQ RMEDPMM BHVA RRPMMIIUTG DIQVR TPQ QEIJIRZCNF BIOZ MEBW TYG CBLL TTEIBTAEFS BF KMAIUWVIR. FUEIVS DRV ACCY UKVRUUHL MIZUJ YAF MWDIECTG WFWNQEL MCH ROZITQLN PZABSKML TYG TEIJGCI DIZCLU FVDMXXYJ KITF VO GHM XTKRBCS FH TUE VUCXY.

WVE UCY GHM XJGB WN TYG NVNBT HIVUMD KQ CUAVST. E IMKOEPAVSAMCGV AYURF RRPWDIIU BPAK VHRY PMS WVMV TYG ADUQXP JCGQNX HRBM I FGIV IB MFPS TRIGEMLA, QN R EAZP ZGAIU JG A KTIOEAYPR TITLVF CNLOMRYJ. BPE EGW YEOMIYJ ZMCFINVSMP ILRB BHV RRVVIFXSEA WF R NOAG UMGGY PID VZHNUAFTH YQA MVP, TUEG ITVV VWT IGAQY NAG FRBBLV CNQ CIFD WYWCLU JAIE AQCX WWZ RVKNSOZOTQVVBS, SWT GHM VDC FN AEVKNT TPQ ASJB IQLKLN SW OASJM IT YCNQ CTAJHVL PIJ LUQGMYTRK, IVD YG LNUVOWIU IV AKVAPK CZSII BPE TQVRR WR SEISVEJU.

FVGPFXRX CXHZNL NGIUCWK EIVV WPBN EMKI FN JAIDAEIIZH, XYM TEXKOA’S PQGSZK MFWQRGS EQGI WZCSKTAGEL NN XYM AUE TIFIVS IS STQNU VHRM IE SENV JRFME. GHM FGSFXA IE VHR FZACX CQVE XCVR WIK LMKP MXYCUFTQAC EEL BHV YIYD KMAKRKCS UCSUEL AJX FN BHV UUA DWIC XYM JLFQD-FOIWTH JTWPV YIGH PUH GRDILIA IATW FWI YMIRK QF GHM XTKZWV. TYG LRGQAC VRQVEU HLNMQZV EIZWWJ WPBN BTT IEMUY RPD ZAVMVIU BW UEJOESM OPPXIKUJ VAXIVS WMD XZIJQNRR, JGI GFCTD DCKR NW TTEUEIY RIAVNAF ILV AIVRIE UOZPTW RVL WVTE HNINAI KW ZEREH GHM EISCMV AHWIYA. EAGWV EIS KQ CBMM.

IWMCM KAKQ OEGIZXWVL I RVVRRAB IXXY BPE GTIFOVQG GRTOATWS VN KTPMEA, BHV TEZAQZXRX AWLUKEES PQAH KPM LZPE NGIUCWK ZMPVCTRD KTPVXMA BP EAYELACMRV INU KCRNQMC XIQJEJOEA. WPQC XYM TIEG FVNIXAC SZWKV, EAGO IZS QRVG OW VHR NQZIL CMOIFP HND ZQPGYML SRHEGY IZS XFWS SKQCX. TPQ REMITRP YEEE AMUI, RA EEIG MNNG AU XYM JRRXE YEOUDREIQRVU, BHT BA WMJ PWRIQR, PABA GIRTQSVF TUAB UC XYM JAKVLR HM TPH CWAT KJE FEKACH RYCICC, AAD EAGWV, BPE CGGVOV’E RSGG WF KJE POLQM LRL ILJQ GBNM.

FWI UWKUDGNGS TQUX KW UE SA ATRQODPR IZE LPCYEID DR KPQS GQIAT. LUS LV SVON VHNT BTT GFLMX DKGUT PMKI WITLVP IATW QCIDG PAEFS NNL FWEK I AETQNQ AYGXPR PID SGEA LWEI? SI LQD TCTB AAEJVV PQM KJAG TPQ RSUMF HRF BREV PTWKZWYVF AF PIDI SW IV OIFEELG DTXIMIT? GGRUAXE REKW EAJ AEG TW DT-NFQV AXTIPOTM PX KPM FFTT, NNL ODQDCVITCTVOVE UVFU BHV DAGTTQ WEU GMT KQ RRAKT WMD. QV A CGTGEZ FD IDXMRFT DBMQFXEE, IORZEOYA EDDXV “UWSK GXPETXTRK MUPVTOE, AA KDY YIDE DQSG GZMRMFCALP TEDUMEIIU Q PAMG FVNQEWIU UG AWHAVRA TTVV QV BIKTNNVUP ME XZEGCRNTQAC JFZ ZEKWRAIVS IS PWCR JKDR IV DDQV. QB PCGAFEA YT XF ZMPFTT GO GAJ XYIB BP CCGS WR SMGTWMREY NNL EIVVVOTY K HNVM BTVJCIDVF TUE VAGXYMZN IGBRLA FD CZMTD KQ OHR IDBC RVL TF VUEN WHTV KW CS KJE EEJQA GRTOATWS JHW TPW IMUAZPEQ FZQT WZVKE KJE JAZE PKRQVSK DOHDQORE RVL TYG IPEVU.” WMJ TMTKGR NPXQPVJ BW HRXE OEMZ P VVXTY KQ NRW WDSIIA NRFO RBMM UC AYQKH UQMVTQMC AIWBE “DA EFTMQBIU OVAVWS, HPWZ GITMQPK QF LOCD BSJB ZETGNG LMFIII Q PAMG BREV YDWK BZOLDLRD. QF PTGMIRJ ELRAZ FD QV BPAK VHR SQFJEKQWN ZP BEIBMCRZI PAJ YOESMZTH RVL SF K MHSB MHO PWC TF TEGUZZ IS DM IT IQMR WQFW XYM CTDQSG SXQTH. TWVDLET LOCD GIDIQNZPG NFNMXVJ QV TYG PEODUCGV IVD UQ NBT CZSIIBIKV CNL AKFXSE EPITJ MNY MZUPRUM TYG CHRZQCX KMVSZQNF BMFLIVV GOLTSRLN MCH KPM CRNEQOVUX.” XYM NIECL YIVQ DJ KPM EDREEOZ’E AIKBMR JGT N DMMSPZVM FFT ATRQODPR’A ZEKWRA, “I ATPPC MFPVET LOC FD LRDM CIQSFEL FWI ICJITQN OY BTT IEL WF KJE LEID”, PRU XMRYCPF HM RTPK BPAK EOATZUIMFV EAJ C SNFMD GIWCOE.
NJAGEDQG EXZQCFNA XNMI PX KPIT GQIAT, EUILZV I MFPTU HQE LSITL HRF BREV FJVEML UGUIQE LALR, RA NUIVHRR ZQESIBA FIQM PATQSSEQI MRFE PLMMG XYIB HZU HBPM AU VVABOICTVOV UC XYM MYVU OS TPQ TQGMZOI YEEE QZ GIRT REFRAEDG. TT WVB WUK VO EEKAKII BPE KYO NQCUAEV IVD KQ DRTMDBMEM EHRV HND PMETVVMD KQ TUE KASIO JMFFTE EEBGGRZVO TF TOZE.

BTDWV EPO IGAQ OV YJWK AMAIEH SOZ FWI KZCTY KN GHM EEMIQBURN HBMM AU XYM IMRBOAS."""

    encrypted_text_6A = """Lelqzq, C xnyhv Isxad en Gkcghhe ufc wbw gem ugejldv maw efjdexq. Rly zzh vwdr fwzhcff xbw LMXSR sjwqenauim gm e qakh agnwy ugemw zvimmh Mwkgoc leeamk u dnx ix msckd evgtx fgnocff jij sly ehwmamk wzztnwq en lgi Nwltfw nj Ujsigar. Xbss kund qy lhqy ln vylqmynd mn xqsg Bnhcw'r jladrx ss xbw Avclhwb Etwyml, ababl bsr e wgkpyusmif nj ujsizsbxm xqsg lgi Nwltfw. Isxad wuqr wbw jryo vi qgtpx xhkojd sol vlyjd xi yn ryps. Xbw bpow vem am xbw kswssmifr. Xbw emlks shw vem ss xbw Fvyss Tsjzqcv, sly kdgifc en lgi zgqx ix Peclaes ogmwz vem ttmfl evie sly jtmhk nj nzd Pcygxbgtwy ss Efwwehvqmu smh nzd xbaqh if Qlivdw, uenra lgi lmhrm ge xbw Bsfgrwok. Sly Cmmazs'w wsrxfw zx Vgcvoe vem ttmfl evie sly jdquamw ix sly Ezymgkioe zx Bskmwsqrukrym, zdrww sly “yqepw” semc nj amzvxamk nzd figj. Wi skp zaui wzztnwqw qwqi zgtrx ss xbw rmnw nj ifd sz lgi Mwuih Onrxwqw ix sly smgcwmx qgqpx. Lgi ifkc lwlecfhra dngulhshk zvy lgi Mlzxow nj Twtw ul Npseomu smh nzd Luffmhy Felvdrm ge Futxpif, zrx fn-shw gem smc cvde qzdvy lgi asqhyfr qcygx bsui vwdr, mg sly gmps hkeww vi wgtpx yn ryps mm Gkcghhe. Nzd gfmd en lgi yfc sz ugejldv zaui jghrnk rxlshkbl slyjd, wcfbi faflnamk, vmkp ufc suc zvy skp mqlfidr sz Rdym. Od wbgtpx td wuxd lyjd fyuzymw Isxad'w hwsaijj luk rihl gil ssxuujilk nr ng Kshvnr vq kesamk u dnra lqecd nj zgqkyv cswmlihlr vyndefamk nzd piuzxcgm sz ugejldv zaui. Nzzx vgtkbl tw yfnyaz smgw ss fgbenw zrx vdgchgil ugejldv maw. Ay zzh vwdr nzhreamk utnyn lgi ynnpolhsh ge xbw Hqjwqmud Bmjzdvm. Lzgcltw okdh vgsl nzd Zcydryjd ehv Aiumesll bmjzdvm ogmwz zvy hnpssktbsainab zyjrmifr sz lgi Wsdwuj rlcxs, ehv zx zaqwn od emktqyv slul bluhsil khb qgtpx td ihuqcjldh nzd wued auq. Vi qwqi udlsml qmazs. Qysmabaki Dgcmy'k trypoiwldh nskihl esl xnvawqc bsr kcndr gw zr cvde. C lgmhc vi gafln td evdd xi ltvh gtv yfdqcwr sh gmi ufnxbwq fs wwtfghxcff lyj fmzl, ayn A jrio gil utvlwmx zgbym ar jcytvcff sol vlyjd sh wzvnz sly kdzyfsl qgmhyj lmazs fy."""

    encrypted_text_5B = """Se vsv Rusyrrx fayyx ute Emaz ud Mpjmmwr ec lke kwrl mi hbi nqeqopzvsyi uh Nengepjii Ugxbhkz Giljjm Ymwxbrjivgz Npzhasx Zjrwgz cnk Uievm Purnnrsti Cmetjginxk, Ahjakbtr mrmyy “Hgw woxlnrk mm jtj zitwtn ip Yrqgcmfyckf aq ay ravh qyejnf. Gagwalgq apg qe i Zrlznjamc rdm, hgy vw raq m Yqfiv yjjsowv tiae, mbm qyhwakim kz ixp moj cmni bz r hijqcjfav. Ew xrq pahyn npx ar mhaxpz ci n pemmaa ud onp hz qeia xjw ipjhw. Ute Blyjuvnrujji qxlro pbwjl wswvecyz feijwnfnc zli rvf yaiwc eiox ld elwpihj taa hr aenweww imr tia zbion vw mjusjxy otme ii, mgp fhm gadw jk lwwgebkz. Ap sz jtj mvbr crrf pnxh um tbpg jv uga kwnjgf kqa qipwchfmv mcb ybe jnjgwn uh wxn qpmgyn Nkoeeen. Ad en aaa axub nngufnl hgw Gymau pbwe nwwtixx wyq xmzk ynn hi nqegeagyx nnf yvn ganwz gscp hi xprrex.” Aypjaqecy ybe Jdqaa hrmhldr Gagwalgq avmy bifn khtyy ec lke tjdix Nurenqgi’q hklmnh, hsw xn wixn c lnyp qhlk pbw ayonsvecy Pixwoonfs izm lainwgrwk nuvwx yu edpvaaue tia craidpww vuv wxn Isgjraj avj wxn Guxnf. Zth pfl cxvvsew Ynpejbrjiv hauzjiwmc mnp indjtjj pvpp rvf layijnpim Elraydra rspf p zdrcwae ytnvecygk iizyompej twyb talyi ijua wxn humhk iumxic ikev hky hwshi yeeqiyhyx Bmvu Jnrotsxk. Vzj pyb tdvc hky iznq yk Ndwe trz gaxevlmyawayyx ute xlqcxe se Bjfhizcwr ivx vvscw pfl Eboapr ov ute Blyjuv trz zjwv vlmyurwo bc ute isckivun, jtjsr fbrdor erk nzh. Igywpuxa mcyjx i vllxrv hd Lmbw, bsdwgaihjon nnf mesdqp ynlhnsv jleyb. Ptn ksqhb kiecpej dx op hibl qm muw jk utarjlb qk xslvhasvg, okgx ino oad."""

    encrypted_text_6B = """Ykuf rqhuddq rgz ppismxooeyz ud ogmp prlzqa er lop nlovktvfnx oyzcqdzov dve cxuaibk vg Lbejrvpsl arh dtwobnim lw Dfgg ta uqg onb lr fln mgmd, rp ybvigkx vx dbo Qxugbuv. Ok lqlpgh rov amgi ukrdphdo qeo ei giukx ivd ly Jlzipege lw vfdb Mdbsgzkc Gsysoooi oeo Boptemmk Hcrocvn, lwd jok Sigw codt jgzcvlsy oyv, touriwmhfd lzq kuai ia wbo Oamkh dngeqbto e icugkf, gyucz iesq pu g aka nfmel clztmvin rrir dbo Bvghj. Bqyhgckm sjilpizq jl xeelpz jsn oycqim jmdhkip ia wbo Aobmbo lw e sogl yv oabxcen, dbo Qxugbuv njcdnv bqu uqg vajt cl ioxgb jvpubautd sqyctmnn trh, da ufe kvpfbsnv yv waxk, zq jaa mmrizov jgmmumxel jthybajvcpk. L whmrbv cmw igkqrnv ly rov Aonguk wb oxg gdmvt gj jok Eumctgd, ydh, paytv gnk, Aubsfndm ygn eebo dy lsiv mweiuho iy xii xrzilm inlcrn ybodt qg kac ukpjnv ly rov Gsbigcr’k fyr nldpiaegee. Zsh vgmro rei a wgmeffka illhw pu mt, ipj l beqbadd oriel fi lzd jsn ctqctme. Gr jaa vqqw ei a vtib nfwa oftg ly ddlee rf ybgu ov cmw tll dqawyzoqsq enapv. Pyrhgbk, xntouxtx, wx oxg nisddmoi rvx ric lwkd zeiwin wcun dy jsn qywwi; rvx sy botnvh flaute, cbg ybodtkgb stlly iu delgtia kcvt iykngllqm, qii ngxk aoh ur mnq sinw; tlx xsg a aqbhsg iuqq cp lnatqbh nn jec mkcri pgxivhg md jok pqjw yb xliuod ku qmszsdy. Roxa immsiflx yei sehtoactd ja oxg ksmv-anpird ryruob roth jc drd jqnn toulkgh hs scgkfn yz rov Gsbigcr’k zuksoeo. Zggkumcb kuap bl wisc ke idpferpsej hjc gxpgrpuoi Krsoidkbe Byzkdlyn ta pxi eka efrgbbli yv ddt jriceraq. Ov cmw uoirenv cqro pgakvvvs ric nbohxdo birwe mpdb Aoqvzgnkr ipj icehqm wbopi uc qiyjtgbu Helqgfsy. So yei iesq pxie lzoo dbo nlpa gj jok Qigcj kon igdcgqkx. Klzoirvda kpauk wx iea illhw md jok pqjw iz o qvhpcv uc Qoou zmrto hjc pvrfl gaq tqanto ddt acnnnzi ia Zey et ufe anat gx uqg autnsbkiefi ia Zeboqn Irdnoizyn Hdogiee GS lnz Lyfxoi Hgtpgyh Xyzigthww. “Et irq nffbqnpdy odrxsqg hq gyg zyid mbvgxpogzr rgz cpirl Eumctgd Mnkqdkrv fi hcswdt Uer Paxipo pn hjq Entpxigv Usyqzgun trh miu uwy jpipq un ko up ufe nfwa gx uqg Aubtn Wczkdpigdk. Yuqg heogct Megxsgzg oiy okpgmjd ueqv jok ddpog gx ndsj a fley, vvd iv ald zg nau xecfrgb rov Sgviw pexfpg plt ueiwkey wx oxg yqri hjct dcm ulm lo kbig pxgu uwy jedl zlxr jem. Dcm rnzd sq uqep etukbnpqgzct ctgg g fidrvpgh ksf awqwtefk oxep rov hbakucr Clzumcbp bmw jok Qigcj mbm xa sacnmpe iea iysuybdige gb oxg Zigwb. Kxkqk fln Hebvriimnc ufeuhcddqn tto ktqmuqkm hg urlg skuo cl ric Sgjty, Smzwrwmk pw ez qmdsmdim Xwuln ezj ov kwwj yk yrfbfoj yvngpi ok qoy vgirgnq pxi nkqyaehy ia wbo Emskrslz sqpovti. Ddt Wwjnf cqnq ag bculzednv, yb ald, qgkv aiiskm ezj tkgbmaek oiv srgm pxdl bgd lzq mpiaq. Rgz iu jl lzq hcvobuq fmpj lr fln dtwq nwybm aa Isdpoylo vg cjeuo lzq uaho ia wbo nkelz nnqigb rgz gj jok Qigcj qrnvdv ykqh gb oeko vt igdcgqkx. Yydil rovr qd evhb vn qomdmvz tm jok cbzegzr Ytlyzaemcb Buzhqnp yv Ygg."""

    encrypted_text_7A = """DEYRB UGZWR VEFMY PEOAR GTDHX MGWHR RRQRL GSZVE VVIES UZRTU OEHVR SSLLQ VBCYW YHVRL OEUGV TOTGX DVQWU SBLJN HELFQ ARJLL BIIGK JAEGM QGUTG NQAQC ENRYY VOAIK PNJGC YDRPW VFSOV QOTGK KIAWE TPNIC ZKRZI XUSAW NYEUK XEZWA NKIUE OMQHA TZWWR WTSGN IGZGC ZWYFZ HDNHM GZKRZ GINLO EOGJL RUNZD RTGKD RYABL ROMHE AVDAE CIFOY HKDVN FKTUK QFNZA IJEGW MRBSJ NHELF QANKV NRUNG NKOIL KVFHL FKDRT COEGI EKVFG NMJUX LULXJ SZLNZ MEXKP CDGRV IYGNM YOMHK KSHKL OSGTR DGNUU MNKVI NSVBZ YUIHA UQAAP OBHYA SVGFB LOBHZ UNEHE SHGNM ZEUKL VJTTB QSJOO EEKBU KNAEJ MAYNA EJMAY CEIHI VLOEE UZZGE BVKIW MZTJG VGKJT FFSAX BSRZP RATIE LXVSA EQGVQ ZUAUG EAWET EGTNE KRFIW RUYEP EDVGI OEIYF AVNNH QGROK VKIQG LSOEX VRONX XTPAW HRXAV TZHVV IYSAE EIPNV ZEIVE AQDAL CMNXK IEOYP CAHRO AUZGR XDXRF VWYOD RYONK KICWY GNSWA SASVX QVFIE ERQAG TDZKE CHLNG UPNBK AGDWF LVTUK NHRRC FOPRU AIBTQ NSTOK VYEWO OJZPR ECICO JRWSA OUCGA YDZVQ NFALV TOVZZ OKUCG GMIAJ KUGVT VUWRN LNOAB VLCEV ATYSP NJNIG OYIEL XVKBS CKKGZ NETXV NLVRF ICEOU SZWCJ ASLBB MEIUM VKMFF HTHXI YVXOK HGGAC EAKAF VKRYD TFLOE EKERC OLCIM ASSLL AVYUI KKKIF WJRRZ WSZNE ZAXUD LGVUV GNGTC HEIWZ TUKYH KYTZR RBXOO JCMQK GLNLX UEPDN YIAJS AIBEZ ZHSNI TRBKR ZGINO LSUUC YJREK WLRUV LYKKG UXDVD PJAAH GNMZZ NEIXW FAHNZ GNVGI AEEIC JLTGE ZHZNL VVWVX AHREN RKRBV WVNQL DNTLF NKHRV WHYNE FZMQG CAPZI ZANHG SIXKZ HVWLV WCEFL IYRUU KLXVK HCHTV VTMPC DRNFK IGNQA QOCRQ LRDW"""

    encrypted_text_7B = """OREOM CUFED VTMGJ ELEHG VYHDN VWGKS HMBCY LMIEX LRWFF WXBKF SZEAF KZPLR VSJUE ZTTMC APOAT SJUSP EMCQF LFIXM GSEAT GRTOI EYYMV JVEUA UGLIO XWFTW GFEBF QLFCE PLTDW ZAAEO AJFIM MUARE BTYPC TPTBR GAASR TYFDL OEARG LLXNE EPWEO RPPPZ YZAEF QPRLO NYFLM DDTDR SRWMW PUOEL EPDIE RWAXG GUYTL PQGKT ZCGLP QLOAF CHZRS OTMEK MSCAR USEEA ABHZW PFLRU ZMEDM CGYME HXYQS RGALD AAPYD MAUOQ CADBJ JISMB TNLIQ GXYUK EJATC FSRSE HMVSI ENXCG GETEY UNWMD HXZPJ OTEEY VXWTE GCJSI TCXLU AVPUT CPZPD NZCHL SLDVY QLPCE AUVJV TVVYE TSPEH WTASP PEYKM EGUMF VAZTT ZQESY OHERX VXQFB RTZMP CLKPL SDEAM KLISL HBGMS ZDGRM FMCPX LNZCE DTRJL YLOKS HSWPM TQUAL NAASE LMXBX QVWPD ABJUM XCUWK ILMED LYEWE LOHNP VJCBM CTSHH EDRTG WPRRY GWIET TLHJP TEADP SXXKV ECYTR CDQPT KUPFH OCUMC SGPEA CKLMM NWGJJ GQGHM FMWLE TMHEV GITQC WSJAL SGHVS CYJKD IERLG GFVOO FPEAW YTGBP UPZSX MKPMQ RKMQX MEEHJ JYMAE HCFAM DSXLC SIDAX CWOOY EBGPS PYOPY WGIFC GGPSP SOXQP GEYTV CKAUC ETQTJ EREVA QDMNO MAKKW EYXDN LGSPG UQDES NZPVV XFHHS KLROH XPGZI YRGBC KWEEM PCLZZ TTCOW SCUWG QOIYR GYCMM PPYYR YXWEH MGDLD SAUGV ELIBR KJLHD BHVVI CAHQU VHDIK CKSGY EBMCV MCTBY GUVSR VCVKY DNPCF WIHFG FTQWO RKAPE RANAK CJXPA KECAW LAKFW SYFTB CPSXZ TZYWW HRTWA CMGZF YRJFT PTLDK SIZAT YUQGP AMYWM ADUMN PFMEE GMPEI WRKCU SRZRM FNKSP EHGUK SZETG GFHLE ULJVI DTHLQ SEPCX GVLYW OAMFN HCNZS GKMSR AMJDJ YEWPF DVLLK GESSD EMMOF IWINF KMEED NPRZY LTTAU FLMTL SNLWY RNEQG VPHLP PKMTS XMTLH DLKDT SMTTW KAAIW ENMIS _RUFX QCAUI SFCBT YCXXL ETLPA HTAGZ TLVRO MCGZS ZRYMV LWOMN MXQIY DAPFJ AWALG NWIST XGJSJ PHLQQ FSEOM FNLWS CTBKS AEDVR RGSZA MLQFL CUBCK MMWWJ GCUXS LBRVS WDDUQ JUHTO EEKUJ QNEQU YXEDG CTXIY NVETV INULQ UMWXI MYQJG LAADN FINSA JVLMW RMCQS INDTR NJMLR HGEQP TCLLV QSZRN CKVAP VPJCJ XCFBR VGRLS PSEFQ DFGUK DENSM JUKWT EGQCZ INNKP KGYYE BGGCG YRAMG WELTH RGJOY RDFDM EFEMR NWHOE ATYES SOGYQ GXNUL GCFEM AEYYZ XRHBB RSXTO TGRLX TEXRN WCDEV LPWQC MXQNK XDNVB YZHFI EQOLL LIXNU SXZAA JRKWT PWYPN VASWG VLXXS MRVSE DHKTQ JLOWK ACSXL OOKVS JPTXS CFMRO HCQJV OTWPV LTLRB RGFTZ RZPVD QLNGF JFIYE KCGWS ZOXCC MMOAN CFWGL SHEAW XREGI VVPDT OYIFI YIXLU FGPSX CJKVS IHDEA CPTLG QFKPR VLTAJ POBBK LVERF ATLEL ETRCK REIRU GVCTG OSX_A GQMSP RTYAW EFSKL QZPDN BBRXX GPECJ AGYHX GCJZD CGQKS CMSKG YJKYT XMIAK YCOYR LISDG CISST FBPEF LEOMQ KWEPS TMPSV EEACP JQNHM PEAIW EBMVS MPUGF MSIPN XRFSN EYMMQ GBYIM RCJLX UPRKA MNGPY TDWDO GINVE DTAFC ASYOV RESZE ATFIG SLTLG GGIPH ZQVKE WSAYQ EGZVH PJSWC KZFEF WPEGQ QSRZI ZLGZI ZNMMM XXSTL YKAXZ AWRGF LEHNQ HOGYD UGGGE TCXRR USWTK YGUKQ EGQKL RPIEP QFHPI VCEDW EOAJQ HWCAX GHEEF TEBOM EDANT RFYDY GMGSM PSTMT ZQQLT YJZKY NMMDJ SLOWE NMKJC BSUWX EAFYV WVEUP YJGWD PLRTZ VEEXC KLSLO GLGHW UIHYQ ZTEHM AUEIP ITDEW SLSAQ UGGHE HGPLE CRLZG PMMNM SVLTN EXCQW JTAAV VSGOH XUNWV CCBCF SIHWW GTNXQ TGJUK QGGRZ DWTZM KSUWE LGERG GIZEK FFLTE FZRFS REGWG VVIYH KFNLE XFNGQ TXDAB AXGKQ DECJT SEAXY JXCQR ALHYL YSBZU AVPIL M_HZN PJAZY VFGAX MOWPG LMZSW JQEIZ ETFPS MTMGL VWLLI TSISM XWVYA ZSZCG JVSHZ EGYTJ IZFFP PXRZN MYKWI QUTSE VMWUN AUMSP AWYEW YPIMF NZHGI GPHKV TDXJD VKLOE GPGLD IMKHD LLHAR QAIDA KQPUY SNEMG WMFEA CVDLL EXPPM WQAAF OASCS HDHZX FANLW VWPIM FGEEE EZSKL WZNNC VZEYT XYKMH DDVLA GPEAW TGBHS EICTA ISTEY GWVWO AEFWV CSXMK CGEUG QHDEP WMCEF ISPMY GLEWN HVEVH YEVDG TECOV RWKPF AYDKG WOLMP GVWZT XPPDR WSXJV LACDY ZNVWR TBMGW XPNGC UZELH PZQFO GMGGE WMZOL JPGSX ARFZF XDHBP WSXPS BCKLI TGKRT FVXSX LGAYG NVFGW VCHHR GVWDS MYEFV WELQP SLCTH YJLIJ TTBVG PLOMF KGEYH GCDJX HEAYV SITTB YPJES LZYXG IZAGR KZZZE NYIMI SWXSO JRPOG RFUPB YAVUL VCCBM KMSWS BQTKM CWLWC JMZOM JJVET SMRGZ YWAXC TSGPA ZRCOL YWICG AYZEW GVWKE LNYPJ EZBK_ THIGR SANUT LVZSX HZJTF VPIBP CTWET BENSR AEZFT APLLL PPLES TLRGO WLIHF ELROT RCAUR FAXDR YVEHE YTXME LLPEN COSTG EHHYI KCDEJ PAMUG JIWAB GYWVL QRJVA WQEGB KWEFO FRGMX PFHLU FXDAM RUFIZ LFEZN XEETF ISFOA MRPOW EEBNP AKXLH RQAXW SMQVL MZRKR RAIGE GLQKG LEXCG KSSME GHSSZ ABUVN LNHHW WVMXR TMQCI EEGQT SWHTB KFSSE TKRGW VESVR GFXLL HCGKX CSGNJ LILFM FYLFF OWQGF QFOXQ GQXDO AANOX WHHUG HWSXB BIGYH RXKJA IVTKC GWYVN NNVZH NDWMV DVPEM QGAIL ODFTF RYTBY NLTOW XPKGX ZNHRH SVLUW LJGBT HLSYL MADZP VKILT TKFSS PMAPD MITHF YKJEO ITQVS LTNVC GSWTN MFPAR FHXDT JAPDL WQSMN EWAVJ QCTUR KRXER GCHSY WOHBK SXXSG SULXZ SMMQG LDDBA WAYEE PRQWM SSNKT XILUH PCMST TZLGF VXUXC HJHNW EDGAI YAWCV SRTEA ACWJL PBRPY ZDIMF GGSWR YRGLQ LU"""

    encrypted_text_2B = """T	E	L	U	U	T	R	F	E	T	N	E	H	I	O	Y	I	T	B	E	P	H	N	D	A	H	E	I	O	O	D	R	U	L	T	I	E	R	A	T	E	F	R	E	H	A	E	D	U	G	S	I O	A	F	D	E	I	I	E	L	C	G	N	E	S	R	P	I	N	E	O	E	R	E	O	O	N	R	H	F	T	O	D	C	R	D	V	L	S	R	V	T	D	B	T	M	M	O	S	S	A	N F	N	S	R	W	T	U	R	A	W	T	E	E	R	N	O	F	G	V	P	U	H	E	H	L	O	A	S	T	I	E	E	A	O	O	E	N	X	I	H	B	T	N	R	E	E	S	T	V	A R	M	E	D	H	B	L	C	C	E	L	R	O	C	S	H	N	P	I	C	C	T	V	V	B	U	A	E	S	D	A	D	A	M	A	A	T	I	D	S	N	E	A	T	R	E	S	A	T	M W	R	O	I	T	D	T	E	O	A	F	U	T	O	T	E	E	I	L	T	I	U	S	O	N	G	L	R	O	A	A	E	A	L	L	E	O	F	L	T	S	H	T	N	I	R	W	T	A	L	N R	D	O	R	N	N	E	F	S	O	R	T	Y	O	N	T	F	R	I	T
H	L	E	E	D	O	E	R	D	O	I	G	E	W	I	T	N	E	E	Y	O	L	I	V	M	E	D	T	T	U	D	A	R	E	O	N	D	U	T	H	G	T	I	S	E	T	T	F	B	R	O S	F	N	R	T	M	R	U	T	L	K	W	G	T	O	D	I	N	U	D	O	B	Y	R	F	R	D	E	O	T	T	U	T	S	E	E	E	O	V	Y	E	E	E	A	H	Y	A	U	A	T F	D	O	T	A	I	A	T	R	E	C	I	H	F	N	H	P	A	O	U	E	R	S	E	S	E	L	U	P	S	H	L	S	R	T	O	W	C	G	T	O	E	U	T	O	A	T	R	A	O E	C	E	Y	V	T	E	E	E	I	A	D	E	S	M	A	W	A	E	T	N	A	I	H	I	E	Y	S	R	L	E	M	R	I	T	M	T	C	H	R	E	I	O	R	N	H	E	T	S	T	O	P E	E	G	M	H	I	H	A	U	A	T	B	A	L	O	F	U	N	O	H	O	I	S	H	C	E	O	I	R	N	S	D	G	A	S	B	B	O	O	H	A	E	S	I	L	E	A	I	G	A	G E	E	F	I	W	G	T	Y	H	W	U	H	T	U	E	R	U	T	S	H
E	L	D	E	I	O	N	O	E	F	N	I	I	E	N	H	O	S	A	S	N	O	U	E	I	Y	T	I	H	N	E	T	T	S	N	F	B	S	U	E	I	H	O	S	S	I	U	R	D	E	N	L M	D	O	H	Y	O	M	R	E	A	A	S	I	W	S	T	G	M	T	N	R	A	O	O	C	M	F	I	H	L	N	A	E	D	C	A	R	I	W	R	R	D	T	E	L	N	S	G	J	E R	U	R	N	C	S	O	E	G	T	T	L	F	C	A	A	T	R	I	R	E	L	S	S	R	I	D	O	E	E	O	S	S	I	K	N	U	T	H	N	D	T	L	T	N	R	I	B	B	A	A	D S	E	H	Y	E	T	P	P	B	O	A	M	T	H	D	N	E	G	E	P	I	C	N	J	C	H	F	C	I	Y	P	I	U	I	R	E	E	C	T	L	S	D	E	W	O	D	C	T	E	R	F	I	S E	T	E	T	D	N	N	T	G	A	B	O	P	G	S	E	N	L	A	A	O	D	C	B	E	D	H	T	R	M	O	E	L	R	S	E	R	R	U	U	E	V	S	T	R	W	I	D	F	T B	A	S	B	O	T	T	T	A	H	W	E	A	R	O	L	O
R	I	B	N	C	K	G	M	F	T	T	O	C	R	E	E	V	T	S	E	B	N	M	R	U	R	H	E	E	D	S	E	E	S	I	O	Y	L	S	I	O	E	U	O	I	O	R	O	U	B	T A	O	C	N	E	A	C	A	A	D	L	T	T	N	N	D	E	O	B	E	E	A	N	I	U	E	Y	U	C	E	E	D	C	N	O	I	N	I	C	H	M	M	T	T	E	O	Y	A	A	U W	E	R	O	D	O	L	S	T	I	I	H	E	I	Y	P	R	O	H	L	T	V	O	U	E	E	O	I	S	D	P	U	O	I	O	H	L	R	H	L	A	I	A	E	A	D	U	L	O	E	L	P E	P	A	A	H	N	O	H	T	A	R	N	U	I	I	B	C	D	T	S	H	S	E	T	U	A	I	H	U	L	A	L	C	N	O	O	E	F	A	W	O	A	N	R	E	N	E	H	H	R E	U	V	E	C	F	D	O	I	D	O	T	R	W	L	R	R	T	S	L	S	A	I	V	U	T	A	E	B	I	I	H	I	A	H	E	A	T	S	C	C	W	E	S	D	E	T	W	I	H	N	T E	H	E	T	T	U	U	O	H	H	T	E	I	D	V	T	T	E	R
E	O	Y	B	C	S	T	T	E	H	H	N	E	E	D	T	A	R	T	T	O	D	A	U	M	A	E	S	G	A	E	D	M	U	U	R	V	E	O	X	N	S	S	F	T	N	N	M	I	E H	N	N	O	T	E	T	O	S	V	B	O	L	R	H	W	E	B	U	E	N	T	V	D	S	R	S	C	L	E	B	G	A	T	S	U	S	D	O	T	O	A	I	H	L	N	S	T	N	I	S H	D	O	O	A	L	E	E	H	O	N	R	S	C	P	S	T	N	I	T	H	I	S	P	D	B	N	C	S	O	E	S	F	T	N	E	I	S	E	E	S	E	B	I	W	T	E	W	U	R	E	T N	Y	L	T	A	A	D	E	U	T	D	D	N	O	C	E	R	U	H	A	E	D	I	E	L	E	M	A	R	I	N	O	C	I	N	S	M	O	D	A	N	F	E	W	S	I	S	E	E	O	C L	E	L	R	O	E	F	C	L	D	H	I	A	A	T	I	H	O	E	A	W	D	E	R	H	L	S	E	T	N	A	C	Y	A	N	M	H	O	A	I	H	T	F	T	A	H	A	C	O	E H	A	E	S	L	R	T	W	K	E	O	D	N	L	T	E	H	H	O	N
B	N	Q	O	A	T	H	H	A	E	L	T	N	J	B	R	N	I	H	U	T	I	N	L	T	Z	C	T	R	N	C	O	P	E	S	M	A	G	F	L	O	E	N	T	U	R	E	S	N	L	E D	A	N	E	N	V	N	H	E	A	N	I	E	I	O	S	E	T	R	T	H	E	H	M	F	A	A	C	O	A	R	N	I	U	R	I	G	U	O	E	S	N	E	E	E	T	H	D	N	T U	O	W	P	G	A	F	C	E	N	G	U	S	I	E	I	T	E	S	O	E	O	S	R	T	E	B	C	E	F	R	N	H	U	T	R	F	I	I	G	S	D	A	S	A	H	P	A	T	E	D U	E	R	E	T	D	B	E	R	R	T	E	C	I	N	H	E	Y	S	E	R	R	E	N	D	I	S	S	D	E	T	D	M	O	C	S	S	P	R	E	S	G	E	I	E	U	U	P	S	E	R	A T	H	F	E	R	F	B	C	E	O	A	C	S	M	H	S	E	F	G	Q	A	T	E	A	E	T	T	L	W	T	T	O	A	V	T	E	E	F	E	P	A	O	A	O	L	A	S	O	E	E	E T	T	O	I	E	I	I	N	T	F	A	Y	L	O	L	E	E	F	S"""
    """
    text_1A = Affine(encrypted_text_1A, switch=(1, 5))
    text_4B = Viginere(encrypted_text_4B, key="arcanaimperii")
    text_6A = Viginere(encrypted_text_6A, key="zeus")
    text_6B = AffineViginere(encrypted_text_6B)
    # print(text_6A.encipher())
    # print(len(letters(text_4B.text)))
    text_5B = Viginere(encrypted_text_5B, key="arcanaimperii", beaufort=True)
    # print(y == letters(text_4B.text[0::13]).lower())
    # print(letters(text_4B.text[0::13]).lower())
    # print(collections.Counter(y))
    y = letters(text_4B.text)[0::13].lower()
    # print(text_6B.encipher())
    text_5B = AffineViginere(encrypted_text_5B)
    print(text_5B.encipher())
    """
    #text_7A = Viginere(encrypted_text_7A)
    # print(text_7A.encipher())
    #text_7B = Viginere(encrypted_text_7B)
    # print(text_7B.prob_key)
    z = letters(encrypted_text_2B)
    print(z[2::len(z)//6])
