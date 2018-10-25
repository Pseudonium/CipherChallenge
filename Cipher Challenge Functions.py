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
            len(text) / 100) for char in english_chars
        )
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
            (english_chars.index(char.lower()) + shift) % ENGLISH_LANG_LEN
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
    def __init__(self, text, key=""):
        self.text = text
        self.key = key
        self.auto = not bool(key)

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

    @property
    def prob_key(self):
        text = letters(self.text).lower()
        split_text = [
            "".join(text[offset::self.prob_key_length])
            for offset in range(self.prob_key_length)
        ]
        shifts = []
        for split in split_text:
            modal_char = auto_freq_analyser(split)[0].character
            split_shift = (
                english_chars.index(modal_char)
            ) - english_chars.index("e")
            shifts.append(split_shift)
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
            split = Caesar(
                split,
                shift=ENGLISH_LANG_LEN-english_chars.index(self.key[index])
                # Above added since vigenere keys are the complement, usually
            )
            shifted_split.append(split.encipher())
        enciphered = "".join(
            "".join(chunk)
            for chunk in itertools.zip_longest(*shifted_split, fillvalue=" ")
        )
        return match(self.text, enciphered.rstrip())


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

    text_1A = Affine(encrypted_text_1A, switch=(1, 5))
    text_4B = Viginere(encrypted_text_4B, key="arcanaimperii")
    text_6A = Viginere(encrypted_text_6A, key="zeus")
    # print(text_6A.encipher())
    # print(len(letters(text_4B.text)))
    text_5B = Viginere(encrypted_text_5B, key="arcanaimperii")
    """
    for possible_shift in range(26):
        shifted_text = Caesar(
            letters(text_4B.text).lower()[0::13], shift=possible_shift, forced=True).encipher()
        print(shifted_text)
        print(collections.Counter(shifted_text))
        print(possible_shift, english_1gram_chi(shifted_text))
    """
    y = """oagwtmtawzemramejrbjttzvvdnckzmfskxeprciavkrsgdomgdygnvgaaijosrheglyczebacuiiwzihqsmiektyncoiimtgluuovrelphevrbgtzelzxcqewnhrsrajipjaagmndpdaofkwepjkilganchtfivxdlqincewdetbsnmbnkotaoyjpeqnacfspjielrvfiofprdwixovepqbrrmdztfaalijiicpveveinbsmbcmqzdkbemnqmvbqizea"""
    # print(y == letters(text_4B.text[0::13]).lower())
    # print(letters(text_4B.text[0::13]).lower())
    # print(collections.Counter(y))
    y = letters(text_4B.text)[0::13].lower()
    print(text_4B.encipher())
