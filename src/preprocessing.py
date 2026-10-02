"""
Text Preprocessing Module for Indonesian NLP
============================================
Comprehensive pipeline for text cleaning, slang normalization, stopword removal,
and stemming tailored for Indonesian conversational data.
"""

import re
import string
from typing import Dict, List, Optional, Union

# Try importing Sastrawi components with safe fallback
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False


# Default Indonesian Stopwords fallback list
DEFAULT_INDONESIAN_STOPWORDS = {
    "ada", "adalah", "adanya", "adapun", "agak", "agar", "akan", "akankah", "akhir",
    "akhiri", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar",
    "antara", "antaranya", "apa", "apaan", "apabila", "apakah", "apalagi", "apatah",
    "arti", "artinya", "asal", "asalkan", "atas", "atau", "ataukah", "ataupun", "awal",
    "awalnya", "bagai", "bagaikan", "bagaimana", "bagaimanakah", "bagaimanapun", "bagi",
    "bagian", "bahkan", "bahwa", "bahwasanya", "baik", "bakal", "bakalan", "balik",
    "banyak", "bapak", "beberapa", "begini", "beginian", "beginikah", "beginilah",
    "begitu", "begitukah", "begitulah", "begitupun", "bekerja", "belakang", "belakangan",
    "belum", "belumlah", "benar", "benarkah", "benarlah", "berada", "berakhir", "berakhirlah",
    "berakhirnya", "berapa", "berapakah", "berapalah", "berapapun", "berarti", "berawal",
    "berbagai", "berdatangan", "beri", "berikan", "berikut", "berikutnya", "berjumlah",
    "berkenaan", "berlainan", "berlalu", "berlangsung", "berlebihan", "bermacam",
    "bermaksud", "bermula", "bersama", "bersiap", "bertanya", "berturut", "bertutur",
    "berujar", "berupa", "besar", "betul", "betulkah", "biasa", "biasanya", "bila",
    "bilakah", "bisa", "bisakah", "boleh", "bolehkah", "bolehlah", "buat", "bukan",
    "bukankah", "bukanlah", "bukannya", "cuma", "percuma", "dahulu", "dalam", "dan",
    "dapat", "dari", "daripada", "dekat", "demi", "demikian", "demikianlah", "dengan",
    "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah", "diantara", "diantaranya",
    "diberi", "diberikan", "diberikannya", "dibuat", "dibuatnya", "didapat", "didatangkan",
    "digunakan", "diibaratkan", "diibaratkannya", "diingat", "diingatkan", "diinginkan",
    "dijawab", "dijelaskan", "dijelaskannya", "dikarenakan", "dikatakan", "dikatakannya",
    "dikerjakan", "diketahui", "diketahuinya", "dikira", "dilakukan", "dilalui", "dilihat",
    "dimaksud", "dimaksudkan", "dimaksudkannya", "dimaksudnya", "diminta", "dimintai",
    "dimisalkan", "dimulai", "dimulailah", "dimulainya", "dimungkinkan", "dini", "dipastikan",
    "diperbuat", "diperbuatnya", "dipergunakan", "diperkirakan", "diperlihatkan", "diperlukan",
    "diperlukannya", "dipersoalkan", "dipertanyakan", "dipunyai", "diri", "dirinya",
    "disampaikan", "disebut", "disebutkan", "disebutkannya", "disini", "disinilah",
    "disoalkan", "disuruh", "disampaikan", "ditambahkan", "ditandaskan", "ditanya",
    "ditanyai", "ditanyakan", "ditegaskan", "ditemukan", "ditemukannya", "diterima",
    "diterangkan", "ditujukan", "ditunjuk", "ditunjuki", "ditunjukkan", "ditunjukkannya",
    "ditunjuknya", "dituturkan", "dituturkannya", "diucapkan", "diucapkannya", "diungkapkan",
    "dong", "dua", "dulu", "empat", "enggak", "enggaknya", "entah", "entahlah", "guna",
    "gunakan", "hal", "hampir", "hanya", "hanyalah", "hari", "harus", "haruslah", "harusnya",
    "hendak", "hendaklah", "hendaknya", "hingga", "ia", "ialah", "ibarat", "ibaratkan",
    "ibaratnya", "ibu", "ikut", "ingat", "inginkah", "ini", "inikah", "inilah", "itu",
    "itukah", "itulah", "jadi", "jadilah", "jadinya", "jangan", "jangankan", "janganlah",
    "jauh", "jawab", "jawaban", "jawabnya", "jelas", "jelaskan", "jelaslah", "jelasnya",
    "jika", "jikalau", "juga", "jumlah", "jumlahnya", "justru", "kala", "kalau", "kalaulah",
    "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah",
    "kapanpun", "karena", "karenanya", "kasus", "kata", "katakan", "katakanlah", "katanya",
    "ke", "keadaan", "kebetulan", "kecil", "kedua", "keduanya", "keinginan", "kelen",
    "kelihatan", "kelihatannya", "kelima", "keluar", "kembali", "kemudian", "kemungkinan",
    "kemungkinannya", "kenapa", "kepada", "kepadanya", "kesampaian", "keseluruhan",
    "keseluruhannya", "keterlaluan", "ketika", "khususnya", "kini", "kinilah", "kira",
    "kira-kira", "kiranya", "kita", "kitalah", "kok", "kurang", "lagi", "lagian", "lah",
    "lain", "lainnya", "lalu", "lama", "lamanya", "lanjut", "lanjutnya", "lebih", "lewat",
    "lima", "luar", "macam", "maka", "makanya", "makin", "malah", "malahan", "mampu",
    "mampukah", "mana", "manakala", "manalagi", "masih", "masihkah", "masing", "masing-masing",
    "mau", "maupun", "melainkan", "melakukan", "melalui", "melihat", "melihatnya", "memang",
    "memastikan", "memberi", "memberikan", "membuat", "memerlukan", "memihak", "meminta",
    "memintakan", "memisalkan", "memperbuat", "mempergunakan", "memperkirakan", "memperlihatkan",
    "mempersiapkan", "mempersoalkan", "mempertanyakan", "mempunyai", "memulai", "memungkinkan",
    "menaiki", "menandaskan", "menanti", "menantikan", "menanya", "menanyai", "menanyakan",
    "mendapat", "mendapatkan", "mendatang", "mendatangi", "mendatangkan", "menegaskan",
    "mengakhiri", "mengapa", "mengenai", "mengerjakan", "mengetahui", "menggunakan",
    "menghendaki", "mengibaratkan", "mengibaratkannya", "mengingat", "mengingatkan",
    "menginginkan", "mengira", "mengucapkan", "mengucapkannya", "mengungkapkan", "menjadi",
    "menjawab", "menjelaskan", "menuju", "menunjuk", "menunjuki", "menunjukkan", "menunjuknya",
    "menurut", "menuturkan", "menyampaikan", "menyangkut", "menyatakan", "menyebutkan",
    "menyeluruh", "menyiapkan", "merasa", "mereka", "merekalah", "merupakan", "meski",
    "meskipun", "meyakini", "meyakinkan", "minta", "mirip", "misal", "misalkan", "misalnya",
    "mula", "mulai", "mulailah", "mulanya", "mungkin", "mungkinkah", "nah", "naik", "namun",
    "nanti", "nantinya", "nyaris", "nyatanya", "oleh", "olehnya", "pada", "padahal", "padanya",
    "pak", "paling", "panjang", "pantas", "para", "pasti", "pastilah", "penting", "pentingnya",
    "per", "percuma", "perlu", "perlukah", "perlunya", "pernah", "persoalan", "pertama",
    "pertama-tama", "pertanyaan", "pertanyakan", "pihak", "pihaknya", "pukul", "pula",
    "pun", "punya", "rasa", "rasanya", "rata", "rupanya", "saat", "saatnya", "saja", "sajalah",
    "saling", "sama", "sama-sama", "sambil", "sampai", "sampai-sampai", "sampaikan", "sana",
    "sangat", "sangatlah", "satu", "saya", "sayalah", "se", "sebab", "sebabnya", "sebagai",
    "sebagaimana", "sebagainya", "sebagian", "sebaik", "sebaik-baiknya", "sebaiknya", "sebaliknya",
    "sebanyak", "sebegini", "sebegitu", "sebelum", "sebelumnya", "sebenarnya", "seberapa",
    "sebesar", "sebetulnya", "sebisanya", "sebuah", "sebut", "sebutlah", "sebutnya", "secara",
    "secukupnya", "sedang", "sedangkan", "sedemikian", "sedikit", "sedikitnya", "seenaknya",
    "segala", "segalanya", "segera", "seharusnya", "sehingga", "seingat", "sejak", "sejauh",
    "sejenak", "sejumlah", "sekadar", "sekadarnya", "sekali", "sekalian", "sekaligus",
    "sekalipun", "sekarang", "sekaranglah", "sekecil", "seketika", "sekiranya", "sekitar",
    "sekitarnya", "sekurang-kurangnya", "sekurangnya", "sela", "selain", "selalu", "selama",
    "selama-lamanya", "selamanya", "selanjutnya", "seluruh", "seluruhnya", "semacam", "semakin",
    "semampu", "semampunya", "semasa", "semasih", "semata", "semata-mata", "semaunya",
    "sementara", "semisal", "semisalnya", "sempat", "semua", "semuanya", "semula", "sendiri",
    "sendirinya", "seolah", "seolah-olah", "seorang", "sepanjang", "sepantasnya", "sepantasnyalah",
    "seperlunya", "seperti", "sepertinya", "sepihak", "sering", "seringnya", "serta", "serupa",
    "sesaat", "sesama", "sesampai", "sesegera", "sesekali", "seseorang", "sesuatu", "sesuatunya",
    "sesudah", "sesudahnya", "setelah", "setempat", "setengah", "seterusnya", "setiap",
    "setiba", "setibanya", "setidak-tidaknya", "setidaknya", "setinggi", "seusai", "sewaktu",
    "siap", "siapa", "siapakah", "siapapun", "sini", "sinilah", "soal", "soalnya", "suatu",
    "sudah", "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tahu", "tahun", "tak",
    "tambah", "tambahnya", "tampak", "tampaknya", "tandas", "tandasnya", "tanpa", "tanya",
    "tanyakan", "tanyanya", "tapi", "tegas", "tegasnya", "telah", "tempat", "tengah", "tentang",
    "tentu", "tentulah", "tentunya", "tepat", "terakhir", "terasa", "terbanyak", "terdahulu",
    "terdapat", "terdiri", "terhadap", "terhadapnya", "teringat", "teringat-ingat", "terjadi",
    "terjadilah", "terjadinya", "terkira", "terlalu", "terlebih", "terlihat", "termasuk",
    "termasuklah", "tersebut", "tersebutlah", "tertentu", "tertuju", "terus", "terutama",
    "tetap", "tetapi", "tiap", "tiba", "tiba-tiba", "tidak", "tidakkah", "tidaklah", "tiga",
    "toh", "tuju", "tunjuk", "turut", "tutur", "tuturnya", "ucap", "ucapnya", "ujar", "ujarnya",
    "umum", "umumnya", "ungkap", "ungkapnya", "untuk", "usah", "usai", "waduh", "wah", "wahai",
    "waktu", "waktunya", "walau", "walaupun", "wong", "yaitu", "yakin", "yakni", "yang"
}

# Slang and Abbreviation Normalization Dictionary
DEFAULT_SLANG_DICT = {
    "dmn": "dimana",
    "dgn": "dengan",
    "kpn": "kapan",
    "gmn": "bagaimana",
    "gimana": "bagaimana",
    "bkn": "bukan",
    "sy": "saya",
    "km": "kamu",
    "yg": "yang",
    "tdk": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "gk": "tidak",
    "aja": "saja",
    "udah": "sudah",
    "udh": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "bisa": "bisa",
    "krn": "karena",
    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",
    "trs": "terus",
    "bgt": "banget",
    "bgmn": "bagaimana",
    "brapa": "berapa",
    "brp": "berapa",
    "gmnkah": "bagaimanakah",
    "info": "informasi",
    "pls": "tolong",
    "tlg": "tolong",
    "mhn": "mohon",
    "makasih": "terima kasih",
    "trims": "terima kasih",
    "thx": "terima kasih",
    "min": "admin",
    "admin": "admin",
    "kak": "kakak",
    "gan": "juragan",
    "bro": "saudara",
    "sis": "saudari",
    "y": "ya",
    "kmpus": "kampus",
    "kmpusnya": "kampus",
    "kmpusny": "kampus",
    "bya": "biaya",
    "dftr": "daftar",
    "mhs": "mahasiswa",
    "kulyah": "kuliah",
    "ap": "apa",
    "aj": "saja",
    "ad": "ada",
    "hrus": "harus",
    "jurusan": "program studi",
    "prodi": "program studi",
    "univ": "universitas",
    "biayanya": "biaya",
    "daftarnya": "daftar",
    "pendaftarannya": "pendaftaran"
}


class TextPreprocessor:
    """
    Configurable Indonesian Text Preprocessor supporting case folding,
    slang normalization, stopword filtering, and morphological stemming.
    """

    def __init__(
        self,
        case_folding: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        normalization: bool = True,
        stopword_removal: bool = True,
        stemming: bool = False,
        custom_slang_dict: Optional[Dict[str, str]] = None
    ):
        self.case_folding = case_folding
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.normalization = normalization
        self.stopword_removal = stopword_removal
        self.stemming = stemming

        self.slang_dict = custom_slang_dict or DEFAULT_SLANG_DICT
        self._stemmer = None
        self._stopword_remover = None
        self._stopwords_set = None

    def _get_stemmer(self):
        """Lazy load Sastrawi Stemmer."""
        if self._stemmer is None and SASTRAWI_AVAILABLE:
            factory = StemmerFactory()
            self._stemmer = factory.create_stemmer()
        return self._stemmer

    def _get_stopwords(self) -> set:
        """Returns stopwords set from Sastrawi if available, or fallback."""
        if self._stopwords_set is None:
            if SASTRAWI_AVAILABLE:
                try:
                    factory = StopWordRemoverFactory()
                    self._stopwords_set = set(factory.get_stop_words())
                except Exception:
                    self._stopwords_set = DEFAULT_INDONESIAN_STOPWORDS
            else:
                self._stopwords_set = DEFAULT_INDONESIAN_STOPWORDS
        return self._stopwords_set

    def clean_text(self, text: str) -> str:
        """
        Cleans text: lowercase, removes URLs, mentions, punctuation, numbers, and extra spaces.
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs and handles
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+|\#", "", text)

        # Case folding
        if self.case_folding:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_slang(self, text: str) -> str:
        """Normalizes informal/slang Indonesian words to standard formal words."""
        if not text:
            return ""
        tokens = text.split()
        normalized = [self.slang_dict.get(word.lower(), word) for word in tokens]
        return " ".join(normalized)

    def remove_stopwords(self, text: str) -> str:
        """Removes Indonesian stop words."""
        if not text:
            return ""
        stopwords = self._get_stopwords()
        tokens = text.split()
        filtered = [word for word in tokens if word.lower() not in stopwords]
        return " ".join(filtered) if filtered else text

    def stem_text(self, text: str) -> str:
        """Applies Sastrawi stemming to Indonesian words."""
        if not text:
            return ""
        stemmer = self._get_stemmer()
        if stemmer:
            try:
                return stemmer.stem(text)
            except Exception:
                return text
        return text

    def transform(
        self,
        text: Union[str, List[str]],
        full_pipeline: bool = True,
        case_folding: Optional[bool] = None,
        remove_punct: Optional[bool] = None,
        remove_num: Optional[bool] = None,
        normalize: Optional[bool] = None,
        remove_stop: Optional[bool] = None,
        stem: Optional[bool] = None
    ) -> Union[str, List[str]]:
        """
        Transforms text with fine-grained control over each pipeline stage.
        Supports both single string and list of strings.
        """
        if isinstance(text, (list, tuple)):
            return [
                self.transform(
                    item,
                    full_pipeline=full_pipeline,
                    case_folding=case_folding,
                    remove_punct=remove_punct,
                    remove_num=remove_num,
                    normalize=normalize,
                    remove_stop=remove_stop,
                    stem=stem
                )
                for item in text
            ]

        if not isinstance(text, str):
            return ""

        # Parameter override or instance default
        do_case = case_folding if case_folding is not None else self.case_folding
        do_punct = remove_punct if remove_punct is not None else self.remove_punctuation
        do_num = remove_num if remove_num is not None else self.remove_numbers
        do_norm = normalize if normalize is not None else self.normalization
        do_stop = remove_stop if remove_stop is not None else self.stopword_removal
        do_stem = stem if stem is not None else self.stemming

        # E0 Baseline mode (Minimal preprocessing)
        if not full_pipeline:
            # Minimal: only basic lowercase and strip
            return text.lower().strip()

        # Step 1: Base cleaning (case folding, punct, num)
        # Custom cleaning per flags
        res = text
        if do_case:
            res = res.lower()
        if do_punct:
            res = res.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        if do_num:
            res = re.sub(r"\d+", "", res)
        res = re.sub(r"\s+", " ", res).strip()

        # Step 2: Slang normalization
        if do_norm:
            res = self.normalize_slang(res)

        # Step 3: Stopword removal
        if do_stop:
            res = self.remove_stopwords(res)

        # Step 4: Stemming
        if do_stem:
            res = self.stem_text(res)

        return res.strip()
