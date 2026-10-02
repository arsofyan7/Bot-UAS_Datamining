"""
================================================================================
MODUL: src/preprocessing.py
DESKRIPSI: Pipeline Pra-Pemrosesan Teks Bahasa Indonesia (NLP Preprocessing)
================================================================================

Tujuan Ilmiah / Konsep Teori:
-----------------------------
Teks percakapan manusia (khususnya Bahasa Indonesia informal) memiliki banyak
"derau" (noise) seperti kesalahan ejaan, tanda baca berlebih, singkatan/slang,
dan kata hubung yang tidak membawa informasi niat (intent).

Modul ini bertanggung jawab mengubah kalimat mentah menjadi bentuk baku:
1. Case Folding & Base Cleaning: Menyeragamkan huruf kecil & menghapus URL/angka/simbol.
2. Slang Normalization: Memetakan kata gaul/singkatan ke kata baku (contoh: "dmn" -> "dimana").
3. Stopword Removal: Menghapus kata umum yang minim informasi pembeda.
4. Stemming (Sastrawi): Mengubah kata berimbuhan ke bentuk kata dasarnya (opsional).
"""

import re
import string
from typing import Dict, List, Optional, Union

# -----------------------------------------------------------------------------
# 1. PENGELOLAAN DEPENDENSI SASTRAWI
# -----------------------------------------------------------------------------
# Menggunakan safe import agar program tidak crash jika Sastrawi belum terinstal
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False


# -----------------------------------------------------------------------------
# 2. DAFTAR STOPWORDS BAHASA INDONESIA (FALLBACK LIST)
# -----------------------------------------------------------------------------
# Kumpulan kata hubung dan partikel umum yang dieliminasi jika tidak membawa makna spesifik intent.
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
    "disoalkan", "disuruh", "ditambahkan", "ditandaskan", "ditanya",
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


# -----------------------------------------------------------------------------
# 3. KAMUS NORMALISASI SLANG / KATA GAUL / SINGKATAN PERCAKAPAN
# -----------------------------------------------------------------------------
# Mengapa normalisasi dilakukan sebelum stopword removal?
# Agar singkatan seperti "dmn" -> "dimana" atau "kpn" -> "kapan" berhasil dipulihkan
# ke bentuk kata aslinya terlebih dahulu sehingga vectorizer dapat mengenali fitur tersebut.
DEFAULT_SLANG_DICT = {
    # Kata Tanya & Partikel Arah
    "dmn": "dimana",
    "dgn": "dengan",
    "kpn": "kapan",
    "gmn": "bagaimana",
    "gimana": "bagaimana",
    "bkn": "bukan",
    "bgmn": "bagaimana",
    "brapa": "berapa",
    "brp": "berapa",
    "gmnkah": "bagaimanakah",

    # Kata Ganti & Sapaan
    "sy": "saya",
    "km": "kamu",
    "yg": "yang",
    "min": "admin",
    "admin": "admin",
    "kak": "kakak",
    "gan": "juragan",
    "bro": "saudara",
    "sis": "saudari",
    "y": "ya",

    # Negasi & Slang Umum
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
    "krn": "karena",
    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",
    "trs": "terus",
    "bgt": "banget",

    # Topik Pendidikan & Kampus
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
    "info": "informasi",
    "pls": "tolong",
    "tlg": "tolong",
    "mhn": "mohon",
    "makasih": "terima kasih",
    "trims": "terima kasih",
    "thx": "terima kasih",
    "jurusan": "program studi",
    "prodi": "program studi",
    "univ": "universitas",
    "biayanya": "biaya",
    "daftarnya": "daftar",
    "pendaftarannya": "pendaftaran"
}


# -----------------------------------------------------------------------------
# 4. KELAS UTAMA: TextPreprocessor
# -----------------------------------------------------------------------------
class TextPreprocessor:
    """
    Kelas pemrosesan teks modular untuk Bahasa Indonesia.
    
    Menyediakan kontrol fleksibel untuk menghidupkan/mematikan komponen pra-pemrosesan
    guna mendukung studi perbandingan ablasi (Eksperimen E0 vs E1).
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
        """
        Inisialisasi konfigurasi pipeline pembersihan.
        """
        self.case_folding = case_folding
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.normalization = normalization
        self.stopword_removal = stopword_removal
        self.stemming = stemming

        # Gunakan kamus kustom jika diberikan, atau gunakan kamus bawaan
        self.slang_dict = custom_slang_dict or DEFAULT_SLANG_DICT

        # Lazy loading objek Sastrawi agar inisialisasi awal tetap instan
        self._stemmer = None
        self._stopword_remover = None
        self._stopwords_set = None

    def _get_stemmer(self):
        """Lazy loader untuk objek Sastrawi Stemmer."""
        if self._stemmer is None and SASTRAWI_AVAILABLE:
            factory = StemmerFactory()
            self._stemmer = factory.create_stemmer()
        return self._stemmer

    def _get_stopwords(self) -> set:
        """Mengambil kumpulan stopwords dari Sastrawi atau fallback list."""
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
        Langkah 1: Pembersihan Dasar
        - Menghapus tautan web (HTTP/HTTPS URL) dan mention username (@).
        - Case Folding: Mengubah seluruh huruf menjadi lowercase (a != A).
        - Punctuation Removal: Mengganti tanda baca dengan spasi agar kata tidak menempel.
        - Number Removal: Menghapus angka jika angka tidak signifikan untuk klasifikasi topik.
        - Whitespace Stripping: Menghapus spasi ganda dan whitespace di awal/akhir kalimat.
        """
        if not isinstance(text, str):
            return ""

        # Hapus link dan mention
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+|\#", "", text)

        # Case folding
        if self.case_folding:
            text = text.lower()

        # Penghapusan tanda baca
        if self.remove_punctuation:
            text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

        # Penghapusan angka
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Rapikan spasi berlebih
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_slang(self, text: str) -> str:
        """
        Langkah 2: Normalisasi Slang & Kata Gaul
        Memecah kalimat menjadi token kata, lalu mengganti kata yang cocok di kamus slang.
        Contoh: 'kmpusnya dmn y' -> 'kampus dimana ya'
        """
        if not text:
            return ""
        tokens = text.split()
        normalized = [self.slang_dict.get(word.lower(), word) for word in tokens]
        return " ".join(normalized)

    def remove_stopwords(self, text: str) -> str:
        """
        Langkah 3: Stopword Filtering
        Menyaring kata-kata umum yang tidak membawa nilai diskriminatif untuk klasifikasi intent.
        """
        if not text:
            return ""
        stopwords = self._get_stopwords()
        tokens = text.split()
        filtered = [word for word in tokens if word.lower() not in stopwords]
        return " ".join(filtered) if filtered else text

    def stem_text(self, text: str) -> str:
        """
        Langkah 4: Morphological Stemming (Sastrawi)
        Mengembalikan kata berimbuhan ke akar kata dasarnya.
        Contoh: 'pendaftaran' -> 'daftar', 'membayar' -> 'bayar'
        """
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
        Pipeline Eksekusi Utama:
        Menerima input berupa 1 string kalimat atau kumpulan list kalimat (batch),
        lalu mengeksekusi seluruh tahapan pra-pemrosesan secara berurutan.
        """
        # Penanganan jika input berupa list kalimat (batch processing)
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

        # Modus Eksperimen E0: Minimal Preprocessing (Hanya lowercase dasar)
        if not full_pipeline:
            return text.lower().strip()

        # Tentukan flag aktif berdasarkan parameter override atau default instans
        do_case = case_folding if case_folding is not None else self.case_folding
        do_punct = remove_punct if remove_punct is not None else self.remove_punctuation
        do_num = remove_num if remove_num is not None else self.remove_numbers
        do_norm = normalize if normalize is not None else self.normalization
        do_stop = remove_stop if remove_stop is not None else self.stopword_removal
        do_stem = stem if stem is not None else self.stemming

        # Fase 1: Pembersihan Karakter & Case Folding
        res = text
        if do_case:
            res = res.lower()
        if do_punct:
            res = res.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        if do_num:
            res = re.sub(r"\d+", "", res)
        res = re.sub(r"\s+", " ", res).strip()

        # Fase 2: Normalisasi Kata Gaul / Slang
        if do_norm:
            res = self.normalize_slang(res)

        # Fase 3: Penghapusan Stopwords
        if do_stop:
            res = self.remove_stopwords(res)

        # Fase 4: Stemming Kata Dasar
        if do_stem:
            res = self.stem_text(res)

        return res.strip()
