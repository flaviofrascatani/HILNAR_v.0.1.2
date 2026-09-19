### Definitions for hilnar run.py ###


# It follows as much as possible the PEP 8 convention

#The code has been developed by creating a theoretical scheme and
#after by laying down a simple algorithm.
# It was then expanded and improved (especially for the 
# computational efficiency of the calculations) by the model 
# Opus 4.6 of Anthropic after its output has been double checked.

# Comments (#) are inserted by hand, while descriptions ("""...""")
# have been automatically inserted by Opus 4.7 to save time.
# If you reader find errors or ways to ameliorate the code please 
# write to flavio.frascatani@gmail.com
# I would be glad to receive feedback :)



"""
HILNAR: distributional measures of narrative change across sectors and time.

Author:   Flavio Frascatani
Version:  0.1.2
Licence:  MIT
Cite as:  https://doi.org/10.5281/zenodo.22847128
AI use:   see README.md

Reference implementation of three indices computed on PPMI co-occurrence
vectors projected into a shared low-dimensional frame:

    NSI  Narrative Shift Index    diachronic displacement within a sector
    SAI  Sector Alignment Index   synchronic similarity to the government
    TDI  Totalitarian Deterioration Index     SAI trajectory over time

The default specification is `Config()`.  Alternative specifications
(dimension, frequency threshold, basis, rotation, vocabulary filter) are
evaluated by `specification_grid()`.

Dependencies: torch (numerics), matplotlib (figures, run.py only).
Subword segmentation, deduplication and resampling use the standard library.

Design
------
1.  Linear algebra runs in float64 through torch.linalg.
2.  Subword segmentation uses a byte-pair encoder learned on the corpus,
    parameterised by a merge budget (`n_merges`).  Diagnostics are returned
    by `BPE.report()`.
3.  `token_class` labels each token as content, function, fragment, digit
    or punctuation.  All classes enter the co-occurrence context;
    `vocab_filter` selects which classes enter the indices.
4.  Each index returns a `Weighted` record with per-token contributions and
    the Kish effective sample size.

Notation
--------
Sectors s in {g, m, e, p} (government, media, education, popular culture);
time slices t, t'; v(w,s,t) is the projected vector of token w in slice
(s, t).  Sums run over tokens present in both slices being compared.

    wt(w,s,t)   = log(1 + count(w,s,t))
    delta(w)    = (1 - cos(v(w,s,t), v(w,s,t'))) / 2                in [0, 1]
    NSI(s,t,t') = sum_w wt(w,s,t) delta(w) / sum_w wt(w,s,t)
    NSI(t,t')   = sum_s log(1 + n_docs(s,t)) NSI(s,t,t')
                  / sum_s log(1 + n_docs(s,t))
    SAI(s,t)    = sum_w wt(w,s,t) cos(v(w,s,t), v(w,g,t))
                  / sum_w wt(w,s,t)                                 in [-1, 1]
    TDI(s)      = ( SAI(s,t) )_t
    TDI(t)      = mean of SAI(s,t) over s in {m, e, p}
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch

torch.set_default_dtype(torch.float64)

CARRIERS = ("g", "m", "e", "p")
CARRIER_NAMES = {
    "g": "government",
    "m": "media",
    "e": "education",
    "p": "popular culture",
}
GOVERNMENT = "g"
NON_GOV = ("m", "e", "p")

_PUNCT_CATS = {"P", "S", "Z", "C"}
#Punctuation, symbols, Separators and other (C)

# Closed-class Russian/Belarusian function words.  Used only for classifying
# tokens, never for deleting them from the co-occurrence context.
FUNCTION_WORDS = frozenset("""
и в на с о по не что а но как к за из у от для это же то так вы мы я он она они
его ее их все был была было были быть есть или при до над под между через во со
об ко нас вам нам вас себя свой своя свое свои своей своего своих который
которая которые когда где еще уже только даже тоже там тут кто чем чтобы если
хотя тем том этом этой этого этих ним них ему ей ими бы ли ни без про
после перед вместе также более менее очень тот та те такой такая такие сам
сама сами весь вся всю всех всем чего чему кого кому мой моя мои наш наша наши
ваш ваша ваши да нет ну вот але і у ў з ад да не як што каб жа так гэта гэты
гэтая гэтыя мы вы яны ён яна яго яе іх быў была было былі ёсць або пры да над
пад паміж праз ва са аб ка нас вам нам вас сябе свой свая сваё свае
""".split())










# I normalisation, deduplication I











def normalize(text: str) -> str:
    """NFC, casefold, collapse whitespace, deterministic and idempotent."""
    text = unicodedata.normalize("NFC", text).casefold().strip()                       
    return re.sub(r"\s+", " ", text)
# NFC= Normalization (Composed, No Compatibility Mapping.) 
# No NFD or NFKC or NFKD. NFC: "e" + ◌́ → "é"
# NFC, casefold, collapse whitespace.  
# Deterministic thanks to .casefold() : (Majuscules become minuscules)
# Idempotent (re-apply the function again and again is ok).
# .strip() takes the space at "borders" while .sub() the remaining


def _shingles(text: str, k: int = 3) -> frozenset:
    w = text.split()
    if len(w) < k:
        return frozenset({" ".join(w)})
    return frozenset(" ".join(w[i:i + k]) for i in range(len(w) - k + 1))
# k is an optional parameter
# if len(w) < k, the set is empty so better for to return " ".join(w)
# frozenset for a stable hash and not a list
# moreover without frozenset order wouldn't matter
# k=3 for natural language, otherwise it must be substituted with 
# 5-6 for long texts


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
# important for taking out copies of exact same texts after,
# mainly in case of web scraping
# in case both sets are empty, the return is 1.0 (impossible to do 0/0)


def deduplicate(docs, jaccard_thr: float = 0.9):
    """Exact dedup (SHA-256 of normalised text) then near-dedup (Jaccard on
    3-word shingles), applied globally.

    Returns (kept, report).  `report["borderline"]` lists every pair whose
    Jaccard falls in [thr - 0.15, thr), i.e. the pairs a human must inspect to
    calibrate `jaccard_thr`.  The threshold is a reported parameter: with 24
    documents no pair is near it, so the value is inert here, but on a scraped
    corpus it is the single most consequential preprocessing choice.
    """
    report = {"exact_removed": 0, "near_removed": 0, "borderline": [],
              "killed": [], "jaccard_thr": jaccard_thr}
    kept, seen = [], set()
    streams = defaultdict(list)
    for d in docs:
        norm = normalize(d["text"])
        h = hashlib.sha256(norm.encode()).hexdigest()
        if h in seen:
            report["exact_removed"] += 1
            continue
        sh = _shingles(norm)
        key = (d["carrier"], d["year"])
        dup = False
        pending = [] 
        for prev_i, prev_sh in streams[key]:
            j = jaccard(sh, prev_sh)
            if j >= jaccard_thr:
                dup = True
                report["killed"].append({"stream": f"{key[0]}-{key[1]}",
                                         "against": prev_i,
                                         "jaccard": round(j, 4)})
                break
            if j >= jaccard_thr - 0.15:
                pending.append({"stream": f"{key[0]}-{key[1]}",
                                "against": prev_i,
                                "jaccard": round(j, 4)})
        if dup:
            report["near_removed"] += 1
            continue

        idx = len(kept) 
        for p in pending:
            p["doc"] = idx 
        report["borderline"].extend(pending)

        seen.add(h)
        streams[key].append((len(kept), sh))
        kept.append({**d, "norm": norm})
    return kept, report
# jaccard_thr = 0.9 can be changed
# for now left as standard for possible web scraping
# sha256(...).hexdigest() used to save memory
# first document that remains becomes the reèresentative of the group
# the rest are eliminated.
# the results and therefore the docs selected depend on the input order
# if jaccard_thr = [0.75, 0.9)
# pending added to know both the documents in case of a borderline case












# II subword segmentation (byte-pair encoding, standard library only) II












_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# compile a word or a punctuation mark
# re.UNICODE for not latin characters.

def pretokenize(text: str) -> list:
    """Split on whitespace, keeping punctuation as separate units."""
    return _WORD_RE.findall(text)
# _WORD_RE.findall uses the work of re.compile (made only one time)

class BPE:
    """Byte-pair encoding learned from a corpus."""
    # Parameterised by a merge budget not a vocabulary ceiling.  A
    #vocabulary ceiling larger than the corpus type count could disable
    #segmentation; a merge budget cannot.`min_pair_freq` stops 
    # merging once pairs are too rare to estimate, which is what
    # keeps the segmentation stable on small corpora.


    def __init__(self, n_merges: int = 1000, min_pair_freq: int = 2):
        self.n_merges = n_merges
        self.min_pair_freq = min_pair_freq
        self.merges = []
        self.ranks = {}
        self._cache = {}
    # int = 1000 can be changed (1000 for small corpora)
    # min_pair_freq: int = 2 to distinguish the unique from recurrent
    # self.merges and self.marks are important after to save memory
    # list inside __init__ so new list for each object

    def fit(self, texts):
        words = Counter()
        for t in texts:
            words.update(pretokenize(t))
        seqs = {w: tuple(w) for w in words}
        counts = {w: words[w] for w in words}

#Example: seqs   = {'gatto': ('g','a','t','t','o'),
        #           'gatta': ('g','a','t','t','a'),
        #            'gatti': ('g','a','t','t','i')}
        #counts = {'gatto': 2, 'gatta': 1, 'gatti': 1}

        for _ in range(self.n_merges):
            pairs = Counter()
            for w, s in seqs.items():
                c = counts[w]
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            # deterministic tie-break: highest count, then lexicograph
            # otherwise no reproducibility
            best = max(pairs, key=lambda p: (pairs[p], p))
            if pairs[best] < self.min_pair_freq:
                break
            self.merges.append(best)
            joined = best[0] + best[1]
            for w, s in seqs.items():
                if len(s) < 2:
                    continue
                out, i = [], 0
                while i < len(s):
                    if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                        out.append(joined)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                seqs[w] = tuple(out)
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self._cache.clear()
        return self
        
        # cache is cleared in the end


    def _segment_word(self, word: str):
        if word in self._cache:
            return self._cache[word]
        s = tuple(word)
        while len(s) > 1:
            cand = [(self.ranks[p], i) for i, p in enumerate(zip(s, s[1:]))
                    if p in self.ranks]
            if not cand:
                break
            _, i = min(cand)
            s = s[:i] + (s[i] + s[i + 1],) + s[i + 2:]
        self._cache[word] = s
        return s

    # tuples are hashable, that is why (s[i] ... ,)
    # ._cache to note the result (and avoid more computation)
    # return s creates the _method chaining_ ()
                




    def encode(self, text: str) -> list:
        """Return subword pieces; continuations are prefixed '##'."""
        out = []
        for word in pretokenize(text):
            pieces = self._segment_word(word)
            out.append(pieces[0])
            out.extend("##" + p for p in pieces[1:])
        return out

    def report(self, docs) -> dict:
        """Segmentation diagnostics.  `continuation_rate` > 0 is the evidence
        that the pipeline is genuinely operating on subwords."""
        toks = [t for d in docs for t in d["tokens"]]
        types = Counter(toks)
        cont = sum(1 for t in types if t.startswith("##"))
        return {"n_merges_learned": len(self.merges),
                "n_tokens": len(toks),
                "n_types": len(types),
                "continuation_types": cont,
                "continuation_rate": round(cont / max(1, len(types)), 4),
                "continuation_token_share": round(
                    sum(c for t, c in types.items() if t.startswith("##"))
                    / max(1, len(toks)), 4),
                "mean_pieces_per_word": round(
                    len(toks) / max(1, sum(len(pretokenize(d["norm"]))
                                           for d in docs)), 3)}
    # n_merges_learned to understand whether to change n_merges=1000
    # continuation_types= how many types start with ##
    #continuation_rate = rate of the vocabulary with continuations
    #continuation_token_share= rate of the text with continuations
    # rounding at 4 digits after comma


def tokenize(docs, bpe: BPE):
    for d in docs:
        d["tokens"] = bpe.encode(d["norm"])
    return docs













# III token classification and the analysis vocabulary III












MIN_CONTENT_CHARS = 3


def token_class(tok: str) -> str:
    """'punct' | 'digit' | 'function' | 'fragment' | 'content'.

    'fragment' is a piece too short to carry lexical meaning (< 3 characters).
    On a corpus of this size byte-pair merging saturates while many pieces are
    still single letters, and a single Cyrillic letter has a co-occurrence
    profile driven by orthography rather than by narrative content.  Marking
    them as a class (instead of letting them into the indices)
    is what makes the subword level auditable.
    """
    cont = tok.startswith("##")
    base = tok[2:] if cont else tok
    if not base:
        return "punct"
    if all(unicodedata.category(c)[0] in _PUNCT_CATS for c in base):
        return "punct"
    if base.isdigit():
        return "digit"
    if not cont and base in FUNCTION_WORDS:
        return "function"
    if len(base) < MIN_CONTENT_CHARS:
        return "fragment"
    return "content"
# in this way the remaining types are categorized


VOCAB_FILTERS = {
    "all": lambda k: True,
    "no_punct": lambda k: k != "punct",
    "lexical": lambda k: k in ("content", "function"),
    "content": lambda k: k == "content",
}
# better a vocabulary so that the categorizations are put as data


def build_vocab(docs, min_count: int = 2, vocab_filter: str = "no_punct"):
    """Analysis vocabulary: types with corpus frequency >= min_count passing
    `vocab_filter`.  Returns (vocab, composition)."""
    counts = Counter(t for d in docs for t in d["tokens"])
    keep = VOCAB_FILTERS[vocab_filter]
    vocab = sorted(t for t, c in counts.items() if c >= min_count
                   and keep(token_class(t)))
    comp = Counter(token_class(t) for t in vocab)
    return vocab, {"min_count": min_count, "vocab_filter": vocab_filter,
                   "size": len(vocab), "composition": dict(comp)}
# sorted() orders and gives a list so that it is deterministic 
# and reproducible
# min_count kept in order to point at rare tokens
# also it must be considered that the hapax legomena are half of
# the vocabulary












# IV embedding: PPMI co-occurrence, one space per (carrier, year) IV













@dataclass
class Slice:
    carrier: str
    year: int
    words: list
    matrix: torch.Tensor              # (n_words, n_context)
    freqs: dict
    n_docs: int
    index: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {w: i for i, w in enumerate(self.words)}


def ppmi_slices(docs, vocab, window: int = 5, context_vocab=None):
    """Positive PMI co-occurrence rows, one matrix per (carrier, year).

    `vocab` selects the ROWS (the tokens that will be compared).
    `context_vocab` selects the COLUMNS. defaults to the full corpus
    vocabulary so that function words and punctuation still contribute
    distributional context even when excluded from the indices.
    """
    if context_vocab is None:
        context_vocab = sorted({t for d in docs for t in d["tokens"]})
    cidx = {w: i for i, w in enumerate(context_vocab)}
    ridx = {w: i for i, w in enumerate(vocab)}
    V, C = len(vocab), len(context_vocab)

    streams = defaultdict(list)
    for d in docs:
        streams[(d["carrier"], d["year"])].append(d["tokens"])

    out = {}
    for (carrier, year), token_lists in sorted(streams.items()):
        co = torch.zeros((V, C))
        freqs = Counter()
        for toks in token_lists:
            ids = [(ridx.get(t), cidx.get(t)) for t in toks]
            for t in toks:
                if t in ridx:
                    freqs[t] += 1
            n = len(ids)
            for i, (ri, _) in enumerate(ids):
                if ri is None:
                    continue     #It is a sliding window
                lo, hi = max(0, i - window), min(n, i + window + 1)
                for j in range(lo, hi):
                    cj = ids[j][1]
                    if j != i and cj is not None:
                        co[ri, cj] += 1.0
        total = co.sum()
        if total == 0 or not freqs:
            continue
        row = co.sum(1, keepdim=True)
        col = co.sum(0, keepdim=True)
        denom = row @ col
        pmi = torch.where(
            (co > 0) & (denom > 0), 
            #remember it cannot be used and instead of &
            torch.log(co.clamp(min=1e-300) * total / denom.clamp(min=1e-300)),
            torch.zeros(()),
        )
        ppmi = pmi.clamp(min=0.0)
        words = [w for w in vocab if freqs.get(w, 0) > 0]
        rows = torch.stack([ppmi[ridx[w]] for w in words])
        out[(carrier, year)] = Slice(carrier, year, words, rows,
                                     dict(freqs), len(token_lists))
    return out

#PMI when it is negative it is not reliable as they measure how small
#is the corpus, not the language itself. So PPMI
# Bullinaria & Levy 2007 show it is an acceptable cost to be paid
# PPMI is very high for rare people when the link between two words
# appears just 2 times !!!!!!!! (Need for a greater corpus)
#From there no more strings but tensors
#SVD (Singular value decomposition) used and Gram Matrix













# V projection and alignment V 















def procrustes(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Orthogonal R minimising ||A R - B||_F  (replaces
    scipy.linalg.orthogonal_procrustes).  R = U V^T from SVD of A^T B."""
    U, _, Vh = torch.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vh
#According to Schonemann (1966) the optimal rotation is R=UV^T
#for procrustes alignment 




def _top_right_singular(X: torch.Tensor, k: int) -> torch.Tensor:
    """First `k` right singular vectors of X, as columns of a (p, k) matrix.

    Obtained from the eigendecomposition of the Gram matrix X^T X rather than
    from a full SVD of X.  The right singular vectors of X are the
    eigenvectors of X^T X, so this is exact, not an approximation; it merely
    avoids forming the left singular vectors, which the projection never uses.
    On the prototype corpus this is ~40x faster, which is what
    makes 1000-replicate resampling feasible on a laptop.

    Signs are fixed by requiring the largest-magnitude entry of each vector to
    be positive, so repeated runs give the same bases.
    """
    G = X.T @ X        
    evals, evecs = torch.linalg.eigh(G)   
    idx = torch.argsort(evals, descending=True)[:k]
    B = evecs[:, idx]
    pivot = B.abs().argmax(dim=0)
    signs = torch.sign(B[pivot, torch.arange(B.shape[1])])
    signs[signs == 0] = 1.0
    return B * signs
 # ascending order
 # (p, p), symmetric PSD

def project(slices, standard_key, dim: int = 20, basis: str = "pooled",
            rotate: bool = False, center: bool = False):
    """Map every slice into one common `dim`-dimensional frame.

    basis  : 'pooled'   truncated-SVD basis from all slices stacked.
             'standard'  basis from the standard slice only (the specification
                         printed in the thesis methods chapter).
    rotate : per-slice orthogonal Procrustes onto the standard slice.  The
             thesis methods chapter prescribes this.  It is OFF by default
             here because with a shared basis all slices are already
             co-framed, and rotating each slice onto the standard slice
             removes exactly the directional drift that the NSI measures --
             on the prototype corpus it reverses the sign of the TDI change.
             Both settings must be reported.
    center : global mean-centring (isotropy correction).  Recommended for
             contextual encoders, off for PPMI.

    `basis` and `rotate` are the two specification choices that change the
    published conclusion, so they are explicit arguments -- see
    `specification_grid()`. See also README for further explanation.
    """
    keys = sorted(slices)
    if basis == "standard":
        src = slices[standard_key].matrix
    elif basis == "pooled":
        src = torch.cat([slices[k].matrix for k in keys], 0)
    else:
        raise ValueError(f"basis must be 'pooled' or 'standard', got {basis!r}")

    mu = src.mean(0, keepdim=True)
    k = min(dim, min(src.shape) - 1)
    B = _top_right_singular(src - mu, k)

    reduced = {key: ((slices[key].matrix - mu) @ B) for key in keys}

    if rotate:
        std = slices[standard_key]
        std_red = reduced[standard_key]
        for key in keys:
            if key == standard_key:
                continue
            sl = slices[key]
            shared = [w for w in sl.words if w in std.index]
            if len(shared) < 2:
                continue
            A = reduced[key][[sl.index[w] for w in shared]]
            Bm = std_red[[std.index[w] for w in shared]]
            reduced[key] = reduced[key] @ procrustes(A, Bm)

    if center:
        gmu = torch.cat([reduced[k] for k in keys], 0).mean(0, keepdim=True)
        reduced = {k: v - gmu for k, v in reduced.items()}

    return reduced












# VI indices (NSI, SAI, TDI) VI
















@dataclass
class Weighted:
    """A log-frequency-weighted mean, with its provenance."""
    value: float
    numerator: float
    denominator: float
    n_shared: int
    contributions: list = field(default_factory=list) 
 # (token, stat, weight)
 
    @property
    def effective_n(self) -> float:
        """Kish effective sample size of the weight vector."""
        if not self.contributions:
            return 0.0
        w = torch.tensor([c[2] for c in self.contributions])
        return float(w.sum() ** 2 / (w.pow(2).sum() + 1e-300))
    # Kish effective sample size of the weight vector.


def log_weight(n: int) -> float:
    return math.log1p(n)
#just a function for log
#PPMI overestimate rare words but they have low weight for log.



def _cosines(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity, zero where a row has zero norm."""
    na = A.norm(dim=1)
    nb = B.norm(dim=1)
    ok = (na > 0) & (nb > 0)
    out = torch.zeros(A.shape[0])
    out[ok] = ((A[ok] * B[ok]).sum(1) / (na[ok] * nb[ok]))
    return out.clamp(-1.0, 1.0)
#It calculates the cosine similarity only of (for example row 0 of A
#with row 0 of B) as A and B contain the same words in diff situations



def _weighted(tokens, stats: torch.Tensor, weights: torch.Tensor) -> Weighted:
    den = float(weights.sum())
    num = float((weights * stats).sum())
    return Weighted(num / den if den else float("nan"), num, den, len(tokens),
                    list(zip(tokens, stats.tolist(), weights.tolist())))


def nsi_sector(reduced, slices, carrier, t, tp) -> Weighted:
    """Diachronic narrative shift of one carrier, t -> tp.  In [0, 1]."""
    a, b = (carrier, t), (carrier, tp)
    if a not in slices or b not in slices:
        return Weighted(float("nan"), 0.0, 0.0, 0)
    sa, sb = slices[a], slices[b]
    shared = [w for w in sa.words if w in sb.index]
    if not shared:
        return Weighted(float("nan"), 0.0, 0.0, 0)
    A = reduced[a][[sa.index[w] for w in shared]]
    B = reduced[b][[sb.index[w] for w in shared]]
    delta = (1.0 - _cosines(A, B)) / 2.0
    wt = torch.tensor([log_weight(sa.freqs.get(w, 0)) for w in shared])
    return _weighted(shared, delta, wt)
#see the research paper


def nsi_state(reduced, slices, t, tp):
    """Document-weighted mean NSI over carriers present in both slices."""
    per, num, den = {}, 0.0, 0.0
    for c in CARRIERS:
        if (c, t) in slices and (c, tp) in slices:
            r = nsi_sector(reduced, slices, c, t, tp)
            per[c] = r
            if not math.isnan(r.value):
                w = log_weight(slices[(c, t)].n_docs)
                num += w * r.value
                den += w
    return (num / den if den else float("nan")), per
#see the research paper (also for sai and tdi)


def sai(reduced, slices, sector, t, reference: str = GOVERNMENT) -> Weighted:
    """Synchronic alignment of `sector` with `reference` at t.  In [-1, 1].

    Not symmetric: weights come from the first argument's corpus.
    """
    a, b = (sector, t), (reference, t)
    if a not in slices or b not in slices:
        return Weighted(float("nan"), 0.0, 0.0, 0)
    sa, sb = slices[a], slices[b]
    shared = [w for w in sa.words if w in sb.index]
    if not shared:
        return Weighted(float("nan"), 0.0, 0.0, 0)
    A = reduced[a][[sa.index[w] for w in shared]]
    B = reduced[b][[sb.index[w] for w in shared]]
    cos = _cosines(A, B)
    wt = torch.tensor([log_weight(sa.freqs.get(w, 0)) for w in shared])
    return _weighted(shared, cos, wt)


def tdi(reduced, slices, years, sectors=NON_GOV):
    """Per-sector SAI trajectories and the composite mean at each year."""
    traj = {s: {y: sai(reduced, slices, s, y) for y in years} for s in sectors}
    comp = {}
    for y in years:
        vals = [traj[s][y].value for s in sectors
                if not math.isnan(traj[s][y].value)]
        comp[y] = sum(vals) / len(vals) if vals else float("nan")
    return traj, comp
















# VII clustering (replaces sklearn AgglomerativeClustering) VII













def average_linkage(D: torch.Tensor, n_clusters: int) -> list:
    """Average-linkage agglomerative clustering on a square distance matrix.
    Returns a flat label vector.  Equivalent to
    sklearn.cluster.AgglomerativeClustering(linkage='average',
    metric='precomputed')."""
    n = D.shape[0]
    clusters = {i: [i] for i in range(n)}
    D = D.clone()
    D.fill_diagonal_(float("inf"))
    sizes = {i: 1 for i in range(n)}
    while len(clusters) > max(1, n_clusters):
        keys = sorted(clusters)
        sub = D[keys][:, keys]
        flat = int(torch.argmin(sub))
        i, j = keys[flat // len(keys)], keys[flat % len(keys)]
        ni, nj = sizes[i], sizes[j]
        for k in keys:
            if k in (i, j):
                continue
            d = (ni * D[i, k] + nj * D[j, k]) / (ni + nj)
            D[i, k] = D[k, i] = d
        clusters[i] = clusters[i] + clusters[j]
        sizes[i] = ni + nj
        del clusters[j], sizes[j]
        D[j, :] = float("inf")
        D[:, j] = float("inf")
    labels = [0] * n
    for lab, (_, members) in enumerate(sorted(clusters.items())):
        for m in members:
            labels[m] = lab
    return labels


def cluster_shifts(reduced, slices, carrier, t, tp, n_clusters: int = 5):
    """Cohesion C(K) and mean displacement of each cluster of shared tokens.
    High cohesion with high mean displacement indicates a conceptual field
    moving as a block rather than isolated token drift."""
    a, b = (carrier, t), (carrier, tp)
    sa, sb = slices[a], slices[b]
    shared = [w for w in sa.words if w in sb.index]
    if len(shared) < n_clusters:
        return []
    A = reduced[a][[sa.index[w] for w in shared]]
    B = reduced[b][[sb.index[w] for w in shared]]
    delta = (1.0 - _cosines(A, B)) / 2.0
    An = A / A.norm(dim=1, keepdim=True).clamp(min=1e-300)
    S = (An @ An.T).clamp(-1.0, 1.0)
    labels = average_linkage(1.0 - S, n_clusters)
    out = []
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        if len(idx) < 2:
            continue
        sub = S[idx][:, idx]
        m = len(idx)
        cohesion = float((sub.sum() - m) / (m * (m - 1)))
        out.append({"tokens": [shared[i] for i in idx],
                    "size": m,
                    "cohesion": round(cohesion, 4),
                    "mean_delta": round(float(delta[idx].mean()), 4)})
    return sorted(out, key=lambda c: -c["mean_delta"])













# VIII pipeline VIII















@dataclass
class Config:
    dim: int = 20
    min_count: int = 2
    window: int = 5
    n_merges: int = 1000
    vocab_filter: str = "content"
    basis: str = "pooled"
    rotate: bool = False
    center: bool = False
    standard: tuple = ("g", 2019)
    jaccard_thr: float = 0.9

    def as_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


def prepare(documents, cfg: Config):
    """Deduplicate, fit the BPE and segment.  Done once; the resampling
    routines reuse the result so that the null distributions differ from the
    observed statistic only in the labels, never in the segmentation."""
    docs, dedup = deduplicate(documents, cfg.jaccard_thr)
    bpe = BPE(n_merges=cfg.n_merges).fit([d["norm"] for d in docs])
    docs = tokenize(docs, bpe)
    return docs, {"dedup": dedup, "segmentation": bpe.report(docs)}


def embed(docs, cfg: Config):
    """Vocabulary -> PPMI slices -> shared reduced frame."""
    vocab, vinfo = build_vocab(docs, cfg.min_count, cfg.vocab_filter)
    context = sorted({t for d in docs for t in d["tokens"]})
    slices = ppmi_slices(docs, vocab, cfg.window, context)
    if cfg.standard not in slices:
        raise ValueError(f"standard slice {cfg.standard} absent from corpus")
    reduced = project(slices, cfg.standard, cfg.dim, cfg.basis,
                      cfg.rotate, cfg.center)
    return reduced, slices, vinfo


def analyse(documents, cfg: Config = None, years=(2019, 2025)):
    """Full observed analysis.  Returns a plain dict of results."""
    cfg = cfg or Config()
    docs, prep = prepare(documents, cfg)
    reduced, slices, vinfo = embed(docs, cfg)
    t, tp = years
    state, per = nsi_state(reduced, slices, t, tp)
    traj, comp = tdi(reduced, slices, years)
    return {"config": cfg.as_dict(), "preprocessing": prep, "vocab": vinfo,
            "docs_per_slice": {f"{c}-{y}": s.n_docs
                               for (c, y), s in sorted(slices.items())},
            "nsi": per, "nsi_state": state, "sai": traj, "tdi": comp,
            "_reduced": reduced, "_slices": slices, "_docs": docs}

















# IX resampling inference IX















def _p_add_one(null, obs: float, tail: str) -> float:
    """(1 + #{as extreme}) / (1 + B).  The add-one correction keeps p strictly
    positive and is the standard estimator for a Monte-Carlo permutation
    p-value (Davison & Hinkley 1997, sec. 4.2)."""
    if math.isnan(obs) or not null:
        return float("nan")
    if tail == "greater":
        k = sum(1 for v in null if not math.isnan(v) and v >= obs)
    else:
        k = sum(1 for v in null if not math.isnan(v) and abs(v) >= abs(obs))
    return (1 + k) / (1 + len(null))


def permutation_nsi(documents, cfg: Config = None, years=(2019, 2025),
                    n_perm: int = 2000, seed: int = 0):
    """Exchangeability: within a carrier, year labels carry no information.
    Shuffling year labels within each carrier preserves carrier vocabulary and
    document count, so the null isolates the diachronic signal."""
    cfg = cfg or Config()
    docs, _ = prepare(documents, cfg)
    reduced, slices, _ = embed(docs, cfg)
    t, tp = years
    obs = {c: nsi_sector(reduced, slices, c, t, tp).value for c in CARRIERS}
    obs_state, _ = nsi_state(reduced, slices, t, tp)

    rng = random.Random(seed)
    null = {c: [] for c in CARRIERS}
    null_state = []
    for _ in range(n_perm):
        shuffled = []
        for c in CARRIERS:
            grp = [d for d in docs if d["carrier"] == c]
            ys = [d["year"] for d in grp]
            rng.shuffle(ys)
            shuffled += [{**d, "year": y} for d, y in zip(grp, ys)]
        try:
            r2, s2, _ = embed(shuffled, cfg)
            st, per = nsi_state(r2, s2, t, tp)
        except (ValueError, RuntimeError):
            continue
        null_state.append(st)
        for c in CARRIERS:
            null[c].append(per[c].value if c in per else float("nan"))

    out = {}
    for c in list(CARRIERS) + ["state"]:
        o = obs_state if c == "state" else obs[c]
        nl = [v for v in (null_state if c == "state" else null[c])
              if not math.isnan(v)]
        out[c] = {"observed": o,
                  "null_mean": sum(nl) / len(nl) if nl else float("nan"),
                  "null_sd": _sd(nl), "n_valid": len(nl),
                  "p_one_sided": _p_add_one(nl, o, "greater")}
    return out


def permutation_sai(documents, year, cfg: Config = None, n_perm: int = 2000,
                    seed: int = 1):
    """Exchangeability: within a year, the three non-governmental carrier
    labels are interchangeable.  Government documents are held fixed because
    they define the reference space."""
    cfg = cfg or Config()
    docs, _ = prepare(documents, cfg)
    reduced, slices, _ = embed(docs, cfg)
    obs = {s: sai(reduced, slices, s, year).value for s in NON_GOV}

    pool = [d for d in docs if d["carrier"] != GOVERNMENT and d["year"] == year]
    rest = [d for d in docs if not (d["carrier"] != GOVERNMENT
                                    and d["year"] == year)]
    labels = [d["carrier"] for d in pool]
    rng = random.Random(seed)
    null = {s: [] for s in NON_GOV}
    for _ in range(n_perm):
        perm = labels[:]
        rng.shuffle(perm)
        shuffled = rest + [{**d, "carrier": c} for d, c in zip(pool, perm)]
        try:
            r2, s2, _ = embed(shuffled, cfg)
        except (ValueError, RuntimeError):
            continue
        for s in NON_GOV:
            null[s].append(sai(r2, s2, s, year).value)

    out = {}
    for s in NON_GOV:
        valid = [x for x in null[s] if not math.isnan(x)]
        out[s] = {"observed": obs[s],
                  "null_mean": sum(valid) / len(valid) if valid else float("nan"),
                  "null_sd": _sd(valid),
                  "n_valid": len(valid),
                  "p_two_sided": _p_add_one(valid, obs[s], "two")}
    return out


def bootstrap(documents, cfg: Config = None, years=(2019, 2025),
              n_boot: int = 2000, seed: int = 2):
    """Percentile bootstrap over DOCUMENTS within each (carrier, year) stream.

    With three documents per stream the permutation test has almost no power,
    so the honest summary of uncertainty is an interval, not a p-value.  The
    bootstrap resamples the sampling unit that actually varies -- the
    document -- and so quantifies how much of each index is an artefact of
    which three excerpts happened to be retrievable.
    """
    cfg = cfg or Config()
    docs, _ = prepare(documents, cfg)
    streams = defaultdict(list)
    for d in docs:
        streams[(d["carrier"], d["year"])].append(d)

    rng = random.Random(seed)
    acc = defaultdict(list)
    t, tp = years
    for _ in range(n_boot):
        resampled = []
        for key, grp in streams.items():
            resampled += [rng.choice(grp) for _ in grp]
        try:
            r2, s2, _ = embed(resampled, cfg)
            st, per = nsi_state(r2, s2, t, tp)
            traj, comp = tdi(r2, s2, years)
        except (ValueError, RuntimeError):
            continue
        acc["nsi_state"].append(st)
        for c in CARRIERS:
            acc[f"nsi_{c}"].append(per[c].value if c in per else float("nan"))
        for s in NON_GOV:
            for y in years:
                acc[f"sai_{s}_{y}"].append(traj[s][y].value)
        for y in years:
            acc[f"tdi_{y}"].append(comp[y])
        acc["tdi_delta"].append(comp[years[1]] - comp[years[0]])

    return {k: _percentiles(v) for k, v in sorted(acc.items())}


def specification_grid(documents, years=(2019, 2025), dims=(8, 12, 20, 30, 50),
                       min_counts=(2, 3), bases=("pooled", "standard"),
                       rotations=(False, True),
                       filters=("content", "lexical", "no_punct")):
    """Every reported index under the full cross of specification choices.

    This is the central robustness object: a conclusion that survives the grid
    is a finding, one that does not is a specification artefact.  On the
    prototype corpus the sign of the TDI change is not invariant to `rotate`.
    """
    rows = []
    # Segmentation depends only on (n_merges, jaccard_thr), neither of which
    # the grid varies, so it is fitted once and reused: every cell then differs
    # from every other only in the specification under test.
    _prepared = {}
    for vf in filters:
        for dim in dims:
            for mc in min_counts:
                for basis in bases:
                    for rot in rotations:
                        cfg = Config(dim=dim, min_count=mc, basis=basis,
                                     rotate=rot, vocab_filter=vf)
                        try:
                            ck = (cfg.n_merges, cfg.jaccard_thr)
                            if ck not in _prepared:
                                _prepared[ck] = prepare(documents, cfg)[0]
                            docs = _prepared[ck]
                            red, sl, vinfo = embed(docs, cfg)
                            st, per = nsi_state(red, sl, *years)
                            traj, comp = tdi(red, sl, years)
                        except (ValueError, RuntimeError):
                            continue
                        rows.append({
                            "vocab_filter": vf, "dim": dim, "min_count": mc,
                            "basis": basis, "rotate": rot,
                            "vocab_size": vinfo["size"],
                            "nsi_state": st,
                            **{f"nsi_{c}": per[c].value for c in CARRIERS
                               if c in per},
                            **{f"sai_{s}_{y}": traj[s][y].value
                               for s in NON_GOV for y in years},
                            "tdi_2019": comp[years[0]],
                            "tdi_2025": comp[years[1]],
                            "tdi_delta": comp[years[1]] - comp[years[0]],
                            "nsi_rank": ">".join(
                                sorted(per, key=lambda c: -per[c].value)),
                        })
    return rows
#this is done in order to address the problem of the "garden of the 
#forking paths" (Gelman & Loken 2013) and the study of (Silberzahn et
# al. 2015) where the same dataset was given to 29 different research 
#groups and the results went from no effect to great effect
# (Simonsohn, Simmons & Nelson, Specification Curve Analysis, 2020)


def weight_audit(reduced, slices, years=(2019, 2025)):
    """Share of each index's total weight carried by non-content tokens.

    The prototype's indices were dominated by punctuation and function words;
    this makes that share a reported quantity rather than something a reader
    has to rediscover.
    """
    out = {}
    def row(r):
        tot = sum(w for _, _, w in r.contributions)
        nc = sum(w for tk, _, w in r.contributions
                 if token_class(tk) != "content")
        return {"n_shared": r.n_shared,
                "n_content": sum(1 for tk, _, _ in r.contributions
                                 if token_class(tk) == "content"),
                "non_content_weight_share": round(nc / tot, 4) if tot else None,
                "effective_n": round(r.effective_n, 2)}
    for s in NON_GOV:
        for y in years:
            r = sai(reduced, slices, s, y)
            if r.contributions:
                out[f"sai_{s}_{y}"] = row(r)
    for c in CARRIERS:
        r = nsi_sector(reduced, slices, c, *years)
        if r.contributions:
            out[f"nsi_{c}"] = row(r)
    return out
#just for the prototype's indices

def top_movers(reduced, slices, carrier, years=(2019, 2025), k: int = 10):
    """Tokens contributing most to a carrier's NSI (weight x displacement)."""
    r = nsi_sector(reduced, slices, carrier, *years)
    rows = [{"token": t, "delta": round(d, 4), "log_weight": round(w, 4),
             "contribution": round(d * w, 4), "class": token_class(t)}
            for t, d, w in r.contributions]
    return sorted(rows, key=lambda x: -x["contribution"])[:k]



# small statistical helpers (stdlib)


def _sd(xs) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _percentiles(xs, qs=(2.5, 50, 97.5)) -> dict:
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return {"n": 0}
    def q(p):
        if len(xs) == 1:
            return xs[0]
        i = (len(xs) - 1) * p / 100.0
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
    return {"n": len(xs), "mean": sum(xs) / len(xs),
            "lo95": q(qs[0]), "median": q(qs[1]), "hi95": q(qs[2])}
