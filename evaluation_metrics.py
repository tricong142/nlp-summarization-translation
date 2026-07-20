"""
evaluation_metrics.py  ─ v6
==============================
Full evaluation suite:
ROUGE-1/2/L + BLEU + METEOR + BERTScore + Coverage + Redundancy
Báo cáo có cột Δ (BART - Lead3).
"""
import re
import math
import warnings
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import evaluate as hf_evaluate
    HF_EVAL = True
except ImportError:
    HF_EVAL = False

try:
    from nltk.translate.bleu_score import (
        sentence_bleu, corpus_bleu, SmoothingFunction)
    from nltk.tokenize import word_tokenize
    NLTK_OK = True
except ImportError:
    NLTK_OK = False

# ── ROUGE ─────────────────────────────────────────────────────── #
class RougeCalculator:
    def tokenize(self, text: str) -> List[str]:
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    def get_ngrams(self, tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i: i + n])
                       for i in range(len(tokens) - n + 1))

    def overlap(self, gen: List[str], ref: List[str], n: int) -> Dict:
        g, r    = self.get_ngrams(gen, n), self.get_ngrams(ref, n)
        ov      = sum((g & r).values())
        gc, rc  = sum(g.values()), sum(r.values())
        p = ov / gc if gc else 0.0
        r_ = ov / rc if rc else 0.0
        f  = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        return {"precision": round(p, 4),
                "recall":    round(r_, 4),
                "f1":        round(f, 4)}

    def lcs_length(self, x: List[str], y: List[str]) -> int:
        m, n       = len(x), len(y)
        prev, curr = [0] * (n + 1), [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                curr[j] = (prev[j - 1] + 1 if x[i - 1] == y[j - 1]
                           else max(curr[j - 1], prev[j]))
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def rouge_l(self, gen: List[str], ref: List[str]) -> Dict:
        l   = self.lcs_length(gen, ref)
        p   = l / len(gen) if gen else 0.0
        r   = l / len(ref) if ref else 0.0
        b   = 1.2
        f   = (1 + b**2) * p * r / (b**2 * p + r) if p + r else 0.0
        return {"precision": round(p, 4),
                "recall":    round(r, 4),
                "f1":        round(f, 4)}

    def compute_all_rouge(self, gen: str, ref: str) -> Dict:
        g, r = self.tokenize(gen), self.tokenize(ref)
        return {
            "rouge1": self.overlap(g, r, 1),
            "rouge2": self.overlap(g, r, 2),
            "rougeL": self.rouge_l(g, r),
        }

# ── BLEU ──────────────────────────────────────────────────────── #
class BLEUCalculator:
    def compute_bleu(self, gen: str, ref: str, max_n: int = 4) -> float:
        if NLTK_OK:
            g, r = word_tokenize(gen.lower()), word_tokenize(ref.lower())
            try:
                return round(float(sentence_bleu(
                    [r], g, weights=[1/max_n] * max_n,
                    smoothing_function=SmoothingFunction().method1)), 4)
            except Exception:
                pass

        rc      = RougeCalculator()
        g, r    = gen.lower().split(), ref.lower().split()
        ls, cnt = 0.0, 0
        for n in range(1, max_n + 1):
            if len(g) < n or len(r) < n:
                continue
            p = rc.overlap(g, r, n)["precision"]
            ls += math.log(p) if p > 0 else math.log(1e-10)
            cnt += 1

        if cnt == 0:
            return 0.0

        bp = (math.exp(1 - len(r) / max(len(g), 1))
              if len(g) < len(r) else 1.0)
        return round(float(bp * math.exp(ls / cnt)), 4)

    def corpus_bleu_score(self, gens: List[str], refs: List[str]) -> float:
        if NLTK_OK:
            g = [word_tokenize(x.lower()) for x in gens]
            r = [[word_tokenize(x.lower())] for x in refs]
            try:
                return round(float(corpus_bleu(
                    r, g,
                    smoothing_function=SmoothingFunction().method1)), 4)
            except Exception:
                pass

        scores = [self.compute_bleu(g_, r_) for g_, r_ in zip(gens, refs)]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

# ── METEOR ────────────────────────────────────────────────────── #
class METEORCalculator:
    def __init__(self):
        self.hf = None
        if HF_EVAL:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.hf = hf_evaluate.load("meteor")
                    print("  ✓ METEOR loaded")
            except Exception:
                pass

    def compute(self, gen: str, ref: str) -> float:
        if self.hf:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self.hf.compute(
                        predictions=[gen], references=[ref])
                    return round(result.get("meteor", 0.0), 4)
            except Exception:
                pass

        if NLTK_OK:
            try:
                from nltk.translate.meteor_score import meteor_score
                g = word_tokenize(gen.lower())
                r = word_tokenize(ref.lower())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return round(float(meteor_score([r], g)), 4)
            except Exception:
                pass

        # Manual approximation
        rc  = RougeCalculator()
        r1  = rc.overlap(rc.tokenize(gen), rc.tokenize(ref), 1)
        p, r = r1["precision"], r1["recall"]
        if p + r == 0:
            return 0.0
        return round((p * r) / (0.9 * p + 0.1 * r), 4)

    def corpus(self, gens: List[str], refs: List[str]) -> float:
        scores = [self.compute(g, r) for g, r in zip(gens, refs)]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

# ── BERTScore ─────────────────────────────────────────────────── #
def bertscore_batch(gens: List[str], refs: List[str]) -> Dict:
    if not HF_EVAL:
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bs  = hf_evaluate.load("bertscore")
            res = bs.compute(predictions=gens, references=refs, lang="en")
            return {
                "bertscore_P": round(
                    sum(res["precision"]) / len(res["precision"]), 4),
                "bertscore_R": round(
                    sum(res["recall"]) / len(res["recall"]), 4),
                "bertscore_F": round(
                    sum(res["f1"]) / len(res["f1"]), 4),
            }
    except Exception as e:
        print(f"  ⚠ BERTScore: {e}")
        return {}

# ── Coverage & Redundancy ─────────────────────────────────────── #
def coverage_score(doc: str, summary: str) -> float:
    """Fraction of doc key-words xuất hiện trong summary."""
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english"))
    except Exception:
        sw = set()

    def cw(text: str) -> Counter:
        tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
        return Counter(t for t in tokens if t not in sw and len(t) > 3)

    doc_cw    = cw(doc)
    sum_words = set(cw(summary).keys())
    key_words = ({w for w, c in doc_cw.items() if c >= 2}
                 or set(doc_cw.keys()))

    if not key_words:
        return 0.0
    return round(len(key_words & sum_words) / len(key_words), 4)

def redundancy_score(summary: str, n: int = 3) -> float:
    """Intra-summary n-gram repetition (0=không lặp, ideal)."""
    tokens = re.sub(r"[^\w\s]", "", summary.lower()).split()
    if len(tokens) < n:
        return 0.0
    ng = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    return round(1.0 - len(set(ng)) / len(ng), 4) if ng else 0.0

# ── EvaluationPipeline ────────────────────────────────────────── #
class EvaluationPipeline:
    """
    Full evaluation: ROUGE + BLEU + METEOR + BERTScore + Coverage + Redundancy.
    Auto-detects BART và Lead-3, in side-by-side với Δ column.
    """
    def __init__(
        self,
        use_hf_evaluate: bool = True,
        use_meteor:      bool = True,
        use_bertscore:   bool = False,
    ):
        self.rouge_c  = RougeCalculator()
        self.bleu_c   = BLEUCalculator()
        self.meteor   = METEORCalculator() if use_meteor else None
        self.use_bs   = use_bertscore
        self.hf_rouge = None

        if use_hf_evaluate and HF_EVAL:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.hf_rouge = hf_evaluate.load("rouge")
                    print("  ✓ HF ROUGE loaded")
            except Exception:
                print("  Using manual ROUGE")

    def eval_single(
        self,
        gen: str,
        ref: str,
        doc: str = "",
    ) -> Dict:
        r = self.rouge_c.compute_all_rouge(gen, ref)
        result = {
            "rouge1_f1":        r["rouge1"]["f1"],
            "rouge1_recall":    r["rouge1"]["recall"],
            "rouge1_precision": r["rouge1"]["precision"],
            "rouge2_f1":        r["rouge2"]["f1"],
            "rouge2_recall":    r["rouge2"]["recall"],
            "rougeL_f1":        r["rougeL"]["f1"],
            "bleu":             self.bleu_c.compute_bleu(gen, ref),
            "generated_length": len(gen.split()),
            "reference_length": len(ref.split()),
            "redundancy":       redundancy_score(gen),
            "coverage":         coverage_score(doc, gen) if doc else 0.0,
            "compression_ratio": round(
                len(gen.split()) / max(len(doc.split()), 1), 4) if doc else 0.0,
        }
        if self.meteor:
            result["meteor"] = self.meteor.compute(gen, ref)
        return result

    def _score_col(
        self,
        dataset:    List[Dict],
        gen_key:    str,
        ref_key:    str = "summary",
        doc_key:    str = "document",
    ) -> Tuple[Dict, List[Dict]]:
        per, gens, refs = [], [], []

        for s in dataset:
            g, r, d = s.get(gen_key, ""), s.get(ref_key, ""), s.get(doc_key, "")
            if not g or not r:
                continue
            per.append(self.eval_single(g, r, d))
            gens.append(g)
            refs.append(r)

        if not per:
            return {}, []

        def avg(k: str) -> float:
            vals = [s[k] for s in per if k in s]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        avgs: Dict = {
            "rouge1_f1":        avg("rouge1_f1"),
            "rouge1_recall":    avg("rouge1_recall"),
            "rouge1_precision": avg("rouge1_precision"),
            "rouge2_f1":        avg("rouge2_f1"),
            "rouge2_recall":    avg("rouge2_recall"),
            "rougeL_f1":        avg("rougeL_f1"),
            "bleu_sent":        avg("bleu"),
            "bleu_corpus":      self.bleu_c.corpus_bleu_score(gens, refs),
            "avg_gen_length":   avg("generated_length"),
            "avg_ref_length":   avg("reference_length"),
            "compression_ratio": avg("compression_ratio"),
            "avg_coverage":     avg("coverage"),
            "avg_redundancy":   avg("redundancy"),
            "num_samples":      len(per),
        }

        if self.meteor:
            avgs["meteor"] = self.meteor.corpus(gens, refs)

        dl = [len(s.get("document", "").split())
              for s in dataset if s.get("document")]
        if dl:
            avgs["avg_input_length"] = round(sum(dl) / len(dl), 1)

        if self.hf_rouge:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hf = self.hf_rouge.compute(
                        predictions=gens, references=refs)
                avgs.update(
                    hf_rouge1=round(hf.get("rouge1", 0), 4),
                    hf_rouge2=round(hf.get("rouge2", 0), 4),
                    hf_rougeL=round(hf.get("rougeL", 0), 4),
                )
            except Exception as e:
                print(f"  HF ROUGE err: {e}")

        if self.use_bs:
            avgs.update(bertscore_batch(gens, refs))

        return avgs, per

    def evaluate_dataset(
        self,
        dataset:        List[Dict],
        generated_key:  str             = "generated_summary",
        reference_key:  str             = "summary",
        inference_time: Optional[float] = None,
    ) -> Dict:
        print(f"\nEvaluating {len(dataset)} samples...")

        results: Dict = {}
        systems = {generated_key: "BART"}
        if any("lead3_summary" in s for s in dataset):
            systems["lead3_summary"] = "Lead-3"

        per_main: List[Dict] = []
        for key, name in systems.items():
            avgs, per = self._score_col(dataset, key, reference_key)
            results[name] = avgs
            if key == generated_key:
                per_main = per

        if inference_time and per_main:
            results.setdefault("BART", {})["inference_time_per_sample"] = \
                round(inference_time / len(per_main), 3)
        results["per_sample_scores"] = per_main

        return results

    def print_results(self, scores: Dict) -> None:
        print("\n" + "=" * 86)
        print("  EVALUATION REPORT")
        print("=" * 86)

        systems = [k for k in scores if k != "per_sample_scores"]
        if not systems:
            print("  No results.")
            return

        metrics = [
            ("ROUGE-1 F1",        "rouge1_f1"),
            ("ROUGE-1 Recall",    "rouge1_recall"),
            ("ROUGE-1 Precision", "rouge1_precision"),
            ("ROUGE-2 F1",        "rouge2_f1"),
            ("ROUGE-2 Recall",    "rouge2_recall"),
            ("ROUGE-L F1",        "rougeL_f1"),
            ("METEOR",            "meteor"),
            ("BLEU (sentence)",   "bleu_sent"),
            ("BLEU (corpus)",     "bleu_corpus"),
            ("Avg Input Length",  "avg_input_length"),
            ("Avg Gen Length",    "avg_gen_length"),
            ("Avg Ref Length",    "avg_ref_length"),
            ("Compression Ratio", "compression_ratio"),
            ("Coverage Score ↑",  "avg_coverage"),
            ("Redundancy Score ↓","avg_redundancy"),
        ]

        cw        = 13
        has_delta = "BART" in scores and "Lead-3" in scores

        hdr = f"  {'Metric':<28}" + "".join(f"{s:>{cw}}" for s in systems)
        if has_delta:
            hdr += f"  {'Δ BART-L3':>13}"
        print(hdr)
        print("  " + "─" * (28 + cw * len(systems) + (15 if has_delta else 0)))

        for disp, key in metrics:
            if not any(key in scores.get(s, {}) for s in systems):
                continue

            row  = f"  {disp:<28}"
            bv   = scores.get("BART",   {}).get(key)
            lv   = scores.get("Lead-3", {}).get(key)

            for sn in systems:
                v = scores.get(sn, {}).get(key)
                row += (f"{v:>{cw}.4f}" if isinstance(v, float)
                        else f"{'N/A':>{cw}}")

            if has_delta and isinstance(bv, float) and isinstance(lv, float):
                d = bv - lv
                row += f"  {'+' if d >= 0 else ''}{d:>+.4f}"

            print(row)

        # Timing
        if "BART" in scores and "inference_time_per_sample" in scores["BART"]:
            t = scores["BART"]["inference_time_per_sample"]
            print(f"\n  Inference time/sample : {t:.3f}s")

        # HF ROUGE
        if "BART" in scores and "hf_rouge1" in scores["BART"]:
            b = scores["BART"]
            print(f"\n  [HF ROUGE stemmed]  "
                  f"R1={b.get('hf_rouge1',0):.4f}  "
                  f"R2={b.get('hf_rouge2',0):.4f}  "
                  f"RL={b.get('hf_rougeL',0):.4f}")

        # BERTScore
        if "BART" in scores and "bertscore_F" in scores["BART"]:
            b = scores["BART"]
            print(f"  [BERTScore]  "
                  f"P={b.get('bertscore_P',0):.4f}  "
                  f"R={b.get('bertscore_R',0):.4f}  "
                  f"F1={b.get('bertscore_F',0):.4f}")

        n = (scores.get("BART") or
             scores.get(systems, {})).get("num_samples", 0)
        print(f"\n  Evaluated on: {n} samples")
        print("=" * 86)

        # Guide
        print("\n  📊 ROUGE TARGETS (abstractive summarization):")
        print("  ─────────────────────────────────────────────────────")
        print("  distilBART zero-shot   : R1 ~0.28-0.33")
        print("  BART-large zero-shot   : R1 ~0.33-0.38")
        print("  distilBART fine-tuned  : R1 ~0.40-0.44")
        print("  BART-large fine-tuned  : R1 ~0.43-0.47  ← mục tiêu")
        print("  PEGASUS fine-tuned     : R1 ~0.44-0.47")
        print()
        print("  📖 Metric guide:")
        print("  Coverage  ↑ : fraction of doc key-words in summary")
        print("  Redundancy ↓: intra-summary repetition (0=ideal)")
        print("  Δ>0: BART outperforms Lead-3 baseline")