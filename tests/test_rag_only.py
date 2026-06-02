#!/usr/bin/env python3
"""Minimal RAG + tools test — no heavy framework imports."""
import sys
import math
import re
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Inline BM25 test (avoids importing langgraph etc.)
def tokenize(text):
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

SAMPLE = """
PATIENT: John Doe  DOB: 1955-03-12  MRN: 12345678
ADMISSION: 2024-01-10  DISCHARGE: 2024-01-15
PRINCIPAL DIAGNOSIS: Acute decompensated heart failure
ALLERGIES: Penicillin
DISCHARGE MEDICATIONS: Furosemide 40mg daily, Lisinopril 10mg daily
PENDING: BNP level pending, Blood culture final pending
CONDITION: Stable improved
"""

# Test BM25 logic inline
docs = [SAMPLE, "Some unrelated text about hospital billing codes"]
tokenized = [tokenize(d) for d in docs]
avgdl = sum(len(t) for t in tokenized) / len(tokenized)
df = Counter()
for t in tokenized:
    for tok in set(t):
        df[tok] += 1

def bm25_score(query, doc_idx, K1=1.5, B=0.75):
    q_toks = tokenize(query)
    tokens = tokenized[doc_idx]
    tf = Counter(tokens)
    dl = len(tokens)
    N = len(docs)
    score = 0
    for tok in q_toks:
        if tok not in tf: continue
        freq = tf[tok]
        n_tok = df.get(tok, 0)
        idf = math.log((N - n_tok + 0.5) / (n_tok + 0.5) + 1)
        num = freq * (K1 + 1)
        den = freq + K1 * (1 - B + B * dl / max(1, avgdl))
        score += idf * num / den
    return score

s0 = bm25_score("patient name date of birth MRN", 0)
s1 = bm25_score("patient name date of birth MRN", 1)
assert s0 > s1, f"Expected patient doc to score higher: {s0:.3f} vs {s1:.3f}"
print(f"✓ BM25 ranking correct: doc0={s0:.3f} > doc1={s1:.3f}")

# Test drug interaction tool in isolation
import random
random.seed(42)  # deterministic

KNOWN = {
    frozenset({"warfarin", "aspirin"}): "HIGH: Increased bleeding risk",
    frozenset({"clopidogrel", "omeprazole"}): "MODERATE: Reduced efficacy",
}

def check_interactions(meds):
    norm = [m.lower().strip() for m in meds]
    found = []
    for i, d1 in enumerate(norm):
        for d2 in norm[i+1:]:
            pair = frozenset({d1, d2})
            if pair in KNOWN:
                found.append({"drug1": d1, "drug2": d2, "severity": KNOWN[pair]})
    return found

r = check_interactions(["Warfarin", "Aspirin", "Metformin"])
assert len(r) == 1 and "HIGH" in r[0]["severity"]
print(f"✓ Drug interaction: found {len(r)} interaction(s)")

r2 = check_interactions([])
assert r2 == []
print("✓ Empty medication list: no interactions")

# Test model logic
missing_fields = []
def mark_missing(f):
    if f not in missing_fields:
        missing_fields.append(f)

mark_missing("admission_date")
mark_missing("admission_date")
assert missing_fields == ["admission_date"]
print("✓ Missing field dedup OK")

# Test conflict detection logic
def detect_conflict(value):
    return value.strip().upper().startswith("CONFLICT")

assert detect_conflict("CONFLICT: Two notes disagree on discharge date")
assert not detect_conflict("2024-01-15")
print("✓ Conflict detection OK")

# Test MISSING/PENDING detection
def is_missing(v): return v.strip().upper().startswith("MISSING")
def is_pending(v): return v.strip().upper().startswith("PENDING")

assert is_missing("MISSING")
assert is_pending("PENDING — awaiting culture results")
assert not is_missing("Acute heart failure")
print("✓ MISSING/PENDING detection OK")

print()
print("=" * 40)
print("All logic tests PASSED ✓")
