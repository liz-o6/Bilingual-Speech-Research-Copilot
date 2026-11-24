"""Filter candidates locally with conservative heuristics, not full section detection."""
import re


def is_reference_list(text):
    # Detect repeated author-year-title entries, not inline citations.
    entries = re.findall(
        r"(?m)^[A-ZÀ-ÖØ-Þ][^\n]{0,180}\((?:19|20)\d{2}[a-z]?\)\s*\.\s*\S", text
    )
    # Vancouver entries use years without parentheses, e.g. 2004; 35:5–15.
    journal_entries = re.findall(r"\b(?:19|20)\d{2};\s*\d+(?:\([^)]*\))?\s*:", text)
    if len(journal_entries) >= 3:
        return True
    dois = re.findall(r"\b10\.\d{4,9}/", text)
    heading = re.search(r"(?im)^\s*(references|bibliography|参考文献)\s*$", text)
    return len(entries) >= 3 or (len(entries) >= 2 and len(dois) >= 2) or bool(heading and len(entries) >= 1)


def select_evidence(candidates, k=4):
    selected = []
    fingerprints = []
    for hit in candidates:
        text = hit.get('text', '')
        if not text.strip() or is_reference_list(text):
            continue
        words = re.findall(r'\w+', text.lower())
        # Use five-word sequences to detect duplicate passages across pages.
        fingerprint = set()
        for i in range(max(1, len(words) - 4)):
            fingerprint.add(tuple(words[i:i + 5]))
        duplicate = False
        for earlier in fingerprints:
            overlap = len(fingerprint & earlier) / min(len(fingerprint), len(earlier))
            if overlap >= 0.8:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(hit)
        fingerprints.append(fingerprint)
        if len(selected) == k:
            break
    return selected
