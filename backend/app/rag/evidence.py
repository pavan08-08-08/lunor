import re
import pymupdf


def extract_evidence_passage(chunk_text: str, query: str = "", answer: str = "") -> str:
    """Extract the most relevant, concise evidence sentence/passage from a retrieved chunk.

    Prioritizes:
    - Sentences containing exact answer terms or key values (e.g. 0.82, metrics)
    - Sentences with distinctive query terms (e.g. recall@5, RRF)
    - Acronym definitions (e.g. Reciprocal Rank Fusion (RRF))
    - Sentences containing numeric or factual claims

    Heavily penalizes standalone headings (e.g. 'Project Atlas', 'Architecture', 'Evaluation').
    """
    if not chunk_text or not chunk_text.strip():
        return ""

    paragraphs = re.split(r"\n\s*\n", chunk_text.strip())
    candidates: list[str] = []

    for para in paragraphs:
        lines = [l.strip() for l in para.splitlines() if l.strip()]
        if not lines:
            continue

        current_para: list[str] = []
        for line in lines:
            # Detect standalone section headings: short, few words, no end punctuation
            if len(line.split()) <= 3 and not line.endswith((".", "!", "?", ":")):
                if current_para:
                    joined = " ".join(current_para)
                    for s in re.split(r"(?<=[.!?])\s+", joined):
                        if s.strip():
                            candidates.append(s.strip())
                    current_para = []
                candidates.append(line)
            else:
                current_para.append(line)

        if current_para:
            joined = " ".join(current_para)
            for s in re.split(r"(?<=[.!?])\s+", joined):
                if s.strip():
                    candidates.append(s.strip())

    if not candidates:
        return chunk_text.strip()

    stopwords = {
        "what", "is", "was", "the", "by", "for", "does", "stand", "a", "an",
        "in", "of", "to", "on", "it", "its", "and", "or", "as", "at", "are",
        "be", "this", "that", "with", "from", "how", "why", "which"
    }

    q_tokens = [w.lower() for w in re.findall(r"[\w@\.-]+", query) if w.lower() not in stopwords]
    a_tokens = [w.lower() for w in re.findall(r"[\w@\.-]+", answer) if w.lower() not in stopwords]

    distinctive_terms = set()
    for tok in q_tokens + a_tokens:
        if re.search(r"\d", tok) or tok.isupper() or len(tok) >= 4:
            distinctive_terms.add(tok)

    scored: list[tuple[float, str]] = []

    for c in candidates:
        score = 0.0
        c_lower = c.lower()
        c_tokens = set(re.findall(r"[\w@\.-]+", c_lower))

        # Heavily penalize standalone section headers
        is_heading = len(c.split()) <= 3 and not c.endswith((".", "!", "?"))
        if is_heading:
            score -= 20.0

        for dt in distinctive_terms:
            if dt in c_lower:
                score += 5.0
                if re.search(r"\d", dt):
                    score += 8.0  # boost numeric metrics (e.g. 0.82, recall@5)

        for qt in q_tokens:
            if qt in c_tokens:
                score += 2.5

        for at in a_tokens:
            if at in c_tokens:
                score += 3.5

        # Direct acronym expansion match (e.g. query has "RRF" -> matches "... (RRF)")
        for word in query.split():
            clean = re.sub(r"[^\w]", "", word)
            if clean.isupper() and len(clean) >= 2:
                if f"({clean})" in c or f"({clean.lower()})" in c_lower:
                    score += 30.0

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def locate_evidence_rectangles(page: pymupdf.Page, evidence_text: str) -> list[dict[str, float]]:
    """Locate precise bounding rectangles for an evidence passage on a PDF page."""
    if not evidence_text or not evidence_text.strip():
        return []

    rects = []
    clean_ev = evidence_text.strip()

    # 1. Direct search for entire passage
    found = page.search_for(clean_ev)
    if found:
        rects.extend(found)

    # 2. Search line by line to handle multiline wraps in the PDF layout
    if not rects:
        lines = [l.strip() for l in clean_ev.splitlines() if l.strip()]
        for line in lines:
            f = page.search_for(line)
            if f:
                rects.extend(f)

    # 3. Search 3-to-5 word window phrases if punctuation/spacing slightly differs
    if not rects:
        words = clean_ev.split()
        if len(words) >= 3:
            for i in range(0, len(words), 3):
                phrase = " ".join(words[i:i + 4])
                f = page.search_for(phrase)
                if f:
                    rects.extend(f)

    # Deduplicate overlapping bounding boxes and convert to dictionary format
    results: list[dict[str, float]] = []
    seen: set[tuple[float, float, float, float]] = set()

    for r in rects:
        box = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
        if box not in seen:
            seen.add(box)
            results.append({
                "x": round(r.x0, 2),
                "y": round(r.y0, 2),
                "width": round(r.width, 2),
                "height": round(r.height, 2),
            })

    return results
