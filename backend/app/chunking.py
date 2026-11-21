"""Structure-aware semantic child chunks with larger, source-local parent context."""
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

CHUNKING_VERSION = 2


@dataclass
class Chunk:
    text: str
    page: int | None = None
    section: str | None = None
    parent_text: str = ""
    parent_id: str = ""


class SemanticChunker:
    def __init__(self, tokenizer=None, child_tokens=360, parent_tokens=1600, minimum_tokens=100):
        if child_tokens < 16 or parent_tokens < child_tokens:
            raise ValueError("Invalid chunk budgets")
        self.tokenizer = tokenizer
        self.child_tokens = child_tokens
        self.parent_tokens = parent_tokens
        self.minimum_tokens = minimum_tokens

    def count(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Offline extraction/testing fallback; ingestion supplies the active model tokenizer.
        return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))

    def units(self, text: str) -> list[str]:
        """Preserve paragraphs, list entries, and complete sentences before applying budgets."""
        output = []
        for paragraph in re.split(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)", text):
            paragraph = re.sub(r"(?<!\n)\n(?!\n)", " ", paragraph).strip()
            if not paragraph:
                continue
            if self.count(paragraph) <= self.child_tokens:
                output.append(paragraph)
                continue
            sentences = re.split(r"(?<=[.!?。！？])\s+(?=[\w\"'“(])", paragraph)
            for sentence in sentences:
                if self.count(sentence) <= self.child_tokens:
                    output.append(sentence)
                else:
                    # Only a pathological sentence/table row is split at word boundaries.
                    words, current = sentence.split(), []
                    for word in words:
                        if current and self.count(" ".join([*current, word])) > self.child_tokens:
                            output.append(" ".join(current))
                            current = []
                        if self.count(word) > self.child_tokens:
                            if current:
                                output.append(" ".join(current)); current = []
                            fragment = ""
                            for char in word:
                                if fragment and self.count(fragment + char) > self.child_tokens:
                                    output.append(fragment); fragment = ""
                                fragment += char
                            if fragment:
                                output.append(fragment)
                        else:
                            current.append(word)
                    if current:
                        output.append(" ".join(current))
        return output

    def topic_breaks(self, units: list[str]) -> set[int]:
        if len(units) < 2:
            return set()
        # Compare intact paragraphs/sentences so genuine topic transitions stay visible.
        similarities = []
        # Sparse TF-IDF overlap uses document vocabulary and requires no embedding inference.
        stopwords = {"the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "with", "was", "were"}
        counts = [Counter(term for term in re.findall(r"\w+", unit.lower()) if term not in stopwords) for unit in units]
        frequency = Counter(term for count in counts for term in count)
        weights = [{term: count * math.log(1 + len(units) / frequency[term]) for term, count in terms.items()} for terms in counts]
        norms = [math.sqrt(sum(value * value for value in weight.values())) for weight in weights]
        for index, (left, right) in enumerate(zip(weights, weights[1:])):
            dot = sum(left[term] * right[term] for term in left.keys() & right.keys())
            similarities.append(dot / max(norms[index] * norms[index + 1], 1e-12))
        ordered = sorted(similarities)
        threshold = min(.78, ordered[max(0, int(len(ordered) * .25) - 1)])
        return {index + 1 for index, score in enumerate(similarities) if score <= threshold}

    def chunk(self, text: str, *, page=None, section=None) -> list[Chunk]:
        text = text.replace("\x00", "").strip()
        units = self.units(text)
        if not units:
            return []
        breaks = self.topic_breaks(units)
        # A parent is a coherent section/page window, never a different source or page.
        parents, current, start = [], [], 0
        for index, unit in enumerate(units):
            if current and self.count("\n\n".join([*current, unit])) > self.parent_tokens:
                parents.append((start, current)); current, start = [], index
            current.append(unit)
        if current:
            parents.append((start, current))
        result = []
        for start, parent_units in parents:
            parent = "\n\n".join(parent_units)
            parent_id = hashlib.sha256(f"{page}|{section}|{parent}".encode()).hexdigest()[:24]
            children, child = [], []
            for index, unit in enumerate(parent_units, start):
                topic_boundary = index in breaks and child and self.count("\n\n".join(child)) >= self.minimum_tokens
                overflow = child and self.count("\n\n".join([*child, unit])) > self.child_tokens
                if topic_boundary or overflow:
                    children.append(child)
                    # Overlap uses a whole final sentence, not an arbitrary character slice.
                    last = re.split(r"(?<=[.!?。！？])\s+", child[-1])[-1]
                    child = [last] if not topic_boundary and self.count(last) <= 60 and self.count(last + " " + unit) <= self.child_tokens else []
                child.append(unit)
            if child:
                children.append(child)
            for child in children:
                result.append(Chunk("\n\n".join(child), page, section, parent, parent_id))
        return result


_HEADING = re.compile(r"^(?:#{1,6}\s+|\d+(?:\.\d+)*[.)]?\s+[A-Z]|(?:ABSTRACT|INTRODUCTION|METHODS|METHODOLOGY|RESULTS|DISCUSSION|CONCLUSIONS?|REFERENCES|LIMITATIONS|BACKGROUND|EXPERIMENTAL SETUP)\s*$)", re.I)


def structured_chunks(text: str, chunker: SemanticChunker, *, page=None, section=None) -> list[Chunk]:
    """Detect explicit research headings, retaining section and page provenance."""
    chunks, lines = [], []
    current_section = section
    for line in text.splitlines():
        stripped = line.strip()
        # Avoid treating numbered findings/list entries ending in punctuation as headings.
        heading = len(stripped) <= 100 and bool(_HEADING.match(stripped)) and not stripped.endswith((".", ";", ","))
        if heading:
            chunks.extend(chunker.chunk("\n".join(lines), page=page, section=current_section))
            current_section = re.sub(r"^#{1,6}\s+", "", stripped)
            lines = [stripped]
        else:
            lines.append(line)
    chunks.extend(chunker.chunk("\n".join(lines), page=page, section=current_section))
    return chunks

