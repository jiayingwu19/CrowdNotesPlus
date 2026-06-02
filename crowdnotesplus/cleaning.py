"""Markdown cleanup helpers for text returned by Jina Reader."""

from __future__ import annotations

import regex as re

NAV_PATTERNS = [
    r'^\s*\[Skip directly.*\]\(.*\)\s*$',
    r'^\s*\[Menu\]\(.*\)\s*$',
    r'^\s*\[View all\]\(.*\)\s*$',
    r'^\s*\[Here\'?s how you know\]\(.*\)\s*$',
    r'^\s*(Search|clear search)\s*$',
    r'^\s*Explore Topics\s*$',
    r'^\s*Related Topics:\s*$',
    r'^\s*\[_?search_?\s*_?close search_?\]\(.*\)\s*$',
]

SECTION_TAILS = [
    r'^Related Topics:?$',
    r'^External links$',
    r'^See also$',
    r'^References$',
]

SECTION_HEADERS_TO_DROP = [
    r'^On This Page$',
    r'^Related Pages$',
    r'^Back to Top$',
    r'^Sources\b$',
    r'^Content Source\b$',
    r'^Languages$',
    r'^Language Assistance$',
    r'^For Everyone$',
    r'^Health Care Providers$',
    r'^SourcesPrintShare$',
]

LANG_WORDS = set(
    "Espa\u00f1ol English Fran\u00e7ais Deutsch Italiano Portugu\u00eas Polski "
    "\u0420\u0443\u0441\u0441\u043a\u0438\u0439 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 "
    "\u0641\u0627\u0631\u0633\u06cc \u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4 "
    "\u7e41\u9ad4\u4e2d\u6587 Ti\u1ebfng Vi\u1ec7t Tagalog Krey\u00f2l Ayisyen"
    .split()
)

REFS_HEADING_RE = re.compile(
    r'^[#>\s\-\*\u2022]*\breferences\b\s*[:\uff1a\-\u2013\u2014]*\s*$',
    flags=re.IGNORECASE,
)

TABLE_SEP_ROW = re.compile(
    r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$'
)


def _cut_at_references_line(text: str) -> str:
    """Remove the references section when a references heading is detected."""
    if 'references' not in text.casefold():
        return text
    return re.split(r'references\n', text, flags=re.IGNORECASE)[0]


def is_image_or_link_only(line: str) -> bool:
    return bool(re.match(
        r'^\s*(?:\*\*|\*|_)?(?:!\[[^\]]*\]\([^)]+\)|\[[^\]]+\]\([^)]+\))(?:\*\*|\*|_)?\s*$',
        line.strip(),
    ))


def looks_like_nav(line: str) -> bool:
    return any(re.match(pattern, line.strip(), flags=re.IGNORECASE) for pattern in NAV_PATTERNS)


def is_short_wordlist_line(text: str) -> bool:
    if len(text.split()) <= 3 and not re.search(r'[.!?;:]', text):
        tokens = [token.strip('*_`-\u2022\u00b7') for token in text.split()]
        title_like_ratio = sum(1 for token in tokens if token and token[0].isupper()) / max(1, len(tokens))
        has_language_word = any(token in LANG_WORDS for token in tokens)
        return title_like_ratio >= 0.67 or has_language_word
    return False


def block_is_listy(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return False
    short_count = sum(is_short_wordlist_line(line) for line in lines)
    bullet_count = sum(bool(re.match(r'^[-*\u2022\u00b7]\s+', line)) for line in lines)
    punct_count = sum(bool(re.search(r'[.!?]', line)) for line in lines)
    return (short_count + bullet_count) / len(lines) >= 0.6 and punct_count <= 1


def block_is_header_only(block: str) -> bool:
    text = block.strip()
    if '\n' in text:
        return False
    return len(text.split()) <= 6 and not re.search(r'[.!?;:]', text)


def block_is_table(block: str) -> bool:
    """Detect GitHub-style Markdown tables and table-like blocks."""
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if any(TABLE_SEP_ROW.match(line) for line in lines):
        return True
    pipe_counts = [line.count('|') for line in lines]
    multi_pipe_count = sum(1 for count in pipe_counts if count >= 2)
    return multi_pipe_count / len(lines) >= 0.6


def clean_jina_markdown(markdown: str) -> str:
    """Remove navigation, images, tables, links, and trailing reference sections."""
    lines = markdown.splitlines()
    output = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            if (re.match(r'^(#+\s+.+)$', stripped) and not looks_like_nav(stripped)) or (
                len(stripped) > 60 and not is_image_or_link_only(stripped) and not looks_like_nav(stripped)
            ):
                started = True
            else:
                continue
        if looks_like_nav(stripped) or is_image_or_link_only(stripped):
            continue
        output.append(line)

    text = "\n".join(output).strip()
    text = _cut_at_references_line(text)

    tail_pattern = r'\n(?:' + "|".join(SECTION_TAILS) + r')\n.*\Z'
    text = re.sub(tail_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'!\[[^\]]*\]\(?\s*\)?', '', text)
    text = re.sub(r'\[\s*\]\(?\s*\)?', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    blocks = [block for block in re.split(r'\n\s*\n', text) if block.strip()]
    cleaned = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if any(re.match(pattern, block.strip(), flags=re.IGNORECASE) for pattern in SECTION_HEADERS_TO_DROP):
            if index + 1 < len(blocks) and block_is_listy(blocks[index + 1]):
                index += 2
                continue
            index += 1
            continue
        if block_is_table(block) or block_is_listy(block):
            index += 1
            continue
        if block_is_header_only(block) and index + 1 < len(blocks) and block_is_listy(blocks[index + 1]):
            index += 2
            continue
        cleaned.append(block.strip())
        index += 1

    text = "\n\n".join(cleaned).strip()
    return re.sub(r'\n{3,}', '\n\n', text).strip()
