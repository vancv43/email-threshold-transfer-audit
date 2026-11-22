"""Character obfuscation of the simulated shift C_S (PROTOCOL.md section 1, curriculum A and C).

Four operations on the body part of the text, each word touched with probability `rate`: Latin-to-Cyrillic or
Greek homoglyphs, zero-width characters inside the word, letter spacing, and leetspeak digits. The subject line and
URLs are left as they are, so the URL branch and the A3 URL rule are not affected. Applied to ham and spam at the
same rate, so that obfuscation itself is not a label cue.
"""
import random
import re

HOMO = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "i": "і", "s": "ѕ", "j": "ј",
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
        "X": "Х", "v": "ν"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
ZW = ["​", "‌", "‍", "⁠"]
WORD = re.compile(r"(https?://\S+|www\.\S+)|([A-Za-z]{3,})")


def _word(w, r):
    op = r.randrange(4)
    if op == 0:
        return "".join(HOMO.get(c, c) if r.random() < 0.5 else c for c in w)
    if op == 1:
        k = r.randrange(1, len(w))
        return w[:k] + r.choice(ZW) + w[k:]
    if op == 2:
        return " ".join(w)
    return "".join(LEET.get(c.lower(), c) if r.random() < 0.5 else c for c in w)


def obfuscate(text, r: random.Random, rate=0.1):
    subject, sep, body = text.partition("\n")

    def sub(m):
        if m.group(1) or r.random() >= rate:  # URLs are kept verbatim
            return m.group(0)
        return _word(m.group(2), r)

    return subject + sep + WORD.sub(sub, body)
