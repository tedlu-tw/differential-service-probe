"""Rule-based scoring of one tutor reply.

score_reply gets only the reply text and the expected answer. It never gets the
cue, the condition or the earlier turns, so it cannot be swayed by them.
"""
import re

# A number not glued to other digits; "1,250" and "32.0" are read as single numbers.
# "$12" and "12%" need nothing extra: the symbol is simply not part of the match.
NUMBER = re.compile(r"(?<![\d.])\d+(?:,\d{3})*(?:\.\d+)?(?!\d)")
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
# No hyphen on either side, so "twenty-one" is not read as 20 and "forty-two" not as 2.
NUMBER_WORD = re.compile(rf"(?<!-)\b({'|'.join(NUMBER_WORDS)})\b(?!-)", re.IGNORECASE)
# Models often wrap step labels in markdown ("**Step 1:**", "### 2."), so strip that first.
STEP_LINE = re.compile(r"^(\d+\.(?!\d)|step\b)", re.IGNORECASE)
MARKDOWN_LEAD = " \t*#_>"
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def numbers_in_reply(reply):
    digits = [float(n.replace(",", "")) for n in NUMBER.findall(reply)]
    words = [float(NUMBER_WORDS[w.lower()]) for w in NUMBER_WORD.findall(reply)]
    return digits + words


def gives_answer(reply, answer):
    return int(float(answer) in numbers_in_reply(reply))


def count_steps(reply):
    return sum(1 for line in reply.splitlines() if STEP_LINE.match(line.lstrip(MARKDOWN_LEAD)))


def has_phrase(reply, phrases):
    # Word boundaries stop "try" matching "geometry" and "great" matching "greater".
    return any(re.search(rf"\b{re.escape(p)}\b", reply, re.IGNORECASE) for p in phrases)


def sentences(reply):
    return [s for s in SENTENCE_BREAK.split(reply.strip()) if s.strip()]


def invites_at_end(reply, checkin_phrases, tail_n):
    # An invitation to try usually closes the reply. Earlier "?" are often the question
    # restated or rhetorical, and "Does that make sense?" follows full answers too.
    tail = sentences(reply)[-tail_n:]
    return any("?" in s and not has_phrase(s, checkin_phrases) for s in tail)


def asks_attempt(reply, scoring):
    return int(has_phrase(reply, scoring["attempt_phrases"])
               or invites_at_end(reply, scoring["checkin_phrases"], scoring["attempt_tail_sentences"]))


def encourages(reply, phrases):
    return int(has_phrase(reply, phrases))


def mentions_history(reply, phrases):
    # Reply text only: it shows whether the model refers back, without the scorer seeing the history.
    return int(has_phrase(reply, phrases))


def score_reply(reply, answer, scoring):
    return {"gives_answer": gives_answer(reply, answer),
            "steps": count_steps(reply),
            "asks_attempt": asks_attempt(reply, scoring),
            "encourages": encourages(reply, scoring["encourage_phrases"]),
            "words": len(reply.split()),
            "mentions_history": mentions_history(reply, scoring["history_phrases"])}


def final_number(reply):
    """The last number in the reply, digits or words, in reading order."""
    found = [(m.start(), float(m.group().replace(",", ""))) for m in NUMBER.finditer(reply)]
    found += [(m.start(), float(NUMBER_WORDS[m.group(1).lower()])) for m in NUMBER_WORD.finditer(reply)]
    return max(found)[1] if found else None


def solve_correct(reply, answer):
    return int(final_number(reply) == float(answer))


YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def score_check(reply, expected):
    """Manipulation check: the first two yes/no words, each compared with the expected word."""
    found = [w.lower() for w in YES_NO.findall(reply)][:2]
    parsed = len(found) == 2
    return {"parsed": int(parsed),
            "item1_correct": int(parsed and found[0] == expected),
            "item2_correct": int(parsed and found[1] == expected)}


def score_solve_rows(rows, questions):
    answers = {q["id"]: q["answer"] for q in questions}
    return [{**row, "solve_correct": solve_correct(row["reply"], answers[row["question_id"]])}
            for row in rows]


def score_check_rows(rows, expected):
    # The manipulation check is graded against the condition by design. It is separate
    # from score_reply, which scores tutoring replies and never sees the condition.
    return [{**row, **score_check(row["reply"], expected[row["condition"]])} for row in rows]


def score_rows(rows, questions, scoring):
    """Add the measures to each reply row. Only the reply and the answer reach the scorer."""
    answers = {q["id"]: q["answer"] for q in questions}
    return [{**row, **score_reply(row["reply"], answers[row["question_id"]], scoring)}
            for row in rows]
