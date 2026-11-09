"""Rule-based scoring of one tutor reply.

score_reply gets only the reply text and the expected answer. It never gets the
cue, the condition or the earlier turns, so it cannot be swayed by them.
"""
import re

# A number not glued to other digits; "1,250" and "32.0" are read as single numbers.
NUMBER = re.compile(r"(?<![\d.])\d+(?:,\d{3})*(?:\.\d+)?(?!\d)")
# Models often wrap step labels in markdown ("**Step 1:**", "### 2."), so strip that first.
STEP_LINE = re.compile(r"^(\d+\.(?!\d)|step\b)", re.IGNORECASE)
MARKDOWN_LEAD = " \t*#_>"
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def numbers_in_reply(reply):
    return [float(n.replace(",", "")) for n in NUMBER.findall(reply)]


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


def score_reply(reply, answer, scoring):
    return {"gives_answer": gives_answer(reply, answer),
            "steps": count_steps(reply),
            "asks_attempt": asks_attempt(reply, scoring),
            "encourages": encourages(reply, scoring["encourage_phrases"]),
            "words": len(reply.split())}


def score_rows(rows, questions, scoring):
    """Add the measures to each reply row. Only the reply and the answer reach the scorer."""
    answers = {q["id"]: q["answer"] for q in questions}
    return [{**row, **score_reply(row["reply"], answers[row["question_id"]], scoring)}
            for row in rows]
