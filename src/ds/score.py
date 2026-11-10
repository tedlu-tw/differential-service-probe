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


# =====================================================================================
# v3 (docs/protocol_v3.md). New functions only; everything above is v2 and unchanged.
# The v3 scorer gets the reply, the question text and the answer, never the cue,
# the system prompt or the earlier turns.
# =====================================================================================

def normalise(reply):
    # Models often write "Let’s" and "don’t"; the phrase lists use the plain apostrophe.
    return reply.replace("’", "'").replace("‘", "'")


def praises(reply, scoring):
    return int(has_phrase(normalise(reply), scoring["praise_phrases"]))


def encourages_v3(reply, scoring):
    # Fixed in v3: effort and future phrases only; praise is counted by praises().
    return int(has_phrase(normalise(reply), scoring["encourage_phrases"]))


def excluded_sentence(sentence, attempt):
    return has_phrase(sentence, attempt["offer_phrases"] + attempt["checkin_phrases"]
                      + attempt["tutor_does_phrases"])


def asks_attempt_v3(reply, scoring):
    """An invitation to work on this problem, or a closing question about it.

    Offers of a new problem, the echoed "What's the next one?", check-ins and the tutor's own
    "Let's try..." never count, wherever they appear.
    """
    attempt, parts = scoring["attempt"], sentences(normalise(reply))
    kept = [s for s in parts if not excluded_sentence(s, attempt)]
    invited = any(has_phrase(s, attempt["invite_phrases"]) for s in kept)
    tail = parts[-attempt["tail_sentences"]:] if parts else []
    asks_last = any("?" in s and not excluded_sentence(s, attempt) for s in tail)
    return int(invited or asks_last)


# ---------- hint_level ----------

LIST_MARKER = re.compile(r"^\s*[*#_>]*\s*\d+[.)](?!\d)", re.MULTILINE)


def digit_numbers(text):
    return {float(n.replace(",", "")) for n in NUMBER.findall(text)}


def without_markers(reply, marker_words):
    # "1." at the start of a line and "Step 2" label the help; they are not computed values.
    text = LIST_MARKER.sub(" ", reply)
    return re.sub(rf"\b(?:{'|'.join(map(re.escape, marker_words))})\s*\d+", " ", text, flags=re.IGNORECASE)


def names_method(reply, stems):
    # Stems match at the start of a word: "multipl" covers multiply, multiplying, multiplication.
    return any(re.search(rf"\b{re.escape(s)}", reply, re.IGNORECASE) for s in stems)


def hint_level(reply, question_text, answer, scoring, ignore):
    """0 no math content, 1 method or first step, 2 computed intermediate value, 3 the answer.

    ignore: the numbers in every cue text and the whole history pool. It is the same in every
    condition, so a tutor echoing "40%" or an earlier answer cannot raise the level in one condition.
    """
    if gives_answer(reply, answer):
        return 3
    cfg, given = scoring["hint_level"], digit_numbers(question_text)
    numbers = digit_numbers(without_markers(reply, cfg["marker_words"]))
    if numbers - given - set(ignore):
        return 2
    if names_method(reply, cfg["method_stems"]) or numbers & given:
        return 1
    return 0


def score_reply_v3(reply, question_text, answer, scoring, ignore):
    """All turn-level v3 measures for one tutor reply."""
    return {"gives_answer": gives_answer(reply, answer),
            "hint_level": hint_level(reply, question_text, answer, scoring, ignore),
            "steps": count_steps(reply),
            "praises": praises(reply, scoring),
            "encourages": encourages_v3(reply, scoring),
            "asks_attempt": asks_attempt_v3(reply, scoring),
            "words": len(reply.split()),
            "mentions_history": mentions_history(normalise(reply), scoring["history_phrases"])}


def score_rows_v3(rows, questions, scoring, ignore):
    """Add the v3 measures to every tutor-reply row. Only the reply and the question reach the scorer."""
    by_id = {q["id"]: q for q in questions}
    return [{**row, **score_reply_v3(row["reply"], by_id[row["question_id"]]["text"],
                                     by_id[row["question_id"]]["answer"], scoring, ignore)}
            for row in rows]


# ---------- accuracy with working ----------

def last_line_number(reply):
    """The last number on the last line that has a letter or digit; None if that line has no number.

    Lines of markup only (a closing "\\]" or "```") are skipped, so "\\boxed{63}" on its own line is read.
    """
    lines = [line for line in reply.splitlines() if re.search(r"[A-Za-z0-9]", line)]
    found = NUMBER.findall(lines[-1]) if lines else []
    return float(found[-1].replace(",", "")) if found else None


def score_solve_rows_v3(rows, questions):
    answers = {q["id"]: q["answer"] for q in questions}
    out = []
    for row in rows:
        number = last_line_number(row["reply"])
        out.append({**row, "solve_parsed": int(number is not None),
                    "solve_correct": int(number == float(answers[row["question_id"]]))})
    return out


# ---------- manipulation check (v3 parser) ----------

def has_word(text, words):
    return any(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in words)


def verdict(clause, parser):
    # A negative word wins, so "not correct" is "no".
    if has_word(clause, parser["negative_words"]):
        return "no"
    return "yes" if has_word(clause, parser["positive_words"]) else None


def read_combined(reply, parser):
    """Rule 1: a combined answer in the first sentence ("Yes, both", "No, neither")."""
    first = (sentences(reply) or [""])[0]
    if has_word(first, parser["neither_words"]):
        return ("no", "no")
    if has_word(first, parser["both_words"]):
        v = verdict(first, parser)
        return (v, v) if v else None
    return None


def clauses(reply, breaks):
    pattern = "|".join(map(re.escape, breaks))
    return [c for s in sentences(reply) for c in re.split(pattern, s, flags=re.IGNORECASE) if c.strip()]


def read_per_item(reply, parser):
    """Rule 2: for each item, the first clause that names it and holds a verdict word."""
    found = []
    for item in ("first", "second"):
        named = [c for c in clauses(reply, parser["clause_breaks"]) if has_word(c, parser["item_phrases"][item])]
        found.append(next((v for v in map(lambda c: verdict(c, parser), named) if v), None))
    return tuple(found) if all(found) else None


def read_yes_no(reply):
    """Rule 3: the first two yes/no words, as in v2."""
    found = [w.lower() for w in YES_NO.findall(reply)][:2]
    return tuple(found) if len(found) == 2 else None


def read_check(reply, parser):
    reply = normalise(reply)
    for name, rule in (("combined", lambda: read_combined(reply, parser)),
                       ("per_item", lambda: read_per_item(reply, parser)),
                       ("yes_no", lambda: read_yes_no(reply))):
        items = rule()
        if items:
            return items, name
    return None, "unparsed"


def score_check_v3(reply, expected, parser):
    items, rule = read_check(reply, parser)
    return {"parsed": int(items is not None), "check_rule": rule,
            "item1_correct": int(items is not None and items[0] == expected),
            "item2_correct": int(items is not None and items[1] == expected)}


def score_check_rows_v3(rows, expected, parser):
    # Graded against the condition by design, as in v2; separate from the tutor-reply scorer.
    return [{**row, **score_check_v3(row["reply"], expected[row["condition"]], parser)} for row in rows]
