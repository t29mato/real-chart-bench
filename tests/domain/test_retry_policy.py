"""The redo loop of 方式D「検証とやり直し」: verify's verdict decides, the
orchestrator's wording does not."""

from __future__ import annotations

from real_chart_bench.domain.orchestration import (
    MAX_RETRIES,
    answer_key,
    may_answer,
    retry_decision,
)


def att(key, accept, score):
    return {"key": key, "accept": accept, "score": score}


def test_answer_key_ignores_order_labels_and_duplicates():
    a = answer_key([{"from": "r2", "index": 0, "label": "a"},
                    {"from": "r3", "index": -1}], "r1")
    b = answer_key([{"from": "r3", "index": -1, "label": "x"},
                    {"from": "r2", "index": 0}, {"from": "r2", "index": 0}], "r1")
    assert a == b
    assert a != answer_key([{"from": "r2", "index": 0}], "r1")
    assert a != answer_key([{"from": "r2", "index": 0}, {"from": "r3", "index": -1}], "")


def test_retry_decision_accept_redo_then_best():
    assert MAX_RETRIES == 2
    assert retry_decision([att("a", True, 0.9)]) == {"action": "accept", "pick": 0,
                                                     "retries_left": 2}
    assert retry_decision([att("a", False, 0.5)])["action"] == "redo"
    assert retry_decision([att("a", False, 0.5), att("b", False, 0.7)]) == {
        "action": "redo", "pick": None, "retries_left": 0}
    d = retry_decision([att("a", False, 0.5), att("b", False, 0.7), att("c", False, 0.6)])
    assert d == {"action": "best", "pick": 1, "retries_left": 0}
    # an accepted retry ends the loop
    d = retry_decision([att("a", False, 0.5), att("b", True, 0.4)])
    assert d == {"action": "accept", "pick": 1, "retries_left": 1}


def test_retry_decision_ties_keep_the_earlier_attempt():
    d = retry_decision([att("a", False, 0.5), att("b", False, 0.5), att("c", False, 0.5)])
    assert d["pick"] == 0


def test_may_answer():
    ok, msg = may_answer([], "a")
    assert not ok and "verify" in msg
    ok, _ = may_answer([att("a", True, 0.9)], "a")
    assert ok
    ok, msg = may_answer([att("a", True, 0.9)], "b")
    assert not ok and "verify" in msg  # not the verified series
    ok, msg = may_answer([att("a", False, 0.5)], "a")
    assert not ok and "redo" in msg and "2" in msg
    hist = [att("a", False, 0.5), att("b", False, 0.7), att("c", False, 0.6)]
    assert may_answer(hist, "b")[0]
    ok, msg = may_answer(hist, "c")
    assert not ok and "best" in msg
    # an accepted attempt can be answered even after later failed ones
    assert may_answer([att("a", True, 0.8), att("b", False, 0.9)], "a")[0]
