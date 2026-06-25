"""Falsifier Haiku JSON parsing must degrade per-item, not all-or-nothing.

Bug (2026-06-24 run): `_structure_with_haiku` did json.loads(text[start:end]) on
the whole array, so a single missing comma OR a response truncated at max_tokens
("Expecting ',' delimiter: line 99 column 73") discarded EVERY structured
falsifier and silently dropped the entire registry to news watches. A tolerant
object-scanner must recover the well-formed objects and only lose the genuinely
broken / truncated tail.
"""
from ascent.strategy.falsifier_registry import _parse_json_objects


def test_recovers_objects_when_array_truncated_midway():
    """Response cut off at max_tokens mid-last-object: keep the complete ones."""
    text = '[{"i": 0, "kind": "price", "value": -0.08}, {"i": 1, "kind": "macro", "val'
    out = _parse_json_objects(text)
    assert [o["i"] for o in out] == [0]
    assert out[0]["kind"] == "price"


def test_recovers_objects_despite_missing_comma_between_them():
    """The exact failure class: a missing ',' delimiter between two objects."""
    text = '[{"i": 0, "fired": true} {"i": 1, "fired": false}]'  # no comma
    out = _parse_json_objects(text)
    assert [o["i"] for o in out] == [0, 1]


def test_braces_inside_string_values_do_not_break_parsing():
    text = '[{"i": 0, "evidence": "guidance cut { see note }"}]'
    out = _parse_json_objects(text)
    assert len(out) == 1 and out[0]["evidence"] == "guidance cut { see note }"


def test_skips_one_malformed_object_keeps_the_rest():
    text = '[{"i": 0, "value": 1}, {"i": 1, "value": }, {"i": 2, "value": 3}]'
    out = _parse_json_objects(text)
    assert [o["i"] for o in out] == [0, 2]


def test_empty_or_garbage_returns_empty_list():
    assert _parse_json_objects("") == []
    assert _parse_json_objects("no json here") == []
