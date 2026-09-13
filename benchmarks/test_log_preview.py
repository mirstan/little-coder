"""preview_tool_result() -- the trajectory-log tool-result shortener.

The three call sites this replaces (harbor_adapter's live log and final log,
tb_adapter's log) each did a raw `text[:400]` character slice, which cut
mid-word and -- worse -- dropped the trailing `[exit=... cwd=... timed_out=...
backend=...]` footer for every result longer than 400 chars. That footer is
the one line that says whether the command actually worked, so every
interesting result lost exactly the part worth reading. Untested until now.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rpc_client import preview_tool_result  # noqa: E402

FOOTER = "[exit=0 cwd=/app timed_out=false backend=harbor-env]"


def _long_body(n_lines=40):
    return "\n".join(f"line {i} of output with several words on it" for i in range(n_lines))


def _formatted(body=None, footer=FOOTER):
    return f"{body if body is not None else _long_body()}\n{footer}"


def test_short_output_passes_through_unchanged():
    text = _formatted("tiny")
    assert len(text) < 400
    assert preview_tool_result(text) == text


def test_footer_survives_a_long_body():
    """The whole point: a 1500-char result used to end at character 400,
    which is never where the footer is."""
    text = _formatted()
    assert len(text) > 400
    out = preview_tool_result(text)
    assert out.endswith(FOOTER)
    assert "truncated" in out


def test_stays_within_the_budget():
    out = preview_tool_result(_formatted())
    assert len(out) <= 400


def test_body_is_cut_at_a_boundary_not_mid_word():
    body = _long_body()
    out = preview_tool_result(_formatted(body))
    kept = out.split("\n… [+")[0]
    assert body.startswith(kept)
    # The character the original body continues with must be whitespace --
    # i.e. we stopped at the end of a line or a word, not inside one.
    assert body[len(kept)] in ("\n", " ")


def test_elision_marker_reports_how_much_was_dropped():
    body = _long_body()
    out = preview_tool_result(_formatted(body))
    kept = out.split("\n… [+")[0]
    assert f"[+{len(body) - len(kept)} chars truncated]" in out


def test_output_without_a_footer_is_handled():
    """Not every logged result comes from _format_output -- a proxy error
    string, or a tool that isn't ShellSession, has no bracketed trailer."""
    text = "word " * 300
    out = preview_tool_result(text)
    assert len(out) <= 400
    assert "truncated" in out
    kept = out.split("\n… [+")[0]
    assert text.startswith(kept)
    assert not kept.endswith(" "), "cut should land before the trailing space"


def test_a_trailing_bracket_line_is_not_confused_with_a_footer_when_short():
    """A result that fits the budget is returned verbatim, footer or not --
    no marker, no reflowing."""
    text = "output\n[exit=1 cwd=/ timed_out=false backend=tmux-proxy]"
    assert preview_tool_result(text) == text


def test_footer_is_preserved_even_when_it_eats_the_whole_budget():
    """A pathologically long footer still wins over the body: losing it is
    the bug this function exists to fix, so it is never the thing dropped."""
    footer = "[exit=0 cwd=" + "/deep" * 80 + " timed_out=false]"
    text = _formatted("some body text that will not survive", footer)
    out = preview_tool_result(text)
    assert out.endswith(footer)
    assert "truncated" in out
    assert not out.startswith("\n")


def test_tb_adapter_footer_shape_also_recognized():
    footer = "[exit=127 cwd=/app timed_out=false backend=tmux-proxy]"
    out = preview_tool_result(_formatted(footer=footer))
    assert out.endswith(footer)


@pytest.mark.parametrize("text", ["", None])
def test_empty_and_none_are_safe(text):
    assert preview_tool_result(text) == ""


def test_custom_limit_is_honoured():
    out = preview_tool_result(_formatted(), limit=1200)
    assert out.endswith(FOOTER)
    assert len(out) <= 1200


def test_unbroken_token_wider_than_the_budget_still_keeps_the_footer():
    """No boundary to back up to. The body cut is then unavoidably hard --
    but the footer must still be there."""
    text = _formatted("x" * 2000)
    out = preview_tool_result(text)
    assert out.endswith(FOOTER)
    assert "truncated" in out


def test_a_footer_only_result_over_the_limit_is_kept_whole():
    """One line, no body, past the limit. Recognising the footer used to
    require a SECOND line, so this fell through as an ordinary body and got
    sliced mid-footer -- losing exactly what the function exists to keep."""
    footer = "[exit=0 cwd=" + "/deep" * 90 + " timed_out=false]"
    assert len(footer) > 400
    assert preview_tool_result(footer) == footer


def test_a_footer_with_an_empty_body_does_not_open_on_a_blank_line():
    footer = "[exit=0 cwd=" + "/deep" * 90 + " timed_out=false]"
    assert preview_tool_result("\n" + footer) == footer


def test_a_trailing_newline_after_the_footer_does_not_defeat_detection():
    """`text.rpartition("\\n")` on input ending in "\\n" yields an empty
    `last`, which fails the footer test and lets the real footer get cut
    mid-line by the body-truncation path -- rstrip before anything else."""
    footer = "[exit=0 cwd=" + "/deep" * 90 + " timed_out=false]"
    body = "x" * 500
    assert preview_tool_result(body + "\n" + footer + "\n") == preview_tool_result(
        body + "\n" + footer
    )
    assert preview_tool_result(body + "\n" + footer + "\n").endswith(footer)
