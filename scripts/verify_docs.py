#!/usr/bin/env python
"""
Documentation drift guard
=========================

CLAUDE.md is loaded into every AI coding session. A stale claim in it does not
sit there harmlessly -- it becomes a confidently-held wrong belief that gets
acted on. Same for CURRENT_VERIFIED_NUMBERS.md, which declares itself the
single source of truth for performance figures.

This script makes those claims machine-checkable. Every check is an assertion
about the code or an artifact as it exists RIGHT NOW. Drift fails loudly here
instead of quietly becoming a hallucination.

    .venv/bin/python scripts/verify_docs.py           # human-readable
    .venv/bin/python scripts/verify_docs.py --quiet   # only failures
    .venv/bin/python scripts/verify_docs.py --json    # machine-readable

Exit code 0 = all checks pass, 1 = at least one FAIL.

Adding a check
--------------
Write a function returning (ok: bool, detail: str) and register it in CHECKS
with a stable name and the doc claim it defends. Prefer checks that read the
code over checks that restate it -- a check that hardcodes the same fact twice
drifts in lockstep and defends nothing.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SQRT252 = 252 ** 0.5

Result = Tuple[bool, str]


# ---------------------------------------------------------------- helpers ----

def _read(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(rel)
    return p.read_text(encoding="utf-8", errors="replace")


def _module_consts(rel: str) -> dict:
    """Top-level literal assignments in a module, without importing it.

    Importing would pull in heavy deps and side effects; ast keeps this fast
    enough to run on every commit.
    """
    tree = ast.parse(_read(rel))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                try:
                    out[tgt.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    pass
    return out


def _code_only(rel: str) -> List[str]:
    """Source lines with comments and string literals blanked out.

    Without this, every check that greps for a forbidden pattern also matches
    the docstring that documents the prohibition -- which is how a drift guard
    trains its owner to ignore it.
    """
    import io
    import tokenize

    src = _read(rel)
    lines = src.splitlines()
    blanked = list(lines)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return lines
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (srow, scol), (erow, ecol) = tok.start, tok.end
        for row in range(srow, erow + 1):
            i = row - 1
            if i >= len(blanked):
                continue
            line = blanked[i]
            a = scol if row == srow else 0
            b = ecol if row == erow else len(line)
            blanked[i] = line[:a] + " " * (b - a) + line[b:]
    return blanked


def _no_comments(rel: str) -> List[str]:
    """Source lines with comments blanked but string literals KEPT.

    Use this when the thing you are looking for is legitimately a string --
    dict keys like `.get("position_changes")` vanish under _code_only and make
    a working check report the feature as missing.
    """
    import io
    import tokenize

    src = _read(rel)
    blanked = src.splitlines()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return blanked
    for tok in toks:
        if tok.type != tokenize.COMMENT:
            continue
        (srow, scol), (erow, ecol) = tok.start, tok.end
        for row in range(srow, erow + 1):
            i = row - 1
            if i >= len(blanked):
                continue
            line = blanked[i]
            a = scol if row == srow else 0
            b = ecol if row == erow else len(line)
            blanked[i] = line[:a] + " " * (b - a) + line[b:]
    return blanked


def _defines(rel: str, *names: str) -> Result:
    """Check that a module defines each name (function, class, or assignment)."""
    try:
        tree = ast.parse(_read(rel))
    except FileNotFoundError:
        return False, f"{rel} does not exist"
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.add(node.id)
    missing = [n for n in names if n not in found]
    if missing:
        return False, f"{rel} is missing {', '.join(missing)}"
    return True, f"{rel} defines {', '.join(names)}"


def _source_files() -> List[Path]:
    """Real project sources. Excludes vendored/stale trees."""
    skip = (".venv", ".worktrees", "__pycache__", "graphify-out", ".git")
    files = []
    for pkg in ("ascent", "agents", "orchestrator", "debate", "scripts"):
        for p in (ROOT / pkg).rglob("*.py"):
            if not any(s in p.parts for s in skip):
                files.append(p)
    rt = ROOT / "run_all_agents.py"
    if rt.exists():
        files.append(rt)
    return files


# ------------------------------------------------- checks: LLM client ----

def check_model_constants() -> Result:
    """CLAUDE.md quotes the three model IDs verbatim."""
    c = _module_consts("ascent/llm/client.py")
    expect = {
        "DEFAULT_MODEL": "claude-opus-5",
        "SONNET_MODEL": "claude-sonnet-5",
        "HAIKU_MODEL": "claude-haiku-4-5-20251001",
    }
    bad = [f"{k}={c.get(k)!r} (doc says {v!r})"
           for k, v in expect.items() if c.get(k) != v]
    if bad:
        return False, "; ".join(bad)
    return True, "all three model IDs match CLAUDE.md"


def check_extract_text_exists() -> Result:
    """The 'never index content[0].text, use extract_text' rule needs the helper."""
    return _defines("ascent/llm/client.py", "extract_text")


def check_min_tokens_with_thinking() -> Result:
    c = _module_consts("ascent/llm/client.py")
    v = c.get("_MIN_TOKENS_WITH_THINKING")
    if v != 4096:
        return False, f"_MIN_TOKENS_WITH_THINKING={v}, CLAUDE.md says 4096"
    return True, "_MIN_TOKENS_WITH_THINKING == 4096"


def check_no_thinking_budget_kwarg() -> Result:
    """thinking={'type':'enabled','budget_tokens':N} 400s on Claude 5.

    ascent/llm/client.py is the sanctioned exception: it is the wrapper that
    serves both generations, and its budget_tokens call sits on the legacy
    branch gated by _CLAUDE_5_MODELS. Anywhere else, a direct budget_tokens is
    a latent 400 the moment that call is pointed at Opus or Sonnet.
    """
    offenders = []
    for p in _source_files():
        rel = str(p.relative_to(ROOT))
        if rel in ("ascent/llm/client.py", "scripts/verify_docs.py"):
            continue
        for i, line in enumerate(_code_only(rel), 1):
            if "budget_tokens" in line:
                offenders.append(f"{rel}:{i}")
    if offenders:
        return False, ("budget_tokens passed outside the llm/client.py wrapper: "
                       + ", ".join(offenders))
    if "_CLAUDE_5_MODELS" not in _read("ascent/llm/client.py"):
        return False, "client.py lost its _CLAUDE_5_MODELS gate for the legacy path"
    return True, "budget_tokens confined to the gated legacy path in llm/client.py"


def check_content_zero_text_only_on_haiku() -> Result:
    """`content[0].text` is safe only where thinking is off (Haiku legacy path).

    CLAUDE.md states the rule absolutely, but every live use is a Haiku call
    where block 0 really is the text. Enforce the invariant that actually
    matters: each indexing site must have a Haiku model in its immediate
    vicinity, so repointing it at Claude 5 trips this check.
    """
    offenders = []
    pat = re.compile(r"\.content\[0\]\s*\.\s*text")
    window = 20
    for p in _source_files():
        rel = str(p.relative_to(ROOT))
        if rel == "scripts/verify_docs.py":
            continue
        lines = _code_only(rel)
        for i, line in enumerate(lines):
            if not pat.search(line):
                continue
            near = "\n".join(lines[max(0, i - window): i + window])
            if not re.search(r"HAIKU_MODEL|claude-haiku", near):
                offenders.append(f"{rel}:{i + 1}")
    if offenders:
        return False, ("content[0].text with no Haiku model in scope (thinking is "
                       "on by default, so block 0 may be a thinking block): "
                       + ", ".join(offenders))
    return True, "every content[0].text site is a Haiku-only call"


# --------------------------------------------- checks: integrity rules ----

def check_alpha_weights_agree() -> Result:
    """Integrity constraint 6: DEFAULT_ALPHA_WEIGHTS must match in both files."""
    a = _module_consts("ascent/alpha/stack.py").get("DEFAULT_ALPHA_WEIGHTS")
    b = _module_consts("ascent/research/self_improve.py").get("DEFAULT_ALPHA_WEIGHTS")
    if a is None or b is None:
        return False, "DEFAULT_ALPHA_WEIGHTS missing from stack.py or self_improve.py"
    if set(a) != set(b):
        only_a = set(a) - set(b)
        only_b = set(b) - set(a)
        return False, (f"sleeve keys diverged — only in stack.py: {sorted(only_a)}; "
                       f"only in self_improve.py: {sorted(only_b)}")
    return True, f"{len(a)} sleeve keys agree across both files"


def check_fundamental_sleeve_disabled() -> Result:
    """Integrity constraint 7: fundamental is an anti-signal, must stay 0."""
    a = _module_consts("ascent/alpha/stack.py").get("DEFAULT_ALPHA_WEIGHTS") or {}
    key = next((k for k in a if "fundamental" in k), None)
    if key is None:
        return True, "no fundamental sleeve key present"
    if a[key] != 0.0:
        return False, f"{key}={a[key]} — constraint 7 requires 0.0 (IC-t -4.75)"
    return True, f"{key} == 0.0"


def check_water_fill_cap_exists() -> Result:
    """Integrity constraint 3: max-weight hard cap."""
    return _defines("ascent/portfolio/optimizer.py", "_water_fill_cap")


def check_point_in_time_helpers() -> Result:
    return _defines("ascent/data/store/point_in_time.py", "as_of_join", "as_of_merge")


def check_kill_switches_off() -> Result:
    """All four kill switches must stay False until paper validation clears."""
    spec = {
        "ascent/execution/event_runner.py": "EVENT_TRADING_ENABLED",
        "ascent/execution/twap_executor.py": "TWAP_ENABLED",
        "ascent/research/self_improve.py": "SELF_MODIFY_ENABLED",
        "run_all_agents.py": "LONG_SHORT_ENABLED",
    }
    bad = []
    for rel, name in spec.items():
        val = _module_consts(rel).get(name, "<missing>")
        if val is not False:
            bad.append(f"{name}={val!r} in {rel}")
    if bad:
        return False, "kill switch not False: " + "; ".join(bad)
    return True, "all four kill switches are False"


def check_judge_change_is_authority_capped() -> Result:
    """Integrity constraint 5: the one sanctioned debate write must be gated."""
    txt = _read("debate/judge.py")
    if "allowed_change_pct" not in txt:
        return False, "debate/judge.py no longer reads allowed_change_pct"
    return True, "judge.py gates position change on allowed_change_pct"


def check_single_position_change() -> Result:
    """Constraint 5: at most ONE judge position change is ever applied.

    Asserts the invariant, not a literal spelling: the code must read
    position_changes, take only its first element, and never iterate over it.
    Pinning the exact expression made this check fail on a pure refactor.
    """
    # _no_comments, not _code_only: "position_changes" is a dict key, so it
    # lives inside a string literal.
    lines = _no_comments("run_all_agents.py")
    src = "\n".join(lines)
    if "position_changes" not in src:
        return False, ("run_all_agents.py no longer reads position_changes at all — "
                       "constraint 5 is stale or the judge write was removed")
    # Whatever the list is bound to, only index 0 may be taken from it.
    takes_first = re.search(r"(position_changes|changes)\s*\[\s*0\s*\]", src)
    if not takes_first:
        return False, ("run_all_agents.py reads position_changes but never takes "
                       "[0] — constraint 5 (max one change) is unenforced")
    # And it must not loop over them.
    loops = [f"line {i}" for i, ln in enumerate(lines, 1)
             if re.search(r"for\s+\w+\s+in\s+(position_)?changes\b", ln)]
    if loops:
        return False, (f"run_all_agents.py iterates position_changes at {', '.join(loops)} "
                       f"— constraint 5 allows at most one")
    return True, "at most one judge position change is applied (takes [0], no loop)"


def check_no_loguru() -> Result:
    """loguru is not installed; an import would crash at runtime."""
    offenders = []
    for p in _source_files():
        txt = p.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^\s*(from\s+loguru|import\s+loguru)", txt, re.M):
            offenders.append(str(p.relative_to(ROOT)))
    if offenders:
        return False, "loguru imported in " + ", ".join(offenders)
    return True, "no loguru imports"


def check_main_returns_10_tuple() -> Result:
    """eod_runner and us_equities_agent unpack run_pipeline's return positionally.

    Scoped to run_pipeline: the module's private loaders legitimately return
    2-tuples, and counting those would make this check permanently red.
    """
    tree = ast.parse(_read("ascent/main.py"))
    fn = next((n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "run_pipeline"), None)
    if fn is None:
        return False, "ascent/main.py no longer defines run_pipeline"
    arities = {len(n.value.elts) for n in ast.walk(fn)
               if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)}
    if not arities:
        return False, "run_pipeline has no tuple return"
    if arities != {10}:
        return False, (f"run_pipeline returns tuples of arity {sorted(arities)} — "
                       "CLAUDE.md says 10; eod_runner.py and us_equities_agent.py "
                       "unpack positionally and will break")
    return True, "run_pipeline returns a 10-tuple"


def check_regime_engine_takes_dict() -> Result:
    """RegimeEngine(config=dict), not a Config object."""
    tree = ast.parse(_read("ascent/regime/engine.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RegimeEngine":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                    ann = ""
                    for a in sub.args.args:
                        if a.arg == "config" and a.annotation is not None:
                            ann = ast.unparse(a.annotation)
                    if "Dict" not in ann and "dict" not in ann:
                        return False, f"RegimeEngine.__init__ config annotated {ann!r}, expected a dict"
                    return True, f"RegimeEngine.__init__ takes config: {ann}"
    return False, "RegimeEngine.__init__ not found"


def check_discovery_guards_exist() -> Result:
    return _defines("run_all_agents.py",
                    "_is_near_scheduled_rebalance", "_insert_candidate_weights")


def check_sortino_annualized_once() -> Result:
    """The wf Sortino bug: annualizing numerator and denominator both."""
    txt = _read("ascent/research/wf_framework/metrics.py")
    m = re.search(r"def sortino.*?(?=\n    def )", txt, re.S)
    if not m:
        return False, "sortino() not found in wf_framework/metrics.py"
    body = m.group(0)
    dv_line = re.search(r"dv\s*=\s*downside\.std\(\)(.*)", body)
    if dv_line and "sqrt" in dv_line.group(1):
        return False, ("sortino() annualizes the downside deviation AND the "
                       "numerator — every result is the true value / sqrt(252)")
    return True, "sortino() annualizes exactly once"


def check_market_time_used_for_vendor_epochs() -> Result:
    """Vendor epoch timestamps must convert in market time, not host time.

    This host runs at UTC+7, so `datetime.fromtimestamp(ts).date()` shifts every
    post-close bar forward one calendar day. That misdated published equity bars
    and corrupted day-level attribution in get_portfolio_history(). Modules that
    turn a vendor epoch into a session date must use market_date_from_epoch.
    """
    try:
        _read("ascent/utils/market_time.py")
    except FileNotFoundError:
        return False, ("ascent/utils/market_time.py is missing — it has been lost "
                       "once before, leaving only a stale .pyc")
    # Files whose local-time conversion is legitimate: filesystem mtimes and
    # cosmetic display of news publication times.
    allowed = {
        "ascent/utils/market_time.py",
        "ascent/monitoring/pre_rebalance_checklist.py",  # file mtime, host-local is right
        "agents/ai_pm_agent.py",                         # news timestamp, display only
        "ascent/reporting/catalyst_scanner.py",          # utcfromtimestamp, host-independent
        "scripts/verify_docs.py",
    }
    offenders = []
    for p in _source_files():
        rel = str(p.relative_to(ROOT))
        if rel in allowed:
            continue
        for i, line in enumerate(_code_only(rel), 1):
            if re.search(r"(?<!utc)fromtimestamp\s*\(", line) and "tz=" not in line:
                offenders.append(f"{rel}:{i}")
    if offenders:
        return False, ("naive fromtimestamp on a vendor epoch (use "
                       "market_date_from_epoch): " + ", ".join(offenders))
    return True, "vendor epochs convert via market_time"


# ------------------------------------------------ checks: doc hygiene ----

# Metrics that must never be hardcoded into CLAUDE.md. They go stale within
# days and CLAUDE.md is loaded into every session, so a stale figure here is
# the single highest-leverage source of confidently-wrong output.
_NUMERIC_CLAIM_PATTERNS = [
    (r"Sharpe\s+[\d.]+", "Sharpe value"),
    (r"CAGR\s*[+\-]?[\d.]+%", "CAGR value"),
    (r"max\s*DD\s*[\-−][\d.]+%", "max drawdown value"),
    (r"[\-−+][\d.]+pp\b", "pp gap (counterfactual/attribution)"),
    (r"\bbeta\s+[\d.]+", "beta value"),
]


def check_claude_md_has_no_numbers() -> Result:
    """CLAUDE.md must point at artifacts, never restate their numbers."""
    txt = _read("CLAUDE.md")
    hits = []
    for line_no, line in enumerate(txt.splitlines(), 1):
        if line.lstrip().startswith(">"):  # quoted/archival context is exempt
            continue
        for pat, label in _NUMERIC_CLAIM_PATTERNS:
            for m in re.finditer(pat, line, re.I):
                hits.append(f"L{line_no} {label}: {m.group(0).strip()!r}")
    if hits:
        return False, (f"{len(hits)} hardcoded performance number(s) in CLAUDE.md — "
                       "cite CURRENT_VERIFIED_NUMBERS.md or the artifact instead: "
                       + "; ".join(hits[:8])
                       + (" ..." if len(hits) > 8 else ""))
    return True, "CLAUDE.md states no performance numbers"


def check_claude_md_paths_exist() -> Result:
    """Every repo path CLAUDE.md names must exist, or it misdirects every session."""
    txt = _read("CLAUDE.md")
    top = ("ascent", "agents", "debate", "orchestrator", "scripts", "tests",
           "docs", "data_cache", "logs", "outputs", "dashboard")
    # Only rooted, repo-relative paths. A bare `main.py` is prose, not a
    # citation, and absolute paths point outside the repo.
    cand = set()
    for m in re.finditer(r"`([A-Za-z0-9_./*-]+)`", txt):
        tok = m.group(1)
        if "/" not in tok:
            continue  # dotted module names and bare filenames are prose
        if tok.split("/", 1)[0] not in top:
            continue  # not rooted in this repo (e.g. the ascent-agri project)
        if "*" in tok or "YYYY" in tok:
            continue  # glob/placeholder, not a literal path
        cand.add(tok)
    cand.add("run_all_agents.py")  # the documented daily entrypoint
    missing = [c for c in sorted(cand) if not (ROOT / c).exists()]
    if missing:
        return False, (f"{len(missing)} path(s) named in CLAUDE.md do not exist: "
                       + ", ".join(missing))
    return True, f"all {len(cand)} repo paths named in CLAUDE.md resolve"


def check_claude_md_no_line_number_citations() -> Result:
    """Line numbers drift silently. Cite greppable symbol names instead."""
    txt = _read("CLAUDE.md")
    hits = []
    for line_no, line in enumerate(txt.splitlines(), 1):
        if line.lstrip().startswith(">"):
            continue
        for m in re.finditer(r"(?:~?line\s+\d+|\.py:\d+)", line, re.I):
            hits.append(f"L{line_no}: {m.group(0)!r}")
    if hits:
        return False, ("CLAUDE.md cites line numbers, which drift: "
                       + "; ".join(hits[:6]))
    return True, "CLAUDE.md cites no line numbers"


def check_repo_map_pointers_resolve() -> Result:
    """A wrong pointer in REPO_MAP.md is worse than a missing one.

    Agents trust this file to avoid grepping, so a stale symbol name sends them
    looking for something that does not exist and costs more than no map at all.
    """
    try:
        txt = _read("docs/REPO_MAP.md")
    except FileNotFoundError:
        return False, "docs/REPO_MAP.md is missing but CLAUDE.md points at it"
    top = ("ascent", "agents", "debate", "orchestrator", "scripts", "tests",
           "docs", "data_cache", "logs", "outputs", "dashboard")
    missing = []
    for m in re.finditer(r"`([A-Za-z0-9_./*<>-]+)`", txt):
        tok = m.group(1)
        if "/" not in tok or tok.split("/", 1)[0] not in top:
            continue
        if any(c in tok for c in "*<>") or "YYYY" in tok:
            continue
        if not (ROOT / tok).exists():
            missing.append(tok)
    if missing:
        return False, (f"{len(missing)} path(s) in REPO_MAP.md do not exist: "
                       + ", ".join(sorted(set(missing))))
    return True, "every path in REPO_MAP.md resolves"


def check_verified_numbers_matches_artifact() -> Result:
    """Every WF figure in CURRENT_VERIFIED_NUMBERS.md must match its artifact."""
    art_rel = "outputs/wf_results/wf_report_clean_2026-06-22.json"
    try:
        art = json.loads(_read(art_rel))
    except FileNotFoundError:
        return False, f"canonical artifact {art_rel} is missing"
    txt = _read("CURRENT_VERIFIED_NUMBERS.md")
    checks = [
        ("sharpe", art["sharpe"], 0.01, "Sharpe"),
        ("cagr", art["cagr"] * 100, 0.1, "CAGR %"),
        ("max_drawdown", abs(art["max_drawdown"]) * 100, 0.1, "max DD %"),
        ("beta", art["beta"], 0.01, "beta"),
    ]
    problems = []
    for field, expect, tol, label in checks:
        # find the doc's stated value for this metric
        pat = {
            "sharpe": r"Sharpe[^\d\n]{0,20}([\d.]+)",
            "cagr": r"CAGR[^\d\n]{0,20}\+?([\d.]+)%",
            "max_drawdown": r"[Mm]ax\s*(?:DD|drawdown)[^\d\n]{0,20}[\-−]?([\d.]+)%",
            "beta": r"[Bb]eta[^\d\n]{0,20}([\d.]+)",
        }[field]
        m = re.search(pat, txt)
        if not m:
            problems.append(f"{label} not stated in CURRENT_VERIFIED_NUMBERS.md")
            continue
        got = float(m.group(1))
        if abs(got - expect) > tol:
            problems.append(f"{label}: doc says {got}, artifact says {expect:.4f}")
    if problems:
        return False, "; ".join(problems)
    return True, "WF figures match the canonical artifact"


def check_no_unsourced_sharpe_published() -> Result:
    """0.518 matches no artifact in the repo but was published as 'the rigorous figure'."""
    art_dir = ROOT / "outputs" / "wf_results"
    known = set()
    for p in art_dir.glob("wf_report_*.json"):
        try:
            known.add(round(float(json.loads(p.read_text())["sharpe"]), 3))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    # A number is "cited" only when presented as current fact. Text that
    # documents a withdrawal must be able to name the withdrawn value.
    retraction = re.compile(
        r"withdraw|retract|no artifact|used to|prior version|superseded|"
        r"formerly|was hardcoded|matched no", re.I)
    offenders = []
    targets = ["scripts/generate_performance_page.py", "docs/methodology.md",
               "ascent/research/self_improve.py", "docs/index.html",
               "README.md", "CURRENT_VERIFIED_NUMBERS.md"]
    for rel in targets:
        try:
            txt = _read(rel)
        except FileNotFoundError:
            continue
        # In source files, only live code counts; a docstring explaining the
        # removal is not a citation.
        lines = _code_only(rel) if rel.endswith(".py") else txt.splitlines()
        for i, line in enumerate(lines, 1):
            if retraction.search(line):
                continue
            # Only walk-forward / OOS Sharpe claims are checked against the wf
            # artifacts. A live-account Sharpe is a different measurement and
            # legitimately has no wf_report behind it.
            if not re.search(r"walk-?forward|\bOOS\b|\bwf\b", line, re.I):
                continue
            for m in re.finditer(r"(?:Sharpe|sharpe)[^\d\n]{0,24}(0\.\d{3})", line):
                val = round(float(m.group(1)), 3)
                if val not in known:
                    offenders.append(f"{rel}:{i} cites walk-forward Sharpe {val} "
                                     f"as fact (no artifact has it)")
    if offenders:
        return False, "; ".join(offenders)
    return True, f"every cited Sharpe traces to one of {len(known)} artifacts"


# ------------------------------------------------------------ registry ----

CHECKS: List[Tuple[str, str, Callable[[], Result]]] = [
    # (name, the doc claim this defends, fn)
    ("model_constants", "CLAUDE.md LLM models block", check_model_constants),
    ("extract_text", "CLAUDE.md: use extract_text, never content[0]", check_extract_text_exists),
    ("min_tokens_thinking", "CLAUDE.md: wrappers raise to 4096", check_min_tokens_with_thinking),
    ("no_thinking_budget", "CLAUDE.md: never pass budget_tokens", check_no_thinking_budget_kwarg),
    ("content0_haiku_only", "CLAUDE.md: never index content[0].text", check_content_zero_text_only_on_haiku),
    ("alpha_weights_agree", "integrity constraint 6", check_alpha_weights_agree),
    ("fundamental_disabled", "integrity constraint 7", check_fundamental_sleeve_disabled),
    ("water_fill_cap", "integrity constraint 3", check_water_fill_cap_exists),
    ("point_in_time", "CLAUDE.md: always use as_of_join/as_of_merge", check_point_in_time_helpers),
    ("kill_switches_off", "CLAUDE.md kill switches", check_kill_switches_off),
    ("judge_authority_cap", "integrity constraint 5", check_judge_change_is_authority_capped),
    ("single_position_change", "integrity constraint 5", check_single_position_change),
    ("no_loguru", "CLAUDE.md: loguru not installed", check_no_loguru),
    ("main_10_tuple", "CLAUDE.md: main.py returns a 10-tuple", check_main_returns_10_tuple),
    ("regime_engine_dict", "CLAUDE.md: RegimeEngine takes a dict", check_regime_engine_takes_dict),
    ("discovery_guards", "CLAUDE.md discovery mini-rebalance gotchas", check_discovery_guards_exist),
    ("sortino_annualized_once", "CURRENT_VERIFIED_NUMBERS: Sortino is trustworthy", check_sortino_annualized_once),
    ("market_time_epochs", "CLAUDE.md: market dates via ascent/utils/market_time.py", check_market_time_used_for_vendor_epochs),
    ("claude_md_no_numbers", "CLAUDE.md must cite, not restate, figures", check_claude_md_has_no_numbers),
    ("claude_md_paths", "CLAUDE.md paths must resolve", check_claude_md_paths_exist),
    ("claude_md_no_line_nums", "CLAUDE.md must not cite line numbers", check_claude_md_no_line_number_citations),
    ("repo_map_pointers", "docs/REPO_MAP.md is a trustworthy index", check_repo_map_pointers_resolve),
    ("verified_numbers_match", "CURRENT_VERIFIED_NUMBERS is the SSOT", check_verified_numbers_matches_artifact),
    ("no_unsourced_sharpe", "no published number without an artifact", check_no_unsourced_sharpe_published),
]


def run() -> Tuple[int, int, list]:
    rows = []
    for name, claim, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:  # a check that cannot run is a failure
            ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
        rows.append({"check": name, "defends": claim, "ok": ok, "detail": detail})
    passed = sum(1 for r in rows if r["ok"])
    return passed, len(rows) - passed, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--quiet", action="store_true", help="print failures only")
    args = ap.parse_args()

    passed, failed, rows = run()

    if args.json:
        print(json.dumps({"passed": passed, "failed": failed, "checks": rows}, indent=2))
        return 1 if failed else 0

    for r in rows:
        if r["ok"] and args.quiet:
            continue
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"[{mark}] {r['check']:<26} {r['detail']}")
        if not r["ok"]:
            print(f"       defends: {r['defends']}")

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        print("Docs have drifted from the code. Fix the doc or the code, not this script.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
