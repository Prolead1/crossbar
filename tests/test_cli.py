"""Tests for the interactive ``crossbar price`` CLI.

Covers the non-interactive flag path and the one-question-at-a-time
interactive path, the barrier vs. vanilla reference table, the optional
vol-surface input, the prompt parser, and the error paths that ``main``
turns into exit code 2.
"""

import json
import runpy
import sys
from pathlib import Path

import pytest

from crossbar import EXAMPLE_QUOTES
from crossbar.cli import (
    PDE_MESH,
    _non_negative,
    _positive,
    _prompt,
    build_parser,
    main,
)


@pytest.fixture(autouse=True)
def _fast_pde(monkeypatch):
    """Shrink the fixed PDE mesh so the CLI tests stay quick."""
    monkeypatch.setitem(PDE_MESH, "M", 120)
    monkeypatch.setitem(PDE_MESH, "N", 120)


@pytest.fixture
def quotes_file(tmp_path):
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(EXAMPLE_QUOTES), encoding="utf-8")
    return str(path)


def _inputs(monkeypatch, mapping):
    """Answer interactive prompts from ``mapping``, keyed by a label substring."""

    def fake_input(prompt):
        for key, value in mapping.items():
            if key in prompt:
                return value
        raise AssertionError(f"unexpected prompt: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)


def _fx(vol="0.07"):
    return ["-S", "1.10", "-v", vol, "-T", "0.5", "-r", "0.04", "-q", "0.03"]


FX = _fx()
UP = ["-H", "1.20", "--type", "up-and-out", "--call"]
MC = ["--paths", "2000", "--steps", "20"]

BARRIER_PROMPTS = {
    "Spot S0": "1.10",
    "Vol (decimal or surface JSON)": "0.07",
    "Maturity": "0.5",
    "Risk-free": "0.04",
    "Carry": "0.03",
    "Barrier type": "uo",
    "Monitoring": "c",
    "Barrier level": "1.20",
    "Rebate": "",
    "Strike": "1.15",
    "Option type": "c",
}


# --------------------------------------------------------------------------
# Parser / entry point plumbing
# --------------------------------------------------------------------------


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "crossbar" in capsys.readouterr().out


def test_price_subcommand_is_required():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_parser_exposes_only_price():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions and set(actions[0].choices) == {"price"}


def test_parser_has_no_pde_mesh_or_instrument_flags():
    parser = build_parser()
    price = next(a for a in parser._actions if a.dest == "command").choices["price"]
    long_options = {opt for action in price._actions for opt in action.option_strings}
    assert {
        "--M",
        "--N",
        "--rannacher",
        "--engine",
        "--instrument",
        "--surface",
    }.isdisjoint(long_options)


def test_example_surface_matches_builtin_quotes():
    path = Path(__file__).resolve().parents[1] / "examples" / "vol_surface.json"
    sample = json.loads(path.read_text(encoding="utf-8"))
    assert sample == json.loads(json.dumps(EXAMPLE_QUOTES))


def test_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["crossbar", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("crossbar", run_name="__main__")
    assert exc.value.code == 0
    assert "crossbar" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Prompt parser
# --------------------------------------------------------------------------


def test_prompt_takes_default_on_blank(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert _prompt("X", 0.5, float) == 0.5


def test_prompt_allow_blank_returns_none(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert _prompt("X", None, str, allow_blank=True) is None


def test_prompt_fails_when_required_value_is_blank(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    with pytest.raises(ValueError, match="a value is required"):
        _prompt("X", None, float)


def test_prompt_fails_on_bad_choice(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "maybe")
    with pytest.raises(ValueError, match="choose one of"):
        _prompt("X", "up-and-out", str, ("up-and-out", "down-and-out"))


def test_prompt_fails_on_bad_cast(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "abc")
    with pytest.raises(ValueError, match="could not parse"):
        _prompt("X", None, float)


def test_prompt_runs_positive_validator(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "-1")
    with pytest.raises(ValueError, match="strictly positive"):
        _prompt("X", None, float, validate=_positive("X"))


def test_prompt_runs_non_negative_validator(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "-1")
    with pytest.raises(ValueError, match="non-negative"):
        _prompt("X", None, float, validate=_non_negative("X"))


# --------------------------------------------------------------------------
# Non-interactive flag path
# --------------------------------------------------------------------------


def test_price_barrier_table_shows_benchmark_and_engines(capsys):
    assert main(["price", *FX, *UP, *MC], interactive=False) == 0
    out = capsys.readouterr().out
    assert "up-and-out call, continuous" in out
    assert "benchmark : vanilla = 0.024153" in out
    assert "paths=2000, steps=20" in out
    assert "M=120, N=120, rannacher=1" in out
    assert "engine" in out and "std_error" in out and "delta" in out and "gamma" in out
    assert "analytic" in out and "mc" in out and "pde" in out
    assert "0.015186" in out  # closed-form barrier


def test_price_barrier_json_has_benchmark_and_three_engines(capsys):
    assert main(["price", *FX, *UP, *MC, "--json"], interactive=False) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["vanilla_benchmark"] == pytest.approx(0.024153, abs=1e-5)
    assert [row["engine"] for row in data["prices"]] == ["analytic", "mc", "pde"]
    assert data["mc_options"] == {
        "paths": 2000,
        "steps": 20,
        "seed": 0,
        "control_variate": True,
    }
    assert data["pde_options"] == {"M": 120, "N": 120, "rannacher": 1}
    assert data["quotes"] is None
    analytic, mc, pde = data["prices"]
    assert analytic["price"] == pytest.approx(0.015186, abs=1e-5)
    assert analytic["delta"] == pytest.approx(0.193211, abs=1e-5)
    assert analytic["gamma"] == pytest.approx(-3.213591, abs=1e-4)
    assert mc["std_error"] > 0.0
    assert mc["delta"] is not None
    assert mc["gamma"] is not None
    assert pde["price"] > 0.0
    assert pde["delta"] is not None
    assert pde["gamma"] is not None


def test_price_barrier_no_control_variate(capsys):
    flags = ["price", *FX, *UP, *MC, "--no-control-variate", "--json"]
    assert main(flags, interactive=False) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mc_options"]["control_variate"] is False


def test_price_with_surface_keeps_flat_vol_analytic(quotes_file, capsys):
    flags = ["price", *_fx(quotes_file), *UP]
    assert main(flags, interactive=False) == 0
    out = capsys.readouterr().out
    assert quotes_file in out
    assert "control_variate" not in out
    assert "vol       : 0.076000 (ATM from surface)" in out
    assert "benchmark : vanilla = 0.025974" in out
    assert "0.014261" in out  # flat ATM closed form is still reported
    assert "delta" in out
    assert "note: analytic" not in out
    assert "note: mc skipped" in out


def test_price_with_surface_json(quotes_file, capsys):
    flags = ["price", *_fx(quotes_file), *UP, "--json"]
    assert main(flags, interactive=False) == 0
    data = json.loads(capsys.readouterr().out)
    assert "mc_options" not in data
    assert data["vol_source"] == "ATM from surface"
    assert data["market"]["vol"] == pytest.approx(0.076)
    assert data["vanilla_benchmark"] == pytest.approx(0.0259738, abs=1e-6)
    analytic, mc, pde = data["prices"]
    assert analytic["price"] == pytest.approx(0.0142608, abs=1e-6)
    assert analytic["delta"] is not None
    assert mc["price"] is None and mc["delta"] is None and "surface" in mc["note"]
    assert mc["gamma"] is None
    assert pde["price"] is not None
    assert pde["delta"] is not None
    assert pde["price"] != analytic["price"]


def test_price_accepts_short_codes(capsys):
    flags = ["price", *_fx(), "-H", "1.20", "--type", "ui", "--monitor", "d", *MC, "--json"]
    assert main(flags, interactive=False) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["contract"]["type"] == "up-and-in"
    assert data["contract"]["monitor"] == "discrete"


def test_price_discrete_still_shows_analytic(capsys):
    flags = ["price", *FX, *UP, "--monitor", "discrete", *MC, "--json"]
    assert main(flags, interactive=False) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["vanilla_benchmark"] is not None
    analytic, mc, pde = data["prices"]
    assert analytic["price"] == pytest.approx(0.015186, abs=1e-5)  # continuous closed form
    assert analytic["delta"] is not None
    assert "note" not in analytic
    assert mc["price"] is not None and mc["delta"] is not None
    assert pde["price"] is not None and pde["delta"] is not None


# --------------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------------


def test_price_rejects_invalid_barrier(capsys):
    flags = ["price", *FX, "-H", "1.00", "--type", "up-and-out"]
    assert main(flags, interactive=False) == 2
    assert "crossbar: error" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bad",
    [
        ["-S", "0.0", "-v", "0.07", "-T", "0.5", "-H", "1.20"],
        ["-S", "1.1", "-v", "0.0", "-T", "0.5", "-H", "1.20"],
        ["-S", "1.1", "-v", "0.07", "-T", "0.0", "-H", "1.20"],
    ],
)
def test_price_validates_market(bad, capsys):
    assert main(["price", *bad], interactive=False) == 2
    assert "strictly positive" in capsys.readouterr().err


def test_price_missing_required_non_interactive(capsys):
    assert main(["price", "-S", "1.10"], interactive=False) == 2
    assert "missing required option --vol" in capsys.readouterr().err


def test_price_requires_a_barrier(capsys):
    assert main(["price", *FX], interactive=False) == 2
    assert "missing required option --barrier" in capsys.readouterr().err


def test_price_missing_surface_file(capsys):
    flags = ["price", *_fx("/does/not/exist.json"), *UP]
    assert main(flags, interactive=False) == 2
    assert "could not read vol surface" in capsys.readouterr().err


def test_price_rejects_non_positive_vol(capsys):
    flags = ["price", *_fx("0.0"), *UP]
    assert main(flags, interactive=False) == 2
    assert "vol must be strictly positive" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Interactive path
# --------------------------------------------------------------------------


def test_interactive_barrier_runs_all_engines(monkeypatch, capsys):
    _inputs(monkeypatch, BARRIER_PROMPTS)
    assert main(["price", *MC], interactive=True) == 0
    out = capsys.readouterr().out
    assert "up-and-out call" in out
    assert "K=1.15" in out
    assert "benchmark : vanilla = " in out
    assert "paths=2000, steps=20" in out
    assert "gamma" in out
    for label in ("analytic", "mc", "pde"):
        assert label in out


def test_interactive_prompts_show_full_choice_names(monkeypatch, capsys):
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        for key, value in BARRIER_PROMPTS.items():
            if key in prompt:
                return value
        raise AssertionError(f"unexpected prompt: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)
    assert main(["price", *MC], interactive=True) == 0
    joined = "\n".join(prompts)
    assert "(up-and-out/up-and-in/down-and-out/down-and-in) [uo]" in joined
    assert "(continuous/discrete) [c]" in joined
    assert "(call/put) [c]" in joined


def test_interactive_accepts_full_choice_names(monkeypatch, capsys):
    prompts = dict(BARRIER_PROMPTS)
    prompts["Barrier type"] = "up-and-in"
    prompts["Monitoring"] = "discrete"
    prompts["Option type"] = "put"
    _inputs(monkeypatch, prompts)
    assert main(["price", *MC, "--json"], interactive=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["contract"]["type"] == "up-and-in"
    assert data["contract"]["monitor"] == "discrete"
    assert data["contract"]["call"] is False


def test_interactive_barrier_with_surface_skips_mc(monkeypatch, quotes_file, capsys):
    prompts = dict(BARRIER_PROMPTS)
    prompts["Vol (decimal or surface JSON)"] = quotes_file
    prompts["Strike"] = ""  # blank -> spot
    _inputs(monkeypatch, prompts)
    assert main(["price"], interactive=True) == 0
    out = capsys.readouterr().out
    assert quotes_file in out
    assert "control_variate" not in out
    assert "note: mc skipped" in out


def test_interactive_with_all_flags_never_prompts(monkeypatch, quotes_file, capsys):
    def fail(_):
        raise AssertionError("should not prompt")

    monkeypatch.setattr("builtins.input", fail)
    flags = [
        "price",
        *_fx(quotes_file),
        *UP,
        "-K",
        "1.10",
        "--monitor",
        "continuous",
        "--rebate",
        "0",
    ]
    assert main(flags, interactive=True) == 0
    assert "pde" in capsys.readouterr().out


def test_interactive_bad_number_fails_fast(monkeypatch, capsys):
    _inputs(monkeypatch, {"Spot S0": "abc"})
    assert main(["price"], interactive=True) == 2
    assert "could not parse" in capsys.readouterr().err


def test_interactive_bad_choice_fails_fast(monkeypatch, capsys):
    prompts = dict(BARRIER_PROMPTS)
    prompts["Barrier type"] = "sideways"
    _inputs(monkeypatch, prompts)
    assert main(["price"], interactive=True) == 2
    assert "choose one of" in capsys.readouterr().err


def test_interactive_blank_required_fails_fast(monkeypatch, capsys):
    _inputs(monkeypatch, {"Spot S0": ""})
    assert main(["price"], interactive=True) == 2
    assert "a value is required" in capsys.readouterr().err


def test_interactive_positive_validation_fails_fast(monkeypatch, capsys):
    _inputs(monkeypatch, {"Spot S0": "0"})
    assert main(["price"], interactive=True) == 2
    assert "S0 must be strictly positive" in capsys.readouterr().err


def test_interactive_non_negative_validation_fails_fast(monkeypatch, capsys):
    prompts = dict(BARRIER_PROMPTS)
    prompts["Rebate"] = "-1"
    _inputs(monkeypatch, prompts)
    assert main(["price"], interactive=True) == 2
    assert "rebate must be non-negative" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Interactive auto-detection
# --------------------------------------------------------------------------


def test_auto_non_interactive_without_tty(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert main(["price", "-S", "1.10"]) == 2
    assert "missing required option" in capsys.readouterr().err


def test_auto_interactive_with_tty(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    _inputs(monkeypatch, BARRIER_PROMPTS)
    assert main(["price", *MC]) == 0
    assert "benchmark : vanilla" in capsys.readouterr().out


def test_non_interactive_flag_overrides_tty(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert main(["price", "--non-interactive", "-S", "1.10"]) == 2
    assert "missing required option" in capsys.readouterr().err
