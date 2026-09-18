"""Tests for the command-line interface.

Exercises every subcommand in both table and ``--json`` modes, the
engine dispatch (analytic / Monte Carlo / PDE), the volatility-surface
quotes file, and the error paths that ``main`` turns into exit code 2.
"""

import json
import runpy
import sys

import numpy as np
import pytest

from crossbar import EXAMPLE_QUOTES
from crossbar.cli import build_parser, main


@pytest.fixture
def quotes_file(tmp_path):
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(EXAMPLE_QUOTES), encoding="utf-8")
    return str(path)


FX = ["-S", "1.10", "-v", "0.07", "-T", "0.5", "-r", "0.04", "-q", "0.03"]
UP = ["-H", "1.20", "--type", "up-and-out", "--call"]
# A coarse but non-degenerate PDE mesh (the barrier sits near the strike).
PDE = ["--M", "120", "--N", "120"]
MC = ["--paths", "2000", "--steps", "20"]


# --------------------------------------------------------------------------
# Parser / entry point plumbing
# --------------------------------------------------------------------------


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "crossbar" in capsys.readouterr().out


def test_subcommand_is_required():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_build_parser_has_all_subcommands():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions and set(actions[0].choices) == {
        "vanilla",
        "price",
        "variants",
        "greeks",
        "surface",
        "plot",
    }


def test_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["crossbar", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("crossbar", run_name="__main__")
    assert exc.value.code == 0
    assert "crossbar" in capsys.readouterr().out


# --------------------------------------------------------------------------
# vanilla
# --------------------------------------------------------------------------


def test_vanilla_table_put_and_default_strike(capsys):
    assert main(["vanilla", *FX, "--put", "-K", "1.20"]) == 0
    out = capsys.readouterr().out
    assert "vanilla put" in out
    assert "strike=1.2" in out


def test_vanilla_json_uses_spot_as_default_strike(capsys):
    assert main(["vanilla", *FX, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["is_call"] is True
    assert data["strike"] == pytest.approx(1.10)
    assert data["price"] > 0.0


# --------------------------------------------------------------------------
# price
# --------------------------------------------------------------------------


def test_price_analytic_table(capsys):
    assert main(["price", *FX, *UP, "--engine", "analytic"]) == 0
    out = capsys.readouterr().out
    assert "price (analytic)" in out


def test_price_pde_json(capsys):
    assert main(["price", *FX, *UP, "--engine", "pde", *PDE, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["engine"] == "pde"
    assert data["price"] > 0.0


def test_price_mc_table_without_control_variate(capsys):
    flags = ["price", *FX, *UP, "--engine", "mc", *MC, "--no-control-variate"]
    assert main(flags) == 0
    out = capsys.readouterr().out
    assert "price (mc)" in out
    assert "+/-" in out


def test_price_mc_default_control_variate(capsys):
    assert main(["price", *FX, *UP, "--engine", "mc", *MC, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["std_error"] > 0.0


def test_price_pde_with_surface_file(quotes_file, capsys):
    flags = ["price", *FX, *UP, "--engine", "pde", *PDE, "--surface", quotes_file]
    assert main(flags) == 0
    assert "price (pde)" in capsys.readouterr().out


def test_price_analytic_rejects_discrete_monitor(capsys):
    flags = ["price", *FX, *UP, "--engine", "analytic", "--monitor", "discrete"]
    assert main(flags) == 2
    assert "continuous" in capsys.readouterr().err


def test_price_rejects_invalid_inputs(capsys):
    # up barrier below spot fails validate_inputs inside the handler
    flags = ["price", *FX, "-H", "1.00", "--type", "up-and-out"]
    assert main(flags) == 2
    assert "crossbar: error" in capsys.readouterr().err


# --------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------


def test_variants_pde_table(capsys):
    assert main(["variants", *FX, *PDE]) == 0
    out = capsys.readouterr().out
    assert "up-and-out call" in out
    assert "down-and-in put" in out


def test_variants_analytic_json(capsys):
    assert main(["variants", *FX, "--engine", "analytic", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["variants"]) == 8
    assert data["engine"] == "analytic"


def test_variants_mc_json(capsys):
    assert main(["variants", *FX, "--engine", "mc", *MC, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["variants"]) == 8


def test_variants_pde_with_surface(quotes_file, capsys):
    flags = ["variants", *FX, *PDE, "--surface", quotes_file, "--json"]
    assert main(flags) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["variants"]) == 8


def test_variants_analytic_rejects_discrete_monitor(capsys):
    flags = ["variants", *FX, "--engine", "analytic", "--monitor", "discrete"]
    assert main(flags) == 2
    assert "continuous" in capsys.readouterr().err


# --------------------------------------------------------------------------
# greeks
# --------------------------------------------------------------------------


def test_greeks_pde_explicit_spots(capsys):
    flags = ["greeks", *FX, *UP, "--engine", "pde", *PDE, "--spots", "1.0,1.1,1.15"]
    assert main(flags) == 0
    out = capsys.readouterr().out
    assert "spot" in out and "delta" in out and "gamma" in out


def test_greeks_mc_default_spots_json(capsys):
    flags = ["greeks", *FX, *UP, "--engine", "mc", *MC, "--json"]
    assert main(flags) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["profile"]) == 5
    assert data["engine"] == "mc"
    assert np.isfinite(data["profile"][0]["price"])


# --------------------------------------------------------------------------
# surface
# --------------------------------------------------------------------------


def test_surface_term_structure_table(capsys):
    assert main(["surface"]) == 0
    out = capsys.readouterr().out
    assert "atm" in out and "25c" in out
    assert out.count("\n") >= len(EXAMPLE_QUOTES)


def test_surface_term_structure_json_from_file(quotes_file, capsys):
    assert main(["surface", "--quotes", quotes_file, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "term-structure"
    assert len(data["maturities"]) == len(EXAMPLE_QUOTES)


def test_surface_sigma_lookup(capsys):
    assert main(["surface", "-S", "1.10", "--strike", "1.10", "--maturity", "0.5"]) == 0
    assert "sigma(K=" in capsys.readouterr().out


def test_surface_strike_lookup_table_and_json(capsys):
    assert main(["surface", "--delta", "0.25", "--maturity", "0.5"]) == 0
    assert "strike(delta=" in capsys.readouterr().out

    assert main(["surface", "--delta", "0.25", "--maturity", "0.5", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "strike"
    assert data["strike"] > 0.0


def test_surface_lookup_requires_maturity(capsys):
    assert main(["surface", "--delta", "0.25"]) == 2
    assert "maturity" in capsys.readouterr().err


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def test_plot_writes_image(quotes_file, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    out = tmp_path / "surface.png"
    flags = [
        "plot",
        "-S",
        "1.10",
        "--quotes",
        quotes_file,
        "--out",
        str(out),
        "--strikes",
        "7",
        "--maturities",
        "4",
    ]
    assert main(flags) == 0
    assert out.exists() and out.stat().st_size > 0

    # default strike/maturity grids and the built-in example surface
    out2 = tmp_path / "surface_default.png"
    assert main(["plot", "-S", "1.10", "--out", str(out2)]) == 0
    assert out2.exists() and out2.stat().st_size > 0
