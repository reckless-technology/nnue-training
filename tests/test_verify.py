# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""The verification gate's net loader and FEN collector, and its comparison against a stand-in engine."""
import stat
import sys

import numpy as np
import pytest

import features
import rnnue1
import verify


@pytest.fixture
def net_path(tmp_path):
    """A net whose weights are small and varied, so every section lands somewhere distinct."""
    generator = np.random.default_rng(7)
    payload = (generator.integers(-3, 4, rnnue1.INPUT_FEATURES * rnnue1.HIDDEN_SIZE, dtype=np.int16).tobytes()
               + generator.integers(-3, 4, rnnue1.HIDDEN_SIZE, dtype=np.int16).tobytes()
               + generator.integers(-3, 4, rnnue1.NUM_OUTPUT_BUCKETS * 2 * rnnue1.HIDDEN_SIZE,
                                    dtype=np.int16).tobytes()
               + generator.integers(-100, 100, rnnue1.NUM_OUTPUT_BUCKETS, dtype=np.int32).tobytes())
    path = tmp_path / "net.nnue"
    rnnue1.write_net(str(path), payload)
    return str(path), payload


def test_load_net_splits_the_payload_into_its_four_sections(net_path):
    path, payload = net_path
    transformer, transformer_bias, output_weight, output_bias = verify.load_net(path)
    assert transformer.shape == (rnnue1.INPUT_FEATURES, rnnue1.HIDDEN_SIZE)
    assert transformer_bias.shape == (rnnue1.HIDDEN_SIZE,)
    assert output_weight.shape == (rnnue1.NUM_OUTPUT_BUCKETS, 2 * rnnue1.HIDDEN_SIZE)
    assert output_bias.shape == (rnnue1.NUM_OUTPUT_BUCKETS,)
    assert transformer[0, 0] == np.frombuffer(payload, dtype=np.int16, count=1)[0]
    assert output_bias[-1] == np.frombuffer(payload[-4:], dtype=np.int32)[0]


def test_collect_fens_takes_the_first_field_skips_blanks_and_stops_at_the_count(tmp_path):
    path = tmp_path / "shard.txt"
    path.write_text("a b;1;2\n\n  \nc d;3;4\ne f;5;6\n")
    assert verify.collect_fens(str(path), 2) == ["a b", "c d"]
    assert verify.collect_fens(str(path), 10) == ["a b", "c d", "e f"]


FENS = ["4k3/8/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1"]


def write_engine(tmp_path, answers):
    engine = tmp_path / "engine.py"
    engine.write_text(f"#!{sys.executable}\nprint({answers!r})\n")
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    return str(engine)


def reference_answers(path):
    net = verify.load_net(path)
    return [features.integer_eval(*net, *features.position_features(fen)[1:]) for fen in FENS]


def run_main(monkeypatch, net, fens, engine):
    monkeypatch.setattr(sys, "argv", ["verify.py", "--net", net, "--fens", fens, "--count", "5",
                                      "--engine", engine])
    verify.main()


def test_main_passes_when_the_engine_agrees(net_path, tmp_path, monkeypatch, capsys):
    path, _ = net_path
    fens = tmp_path / "fens.txt"
    fens.write_text("\n".join(FENS) + "\n")
    engine = write_engine(tmp_path, " ".join(str(value) for value in reference_answers(path)))
    run_main(monkeypatch, path, str(fens), engine)
    assert "PASS" in capsys.readouterr().out


def test_main_fails_when_the_engine_differs(net_path, tmp_path, monkeypatch, capsys):
    path, _ = net_path
    fens = tmp_path / "fens.txt"
    fens.write_text("\n".join(FENS) + "\n")
    wrong = [value + 1 for value in reference_answers(path)]
    engine = write_engine(tmp_path, " ".join(str(value) for value in wrong))
    with pytest.raises(SystemExit) as raised:
        run_main(monkeypatch, path, str(fens), engine)
    assert raised.value.code == 1 and "FAIL" in capsys.readouterr().out


def test_main_fails_on_a_count_mismatch(net_path, tmp_path, monkeypatch):
    path, _ = net_path
    fens = tmp_path / "fens.txt"
    fens.write_text("\n".join(FENS) + "\n")
    engine = write_engine(tmp_path, "1")
    with pytest.raises(SystemExit, match="count mismatch"):
        run_main(monkeypatch, path, str(fens), engine)
