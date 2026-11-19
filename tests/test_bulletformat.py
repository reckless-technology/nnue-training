# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""The bulletformat reader and featuriser, and the check tool that gates them."""
import sys

import numpy as np
import pytest

import bulletformat
import bulletformat_check
import features
from conftest import FENS, record_for


def test_record_layout_is_32_bytes_and_fields_decode(records):
    assert bulletformat.RECORD_DTYPE.itemsize == bulletformat.RECORD_BYTES == 32
    assert len(records) == len(FENS)
    assert int(records[0]["score"]) == -4 * 37
    assert int(records[4]["result"]) == 1


def test_record_count_of_refuses_a_partial_record(tmp_path):
    path = tmp_path / "partial.data"
    path.write_bytes(bytes(33))
    with pytest.raises(ValueError, match="not a whole number"):
        bulletformat.record_count_of(str(path))


def test_read_records_slices_and_refuses_the_past_the_end(records_file):
    assert len(bulletformat.read_records(records_file)) == len(FENS)
    window = bulletformat.read_records(records_file, count=3, first_record=2)
    assert len(window) == 3 and int(window[0]["score"]) == (2 - 4) * 37
    assert len(bulletformat.read_records(records_file, count=1000, first_record=8)) == 2
    with pytest.raises(ValueError, match="no records at or after"):
        bulletformat.read_records(records_file, first_record=len(FENS))


def test_record_to_fen_round_trips_the_placement(records):
    for fen, record in zip(FENS, records):
        assert bulletformat.record_to_fen(record) == fen.split()[0] + " w - - 0 1"


def test_vectorised_features_equal_the_text_path_for_every_position(records):
    own, opponent, _, _ = bulletformat.featurise_records(records)
    assert own.shape == (len(FENS), features.MAX_ACTIVE_FEATURES)
    for row, fen in enumerate(FENS):
        _, text_own, text_opponent = features.position_features(fen)
        pad = features.PADDING_INDEX
        assert sorted(own[row][own[row] != pad].tolist()) == sorted(text_own)
        assert sorted(opponent[row][opponent[row] != pad].tolist()) == sorted(text_opponent)
        assert int((own[row] != pad).sum()) == len(text_own)


def test_scores_and_results_are_scaled(records):
    _, _, scores, results = bulletformat.featurise_records(records)
    assert scores.dtype == np.float32 and scores[1] == (1 - 4) * 37
    assert results[:3].tolist() == [0.0, 0.5, 1.0]


def test_featurise_refuses_more_than_thirty_two_pieces():
    record = bytearray(record_for(FENS[0]))
    record[0:8] = (0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
    with pytest.raises(ValueError, match="at most 32"):
        bulletformat.featurise_records(np.frombuffer(bytes(record), dtype=bulletformat.RECORD_DTYPE))


def test_a_single_record_block_featurises(records):
    own, _, _, _ = bulletformat.featurise_records(records[:1])
    assert own.shape[0] == 1


def test_layout_check_accepts_good_records_and_reports_each_fault(records, capsys):
    assert bulletformat_check.check_record_layout(records) == 0
    bad = records.copy()
    bad[0]["own_king_square"] = 1                      # wrong ksq
    bad[1]["opponent_king_square"] = 1                 # wrong opp_ksq
    bad[2]["result"] = 7                               # not 0, 1 or 2
    bad[3]["extra"] = [1, 0, 0]                        # extra bytes non-zero
    bad[5]["pieces"][0] &= 0x0F                        # the own king (slot 1) becomes a pawn: no own king
    assert bulletformat_check.check_record_layout(bad) == 5
    printed = capsys.readouterr().out
    assert "ksq says" in printed and "opp_ksq says" in printed and "king" in printed


def test_a_bitboard_with_more_pieces_than_fit_counts_as_one_failure_and_does_not_crash(records, capsys):
    bad = records[:1].copy()
    bad[0]["occupancy"] = (1 << 33) - 1   # 33 pieces: one more than the 32 nibbles the record holds
    assert bulletformat_check.check_record_layout(bad) == 1
    assert "33 pieces" in capsys.readouterr().out


def test_feature_check_agrees_and_detects_a_difference(records, capsys):
    assert bulletformat_check.check_features_match(records) == 0
    bad = records.copy()
    bad[0]["pieces"][0] = (bad[0]["pieces"][0] & 0xF0) | 0x00   # a1 rook turned into a pawn: the text path differs
    # The FEN is rebuilt from the same bytes, so the two paths still agree; this proves the check is not vacuous
    # only if it can disagree, so break the vectorised side instead.
    original = bulletformat.featurise_records
    try:
        bulletformat.featurise_records = lambda block: tuple(
            array + 1 if index < 2 else array for index, array in enumerate(original(block)))
        assert bulletformat_check.check_features_match(records) == len(FENS)
    finally:
        bulletformat.featurise_records = original
    assert "differs" in capsys.readouterr().out


def test_benchmark_runs(records_file, capsys):
    bulletformat_check.benchmark(records_file)
    assert "positions/sec" in capsys.readouterr().out


def test_main_passes_on_good_records(records_file, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bulletformat_check.py", records_file, "--positions", "10", "--benchmark"])
    assert bulletformat_check.main() == 0
    assert "[PASS]" in capsys.readouterr().out


def test_main_fails_on_bad_records(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bad.data"
    path.write_bytes(record_for(FENS[0], result=9))
    monkeypatch.setattr(sys, "argv", ["bulletformat_check.py", str(path)])
    assert bulletformat_check.main() == 1
    assert "[FAIL]" in capsys.readouterr().out
