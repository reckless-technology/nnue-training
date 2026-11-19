# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""The stamper: its three accepted inputs and its refusals, run as the script it is."""
import contextlib
import io
import runpy
import sys

import pytest

import rnnue1
import stamp_net

STAMPER = stamp_net.__file__


class Outcome:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def stamp(*arguments):
    """Run the stamper as a script, in this process so that coverage sees it: the CRC is pure Python and a
    12.6 MB payload costs seconds per pass, so a subprocess per case would dominate the suite."""
    out, err = io.StringIO(), io.StringIO()
    returncode = 0
    saved = sys.argv
    sys.argv = [STAMPER, *arguments]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(STAMPER, run_name="__main__")
    except SystemExit as exit_request:
        returncode = 0 if exit_request.code is None else 1
        if isinstance(exit_request.code, str):
            err.write(exit_request.code)
    finally:
        sys.argv = saved
    return Outcome(returncode, out.getvalue(), err.getvalue())


@pytest.fixture
def inputs(tmp_path, bare_payload):
    paths = {
        "bare": tmp_path / "bare.bin",
        "padded": tmp_path / "padded.bin",
        "appended": tmp_path / "appended.bin",
        "short": tmp_path / "short.bin",
    }
    paths["bare"].write_bytes(bare_payload)
    paths["padded"].write_bytes(bare_payload + (b"bullet" * 40)[:40])
    paths["appended"].write_bytes(bare_payload + b"junkjunkju")
    paths["short"].write_bytes(bare_payload[:100])
    return {name: str(path) for name, path in paths.items()}


def test_a_bare_payload_stamps_and_reads_back(inputs, tmp_path, bare_payload):
    result = stamp(inputs["bare"], str(tmp_path / "a.nnue"))
    assert result.returncode == 0, result.stderr
    assert rnnue1.read_payload(str(tmp_path / "a.nnue")) == bare_payload
    assert "king_bucket_map_crc" in result.stdout and "payload_crc" in result.stdout


def test_bullets_padded_output_stamps_identically_to_the_bare_payload(inputs, tmp_path):
    assert stamp(inputs["bare"], str(tmp_path / "a.nnue")).returncode == 0
    padded = stamp(inputs["padded"], str(tmp_path / "b.nnue"))
    assert padded.returncode == 0 and "dropping 40 bytes" in padded.stdout
    assert (tmp_path / "b.nnue").read_bytes() == (tmp_path / "a.nnue").read_bytes()


def test_a_stamped_net_restamps_in_place(inputs, tmp_path):
    net = str(tmp_path / "a.nnue")
    assert stamp(inputs["bare"], net).returncode == 0
    before = (tmp_path / "a.nnue").read_bytes()
    assert stamp(net).returncode == 0
    assert (tmp_path / "a.nnue").read_bytes() == before


@pytest.mark.parametrize("name", ["appended", "short"])
def test_a_wrong_length_input_is_refused(inputs, tmp_path, name):
    result = stamp(inputs[name], str(tmp_path / "no.nnue"))
    assert result.returncode != 0
    assert not (tmp_path / "no.nnue").exists()


def test_a_corrupt_stamped_net_is_not_quietly_reblessed(inputs, tmp_path):
    net = tmp_path / "a.nnue"
    assert stamp(inputs["bare"], str(net)).returncode == 0
    image = bytearray(net.read_bytes())
    image[-1] ^= 0xFF
    net.write_bytes(bytes(image))
    assert stamp(str(net), str(tmp_path / "no.nnue")).returncode != 0


def test_the_wrong_number_of_arguments_prints_usage():
    result = stamp()
    assert result.returncode != 0 and "usage" in result.stderr


def test_payload_of_names_both_lengths_for_an_unrelated_size(tmp_path):
    path = tmp_path / "other.bin"
    path.write_bytes(bytes(1000))
    with pytest.raises(SystemExit, match="1000 bytes"):
        stamp_net.payload_of(str(path))


def test_main_runs_in_process(inputs, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["stamp_net.py", inputs["bare"], str(tmp_path / "m.nnue")])
    stamp_net.main()
    assert "stamped" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["stamp_net.py"])
    with pytest.raises(SystemExit):
        stamp_net.main()
