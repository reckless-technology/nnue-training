# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Shared fixtures: puts `spec/` and `pytorch/` on `sys.path`, and builds `bulletformat` records.

Neither directory is a package (the tools in them are run as scripts), so the tests import their modules
by bare name, the way the scripts themselves do.
"""
import os
import struct
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for directory in ("spec", "pytorch"):
    sys.path.insert(0, os.path.join(REPOSITORY_ROOT, directory))

import bulletformat  # noqa: E402
import rnnue1  # noqa: E402

PIECE_TYPES = "pnbrqk"

# Positions with the side to move as White, which is the frame a bulletformat record is always stored in.
# They cover both king wings (so mirroring fires and does not), both board halves, and piece counts that
# land in several of the eight output buckets.
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/K7 w - - 0 1",
    "7k/8/8/8/8/8/8/K7 w - - 0 1",
    "r3k2r/8/8/3q4/3Q4/8/8/R3K2R w KQkq - 0 1",
    "8/5k2/8/8/8/2K5/6P1/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "2kr3r/ppp2ppp/2n1b3/3q4/3P4/2N2N2/PPP2PPP/2KR3R w - - 0 1",
    "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
    "k7/8/8/8/8/8/8/6QK w - - 0 1",
]


def record_for(fen, score=0, result=1):
    """Pack `fen` (white to move) into one 32-byte bulletformat record, as the crate writes it."""
    board = fen.split()[0]
    pieces = {}
    rank, file = 7, 0
    for character in board:
        if character == "/":
            rank, file = rank - 1, 0
        elif character.isdigit():
            file += int(character)
        else:
            color = 0 if character.isupper() else 1
            pieces[rank * 8 + file] = (color << 3) | PIECE_TYPES.index(character.lower())
            file += 1
    occupancy = 0
    nibbles = []
    own_king = opponent_king = 0
    for square in sorted(pieces):
        occupancy |= 1 << square
        nibbles.append(pieces[square])
        if pieces[square] & 7 == 5:
            if pieces[square] >> 3 == 0:
                own_king = square
            else:
                opponent_king = square ^ 56
    packed = bytearray(16)
    for slot, nibble in enumerate(nibbles):
        packed[slot // 2] |= nibble << (4 * (slot & 1))
    return (struct.pack("<Q", occupancy) + bytes(packed)
            + struct.pack("<hBBB", score, result, own_king, opponent_king) + b"\0\0\0")


@pytest.fixture
def records_file(tmp_path):
    """A bulletformat file holding one record per entry of `FENS`, scores and results all distinct."""
    path = tmp_path / "records.data"
    path.write_bytes(b"".join(record_for(fen, score=(index - 4) * 37, result=index % 3)
                              for index, fen in enumerate(FENS)))
    return str(path)


@pytest.fixture
def bare_payload():
    """A payload of exactly this architecture's size with a non-constant pattern."""
    return bytes((index * 7 + 3) % 251 for index in range(rnnue1.PAYLOAD_BYTES))


@pytest.fixture
def records(records_file):
    return np.array(bulletformat.read_records(records_file))
