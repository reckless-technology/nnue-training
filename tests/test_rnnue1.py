# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""The RNNUE1 container module: the CRC, the map, the header layout and the reader's refusals."""
import struct

import pytest

import rnnue1


def test_crc64_matches_the_published_check_value():
    assert rnnue1.crc64_xz(b"123456789") == rnnue1.CRC64_CHECK_VALUE
    assert rnnue1.crc64_xz(b"") == 0


def test_king_bucket_map_is_the_mirrored_one():
    assert len(rnnue1.KING_BUCKET_MAP) == 64
    assert max(rnnue1.KING_BUCKET_MAP) == rnnue1.NUM_KING_BUCKETS - 1
    assert rnnue1.INPUT_FEATURES == rnnue1.NUM_KING_BUCKETS * rnnue1.BASE_FEATURES
    for square in range(64):
        file, half = square & 7, (square >> 3) // 4
        assert rnnue1.KING_BUCKET_MAP[square] == half * 4 + min(file, 7 - file)


def test_the_unmirrored_map_is_file_pairs(monkeypatch):
    monkeypatch.setattr(rnnue1, "IS_KING_MIRRORING_ENABLED", False)
    unmirrored = rnnue1._build_king_bucket_map()
    assert unmirrored[0:8] == bytes([0, 0, 1, 1, 2, 2, 3, 3])
    assert unmirrored != rnnue1.KING_BUCKET_MAP


def test_payload_size_is_the_documented_one():
    assert rnnue1.PAYLOAD_BYTES == 12_617_760
    assert rnnue1.FORMAT_VERSION == 2
    assert rnnue1.FLAGS == rnnue1.FLAG_KING_MIRRORING


def test_header_fields_and_crcs(bare_payload):
    built = rnnue1.header(bare_payload)
    assert len(built) == rnnue1.HEADER_BYTES
    assert built[:8] == rnnue1.MAGIC
    version, header_bytes, features, buckets, hidden, heads = struct.unpack_from("<HHHHHH", built, 8)
    assert (version, header_bytes) == (2, 64)
    assert (features, buckets, hidden, heads) == (768, 8, 1024, 8)
    assert built[20] == 1 and built[21] == 0
    assert struct.unpack_from("<HHH", built, 22) == (255, 64, 400)
    assert built[28:32] == bytes(4)
    map_crc, payload_bytes, payload_crc, header_crc = struct.unpack_from("<QQQQ", built, 32)
    assert map_crc == rnnue1.crc64_xz(rnnue1.KING_BUCKET_MAP)
    assert payload_bytes == rnnue1.PAYLOAD_BYTES
    assert payload_crc == rnnue1.crc64_xz(bare_payload)
    assert header_crc == rnnue1.crc64_xz(built[:56])


def test_header_refuses_a_payload_of_the_wrong_length():
    with pytest.raises(ValueError, match="payload is 10 bytes"):
        rnnue1.header(bytes(10))


def test_write_and_read_round_trip(tmp_path, bare_payload):
    path = str(tmp_path / "net.nnue")
    assert rnnue1.write_net(path, bare_payload) == rnnue1.HEADER_BYTES + rnnue1.PAYLOAD_BYTES
    assert rnnue1.read_payload(path) == bare_payload


@pytest.fixture
def stamped(tmp_path, bare_payload):
    path = tmp_path / "net.nnue"
    rnnue1.write_net(str(path), bare_payload)
    return path


def rewrite(path, transform):
    image = bytearray(path.read_bytes())
    image = transform(image)
    path.write_bytes(bytes(image))


def test_reader_refuses_a_file_that_is_too_short(tmp_path):
    path = tmp_path / "short.nnue"
    path.write_bytes(b"RNNUE1")
    with pytest.raises(SystemExit, match="not a net"):
        rnnue1.read_payload(str(path))


def test_reader_refuses_the_wrong_magic(stamped):
    rewrite(stamped, lambda image: b"ZNNUE5\0\0" + image[8:])
    with pytest.raises(SystemExit, match="not a net"):
        rnnue1.read_payload(str(stamped))


def test_reader_refuses_a_corrupt_header(stamped):
    def flip(image):
        image[12] ^= 0xFF
        return image
    rewrite(stamped, flip)
    with pytest.raises(SystemExit, match="header is corrupt"):
        rnnue1.read_payload(str(stamped))


def test_reader_refuses_a_truncated_payload(stamped):
    rewrite(stamped, lambda image: image[:-1])
    with pytest.raises(SystemExit, match="payload bytes, not the"):
        rnnue1.read_payload(str(stamped))


def test_reader_refuses_a_corrupt_payload(stamped):
    def flip(image):
        image[-1] ^= 0xFF
        return image
    rewrite(stamped, flip)
    with pytest.raises(SystemExit, match="payload is corrupt"):
        rnnue1.read_payload(str(stamped))
