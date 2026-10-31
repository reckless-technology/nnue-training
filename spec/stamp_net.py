#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Wrap a raw trainer payload in the shared Reckless `RNNUE1` container.

A trainer writes a headerless quantised net. This tool prepends the 64-byte header every Reckless engine
reads, **written entirely from `rnnue1.py`'s own constants** -- the architecture fields, the king-bucket
map's CRC-64, the payload length, and then the payload's and the header's own CRC-64. Nothing is taken
from an argument, so the header always asserts the architecture this repository trained, and a payload of
any other shape gets a header describing *this* shape and is refused by every engine on first load rather
than loading the wrong weights. Never hand-edit a net file; stamp it.

    spec/stamp_net.py checkpoints/quantised.bin nets/pawnstar-v20.nnue
    spec/stamp_net.py nets/pawnstar-v20.nnue                           # re-stamp in place

Three inputs are accepted, and the only difference is where the payload starts and how long it is:

* **A net already in the RNNUE1 container** is read back through its own checks first, so re-stamping a
  corrupt net fails rather than quietly re-blessing it.
* **A bare payload** of exactly `PAYLOAD_BYTES`, straight from the PyTorch pipeline.
* **bullet's quantised output**, which is the same payload padded up to a 64-byte boundary with repeats
  of the word `bullet`. The padding is dropped: RNNUE1 records an exact payload length and the engine's
  loader checks the file holds nothing after it, which is what lets a truncated or an appended-to file be
  told apart from a sound one.

This replaced a `stamp_net.cpp` compiled into each engine against that engine's own constants, which made
`king_bucket_map_crc` agree tautologically -- see README.md, *Why the net stamper lives here*. An engine
carrying a retired container format of its own keeps that conversion in its own tool: this one knows the
shared container and a raw payload, and nothing else.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnnue1  # noqa: E402  (the path above is what makes this importable when run from elsewhere)

BULLET_PADDING_WORD = b"bullet"


def payload_of(path):
    """The weight payload inside `path`, whichever of the three accepted inputs it is.

    Raises SystemExit if it is none of them, naming both lengths, because "wrong architecture" and
    "truncated file" are the two things a length mismatch means and the numbers tell them apart.
    """
    with open(path, "rb") as source:
        image = source.read()

    if image[:8] == rnnue1.MAGIC:
        return rnnue1.read_payload(path)

    if len(image) == rnnue1.PAYLOAD_BYTES:
        return image

    # bullet pads its quantised output up to a 64-byte boundary, so anything in [payload, payload + 64)
    # is this architecture's payload plus padding. Check the padding is what bullet writes before
    # dropping it: a file that is a little too long for another reason must not be silently truncated.
    padding_length = len(image) - rnnue1.PAYLOAD_BYTES
    if 0 < padding_length < 64:
        padding = image[rnnue1.PAYLOAD_BYTES:]
        expected = (BULLET_PADDING_WORD * padding_length)[:padding_length]
        if padding != expected:
            raise SystemExit(f"{path}: {len(image)} bytes, which is this architecture's "
                             f"{rnnue1.PAYLOAD_BYTES} plus {padding_length} trailing bytes that are not "
                             f"bullet's padding ({padding!r}) -- truncated, appended to, or another format")
        print(f"{path}: dropping {padding_length} bytes of bullet's tail padding")
        return image[:rnnue1.PAYLOAD_BYTES]

    raise SystemExit(f"{path}: {len(image)} bytes -- not a net in the RNNUE1 container, and not this "
                     f"architecture's {rnnue1.PAYLOAD_BYTES}-byte payload. A net for another "
                     f"architecture must be stamped by a trainer built for it.")


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: stamp_net.py <in.bin|in.nnue> [out.nnue]   (one argument stamps in place)")
    source_path = sys.argv[1]
    target_path = sys.argv[2] if len(sys.argv) == 3 else source_path

    payload = payload_of(source_path)
    written = rnnue1.write_net(target_path, payload)

    # Print every stamped value. Each engine's net table records these, so a disagreement between two
    # nets of the same size is then a number to compare rather than two weight arrays to read.
    print(f"stamped {target_path}  ({written} bytes: {rnnue1.HEADER_BYTES}-byte RNNUE1 header + "
          f"{len(payload)}-byte payload)")
    print(f"  format_version {rnnue1.FORMAT_VERSION}  features_per_bucket {rnnue1.BASE_FEATURES}  "
          f"king_buckets {rnnue1.NUM_KING_BUCKETS}  hidden_size {rnnue1.HIDDEN_SIZE}  "
          f"output_buckets {rnnue1.NUM_OUTPUT_BUCKETS}")
    print(f"  flags {rnnue1.FLAGS:#04x} (king mirroring "
          f"{'on' if rnnue1.IS_KING_MIRRORING_ENABLED else 'off'})  "
          f"activation {rnnue1.ACTIVATION_SCRELU}  qa {rnnue1.QUANT_ACCUMULATOR}  "
          f"qb {rnnue1.QUANT_OUTPUT}  eval_scale {rnnue1.EVALUATION_SCALE}")
    print(f"  king_bucket_map_crc {rnnue1.crc64_xz(rnnue1.KING_BUCKET_MAP):#018x}")
    print(f"  payload_crc         {rnnue1.crc64_xz(payload):#018x}")


if __name__ == "__main__":
    main()
