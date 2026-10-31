# nnue-training

An NNUE trainer for chess: one network architecture, one net file format, and two independent training
front ends that produce the same artefact. It is the single definition of the architecture, the
king-bucket map, the quantisation and the `RNNUE1` net container, so a change to any of those is made
here rather than in whatever program loads the result.

**What it produces** is a single `.bin` file: a quantised network wrapped in the 64-byte `RNNUE1`
container, which carries every architecture field, the king-bucket map's digest and three CRC-64
checksums, so a loader can refuse a net built for a different shape instead of evaluating from the
wrong weight bank. `RNNUE1` is the format read by the Reckless chess engines —
[`pawnstar`](https://github.com/reckless-technology/pawnstar) (C++23),
[`pawnstar-rs`](https://github.com/reckless-technology/pawnstar-rs) (Rust),
[`pawnstar-go`](https://github.com/reckless-technology/pawnstar-go) (Go) and
[`zenith`](https://github.com/reckless-technology/zenith) (C17) — which is why the container is shaped
the way it is.

**It ships no networks, no binaries and no training data.** You supply the data; see
[The data this trainer reads](#the-data-this-trainer-reads).

## The architecture

```
(768 x 8 king buckets, horizontally mirrored) -> 1024 x2 perspectives -> 8 material output heads -> 1
```

SCReLU; `QA` 255, `QB` 64, evaluation scale 400. The numbers are written down once, in
[`spec/rnnue1.py`](spec/rnnue1.py), and every engine checks a net's header against its own compiled
constants at load time.

- **768 base features** per king bucket: 2 colours x 6 piece types x 64 squares, from the moving side's
  perspective.
- **8 king buckets**: 4 files x 2 board halves. **Horizontal king mirroring** flips a perspective whose
  own king is on files e-h (`square ^ 7`) onto a-d before indexing, so every bucket trains on both
  wings — twice the effective data per bucket at the same file size.
- **8 material output heads**, selected by `(piece_count - 2) / 4`, so the endgame and the middlegame
  do not share one output layer.

Mirroring and the output heads were each measured and SPRT-gated before they became part of the shape:
**+14.3 Elo** and **+14.7 Elo** at fixed depth. Both only paid past about 1.4 billion
training positions — at 190M they were neutral or negative.

### An open question about the bucketing, measured and not decided

A census over 3,859,988,659 training positions found the king bucketing misallocates capacity badly:
buckets 4-7, the king on ranks 5-8 in its own frame, are **3,145,728 weights — 49.86% of the net —
and see 3.747% of the data**. Bucket 4 gets **62 positions per weight** against bucket 1's **4,196**,
a 68x skew. No achievable amount of data fixes it (262 billion positions would be needed, and at 6B
the gap *widens* in absolute terms), and because g1 alone is 31.66% of king-perspective instances,
**no 8-way partition of king squares can beat 3.24x skew on real game data**. Dropping the board-half
split to four file-pair buckets reaches 3.69x, halves the feature transformer and removes a class of
accumulator refresh.

**But it did not translate into a better net.** Measured 2026-10-28: a 4-bucket run at matched compute
ended at 0.012335 held-out loss against the 8-bucket run's 0.012253 — 0.67% worse on 50.1% of the
feature weights, at or below the baseline in only 51 of 193 superbatches and all of those early. So it
is a parameter-efficiency result, not a loss win. A 4-bucket net also needs a 4-bucket engine build,
which the trainer example cannot currently express, so the shape is not yet a configuration either
side can state.

## The `RNNUE1` net container

A trained net is 64 bytes of header followed by the quantised payload — 12,617,760 bytes at this shape.
[`spec/rnnue1.py`](spec/rnnue1.py) is the reference implementation: the magic `RNNUE1\0\0`, a format
version that moves with the architecture, every architecture field, the payload length, and three
CRC-64/XZ digests — one over the king-bucket map, one over the payload and one over the header itself.
It is **stdlib only** on purpose, so a machine with no trainer environment can still generate and check
a net.

The container is self-describing where the formats it replaced were not. Those carried an 8-byte magic
and nothing else, so a net for a different architecture was told apart only by its file size, and two
different king-bucket maps of the same cardinality were indistinguishable — which loaded cleanly and
evaluated from the wrong weight bank. Here every architecture field is checked by the loading engine
against its own compile-time constants, the map is checked by digest, and the CRCs make a corrupt or
truncated file refuse rather than play.

## Layout

```
bullet/     the bullet trainer example and its setup — architecture, save format, environment variables
pytorch/    the PyTorch pipeline, including the vectorised bulletformat reader
pipeline/   harvest, extraction, deduplication, shuffle, pool audit, the cross-port signature gate
spec/       the RNNUE1 container: the reference implementation, and the net stamper
```

## Running a training

The production route is the [bullet](https://github.com/jw1912/bullet) trainer, on a CUDA GPU. bullet
is a dependency rather than something vendored here: the setup script clones it at a pinned commit and
installs this repository's two example sources into it.

```sh
# Once: clone bullet at the pinned commit, install the trainer and reference evaluator as examples,
# and register them in bullet's Cargo.toml. Idempotent.
bullet/setup_bullet.sh                       # BULLET=<dir> to choose the checkout (default ~/pawnstar_nnue/bullet)

# Build and run the trainer. It is configured entirely through the environment:
cd "$BULLET/crates/bullet_lib"
PAWNSTAR_DATA=/path/to/shuffled.data \
PAWNSTAR_BPS=3662 PAWNSTAR_SBS=200 PAWNSTAR_WDL=0.0 PAWNSTAR_OUT=checkpoints \
    cargo run --release --features cuda --example pawnstar
```

| Variable | Meaning | Default |
|---|---|---|
| `PAWNSTAR_DATA` | path to a bulletformat `.data` file | `data/pawnstar.data` |
| `PAWNSTAR_BPS` | batches per superbatch (a batch is 16,384 positions) | 1000 |
| `PAWNSTAR_SBS` | end superbatch | 30 |
| `PAWNSTAR_LR_STEP` | superbatches per learning-rate drop | `PAWNSTAR_SBS / 3` |
| `PAWNSTAR_OUT` | checkpoint directory | `checkpoints` |
| `PAWNSTAR_SAVE_RATE` | checkpoint every this many superbatches | 10 |
| `PAWNSTAR_WDL` | how much of the target is the game result rather than the eval | 0.5 |
| `PAWNSTAR_RESUME` / `PAWNSTAR_START_SB` | checkpoint directory to continue from, and the absolute superbatch to resume at | train from scratch |

A superbatch here is about 60M positions (`PAWNSTAR_BPS` near 3662), which leaves `PAWNSTAR_SBS` to set
how many passes are made over a pool larger than that. It is **not** one superbatch per pass over the
data. The trainer prints the learning-rate schedule it will follow and **refuses to start** one whose
last drop can never apply, which is a failure that had already cost five runs their final drop.

The width is fixed in the trainer example (`HIDDEN_SIZE`) and must equal the engine's own hidden size;
changing the architecture means changing it in `spec/rnnue1.py`, in the trainer example and in every
engine, together.

Other entry points:

- `pipeline/run_gendata.sh` — launch parallel self-play data-generation shards. Workers are given
  disjoint slices of the opening pool, so no two workers in a campaign can generate the same game by
  construction rather than merely unlikely.
- `pipeline/check_duplicates.sh` — audit what fraction of an assembled pool is distinct positions, and
  check the campaign identities in the shards' `.meta` provenance sidecars.
- `pipeline/train_pipeline.sh`, `pipeline/run_experiment.sh` — the end-to-end "text shards in, verified
  net out" and "train, verify, SPRT" wrappers. **These two still call the paths of the engine
  repository they were written in**, and two of the gates they invoke (`verify_net.sh`, `run_sprt.sh`)
  are deliberately engine-side and are not here; see *What is not finished* below.
- `pytorch/train.py`, `pytorch/verify.py` — the independent PyTorch pipeline, which trains from text
  records (`fen;score;wdl`) and has a verification gate that requires the engine's integer evaluation to
  equal the Python reference exactly.
- `pytorch/bulletformat_check.py` — the correctness gate for the bulletformat reader: every record's
  feature indices are compared against the text path's, and the decoded board is checked against chess.
- `pipeline/new-shape-signature.sh` — the cross-port gate. It searches six fixed positions at a fixed
  depth over a net generated from nothing, so a net change and a code change can never be confused for
  one another. Three engine repositories commit this file byte-identically, and the copies agreeing
  *is* the audit; it must only ever be changed in all of them at once.

## What belongs here, and what does not

The split is **producer asserts, consumer verifies**. This repository produces nets and asserts what
they are; each engine verifies against its own compiled constants and refuses loudly on a mismatch.

| Here | In each engine |
|---|---|
| Both trainers — the bullet example and the PyTorch pipeline | The **loader**, in that engine's own language |
| The `RNNUE1` container: header, the CRCs, the bullet-save conversion | Validation of every header field against its own constants |
| The king-bucket map, as **the** definition | `verify_net.sh` — "does this net load *and play* here" is the one question only that engine can answer |
| Data pipeline: harvest, extraction, deduplication, shuffle, pool audit | Self-play datagen, which links the engine's own core |
| The cross-port node-count signature gate | |

### Why the net stamper lives here

`stamp_net` used to live in `pawnstar` and derive the net header from *that engine's* compile-time
constants. That describes the reader's expectation rather than the artefact's truth, and it had a
consequence worth recording.

`king_bucket_map_crc` exists to catch a **trainer/engine map drift** — the failure where the trainer
trains one network and the engine runs another. But a stamper compiled from the engine's own map writes
that map's digest **by construction**, and the loader then checks it against the same map. It agreed
tautologically. The field could only ever fire on a cross-load into a differently built engine, so the
drift it was added for was the one case it could not see.

Written here instead, it becomes a real cross-check: the trainer asserts *the map I trained with*, the
engine verifies *the map I index*, and a drift between them is refused at load. It also collapses one
stamper per architecture into one, and removes the "stamper built at the wrong architecture" failure
class.

### Not vendored into a consumer

Training code is not needed to build a program that loads a net — only by whoever is actually training
one. So this repository is not vendored, subtree'd or submoduled into anything: there is no copy to
keep in sync, and a consumer reads it when it needs it.

## The data this trainer reads

**You supply the training data, and it is your responsibility to confirm that you have the rights to
use it for this purpose.** This repository holds no data, downloads none on your behalf and
redistributes none. Everything below is a statement about *formats*, so that you can tell whether data
you already have is readable here, or convert it.

### The packed record: `bulletformat`

The production path reads a headerless stream of fixed 32-byte records — `bulletformat::ChessBoard`,
crate 1.8.0, `#[repr(C)]` — which is what both the bullet trainer and `pytorch/bulletformat.py`
consume. A file is simply N of these back to back, so its size is always a multiple of 32.

| Offset | Size | Field | Meaning |
|---|---|---|---|
| 0 | 8 | `occ` | occupancy bitboard, little-endian `u64` |
| 8 | 16 | `pcs` | one 4-bit piece per occupied square, **low nibble of each byte first** |
| 24 | 2 | `score` | centipawns, `i16` |
| 26 | 1 | `result` | 0 = loss, 1 = draw, 2 = win |
| 27 | 1 | `ksq` | the side to move's own king square |
| 28 | 1 | `opp_ksq` | the opponent's king square, **already `XOR 56`** |
| 29 | 3 | `extra` | unused |

Two properties of the format do most of the work, and both are easy to get wrong.

- **The record is side-to-move relative, not White relative.** A black-to-move position is flipped
  before storing: every square `^ 56`, every piece's colour bit inverted, `score = -score` and
  `result = 2 - result`. So the stored board always has the side to move playing up the board, a
  nibble's colour bit is 0 for the side to move's own pieces, and `score` and `result` are already
  from the side to move's point of view. Reading the stored score as White relative correlates an
  exported net against its own training labels at about **0.14** instead of about **0.78**.
- **The nibbles are in ascending square order, not piece order.** The n-th nibble belongs to the n-th
  lowest set bit of the occupancy, so each record has a different piece count — which is the one
  genuinely awkward step to vectorise.

`pytorch/bulletformat_check.py` is the gate on all of this: it renders every record back to a FEN, puts
it through the text path's featuriser, and compares the two index sets element for element, so a misread
field offset or nibble order cannot pass as "the net trained, it must be fine".

### The text records

Both front ends also accept plain text, one position per line, which is the easier format to produce
from your own games or engine output:

- the pipeline scripts under `pipeline/` read shards of bullet's text form, `FEN | eval | wdl`, and
  convert them to the packed format (combining, auditing for duplicate positions, and shuffling on the
  way);
- `pytorch/train.py` reads `fen;score;wdl` directly.

In both, the evaluation is centipawns from the side to move's point of view and the result is that same
side's game outcome. `pipeline/check_duplicates.sh` will tell you what fraction of an assembled pool is
distinct positions, which is the cheap check that a pool is not mostly the same game over and over.

### If you distribute a network this trainer produces

Worth knowing, and not obvious: **a network trained on a licensed database may be a "Produced Work" of
that database**, and some database licences attach an attribution obligation to distributing a Produced
Work even though they place no licensing condition on the Produced Work itself. So the licence of
whatever data you train on can reach the net you ship, without making the net carry that licence — and
without this repository's GPL reaching it either. If you distribute a net, check what your data's
licence asks of you. Nothing here can answer that for you, because it never sees your data.

## What is not finished

Most of this code was gathered here out of the engine repositories that had each grown their own copy,
and the gathering is not complete. These are the loose ends, listed because they are visible on arrival:

- **`pytorch/` does not run as placed.** `features.py`, `train.py`, `verify.py` and `bulletformat.py`
  all `import rnnue1`, which resolved when they sat beside it in one directory and does not now that
  the container module lives in `spec/`. They need a `sys.path` entry or a package layout.
- **`spec/stamp_net.cpp` does not compile here.** It still `#include "nnue.h"` and writes the header
  from an engine's compiled constants — the tautology *Why the net stamper lives here* is about.
  Rewriting it against `spec/rnnue1.py` is the actual work of the move.
- **`pipeline/bullet2text_main.c` is an entry point only**, 14 lines calling a function that is still
  in `zenith`.
- **`pipeline/train_pipeline.sh` and `pipeline/run_experiment.sh`** still resolve their helpers through
  the `tools/` layout they were written in rather than this one, and two of the gates they call -
  "does this net load and play" and the match runner - belong to a consumer by design and are not here.
- **`spec/` has no container test vectors yet.** The nearest thing is `bullet/pawnstar_eval.rs`, the
  reference evaluator, which reproduces the engines' 250 committed reference evaluations exactly from
  a stamped net's payload.

## Working on this repository

`.githooks/` holds `pre-commit` and `pre-push` hooks that **enforce the maintainer's own identity** and
refuse any commit not authored *and* committed as `jonny@recklesstechnology.com`. They are not for
contributors: do not run `git config core.hooksPath .githooks` in a clone of your own, or every commit
you make will be refused.

Shell scripts, Markdown and source files are pinned to LF in `.gitattributes`, so a Windows clone does
not hand `bash` a `\r` on every line.

## Licence

**The code in this repository is GPL-3.0-or-later.** [LICENSE](LICENSE) is the GPLv3 text exactly as
`pawnstar`, `pawnstar-rs`, `pawnstar-go`, `zenith` and the shared toolkit carry it. Each source file
carries an `SPDX-License-Identifier: GPL-3.0-or-later` line and a `Copyright (C) 2026 Jonny Reckless`
line; Markdown, `requirements.txt` and the `.gitkeep` placeholders carry none, and
`pipeline/new-shape-signature.sh` deliberately carries none either, because its value is being
byte-identical to the three engine copies.

Three things are licensed separately, and are not the same thing:

- **The code here is GPL.** That is this section.
- **Your training data is under whatever licence you obtained it under.** It is not part of this
  repository and nothing here changes it.
- **A network this trainer produces is not GPL merely because this code trained it.** The GPL covers
  this program and its source; a net file is data produced by running it. What a net may carry is
  whatever its training data's licence attaches to a work derived from that data — see
  [If you distribute a network this trainer produces](#if-you-distribute-a-network-this-trainer-produces).

**bullet, the trainer this depends on, is MIT** (Copyright (c) 2023 Jamie Whiting,
<https://github.com/jw1912/bullet>). It is a dependency, not vendored: `bullet/setup_bullet.sh` clones
it at a pinned commit. But `bullet/pawnstar.rs` was written starting from bullet's own
`examples/simple.rs`, and `bullet/pawnstar_eval.rs` follows that example's inference, so portions of
those two files are bullet's. They carry `GPL-3.0-or-later AND MIT` and bullet's copyright line, and
the MIT notice is kept in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Nothing else here derives
from bullet; reading its on-disk record format in `pytorch/bulletformat.py` is format compatibility
only.
