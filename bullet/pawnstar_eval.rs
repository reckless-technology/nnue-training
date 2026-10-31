// SPDX-License-Identifier: GPL-3.0-or-later AND MIT
// Copyright (C) 2026 Jonny Reckless
// Copyright (c) 2023 Jamie Whiting
//
// Portions of this file follow bullet (https://github.com/jw1912/bullet, MIT licence, pinned commit
// da426a49b2a4b79a55a87031cd51c89bdf1ff030): the SCReLU inference, the Net field layout and the order of the output
// dequantisation follow the example inference at the foot of examples/simple.rs; the per-piece feature index, the
// king-bucket offset and the mirror flip are those of bullet's `Chess768` and `ChessBucketsMirrored` input types
// (crates/bullet_lib/src/game/inputs), and the head selection is that of `MaterialCount`
// (crates/bullet_lib/src/game/outputs.rs). The MIT copyright and permission notice is in THIRD-PARTY-NOTICES.md at
// the root of this repository.
/*
Reference evaluator for cross-checking each engine's in-engine NNUE inference against bullet.

Reads a trained net (bullet's native quantised .bin, NOT a stamped RNNUE1 net) and a list of FENs (one per
line, on stdin or a file argument), and prints "<fen> | <eval_cp>" using the SAME horizontally-mirrored
king-bucketed ChessBucketsMirrored feature formula bullet trains with, the SAME MaterialCount output-head
selection and the SAME SCReLU inference. Each engine's own NNUE suite compares its evaluation to this
output, and agreement within rounding confirms feature indexing, king bucketing, mirroring, orientation,
head selection and dequantisation all match.

THIS IS THE CANONICAL REFERENCE-EVALUATION GENERATOR, and it is the producer side of this repository's
"producer asserts, consumer verifies" split: the values it prints are what every consumer's
`nnue_reference.txt` holds, and a consumer re-deriving them from a local copy instead is how this file
came to be left behind at v12's shape while every engine shipped the new one.

THE SHAPE HERE IS THE SHARED RECKLESS SHAPE - horizontal king mirroring plus 8 material output heads -
which has been the DEFAULT in every engine since their 2026-10-30 releases. It must stay in step with
bullet/pawnstar.rs, which trains the net this reads: KING_BUCKETS_HALF, OUTPUT_BUCKETS, HIDDEN, QA, QB and
SCALE are the same numbers there, and the save order below is that file's `save_format` list.

Usage:  cargo run --release --example pawnstar_eval -- <net.bin> [fens.txt]

Two notes on what to feed it, both worth knowing before a mismatch is read as a bug:

  * <net.bin> is the trainer's own quantised output, not a stamped net. An RNNUE1 file will parse, because
    the reader skips nothing, but its 64-byte header would be read as feature weights.
  * the payload of a stamped RNNUE1 net IS a valid input once the 64-byte header is stripped: at this shape
    bullet/pawnstar.rs quantises l1b to i32, so the trainer's weight image and the container's payload are
    the same bytes in the same order, and the only difference is the trainer's trailing padding, which this
    reader ignores. That is the cheapest way to regenerate a reference file for a net whose checkpoint has
    been cleaned up - `tail -c +65 <stamped net>` - and it is how the comparison in this repository's
    reference-value check is run.
*/
use std::io::{BufRead, BufReader, Read};

use bulletformat::ChessBoard;

const HIDDEN: usize = 1024;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

// The mirrored king-bucket map over HALF a board: 32 entries indexed `rank * 4 + mirrored_file`, where
// `mirrored_file = min(file, 7 - file)`. MUST be byte-identical to KING_BUCKETS_HALF in pawnstar.rs, from
// which every engine's own 64-entry map is generated with bullet's expansion arithmetic.
#[rustfmt::skip]
const KING_BUCKETS_HALF: [usize; 32] = [
    0, 1, 2, 3,
    0, 1, 2, 3,
    0, 1, 2, 3,
    0, 1, 2, 3,
    4, 5, 6, 7,
    4, 5, 6, 7,
    4, 5, 6, 7,
    4, 5, 6, 7,
];
const NUM_BUCKETS: usize = 8;

// Material output heads, selected by bullet's `MaterialCount<8>`: `(piece_count - 2) / ceil(32 / 8)`.
const OUTPUT_BUCKETS: usize = 8;

/// bullet's `ChessBucketsMirrored::new` expansion, verbatim:
/// `expanded[sq] = buckets[(sq / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][sq % 8]]`.
fn expand_king_buckets() -> [usize; 64] {
    let mut expanded = [0; 64];
    for (idx, elem) in expanded.iter_mut().enumerate() {
        *elem = KING_BUCKETS_HALF[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
    }
    expanded
}

fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, QA);
    y * y
}

struct Net {
    feature_weights: Vec<i16>, // 768*NUM_BUCKETS * HIDDEN, column-major: feature * HIDDEN + i
    feature_bias: Vec<i16>,    // HIDDEN
    output_weights: Vec<i16>,  // OUTPUT_BUCKETS * 2 * HIDDEN, bucket-major, stm half first
    output_bias: Vec<i32>,     // OUTPUT_BUCKETS
    king_buckets: [usize; 64],
}

impl Net {
    fn load(path: &str) -> Net {
        let mut bytes = Vec::new();
        std::fs::File::open(path).unwrap().read_to_end(&mut bytes).unwrap();

        // Byte offsets of bullet's save order for this architecture (pawnstar.rs's save_format):
        // l0w int16, l0b int16, l1w int16 (transposed to bucket-major), l1b **int32**.
        let fw_bytes = 768 * NUM_BUCKETS * HIDDEN * 2;
        let fb_bytes = HIDDEN * 2;
        let ow_bytes = OUTPUT_BUCKETS * 2 * HIDDEN * 2;
        let ob_bytes = OUTPUT_BUCKETS * 4;
        let needed = fw_bytes + fb_bytes + ow_bytes + ob_bytes;
        // bullet pads the trailing tensor to a 64-byte boundary, so the file may be slightly larger than
        // the packed weights; read what we need and ignore any trailing padding.
        assert!(bytes.len() >= needed, "net too small: got {} bytes, need {}", bytes.len(), needed);

        let read_i16 = |from: usize, count: usize| -> Vec<i16> {
            bytes[from..from + count * 2]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let ow_at = fw_bytes + fb_bytes;
        let ob_at = ow_at + ow_bytes;
        let output_bias = bytes[ob_at..ob_at + ob_bytes]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Net {
            feature_weights: read_i16(0, 768 * NUM_BUCKETS * HIDDEN),
            feature_bias: read_i16(fw_bytes, HIDDEN),
            output_weights: read_i16(ow_at, OUTPUT_BUCKETS * 2 * HIDDEN),
            output_bias,
            king_buckets: expand_king_buckets(),
        }
    }

    fn eval(&self, board: &ChessBoard) -> i32 {
        let mut us = self.feature_bias.clone();
        let mut them = self.feature_bias.clone();
        // Identical to bullet's ChessBucketsMirrored::map_features: each perspective takes its own king's
        // bucket bank AND its own mirror flip (7 when that king stands on files e-h), the flip applied to
        // the Chess768 feature index, which is where the square's file bits live.
        let get = |ksq: u8| -> (usize, usize) {
            (if ksq % 8 > 3 { 7 } else { 0 }, 768 * self.king_buckets[usize::from(ksq)])
        };
        let (our_flip, our_bucket) = get(board.our_ksq());
        let (opp_flip, opp_bucket) = get(board.opp_ksq());
        for (piece, square) in board.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);
            let stm = our_bucket + ((([0, 384][c] + pc + sq)) ^ our_flip);
            let ntm = opp_bucket + ((([384, 0][c] + pc + (sq ^ 56))) ^ opp_flip);
            for i in 0..HIDDEN {
                us[i] += self.feature_weights[stm * HIDDEN + i];
                them[i] += self.feature_weights[ntm * HIDDEN + i];
            }
        }
        // bullet's MaterialCount<OUTPUT_BUCKETS>, and every engine's own OutputBucket.
        let divisor = 32usize.div_ceil(OUTPUT_BUCKETS);
        let bucket = (board.occ().count_ones() as usize - 2) / divisor;
        let weights = &self.output_weights[bucket * 2 * HIDDEN..(bucket + 1) * 2 * HIDDEN];
        let mut out: i32 = 0;
        for i in 0..HIDDEN {
            out += screlu(us[i]) * i32::from(weights[i]);
            out += screlu(them[i]) * i32::from(weights[HIDDEN + i]);
        }
        out /= QA;
        out += self.output_bias[bucket];
        out *= SCALE;
        out /= QA * QB;
        out
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = Net::load(&args[1]);

    let reader: Box<dyn BufRead> = if args.len() > 2 {
        Box::new(BufReader::new(std::fs::File::open(&args[2]).unwrap()))
    } else {
        Box::new(BufReader::new(std::io::stdin()))
    };

    for line in reader.lines() {
        let fen = line.unwrap();
        let fen = fen.trim();
        if fen.is_empty() {
            continue;
        }
        // ChessBoard::from_str wants "fen | score | wdl"; values are irrelevant for evaluation.
        let record = format!("{fen} | 0 | 0.5");
        match record.parse::<ChessBoard>() {
            Ok(board) => println!("{fen} | {}", net.eval(&board)),
            Err(e) => eprintln!("parse error for '{fen}': {e}"),
        }
    }
}
