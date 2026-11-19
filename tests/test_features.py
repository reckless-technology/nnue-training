# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""The featuriser and the integer forward pass, against values worked out independently here."""
import numpy as np
import pytest

import features
import rnnue1
import spec_path  # noqa: F401


def test_output_bucket_runs_zero_to_seven_over_two_to_thirty_two_pieces():
    assert [features.output_bucket(count) for count in (2, 5, 6, 10, 30, 32)] == [0, 0, 1, 2, 7, 7]


def test_king_mirror_is_files_e_to_h():
    assert [features.king_mirror(square) for square in (0, 3, 4, 7, 8 + 4, 63)] == [
        False, False, True, True, True, True]


def test_king_bucket_reads_the_containers_map():
    assert all(features.king_bucket(square) == rnnue1.KING_BUCKET_MAP[square] for square in range(64))


def test_parse_board_pieces_start_position():
    pieces = features.parse_board_pieces("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    assert len(pieces) == 32
    assert (features.WHITE, 3, 0) in pieces      # white rook a1
    assert (features.BLACK, 5, 60) in pieces     # black king e8
    assert (features.WHITE, 0, 8) in pieces      # white pawn a2


def test_feature_index_white_perspective_unmirrored_king():
    # White king on b1 (file 1): bucket 1, no mirror. Own pawn on e4 = square 28.
    index = features.feature_index(features.WHITE, 1, features.WHITE, 0, 28)
    assert index == 1 * 768 + 0 * 384 + 0 * 64 + 28


def test_feature_index_white_perspective_mirrored_king():
    # King on g1 (file 6) is mirrored: bucket min(6, 1) = 1, squares XOR 7. Opposing knight on c3 = 18.
    index = features.feature_index(features.WHITE, 6, features.BLACK, 1, 18)
    assert index == 1 * 768 + 384 + 1 * 64 + (18 ^ 7)


def test_feature_index_black_perspective_flips_vertically():
    # Black king e8 = 60 -> relative 4 (mirrored, bucket 3). Black pawn on e5 = 36 -> 36 ^ 56 = 28 -> ^7 = 27.
    index = features.feature_index(features.BLACK, 60, features.BLACK, 0, 36)
    assert index == 3 * 768 + 0 + 0 + (28 ^ 7)


def test_position_features_counts_every_piece_for_both_perspectives():
    side, own, opponent = features.position_features(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert side == features.WHITE and len(own) == len(opponent) == 32
    assert len(set(own)) == 32 and max(own) < rnnue1.INPUT_FEATURES


def test_black_to_move_swaps_the_perspectives():
    fen_white = "4k3/8/8/8/8/8/4P3/4K3"
    side, own, _ = features.position_features(fen_white + " b - - 0 1")
    assert side == features.BLACK
    _, white_own, white_opponent = features.position_features(fen_white + " w - - 0 1")
    assert own == white_opponent


def test_truncated_divide_rounds_toward_zero_unlike_floor_division():
    divide = features._truncated_divide
    assert (divide(7, 2), divide(-7, 2), divide(7, -2), divide(-7, -2)) == (3, -3, -3, 3)
    assert -7 // 2 == -4


def worked_net(hidden_weight_value=1):
    transformer = np.zeros((rnnue1.INPUT_FEATURES, rnnue1.HIDDEN_SIZE), dtype=np.int64)
    transformer_bias = np.zeros(rnnue1.HIDDEN_SIZE, dtype=np.int64)
    output_weight = np.zeros((rnnue1.NUM_OUTPUT_BUCKETS, 2 * rnnue1.HIDDEN_SIZE), dtype=np.int64)
    output_bias = np.zeros(rnnue1.NUM_OUTPUT_BUCKETS, dtype=np.int64)
    return transformer, transformer_bias, output_weight, output_bias


def test_integer_eval_of_an_all_zero_net_is_the_output_bias_scaled():
    transformer, bias, output_weight, output_bias = worked_net()
    output_bias[0] = 64  # two kings only -> bucket 0
    _, own, opponent = features.position_features("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    # 64 * 400 // (255 * 64) = 400 // 255 = 1
    assert features.integer_eval(transformer, bias, output_weight, output_bias, own, opponent) == 1
    output_bias[0] = -64
    assert features.integer_eval(transformer, bias, output_weight, output_bias, own, opponent) == -1


def test_integer_eval_applies_screlu_with_clamping_and_the_selected_head():
    transformer, bias, output_weight, output_bias = worked_net()
    _, own, opponent = features.position_features("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    bias[0] = 300                                 # both accumulators clamp from 300 down to 255
    output_weight[0, 0] = 2                       # own half, unit 0
    output_weight[7, 0] = 999                     # a different head must be ignored
    # 255 * 2 * 255 = 130050 -> // 255 = 510 -> 510 * 400 // (255 * 64) = 204000 // 16320 = 12
    assert features.integer_eval(transformer, bias, output_weight, output_bias, own, opponent) == 12
    # A negative accumulator clamps to zero, so the same weights contribute nothing.
    bias[0] = -5
    assert features.integer_eval(transformer, bias, output_weight, output_bias, own, opponent) == 0
    # The opponent half is read too, with its own weights.
    bias[0] = 255
    output_weight[0, 0] = 0
    output_weight[0, rnnue1.HIDDEN_SIZE] = 2
    assert features.integer_eval(transformer, bias, output_weight, output_bias, own, opponent) == 12
