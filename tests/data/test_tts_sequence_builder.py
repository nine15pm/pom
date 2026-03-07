import pytest

from model.constants import IGNORE_INDEX
from train.tts_sequence_builder import build_read_write_sequence


def test_build_read_write_sequence_happy_path():
    text_ids = [10, 11, 12, 13]
    unit_ids = [0, 1, 2, 3, 4]
    input_ids, labels = build_read_write_sequence(
        text_ids,
        unit_ids,
        speech_token_offset=100,
        sep_id=7,
        eos_id=2,
        read_length=3,
        write_length=2,
    )

    expected_input_ids = [10, 11, 12, 100, 101, 13, 7, 102, 103, 104, 2]
    expected_labels = [
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        100,
        101,
        IGNORE_INDEX,
        IGNORE_INDEX,
        102,
        103,
        104,
        2,
    ]
    assert input_ids == expected_input_ids
    assert labels == expected_labels


def test_build_read_write_sequence_masks_text_sep_and_eos():
    text_ids = [42]
    unit_ids = [5]
    input_ids, labels = build_read_write_sequence(
        text_ids,
        unit_ids,
        speech_token_offset=200,
        sep_id=9,
        eos_id=3,
        read_length=3,
        write_length=10,
    )

    assert input_ids == [42, 9, 205, 3]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 205, 3]


def test_build_read_write_sequence_fails_if_text_remains_after_speech():
    text_ids = [1, 2, 3]
    unit_ids = [0]
    with pytest.raises(ValueError, match="text tokens remain"):
        build_read_write_sequence(
            text_ids,
            unit_ids,
            speech_token_offset=100,
            sep_id=7,
            eos_id=2,
            read_length=1,
            write_length=1,
        )


def test_build_read_write_sequence_rejects_out_of_range_unit_ids():
    text_ids = [1]
    unit_ids = [6561]
    with pytest.raises(ValueError, match="out of range"):
        build_read_write_sequence(
            text_ids,
            unit_ids,
            speech_token_offset=100,
            sep_id=7,
            eos_id=2,
            read_length=1,
            write_length=1,
        )


def test_build_read_write_sequence_rejects_empty_units():
    text_ids = [1]
    unit_ids = []
    with pytest.raises(ValueError, match="unit_ids must be non-empty"):
        build_read_write_sequence(
            text_ids,
            unit_ids,
            speech_token_offset=100,
            sep_id=7,
            eos_id=2,
            read_length=1,
            write_length=1,
        )


def test_build_read_write_sequence_rejects_non_positive_lengths():
    text_ids = [1]
    unit_ids = [0]
    with pytest.raises(ValueError, match="read_length"):
        build_read_write_sequence(
            text_ids,
            unit_ids,
            speech_token_offset=100,
            sep_id=7,
            eos_id=2,
            read_length=0,
            write_length=1,
        )
    with pytest.raises(ValueError, match="write_length"):
        build_read_write_sequence(
            text_ids,
            unit_ids,
            speech_token_offset=100,
            sep_id=7,
            eos_id=2,
            read_length=1,
            write_length=-1,
        )
