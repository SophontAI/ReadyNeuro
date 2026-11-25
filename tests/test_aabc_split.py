"""The AABC eval split is user-supplied (it lists AABC subject IDs, so it is not shipped)."""

import json

import pytest

from readyneuro.builder.discover import aabc


def test_eval_split_is_read_from_the_configured_path_and_required(tmp_path):
    with pytest.raises(RuntimeError, match="aabc_eval_split"):
        aabc.eval_split_map(None)
    with pytest.raises(RuntimeError, match="aabc_eval_split"):
        aabc.eval_split_map(str(tmp_path / "missing.json"))
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"HCA0000001": ["train", "V2"]}))
    assert aabc.eval_split_map(str(path)) == {"HCA0000001": ("train", "V2")}
