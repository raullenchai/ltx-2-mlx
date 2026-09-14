import json

import numpy as np
from safetensors.numpy import save_file

from scripts.render_stage2_blind_suite import _review_case


def test_review_case_does_not_reveal_blind_mapping(tmp_path) -> None:
    data = tmp_path / "latent_0007.safetensors"
    save_file({"latents": np.zeros((1,), dtype=np.float32)}, data, metadata={"prompt": "A moving fox"})

    case = _review_case(data, tmp_path / "blind")
    rendered = json.dumps(case)

    assert case["index"] == 7
    assert case["prompt"] == "A moving fox"
    assert case["A"] == "case-07/A.mp4"
    assert "student" not in rendered
    assert "teacher" not in rendered
