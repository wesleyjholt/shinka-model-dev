"""Read this example's model specification without executing candidate code."""
from __future__ import annotations

import ast
import json
from pathlib import Path


def read_model_spec(path: str | Path) -> dict:
    path = Path(path)
    payload = path.read_bytes()
    if len(payload) > 256_000:
        raise ValueError("The example specification exceeds 256 KB")
    if path.suffix == ".json":
        spec = json.loads(payload)
    else:
        tree = ast.parse(payload)
        if sum(1 for _ in ast.walk(tree)) > 10_000:
            raise ValueError("The example specification has too many syntax nodes")
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "MODEL_SPEC" for target in node.targets)]
        if len(assignments) != 1:
            raise ValueError("Supply exactly one literal MODEL_SPEC assignment")
        spec = ast.literal_eval(assignments[0].value)
    if not isinstance(spec, dict) or set(spec) != {"degree", "hypothesis"}:
        raise ValueError("This toy evaluator expects degree and hypothesis fields")
    if isinstance(spec["degree"], bool) or not isinstance(spec["degree"], int) or not 0 <= spec["degree"] <= 8:
        raise ValueError("This toy evaluator supports polynomial degrees zero through eight")
    if not isinstance(spec["hypothesis"], str) or len(spec["hypothesis"]) > 2000:
        raise ValueError("Supply a hypothesis with at most 2,000 characters")
    return spec
