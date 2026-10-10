"""Run the pinned upstream ActionQueue, without installing a moving LeRobot tree.

The module's RTCConfig type import is provided by a narrow configuration shim:
ActionQueue reads only cfg.enabled. Tensor and queue operations are upstream
code using real PyTorch. This is a queue-component test, not an RTC policy test.
"""

import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.request import urlopen

REVISION = "5aa74557f84c54d4b458f8b9643c5aa2982acfed"
URL = f"https://raw.githubusercontent.com/huggingface/lerobot/{REVISION}/src/lerobot/policies/rtc/action_queue.py"
# A full revision pin and content digest are both checked by the loader.
SHA256 = "c71228c61bb50049b5ce5a0a8c282fcf5da6f97bd7ad1b26b4893af89f9c1a39"
MAX_SOURCE_BYTES = 1024 * 1024


def load_queue(cache, *, fetch=False):
    path = Path(cache) / REVISION / "action_queue.py"
    if not path.exists():
        if not fetch:
            raise FileNotFoundError("pinned queue missing; use --fetch-upstream once")
        with urlopen(URL, timeout=30) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_SOURCE_BYTES:
                raise ValueError("upstream queue exceeds download size limit")
            content = response.read(MAX_SOURCE_BYTES + 1)
        if len(content) > MAX_SOURCE_BYTES:
            raise ValueError("upstream queue exceeds download size limit")
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError("upstream queue content digest mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("cached queue content digest mismatch")
    package_name = "_flowedge_pinned_rtc"
    package = ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    config = ModuleType(package_name + ".configuration_rtc")
    config.RTCConfig = SimpleNamespace
    sys.modules[package_name] = package
    sys.modules[config.__name__] = config
    spec = importlib.util.spec_from_file_location(package_name + ".action_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logging.getLogger(module.__name__).setLevel(logging.WARNING)
    return module.ActionQueue


def check_operations(queue_class, operations, *, rtc):
    """Compare public results and leftovers against a list-based specification."""
    import torch

    queue = queue_class(SimpleNamespace(enabled=rtc))
    expected = []  # original scalar, processed scalar, task
    consumed_index = 0
    for index, op in enumerate(operations):
        kind = op["kind"]
        if kind == "merge":
            request, length, delay = op["request"], op["length"], op["delay"]
            task = op.get("task", f"task-{request}")
            original = torch.arange(length, dtype=torch.float32)[:, None] + 1000 * (
                request + 1
            )
            processed = original + 100
            new = [
                (float(a), float(a + 100), task)
                for a in range(1000 * (request + 1), 1000 * (request + 1) + length)
            ]
            queue.merge(original, processed, delay, op.get("index_before"), task=task)
            if rtc:
                expected = [
                    item for slot, item in enumerate(new) if slot >= max(0, delay)
                ]
            else:
                expected.extend(new)
            consumed_index = 0
            # Mutating caller tensors must not alter queued data.
            original.fill_(-900)
            processed.fill_(-901)
        elif kind == "get":
            observed = queue.get_with_task()
            if expected:
                original, value, task = expected.pop(0)
                wanted = ([value], task)
                actual = (
                    None if observed is None else (observed[0].tolist(), observed[1])
                )
                if actual != wanted:
                    return {
                        "verdict": "fail",
                        "operation": index,
                        "expected": wanted,
                        "observed": actual,
                    }
                observed[0].fill_(-902)  # get returns a clone.
                consumed_index += 1
            elif observed is not None:
                return {
                    "verdict": "fail",
                    "operation": index,
                    "reason": "get_from_empty_queue",
                }
        elif kind == "clear":
            queue.clear()
            expected = []
            consumed_index = 0
        else:
            raise ValueError("unsupported operation")
        if queue.qsize() != len(expected) or queue.empty() != (len(expected) == 0):
            return {"verdict": "fail", "operation": index, "reason": "queue_size"}
        if queue.get_action_index() != consumed_index:
            return {
                "verdict": "fail",
                "operation": index,
                "reason": "consumption_index",
            }
        for getter, column in (
            (queue.get_left_over, 0),
            (queue.get_processed_left_over, 1),
        ):
            leftovers = getter()
            observed = [] if leftovers is None else leftovers[:, 0].tolist()
            wanted = [item[column] for item in expected]
            if observed != wanted:
                return {
                    "verdict": "fail",
                    "operation": index,
                    "reason": "leftover_pairing",
                    "expected": wanted,
                    "observed": observed,
                }
            if leftovers is not None:
                leftovers.fill_(-903)
    return {"verdict": "pass", "operations": len(operations)}


def evaluate(cache, *, fetch=False):
    from itertools import product

    queue_class = load_queue(cache, fetch=fetch)
    cases = []
    for rtc, length, delay, consumed, before in product(
        (False, True), (1, 2, 4), (-1, 0, 1, 2, 4, 5), (0, 1, 3), (None, 0, 9)
    ):
        operations = [{"kind": "merge", "request": 0, "length": 4, "delay": 0}]
        operations.extend({"kind": "get"} for _ in range(consumed))
        operations.append(
            {
                "kind": "merge",
                "request": 1,
                "length": length,
                "delay": delay,
                "index_before": before,
            }
        )
        operations.extend({"kind": "get"} for _ in range(9))
        operations.extend(
            [
                {"kind": "clear"},
                {"kind": "get"},
                {"kind": "merge", "request": 2, "length": 1, "delay": 0, "task": None},
                {"kind": "get"},
            ]
        )
        result = check_operations(queue_class, operations, rtc=rtc)
        cases.append(
            dict(
                rtc=rtc,
                length=length,
                delay=delay,
                consumed=consumed,
                index_before=before,
                **result,
            )
        )
    histories = evaluate_histories(queue_class)
    return {
        "scope": "controlled_tests",
        "revision": REVISION,
        "sha256": SHA256,
        "configuration": "cfg.enabled shim; upstream queue and real torch",
        "cases": cases,
        "passed": sum(case["verdict"] == "pass" for case in cases),
        "histories": histories,
        "verdict": "pass"
        if all(case["verdict"] == "pass" for case in cases)
        and histories["verdict"] == "pass"
        else "fail",
    }


def history_operations(symbols):
    """Materialize one bounded history; request identifiers remain unique."""
    templates = {
        "m0": {"kind": "merge", "length": 1, "delay": 0},
        "m1": {
            "kind": "merge",
            "length": 2,
            "delay": 1,
            "index_before": 0,
            "task": None,
        },
        "m2": {
            "kind": "merge",
            "length": 2,
            "delay": 3,
            "index_before": 2,
        },
    }
    operations = []
    request = 0
    for symbol in symbols:
        if symbol in templates:
            operation = dict(templates[symbol], request=request)
            request += 1
        else:
            operation = {"kind": symbol}
        operations.append(operation)
    return operations


def evaluate_histories(queue_class, *, depth=4):
    """Exhaust all histories through ``depth`` over a small public-operation alphabet."""
    from itertools import product

    alphabet = ("m0", "m1", "m2", "get", "clear")
    failures = []
    by_mode = {}
    total = passed = 0
    for rtc in (False, True):
        mode_total = mode_passed = 0
        for length in range(depth + 1):
            for symbols in product(alphabet, repeat=length):
                result = check_operations(
                    queue_class, history_operations(symbols), rtc=rtc
                )
                total += 1
                mode_total += 1
                if result["verdict"] == "pass":
                    passed += 1
                    mode_passed += 1
                elif len(failures) < 10:
                    failures.append({"rtc": rtc, "symbols": symbols, "result": result})
        by_mode[str(rtc).lower()] = {"histories": mode_total, "passed": mode_passed}
    return {
        "alphabet": list(alphabet),
        "maximum_depth": depth,
        "histories": total,
        "passed": passed,
        "by_mode": by_mode,
        "failures": failures,
        "verdict": "pass" if passed == total else "fail",
    }
