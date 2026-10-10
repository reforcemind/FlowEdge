# Action-delivery study reproduction

This directory is the entry point for the review archive. In a repository
checkout, the full study also requires the companion `papers/action-delivery`
evidence, which is intentionally maintained outside the source branch. The
review archive includes that frozen evidence and reproduces the deterministic,
CPU-only portion of the study without a model checkpoint, Java, a robot, or
network access.

## Supported environment

- Python 3.12
- NumPy 2.5.2
- Windows PowerShell or a POSIX shell

Create an isolated environment beside the checkout or extracted archive, not
inside it, and install the pinned dependency. Keeping the environment external
preserves a strict archive for a second manifest-verified run:

```sh
python -m venv ../action-delivery-venv
# POSIX: . ../action-delivery-venv/bin/activate
# PowerShell: ..\action-delivery-venv\Scripts\Activate.ps1
python -m pip install -r reproduction/action-delivery/requirements.txt
python reproduction/action-delivery/run.py \
  --output ../action-delivery-rerun/default \
  --cache ../action-delivery-upstream-cache
```

The driver verifies the archive manifest when one is present, executes the
controlled smoke check, runs every `test_delivery*.py` regression, executes the
development evaluation, frozen held-out evaluation, matched general-tool
baseline, and online-check sensitivity accounting, then compares their summaries
with `expected.json`. By
default, results and the optional upstream cache are written to sibling
directories named from the checkout directory. The explicit paths above do the
same. When a manifest is present, the driver rejects output or cache paths
inside the extracted archive before creating anything, and verifies the
manifest before producing output.

Expected default result: 81 synchronous and 432 asynchronous controlled cases,
four detected development mutations, zero routing failures, a passing static
source inventory, two detected readiness-consistency controls, and 166,950
checked erasure state-operation pairs. The same run preserves 378 held-out
mutant cases with three online rejections and one survivor, records that the
offline projection rejects that survivor, matches 130 bounded states between
Python and TLC, and reports 16/17 designated observable mutations rejected by
the three online checks. The pinned
LeRobot source is deliberately not downloaded by the default flow.

## Optional protocol-fixed extension

To rerun the complete protocol-fixed extension, including mutation enumeration,
search trials, scaling, systematic monitor corruptions, and semantic comparison
against the declared result shape:

```sh
python reproduction/action-delivery/run.py --extension \
  --output ../action-delivery-rerun/full \
  --cache ../action-delivery-upstream-cache
```

This remains network-free but takes several minutes. It expects 513 valid
executions, 432 phase-model projections, detection of all nine mutations
classified as observable, no observation of the delivery-equivalent control,
360 search trials, and 60 scaling runs (51 pass and 9 resource-bounded unknown).
It also expects 5,828 single-corruption checks across 13 classes with no verdict
mismatch or unexplained pass. An unknown scaling or integrity verdict is retained
as unknown rather than counted as pass.

## Optional pinned upstream component

The upstream queue experiment additionally requires PyTorch. The first run may
fetch exactly one revision-pinned, digest-checked Apache-2.0 source file:

```sh
python reproduction/action-delivery/run.py --fetch-upstream \
  --output ../action-delivery-rerun/upstream-fetch \
  --cache ../action-delivery-upstream-cache
```

Later offline reruns use the verified cache:

```sh
python reproduction/action-delivery/run.py --upstream \
  --output ../action-delivery-rerun/upstream-offline \
  --cache ../action-delivery-upstream-cache
```

This adds 324 parameter cases and 1,562 histories over the declared operation
alphabet. It does not install LeRobot or execute its policy, robot client, or
configuration constructor.

## Deliberate exclusions

Fresh TLC execution, native-checkpoint integration, and timing measurements have
separate host or dependency requirements documented in
`papers/action-delivery/reproduce.md`.
The default command does not silently download dependencies and does not turn
unavailable external evidence into a passing result.

The default run includes the matched Python/TLA+ baseline; it is also available
separately:

```sh
python -m tools.delivery_check.baseline \
  --output ../action-delivery-rerun/general-tool-baseline.json
```

The clean archive includes the source-hashed TLC result, so this command can
compare the current TLA+ source and rerun the Python side without Java. Its
stored TLC timing is explicitly historical. A fresh general-tool runtime
requires Java 11 or newer and a local `tla2tools.jar`; pass `--jar PATH` to
use it. The baseline is a matched bounded check, not a claim that either
encoding proves arbitrary source code. Its direct TLA+ source is smaller under
the recorded nonblank-line proxy, and no effort advantage is claimed.

In a full checkout without `artifact-manifest.json`, maintainers may still use
explicit in-tree paths when that is useful, for example `--output
build-delivery/reproduction`. The strict archive intentionally disallows that
choice so consecutive runs remain repeatable.
