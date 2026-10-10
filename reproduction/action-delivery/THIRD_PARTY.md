# Third-party inputs

The default offline reproduction installs NumPy but does not vendor it. Optional
experiments use additional third-party software that remains outside this bundle.
Each dependency is governed by its own upstream license.

| Input | Pin used by the study | How it is used |
|---|---|---|
| NumPy | 2.5.2 | Controlled runners, arrays, and summary calculations |
| PyTorch | 2.14.0+cpu in the recorded WSL run | Optional execution of the pinned queue with real tensors |
| LeRobot `action_queue.py` | revision `5aa74557f84c54d4b458f8b9643c5aa2982acfed`, SHA-256 `c71228c61bb50049b5ce5a0a8c282fcf5da6f97bd7ad1b26b4893af89f9c1a39` | Optional, explicitly fetched source; executed in-process with a narrow configuration shim; not bundled |
| TLA+ tools | release 1.8.0, SHA-256 `9d36716ffb5e49d1ba8fae4651eba59f3189887e12eb90e204a42d2e6e993fef` | Optional TLC baseline; downloaded only with `--fetch`; not bundled |
| ACM `acmart` | local TeX distribution | Typesets the manuscript; class files are not bundled |

The LeRobot and TLC downloaders enforce byte limits and verify the listed
SHA-256 digests before executing downloaded code. Network access is opt-in.
