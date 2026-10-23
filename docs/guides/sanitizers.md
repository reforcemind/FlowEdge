# Sanitizer verification

FlowEdge provides a bounded Linux sanitizer lane for the Core loaders, protocol
codecs, and Relay parser/process tests:

```bash
bash tools/scripts/sanitize.sh
```

The script builds with Clang AddressSanitizer and UndefinedBehaviorSanitizer,
then runs CTest with one test process at a time. Two build workers and one test
worker are intentional peak-RSS limits for CI and small development machines.
ASan reserves a large virtual shadow range, so the lane does not impose an
address-space `ulimit`; doing so would prevent ASan from starting.

Set `FLOWEDGE_SANITIZER_BUILD_DIR` to keep the build outside the repository, or
`FLOWEDGE_SANITIZER_BUILD_JOBS` to lower the build concurrency. On failure,
`Testing/Temporary/sanitizer-build.log` and `sanitizer-tests.log` contain the
compiler, failing test name, and sanitizer stack trace for CI artifact upload.
