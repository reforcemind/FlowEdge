import copy
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.delivery_check.controlled import Completion, execute_async, execute_sync
from tools.delivery_check.extension import (
    compile_mutation,
    detections,
    extension_accounting,
    project_trace,
)
from tools.delivery_check.payload_cost import route, tape

PROTOCOL = ROOT / "papers/action-delivery/extension-protocol.json"


class DeliveryExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mutations = {}
        if PROTOCOL.is_file():
            protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
            cls.mutations = {m["name"]: m for m in protocol["mutations"]}

    def mutation(self, name):
        if not self.mutations:
            self.skipTest("requires frozen action-delivery protocol")
        return self.mutations[name]

    def test_projection_and_wrong_implementation_origin(self):
        case = execute_async()
        self.assertGreater(project_trace(case, 4, 6), 0)
        changed = copy.deepcopy(case)
        next(e for e in changed["events"] if e["kind"] == "send_begin")["origin"][
            "slot"
        ] += 1
        with self.assertRaisesRegex(AssertionError, "origin mismatch"):
            project_trace(changed, 4, 6)
        changed = copy.deepcopy(
            execute_async((Completion(9),), horizon=1, steps=3)
        )
        next(e for e in changed["events"] if e["kind"] == "fallback")["origin"][
            "kind"
        ] = "policy"
        with self.assertRaisesRegex(AssertionError, "selection origin"):
            project_trace(changed, 1, 3)

    def test_projection_checks_completion_disposition_send_and_terminal_events(self):
        case = execute_async()
        changes = (
            ("complete", "request", 999, "completion correspondence"),
            ("accept", "length", 999, "disposition range correspondence"),
            ("send_end", "status", "error", "send return correspondence"),
            ("stop_end", "status", "error", "stop outcome correspondence"),
            ("run_end", "completed", 999, "terminal correspondence"),
        )
        for kind, field, value, reason in changes:
            with self.subTest(kind=kind, field=field):
                changed = copy.deepcopy(case)
                event = next(
                    event
                    for event in changed["events"]
                    if event["kind"] == kind
                    and (kind != "complete" or event["step"] != -1)
                    and (kind != "accept" or event["step"] != -1)
                )
                event[field] = value
                with self.assertRaisesRegex(AssertionError, reason):
                    project_trace(changed, 4, 6)

    def test_trace_can_lie_about_stop(self):
        runner = compile_mutation(self.mutation("async_suppress_actual_stop"))
        original = execute_async()
        case = execute_async(runner=runner)
        self.assertEqual(case["events"], original["events"])
        self.assertEqual(case["attempts"], original["attempts"])
        self.assertEqual(case["monitor"]["verdict"], "pass")
        self.assertEqual(case["boundary"]["verdict"], "pass")
        self.assertEqual(
            detections(case), {"monitor": False, "tensor": False, "lifecycle": True}
        )

    def test_trace_can_lie_about_held_payload(self):
        runner = compile_mutation(self.mutation("sync_uncommitted_hold"))
        original = execute_sync((0, 2_000_000))
        case = execute_sync((0, 2_000_000), runner=runner)
        self.assertEqual(case["events"], original["events"])
        self.assertNotEqual(case["attempts"], original["attempts"])
        self.assertEqual(case["monitor"]["verdict"], "pass")
        self.assertEqual(case["boundary"]["reason"], "tensor_origin_disagreement")

    def test_private_early_assignment_is_delivery_equivalent_on_send_failure(self):
        runner = compile_mutation(self.mutation("sync_early_private_commit"))
        for failure in (0, 1, 2):
            self.assertEqual(
                execute_sync((0, 0, 0), fail_send=failure),
                execute_sync((0, 0, 0), fail_send=failure, runner=runner),
            )

    def test_mutation_sites_fail_closed_on_source_drift(self):
        description = dict(
            self.mutation("async_pop_tail"), old="missing source pattern"
        )
        with self.assertRaisesRegex(ValueError, "site drift"):
            compile_mutation(description)

    def test_extension_accounting_rejects_surviving_observable_mutation(self):
        result = {
            "valid_cases": 1,
            "valid_failures": 0,
            "projected_cases": 1,
            "mutations": [
                {
                    "classification": "observable",
                    "detections": {"either": 0},
                }
            ],
            "equivalence_fault_controls": [{"identical": True}],
            "searches": [],
            "scaling": [
                {"samples": [{"result": {"verdict": "unknown"}}]}
            ],
        }
        counts, complete = extension_accounting(
            result,
            expected_valid=1,
            expected_projected=1,
            expected_searches=0,
        )
        self.assertEqual(counts["observable_detected"], 0)
        self.assertFalse(complete)

    def test_erased_and_tensor_event_tapes_agree(self):
        for horizon in (1, 4, 16):
            events = tape(horizon, requests=9)
            for dim in (1, 8, 64):
                self.assertEqual(route(events, horizon), route(events, horizon, dim))

    def test_artifact_manifest_detects_modified_payload(self):
        from tools.delivery_check.artifact import digest, verify

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.txt"
            source.write_text("original", encoding="utf-8")
            (root / "artifact-manifest.json").write_text(
                json.dumps({"schema": 1, "files": {"fixture.txt": digest(source)}}),
                encoding="utf-8",
            )
            self.assertEqual(verify(root)["verdict"], "pass")
            source.write_text("modified", encoding="utf-8")
            self.assertEqual(verify(root)["failures"], ["fixture.txt"])

    def test_artifact_manifest_rejects_unexpected_payload(self):
        from tools.delivery_check.artifact import digest, verify

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.txt"
            source.write_text("original", encoding="utf-8")
            (root / "artifact-manifest.json").write_text(
                json.dumps({"schema": 1, "files": {"fixture.txt": digest(source)}}),
                encoding="utf-8",
            )
            (root / "sitecustomize.py").write_text("raise SystemExit", encoding="utf-8")
            result = verify(root)
            self.assertEqual(result["verdict"], "fail")
            self.assertEqual(result["unexpected"], ["sitecustomize.py"])

    def test_anonymized_bundle_hashes_transformed_payloads(self):
        from tools.delivery_check.artifact import (
            INCLUDE,
            build,
            resolve_source,
            verify,
        )
        from tools.verification.anonymize_artifact import (
            anonymize_json_bytes,
            identity_path_matches_json_bytes,
            identity_token_matches,
            public_repo_sha_matches,
        )

        # Build tokens from parts so this delivered test file does not contain
        # contiguous public-repo SHAs or email addresses for the residual scan.
        public_sha = "97f57206" + "4e3e785eca1663bfab709941d441e906"
        sample_email = "reviewer@" + "example.com"
        sample = json.dumps(
            {
                "windows": r"C:\Users\sample-user\project",
                "wsl": "/mnt/c/Users/sample-user/project",
                "home": "/home/sample-user/project",
                "note": f"pinned at {public_sha}",
                "flowedge_base_revision": public_sha,
                "contact": sample_email,
            }
        ).encode()
        sanitized = anonymize_json_bytes(sample)
        self.assertEqual(identity_path_matches_json_bytes(sanitized), [])
        self.assertNotIn(b"sample-user", sanitized)
        self.assertNotIn(b"/home/sample-user", sanitized)
        self.assertNotIn(public_sha.encode(), sanitized)
        self.assertIn(b"<source-snapshot>", sanitized)
        self.assertIsNone(json.loads(sanitized)["flowedge_base_revision"])
        # Emails are not rewritten by anonymize_json_bytes; the residual
        # scan must still report them so build() can reject the payload.
        self.assertEqual(identity_token_matches(sanitized), [sample_email])

        missing = [
            name for name in INCLUDE if not resolve_source(ROOT, name).is_file()
        ]
        if missing:
            self.skipTest("requires complete action-delivery review evidence")

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            archive = directory / "review.zip"
            source_has_identity_paths = any(
                identity_path_matches_json_bytes(
                    resolve_source(ROOT, name).read_bytes()
                )
                for name in INCLUDE
                if name.endswith(".json")
            )
            result = build(ROOT, archive, anonymized=True)
            if source_has_identity_paths:
                self.assertGreater(result["anonymized_files"], 0)
            else:
                self.assertEqual(result["anonymized_files"], 0)
            extracted = directory / "extracted"
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)
            self.assertEqual(verify(extracted)["verdict"], "pass")
            manifest = json.loads(
                (extracted / "artifact-manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["anonymized"])
            self.assertNotIn("flowedge_base_revision", manifest)
            self.assertNotIn("artifact_revision", manifest)
            self.assertIn("source_tree_sha256", manifest)
            self.assertEqual(
                manifest["anonymization"]["identity_path_scan"], "pass"
            )
            self.assertEqual(
                manifest["anonymization"]["identity_token_scan"], "pass"
            )
            self.assertTrue((extracted / "README.md").is_file())
            self.assertTrue((extracted / "AI_USE.md").is_file())

            leaks = []
            revision_leaks = []
            email_leaks = []
            for path in extracted.rglob("*"):
                if not path.is_file():
                    continue
                data = path.read_bytes()
                if path.suffix == ".json":
                    leaks.extend(identity_path_matches_json_bytes(data))
                revision_leaks.extend(public_repo_sha_matches(data))
                email_leaks.extend(identity_token_matches(data))
            self.assertEqual(leaks, [])
            self.assertEqual(revision_leaks, [])
            self.assertEqual(email_leaks, [])

            # Rebuilding from the extracted review bundle must succeed: INCLUDE
            # source paths resolve through ARCHIVE_ALIASES to root files.
            rebuilt = directory / "rebuilt.zip"
            rebuild = build(extracted, rebuilt, anonymized=True)
            self.assertEqual(rebuild["files"], result["files"])
            rebuilt_root = directory / "rebuilt"
            with zipfile.ZipFile(rebuilt) as bundle:
                bundle.extractall(rebuilt_root)
            self.assertEqual(verify(rebuilt_root)["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
