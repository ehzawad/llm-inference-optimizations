from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bench_external  # noqa: E402


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None):
    return subprocess.run(
        list(args),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def valid_row(concurrency: int = 1, measured: int = 2) -> dict[str, object]:
    expected = concurrency * measured
    tokens = expected * 256
    return {
        "bench_version": "test",
        "tag": f"c{concurrency:03d}",
        "concurrency": concurrency,
        "measured_per_worker": measured,
        "requests_ok": expected,
        "requests_failed": 0,
        "completion_tokens_total": tokens,
        "server_predicted_tokens_delta": float(tokens),
        "output_tokens_per_s": 100.0,
        "output_tokens_per_min": 6000.0,
        "ttft_s": {"p50": 0.1, "p95": 0.2},
        "latency_s": {"p50": 2.0, "p95": 2.2},
        "telemetry": {},
        "ok": True,
    }


class IntegrityGateTests(unittest.TestCase):
    def test_assertion_based_scripts_refuse_optimized_python(self) -> None:
        for name in ("merge.py", "verify_artifacts.py"):
            with self.subTest(script=name):
                completed = run(sys.executable, "-O", str(SCRIPTS / name))
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("optimized Python", completed.stdout + completed.stderr)

    def test_download_hash_verification_survives_optimization(self) -> None:
        path = SCRIPTS / "01_download.py"
        spec = importlib.util.spec_from_file_location("download_script", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "adapter.safetensors"
            artifact.write_bytes(b"known adapter bytes")
            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(module.verify_sha256(str(artifact), expected), expected)
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                module.verify_sha256(str(artifact), "0" * 64)


class ReportTests(unittest.TestCase):
    def invoke_report(self, run_dir: Path):
        return run(sys.executable, str(SCRIPTS / "report.py"), str(run_dir), cwd=REPO)

    def test_empty_run_fails_without_creating_plausible_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("no benchmark-*.json", completed.stderr)
            self.assertFalse((run_dir / "summary.md").exists())

    def test_failed_point_is_labeled_and_excluded(self) -> None:
        # A valid C=1 baseline plus a failed C=24 point. Because a valid baseline
        # exists, the scaling table IS emitted -- so this genuinely proves the
        # failed point's throughput is absent from scaling, not merely that the
        # table was suppressed for lack of a baseline.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            base = valid_row(concurrency=1, measured=10)
            base["output_tokens_per_s"] = 100.0
            (run_dir / "benchmark-c001.json").write_text(json.dumps(base), encoding="utf-8")

            failed = valid_row(concurrency=24, measured=10)
            failed.update({
                "requests_ok": 239,
                "requests_failed": 1,
                "completion_tokens_total": 239 * 256,
                "server_predicted_tokens_delta": float(240 * 256),
                "output_tokens_per_s": 9999.0,  # distinctive; must NOT reach scaling
                "ok": False,
            })
            (run_dir / "benchmark-c024.json").write_text(json.dumps(failed), encoding="utf-8")

            (run_dir / "experiment.json").write_text(
                json.dumps({
                    "benchmark": {
                        "concurrency_points": [1, 24],
                        "expected_tags": ["c001", "c024"],
                        "measured_per_worker": 10,
                    }
                }),
                encoding="utf-8",
            )

            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            markdown = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Run status: **FAILED / PARTIAL**", markdown)
            self.assertIn("| 24 | FAIL |", markdown)
            # The scaling section exists (valid C=1 baseline) but excludes C=24.
            self.assertIn("## Throughput scaling", markdown)
            scaling = markdown.split("## Throughput scaling", 1)[1]
            self.assertNotIn("9999", scaling)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            failed_entry = next(item for item in summary if item["concurrency"] == 24)
            self.assertFalse(failed_entry["ok"])
            self.assertTrue(failed_entry["failure_reasons"])

    def test_valid_run_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = valid_row()
            (run_dir / "benchmark-c001.json").write_text(json.dumps(row), encoding="utf-8")
            (run_dir / "experiment.json").write_text(
                json.dumps({"benchmark": {"concurrency_points": [1]}}),
                encoding="utf-8",
            )

            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            markdown = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Run status: **PASS**", markdown)
            self.assertIn("## Throughput scaling", markdown)

    def test_missing_configured_point_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "benchmark-c001.json").write_text(
                json.dumps(valid_row()), encoding="utf-8"
            )
            (run_dir / "experiment.json").write_text(
                json.dumps({
                    "benchmark": {
                        "concurrency_points": [1, 30],
                        "expected_tags": ["c001", "c030"],
                    }
                }),
                encoding="utf-8",
            )

            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            markdown = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Missing expected concurrency point(s): 30", markdown)
            self.assertIn("Missing expected benchmark tag(s): `c030`", markdown)

    def _write_run(self, run_dir: Path, row: dict, spec: dict) -> None:
        tag = row.get("tag", "c001")
        (run_dir / f"benchmark-{tag}.json").write_text(json.dumps(row), encoding="utf-8")
        (run_dir / "experiment.json").write_text(json.dumps({"benchmark": spec}), encoding="utf-8")

    def test_report_requires_experiment_spec_by_default(self) -> None:
        # D3: a run directory without experiment.json must not certify as PASS.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "benchmark-c001.json").write_text(
                json.dumps(valid_row()), encoding="utf-8"
            )
            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("experiment.json", completed.stderr)
            self.assertFalse((run_dir / "summary.md").exists())

            # The escape hatch is explicit and opt-in.
            override = run(
                sys.executable, str(SCRIPTS / "report.py"), str(run_dir),
                "--allow-missing-spec", cwd=REPO,
            )
            self.assertEqual(override.returncode, 0, override.stderr)
            self.assertIn("Run status: **PASS**", (run_dir / "summary.md").read_text("utf-8"))

    def test_report_uses_authoritative_measured_not_self_report(self) -> None:
        # D4: the artifact self-reports measured=1 with a single request, but the
        # spec's authoritative measured_per_worker=20 must be enforced.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = valid_row(concurrency=1, measured=1)  # self-reports 1 request
            self._write_run(run_dir, row, {
                "concurrency_points": [1],
                "expected_tags": ["c001"],
                "measured_per_worker": 20,
            })
            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
            self.assertIn("expected 20", " ".join(summary[0]["failure_reasons"]))

    def test_report_honors_per_tag_request_expectations(self) -> None:
        # D4: fine-sweep style per-tag expectations override self-report too.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = valid_row(concurrency=32, measured=10)  # self-reports 320
            row["tag"] = "c032b"
            self._write_run(run_dir, row, {
                "concurrency_points": [32],
                "expected_tags": ["c032b"],
                "expected_requests_by_tag": {"c032b": 999},  # deliberately unmet
            })
            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
            self.assertIn("expected 999", " ".join(summary[0]["failure_reasons"]))

    def test_report_fails_when_advertised_counter_delta_missing(self) -> None:
        # D2: a run that selected a server metric family but has no delta fails.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = valid_row(concurrency=1, measured=2)
            row.pop("server_predicted_tokens_delta", None)
            row["server_generated_tokens_metric"] = "vllm:generation_tokens_total"
            row["server_generated_tokens_delta"] = None
            self._write_run(run_dir, row, {
                "concurrency_points": [1], "expected_tags": ["c001"], "measured_per_worker": 2,
            })
            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("| 1 | FAIL |", (run_dir / "summary.md").read_text("utf-8"))

    def test_report_fails_on_server_counter_mismatch_flag(self) -> None:
        # D2: an explicit server_counter_matches_client=false must fail the row.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = valid_row(concurrency=1, measured=2)
            row["server_generated_tokens_delta"] = float(1 * 2 * 256)
            row["server_counter_matches_client"] = False
            self._write_run(run_dir, row, {
                "concurrency_points": [1], "expected_tags": ["c001"], "measured_per_worker": 2,
            })
            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)

    def test_report_passes_when_no_server_counter_advertised(self) -> None:
        # D2 guard against over-failing: a run with genuinely no server counter
        # (metrics unavailable) is unverified but must not be rejected outright.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = valid_row(concurrency=1, measured=2)
            row.pop("server_predicted_tokens_delta", None)  # no counter at all
            self._write_run(run_dir, row, {
                "concurrency_points": [1], "expected_tags": ["c001"], "measured_per_worker": 2,
            })
            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Run status: **PASS**", (run_dir / "summary.md").read_text("utf-8"))

    def test_report_rejects_boolean_counts_and_nan_tokens(self) -> None:
        # D7: JSON booleans must not satisfy integer counts, and a non-finite
        # token delta must not slip through the numeric comparison.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = valid_row(concurrency=1, measured=1)
            row["requests_ok"] = True          # bool masquerading as the count 1
            row["requests_failed"] = False     # bool masquerading as 0
            row["server_predicted_tokens_delta"] = float("nan")
            self._write_run(run_dir, row, {
                "concurrency_points": [1], "expected_tags": ["c001"], "measured_per_worker": 1,
            })
            completed = self.invoke_report(run_dir)
            self.assertEqual(completed.returncode, 2)
            summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
            reasons = " ".join(summary[0]["failure_reasons"])
            self.assertIn("requests_ok", reasons)
            self.assertIn("requests_failed", reasons)
            self.assertIn("finite", reasons)


class ExternalHarnessTests(unittest.TestCase):
    def test_worker_exceptions_become_failed_records(self) -> None:
        original = bench_external.do_request

        def explode(_base: str, _prompt: str, max_tokens: int = 256):
            del max_tokens
            raise RuntimeError("synthetic worker failure")

        bench_external.do_request = explode
        try:
            records = bench_external.run_phase(
                "http://127.0.0.1:1/", ["prompt"], concurrency=3, per_worker=4
            )
        finally:
            bench_external.do_request = original

        self.assertEqual(len(records), 12)
        self.assertTrue(all(record["ok"] is False for record in records))
        self.assertTrue(all("synthetic worker failure" in record["error"] for record in records))

    def test_success_requires_every_expected_request(self) -> None:
        self.assertTrue(bench_external.successful_run(20, 0, 20, True))
        self.assertTrue(bench_external.successful_run(20, 0, 20, None))
        self.assertFalse(bench_external.successful_run(0, 0, 20, None))
        self.assertFalse(bench_external.successful_run(19, 1, 20, True))
        self.assertFalse(bench_external.successful_run(20, 0, 20, False))

    def test_labelled_prometheus_generation_counters_are_aggregated(self) -> None:
        before = {
            'vllm:generation_tokens_total{model_name="a"}': 10.0,
            'vllm:generation_tokens_total{model_name="b"}': 5.0,
        }
        after = {
            'vllm:generation_tokens_total{model_name="a"}': 16.0,
            'vllm:generation_tokens_total{model_name="b"}': 7.0,
        }
        delta, metric = bench_external.generation_counter_delta(before, after)
        self.assertEqual(delta, 8.0)
        self.assertEqual(metric, "vllm:generation_tokens_total")

        sglang_delta, sglang_metric = bench_external.generation_counter_delta(
            {"sglang:generation_tokens_total{model_name=\"m\"}": 3.0},
            {"sglang:generation_tokens_total{model_name=\"m\"}": 9.0},
        )
        self.assertEqual(sglang_delta, 6.0)
        self.assertEqual(sglang_metric, "sglang:generation_tokens_total")

    def test_empty_prompt_corpus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt corpus is empty"):
                bench_external.load_prompts(path)

    def test_corpus_error_writes_structured_failure_artifact(self) -> None:
        # D7: a setup failure (empty corpus) happens before any request, but the
        # point must still leave a structured ok:false artifact behind, so an
        # aborted point is never silently absent from the run directory.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "empty.jsonl"
            corpus.write_text("", encoding="utf-8")
            outdir = root / "out"
            completed = run(
                sys.executable,
                str(SCRIPTS / "bench_external.py"),
                "--url", "http://127.0.0.1:1",
                "--engine", "vllm",
                "--concurrency", "1",
                "--outdir", str(outdir),
                "--tag", "vllm-c001",
                "--prompts", str(corpus),
                "--measured", "1",
                "--warmup", "0",
                cwd=REPO,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            artifact = outdir / "benchmark-vllm-c001.json"
            self.assertTrue(artifact.exists(), "no structured failure artifact was written")
            data = json.loads(artifact.read_text("utf-8"))
            self.assertFalse(data["ok"])
            self.assertEqual(data["error_type"], "SetupError")

    def test_run_phase_load_balances_across_replicas(self) -> None:
        # Data-parallel dispatch is join-shortest-queue, not static round-robin:
        # with equal-speed replicas the offered load splits ~evenly; with a
        # faster replica, it receives strictly MORE (so a fast A6000 is not held
        # back to the slow A5000's request count). This is the M1 fix.
        import time

        def make_recorder(delays):
            seen: list[str] = []

            def record(base: str, _prompt: str, max_tokens: int = 256):
                seen.append(base)               # append is atomic under the GIL
                time.sleep(delays[base])        # simulate in-flight service time
                return {"ok": True, "completion_tokens": max_tokens,
                        "ttft": 0.01, "latency": 0.05, "t_start": 0.0, "t_end": 1.0}

            return record, seen

        original = bench_external.do_request

        # Equal-speed replicas -> load splits ~evenly.
        rec, seen = make_recorder({"http://a/": 0.005, "http://b/": 0.005})
        bench_external.do_request = rec
        try:
            records = bench_external.run_phase(
                ["http://a/", "http://b/"], ["p0", "p1"],
                concurrency=4, per_worker=6, max_tokens=64,
            )
        finally:
            bench_external.do_request = original
        self.assertEqual(len(records), 24)
        counts = Counter(seen)
        self.assertEqual(set(counts), {"http://a/", "http://b/"})
        self.assertLessEqual(abs(counts["http://a/"] - counts["http://b/"]), 4)
        # every record is tagged with the replica that served it (DP balance audit)
        self.assertTrue(all(r.get("base") in {"http://a/", "http://b/"} for r in records))

        # Heterogeneous replicas -> the faster one gets strictly more load.
        rec, seen = make_recorder({"http://slow/": 0.02, "http://fast/": 0.002})
        bench_external.do_request = rec
        try:
            bench_external.run_phase(
                ["http://slow/", "http://fast/"], ["p0", "p1"],
                concurrency=4, per_worker=8, max_tokens=64,
            )
        finally:
            bench_external.do_request = original
        counts = Counter(seen)
        self.assertGreater(counts["http://fast/"], counts["http://slow/"])

    def test_decode_rate_isolates_decode_window(self) -> None:
        # (gen-1)/(latency-ttft): a clean decode-throughput proxy distinct from
        # aggregate system throughput.
        self.assertAlmostEqual(
            bench_external.decode_rate({"completion_tokens": 256, "ttft": 0.1, "latency": 2.1}),
            255 / 2.0,
        )
        self.assertIsNone(bench_external.decode_rate(
            {"completion_tokens": 1, "ttft": 0.1, "latency": 1.0}))   # no decode window
        self.assertIsNone(bench_external.decode_rate(
            {"completion_tokens": 256, "ttft": None, "latency": 2.0}))  # missing ttft
        self.assertIsNone(bench_external.decode_rate(
            {"completion_tokens": 256, "ttft": 2.0, "latency": 2.0}))   # window <= 0

    def test_aggregate_generation_delta_sums_replicas(self) -> None:
        before = [
            {'vllm:generation_tokens_total{model_name="m"}': 10.0},
            {'vllm:generation_tokens_total{model_name="m"}': 100.0},
        ]
        after = [
            {'vllm:generation_tokens_total{model_name="m"}': 30.0},
            {'vllm:generation_tokens_total{model_name="m"}': 250.0},
        ]
        delta, name = bench_external.aggregate_generation_delta(before, after)
        self.assertEqual(delta, 20.0 + 150.0)
        self.assertEqual(name, "vllm:generation_tokens_total")

        none_delta, none_name = bench_external.aggregate_generation_delta([{}], [{}])
        self.assertIsNone(none_delta)
        self.assertIsNone(none_name)


class ProvenanceTests(unittest.TestCase):
    @staticmethod
    def load_capture(path: Path):
        spec = importlib.util.spec_from_file_location("capture_env_test", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_non_git_directory_is_unknown_not_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            script = scripts / "capture_env.py"
            script.write_text((SCRIPTS / "capture_env.py").read_text(encoding="utf-8"), encoding="utf-8")
            module = self.load_capture(script)
            self.assertEqual(module.git_provenance(root), (None, None))

    def test_clean_and_dirty_git_states_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            script = scripts / "capture_env.py"
            script.write_text((SCRIPTS / "capture_env.py").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
            (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
            self.assertEqual(run("git", "init", "-q", cwd=root).returncode, 0)
            self.assertEqual(run("git", "config", "user.name", "ehzawad", cwd=root).returncode, 0)
            self.assertEqual(run("git", "config", "user.email", "test@example.invalid", cwd=root).returncode, 0)
            self.assertEqual(run("git", "add", ".", cwd=root).returncode, 0)
            self.assertEqual(run("git", "commit", "-qm", "fixture", cwd=root).returncode, 0)

            module = self.load_capture(script)
            commit, dirty = module.git_provenance(root)
            self.assertIsNotNone(commit)
            self.assertFalse(dirty)

            (root / "tracked.txt").write_text("dirty\n", encoding="utf-8")
            commit_after, dirty_after = module.git_provenance(root)
            self.assertEqual(commit_after, commit)
            self.assertTrue(dirty_after)

    def test_git_dir_override_cannot_redirect_provenance(self) -> None:
        # D5: GIT_DIR / GIT_WORK_TREE must not let provenance record another
        # repository's clean commit for a directory that is not itself a repo.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # A real, committed decoy repository we try to leak in via GIT_DIR.
            other = base / "other-repo"
            (other / "scripts").mkdir(parents=True)
            (other / "file.txt").write_text("decoy\n", encoding="utf-8")
            for cmd in (
                ("git", "init", "-q"),
                ("git", "config", "user.name", "ehzawad"),
                ("git", "config", "user.email", "test@example.invalid"),
                ("git", "add", "."),
                ("git", "commit", "-qm", "decoy"),
            ):
                self.assertEqual(run(*cmd, cwd=other).returncode, 0)
            decoy_head = run("git", "rev-parse", "HEAD", cwd=other).stdout.strip()

            # The run's actual root: NOT a git repository.
            root = base / "run-root"
            (root / "scripts").mkdir(parents=True)
            script = root / "scripts" / "capture_env.py"
            script.write_text(
                (SCRIPTS / "capture_env.py").read_text(encoding="utf-8"), encoding="utf-8"
            )
            module = self.load_capture(script)

            override = {
                "GIT_DIR": str(other / ".git"),
                "GIT_WORK_TREE": str(other),
            }
            with unittest.mock.patch.dict(os.environ, override):
                # Control: plain git honours the override and WOULD leak the decoy.
                leaked = run("git", "rev-parse", "HEAD", cwd=root).stdout.strip()
                self.assertEqual(leaked, decoy_head)
                # The hardened provenance ignores the override entirely.
                commit, dirty = module.git_provenance(root)
            self.assertEqual((commit, dirty), (None, None))


class ShellOrchestratorTests(unittest.TestCase):
    def make_fake_tools(self, root: Path, capture_exit: int = 0) -> tuple[Path, Path]:
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        log = root / "calls.log"

        python = fake_bin / "python3"
        python.write_text(
            "#!/usr/bin/env bash\n"
            "set -u\n"
            'echo "$*" >> "$FAKE_CALL_LOG"\n'
            'if [[ "$*" == *"scripts/capture_env.py"* ]]; then\n'
            f"  exit {capture_exit}\n"
            "fi\n"
            'if [[ "$*" == *"scripts/benchmark.py"* && "$*" == *"--tag c030"* ]]; then\n'
            "  exit 2\n"
            "fi\n"
            'if [[ "$*" == *"scripts/benchmark.py"* && "$*" == *"--tag c024"* ]]; then\n'
            "  exit 2\n"
            "fi\n"
            'if [[ "$*" == *"scripts/bench_external.py"* && "$*" == *"-c016"* ]]; then\n'
            "  exit 2\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        python.chmod(0o755)

        nvidia_smi = fake_bin / "nvidia-smi"
        nvidia_smi.write_text("#!/usr/bin/env bash\necho 0\n", encoding="utf-8")
        nvidia_smi.chmod(0o755)

        for name, body in {
            "curl": "#!/usr/bin/env bash\nexit 0\n",
            "fuser": "#!/usr/bin/env bash\nexit 0\n",
            "sleep": "#!/usr/bin/env bash\nexit 0\n",
            "setsid": (
                "#!/usr/bin/env bash\n"
                "echo \"setsid $*\" >> \"$FAKE_CALL_LOG\"\n"
                "exec \"$REAL_SETSID\" \"$REAL_SLEEP\" 300\n"
            ),
        }.items():
            tool = fake_bin / name
            tool.write_text(body, encoding="utf-8")
            tool.chmod(0o755)

        return fake_bin, log

    def fake_env(self, fake_bin: Path, log: Path) -> dict[str, str]:
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["FAKE_CALL_LOG"] = str(log)
        real_setsid = shutil.which("setsid")
        real_sleep = shutil.which("sleep")
        if not real_setsid or not real_sleep:
            raise RuntimeError("tests require setsid and sleep")
        env["REAL_SETSID"] = real_setsid
        env["REAL_SLEEP"] = real_sleep
        return env

    def test_top_level_benchmark_continues_but_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            output = root / "run"
            completed = run(
                "bash",
                "run.sh",
                "benchmark",
                str(output),
                cwd=REPO,
                env=self.fake_env(fake_bin, log),
            )
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("--tag c001", calls)
            self.assertIn("--tag c030", calls)
            self.assertIn("--tag c100", calls)
            self.assertIn("scripts/report.py", calls)

    def test_concurrency_sweep_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            output = root / "sweep"
            completed = run(
                "bash",
                "scripts/sweep_concurrency.sh",
                str(output),
                cwd=REPO,
                env=self.fake_env(fake_bin, log),
            )
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("--tag c024", calls)
            self.assertIn("--tag c128", calls)
            self.assertIn("--tag c032b", calls)
            self.assertIn("scripts/report.py", calls)

    def test_env_capture_failure_continues_and_fails_run(self) -> None:
        # D6: capture_env failure under `set -e` must NOT abort run.sh benchmark;
        # the benchmark points and the report still run, and the run fails at end.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root, capture_exit=7)
            output = root / "run"
            completed = run(
                "bash", "run.sh", "benchmark", str(output),
                cwd=REPO, env=self.fake_env(fake_bin, log),
            )
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("scripts/capture_env.py", calls)
            self.assertIn("--tag c001", calls)
            self.assertIn("--tag c100", calls)  # continued past failed capture
            self.assertIn("scripts/report.py", calls)

    def test_env_capture_failure_continues_and_fails_sweep(self) -> None:
        # D6: same guarantee for the fine sweep.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root, capture_exit=7)
            output = root / "sweep"
            completed = run(
                "bash", "scripts/sweep_concurrency.sh", str(output),
                cwd=REPO, env=self.fake_env(fake_bin, log),
            )
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("scripts/capture_env.py", calls)
            self.assertIn("--tag c001", calls)
            self.assertIn("--tag c128", calls)  # full sweep ran despite capture failure
            self.assertIn("scripts/report.py", calls)

    def test_engine_fair_enables_sglang_metrics(self) -> None:
        # D1: the SGLang server must actually be launched with --enable-metrics,
        # verified against the command the wrapper truly execs (not its source).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            env = self.fake_env(fake_bin, log)
            env.update({"CLIENTS": "1", "MEASURED": "1", "WARMUP": "0"})
            completed = run(
                "bash", "scripts/engine_fair.sh", str(root / "ef"), "sglang",
                cwd=REPO, env=env,
            )
            calls = log.read_text(encoding="utf-8")
            self.assertIn("sglang.launch_server", calls)
            self.assertIn("--enable-metrics", calls)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_engine_compare_continues_past_failed_point(self) -> None:
        # Behavioral continue-through for engine_compare.sh (previously only
        # covered by source-string checks): a middle failing point does not stop
        # the sweep, and the wrapper exits non-zero.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            env = self.fake_env(fake_bin, log)
            env.update({"CLIENTS": "1 16 30", "MEASURED": "1", "WARMUP": "0"})
            completed = run(
                "bash", "scripts/engine_compare.sh", str(root / "ec"),
                cwd=REPO, env=env,
            )
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("--tag llamacpp-c001", calls)
            self.assertIn("--tag llamacpp-c016", calls)  # the injected failure
            self.assertIn("--tag llamacpp-c030", calls)  # ran after the failure

    def test_engine_fair_propagates_client_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            output = root / "engine-fair"
            env = self.fake_env(fake_bin, log)
            env.update({"CLIENTS": "1 8 16", "MEASURED": "1", "WARMUP": "0"})
            completed = run(
                "bash",
                "scripts/engine_fair.sh",
                str(output),
                "llamacpp",
                cwd=REPO,
                env=env,
            )
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("--tag llamacpp-bf16-c001", calls)
            self.assertIn("--tag llamacpp-bf16-c008", calls)
            self.assertIn("--tag llamacpp-bf16-c016", calls)

    def test_model_scale_marks_partial_models_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin, log = self.make_fake_tools(root)
            output = root / "model-scale"
            env = self.fake_env(fake_bin, log)
            env.update({
                "CLIENTS": "1 16",
                "MEASURED": "1",
                "WARMUP": "0",
                "NP": "1",
                "SLOTCTX": "64",
            })
            completed = run(
                "bash",
                "scripts/model_serve_bench.sh",
                str(output),
                cwd=REPO,
                env=env,
            )
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            calls = log.read_text(encoding="utf-8")
            for model in ("qwen3-4b", "qwen3.5-9b", "gemma-4-e2b"):
                self.assertIn(f"--tag {model}-c001", calls)
                self.assertIn(f"--tag {model}-c016", calls)
                status = json.loads(
                    (output / f"status-{model}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(status["status"], "benchmark_failed")

    def test_stop_group_kills_workers_after_session_leader_exits(self) -> None:
        real_bash = shutil.which("bash")
        real_setsid = shutil.which("setsid")
        real_sleep = shutil.which("sleep")
        if not real_bash or not real_setsid or not real_sleep:
            self.skipTest("requires bash, setsid, and sleep")

        for name in ("engine_fair.sh", "engine_compare.sh", "model_serve_bench.sh"):
            with self.subTest(script=name), tempfile.TemporaryDirectory() as tmp:
                text = (SCRIPTS / name).read_text(encoding="utf-8")
                start = text.index("stop_group(){")
                end = text.index("\n}\n", start) + 3
                function = text[start:end]
                root = Path(tmp)
                pidfile = root / "worker.pid"
                probe = root / "probe.sh"
                probe.write_text(
                    f'''#!/usr/bin/env bash
set -uo pipefail
active_pid=""
{function}
"$2" "$1" -c '"$1" 300 & echo $! > "$2"' bash "$3" "$4" &
leader=$!
wait "$leader" 2>/dev/null || true
worker=$(cat "$4")
stop_group "$leader"
for _ in $(seq 1 200); do
  state=$(ps -o stat= -p "$worker" 2>/dev/null | xargs || true)
  case "$state" in
    ""|Z*) exit 0 ;;
  esac
  "$3" 0.01
done
kill -KILL "$worker" 2>/dev/null || true
exit 1
''',
                    encoding="utf-8",
                )
                completed = run(
                    real_bash,
                    str(probe),
                    real_bash,
                    real_setsid,
                    real_sleep,
                    str(pidfile),
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_engine_wrappers_use_scoped_cleanup_and_failure_exit(self) -> None:
        for name in ("engine_fair.sh", "engine_compare.sh", "model_serve_bench.sh"):
            with self.subTest(script=name):
                text = (SCRIPTS / name).read_text(encoding="utf-8")
                self.assertNotIn("pkill -f", text)
                self.assertIn('exit "$fail"', text)
                self.assertIn("if ! python3 scripts/bench_external.py", text)
                self.assertIn("stop_group", text)

    def test_shell_syntax(self) -> None:
        completed = run(
            "bash",
            "-n",
            "run.sh",
            "scripts/sweep_concurrency.sh",
            "scripts/engine_fair.sh",
            "scripts/engine_compare.sh",
            "scripts/model_serve_bench.sh",
            "scripts/gpu_matrix.sh",
            cwd=REPO,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_gpu_matrix_dry_run_pins_gpus_per_config(self) -> None:
        # Behavioral (no GPU): the driver's DRY_RUN plan must pin each config to
        # the right card(s) and wire the data-parallel client to both replicas.
        cases = {
            "vllm-a5000": ["CUDA_VISIBLE_DEVICES=0 setsid", "--gpu-index 0", "--tensor-parallel-size 1"],
            "vllm-a6000": ["CUDA_VISIBLE_DEVICES=1 setsid", "--gpu-index 1"],
            "vllm-tp2": ["CUDA_VISIBLE_DEVICES=0,1 setsid", "--tensor-parallel-size 2", "--gpu-index 0,1"],
            "vllm-dp2": ["CUDA_VISIBLE_DEVICES=0 setsid", "CUDA_VISIBLE_DEVICES=1 setsid",
                         "--url http://127.0.0.1:8400 --url http://127.0.0.1:8401"],
            "llamacpp-a5000": ["CUDA_VISIBLE_DEVICES=0 setsid", "llama-server", "-dev CUDA0", "--main-gpu 0"],
            "llamacpp-a6000": ["CUDA_VISIBLE_DEVICES=1 setsid", "-dev CUDA0", "--main-gpu 0", "--gpu-index 1"],
        }
        env = dict(os.environ)
        env["DRY_RUN"] = "1"
        for config, needles in cases.items():
            with self.subTest(config=config):
                completed = run(
                    "bash", "scripts/gpu_matrix.sh", config, "/tmp/matrix-dryrun", "balanced",
                    cwd=REPO, env=env,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                out = completed.stdout
                for needle in needles:
                    self.assertIn(needle, out, f"{config}: missing {needle!r}")

    def test_gpu_matrix_rejects_unknown_config(self) -> None:
        env = dict(os.environ)
        env["DRY_RUN"] = "1"
        completed = run(
            "bash", "scripts/gpu_matrix.sh", "vllm-a7000", "/tmp/matrix-dryrun", "balanced",
            cwd=REPO, env=env,
        )
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)


class MatrixReportTests(unittest.TestCase):
    def write_config_run(
        self, parent: Path, config: str, shape: str, points: list[int],
        base_tps: float, all_ok: bool = True, measured: int = 10,
    ) -> None:
        run_dir = parent / f"{config}-{shape}"
        run_dir.mkdir(parents=True)
        tags = [f"c{c:03d}" for c in points]
        (run_dir / "experiment.json").write_text(json.dumps({
            "benchmark": {"concurrency_points": points, "expected_tags": tags,
                          "measured_per_worker": measured}
        }), encoding="utf-8")
        (run_dir / "config.json").write_text(json.dumps({
            "config": config, "engine": "vllm", "shape": shape, "precision": "bf16",
        }), encoding="utf-8")
        for c in points:
            ok = all_ok or c != points[-1]  # fail the last point when all_ok is False
            tokens = c * measured * 256
            row = {
                "engine": "vllm", "placement": config, "tag": f"c{c:03d}",
                "concurrency": c, "measured_per_worker": measured,
                "requests_ok": c * measured if ok else c * measured - 1,
                "requests_failed": 0 if ok else 1,
                "completion_tokens_total": tokens,
                "server_generated_tokens_delta": float(tokens),
                "server_generated_tokens_metric": "vllm:generation_tokens_total",
                "server_counter_matches_client": True,
                "output_tokens_per_s": base_tps * c,
                "output_tokens_per_min": base_tps * c * 60,
                "ttft_s": {"p50": 0.05, "p95": 0.09},
                "latency_s": {"p50": 2.0, "p95": 2.5},
                "decode_tokens_per_s": {"p50": base_tps * 0.9, "p95": base_tps * 0.8},
                "telemetry": {"mem_used_mib": {"peak": 9000}, "gpu_util_pct": {"median": 85}},
                "ok": ok,
            }
            (run_dir / f"benchmark-c{c:03d}.json").write_text(json.dumps(row), encoding="utf-8")

    def run_matrix(self, parent: Path, *extra: str):
        return run(sys.executable, str(SCRIPTS / "matrix_report.py"), str(parent), *extra, cwd=REPO)

    def test_matrix_report_emits_tables_and_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            pts = [1, 8, 16]
            self.write_config_run(parent, "vllm-a5000", "balanced", pts, 100.0)
            self.write_config_run(parent, "vllm-a6000", "balanced", pts, 180.0)
            self.write_config_run(parent, "vllm-dp2", "balanced", pts, 260.0)
            self.write_config_run(parent, "vllm-tp2", "balanced", pts, 150.0, all_ok=False)

            completed = self.run_matrix(parent)
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            md = (parent / "matrix-summary.md").read_text("utf-8")
            self.assertIn("placement × concurrency", md)
            self.assertIn("Prefill (TTFT) vs decode split", md)
            self.assertIn("2-GPU (vLLM) vs single-card", md)
            self.assertIn("A5000 vs A6000", md)
            self.assertIn("`vllm-tp2`", md)  # flagged as failed
            summary = json.loads((parent / "matrix-summary.json").read_text("utf-8"))
            self.assertFalse(summary["matrix_ok"])
            tp2 = next(c for c in summary["configs"] if c["config"] == "vllm-tp2")
            self.assertFalse(tp2["ok"])

    def test_matrix_report_passes_when_all_configs_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            pts = [1, 8]
            self.write_config_run(parent, "vllm-a5000", "balanced", pts, 100.0)
            self.write_config_run(parent, "vllm-a6000", "balanced", pts, 180.0)
            completed = self.run_matrix(parent)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("Matrix status: **PASS**", (parent / "matrix-summary.md").read_text("utf-8"))

    def test_matrix_report_flags_missing_expected_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            self.write_config_run(parent, "vllm-a5000", "balanced", [1, 8], 100.0)
            expect = parent / "matrix.json"
            expect.write_text(json.dumps({
                "runs": ["vllm-a5000-balanced", "vllm-a6000-balanced"]
            }), encoding="utf-8")
            completed = self.run_matrix(parent, "--expect", str(expect))
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("vllm-a6000-balanced", (parent / "matrix-summary.md").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
