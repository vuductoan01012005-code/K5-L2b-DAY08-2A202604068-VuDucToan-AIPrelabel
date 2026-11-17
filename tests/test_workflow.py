from __future__ import annotations

import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SOURCE = Path(__file__).resolve().parents[1]


def clean_cp1252_env() -> dict[str, str]:
    """Strip UTF-8 helper vars to simulate a fresh Windows PowerShell session."""
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONIOENCODING", None)
    return env


class Day08RegressionAndWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

        shutil.copytree(SOURCE / "scripts", self.root / "scripts")
        (self.root / "data" / "images").mkdir(parents=True)
        shutil.copyfile(SOURCE / "data" / "split.csv", self.root / "data" / "split.csv")
        shutil.copyfile(SOURCE / "data" / "ATTRIBUTION.csv", self.root / "data" / "ATTRIBUTION.csv")
        shutil.copytree(SOURCE / "data" / "cache", self.root / "data" / "cache")

        self.sub = self.root / "submission"
        for d in ("labels", "prelabel", "masks", "guidelines", "qc", "release"):
            (self.sub / d).mkdir(parents=True)
        shutil.copyfile(SOURCE / "submission" / "taxonomy.json", self.sub / "taxonomy.json")

        cache_y26 = json.loads((self.root / "data" / "cache" / "yolo26m_all_c005.coco.json").read_text(encoding="utf-8"))
        self.img_sizes = {Path(im["file_name"]).name: (im["width"], im["height"]) for im in cache_y26["images"]}
        for name, (w, h) in self.img_sizes.items():
            Image.new("RGB", (w, h), (120, 140, 160)).save(self.root / "data" / "images" / name)

    def run_raw(self, script: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, str(self.root / "scripts" / f"{script}.py"), *args],
            cwd=cwd or self.root,
            capture_output=True,
            env=clean_cp1252_env(),
        )

    def run_script(self, script: str, *args: str, success: bool = True) -> str:
        proc = self.run_raw(script, *args)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        self.assertNotIn("UnicodeEncodeError", stderr, stderr)
        self.assertNotIn("UnicodeDecodeError", stderr, stderr)
        output = stdout + stderr
        self.assertEqual(proc.returncode == 0, success, output)
        return output

    def _split_names(self, split_name: str) -> list[str]:
        with (self.root / "data" / "split.csv").open(encoding="utf-8", newline="") as f:
            return sorted(r["image"] for r in csv.DictReader(f) if r["split"] == split_name)

    def _build_valid_submission(self, mshv: str = "22001234", full_name: str = "Nguyen Van A") -> None:
        (self.root / "README.md").write_text(f"# Day 08 Lab — {mshv} - {full_name}\n", encoding="utf-8")
        (self.root / "STUDENT.md").write_text(
            f"# Thông tin học viên\n- Họ và tên: {full_name}\n- MSHV: {mshv}\n", encoding="utf-8"
        )
        (self.sub / "REPORT.md").write_text(
            "# Báo cáo Day 08\n"
            "## 1. Model\nsam2.1_b.pt\n"
            "## 2. Profile\nok\n"
            "## 3. Bẫy\nok\n"
            "## 4. Đáng dùng\nok\n"
            "## 5. Mask SAM\nsam2.1_b.pt\n"
            "## 6. Calibration\nok\n"
            "## 7. QC\nQuyết định: Rework (mẫu nhỏ)\n"
            "## 8. Ghi chú\nok\n",
            encoding="utf-8",
        )
        (self.sub / "guidelines" / "guideline_v1.1_calibrated.md").write_text(
            "# Bổ sung v1.1\n"
            "## 1. Câu hỏi mở\nimg_004 xe đạp điện\n"
            "## 2. Ca khó mới\nimg_013 tuk-tuk\n"
            "## 3. Lịch sử thay đổi\n| v1.1 | cập nhật | CAL-01 |\n",
            encoding="utf-8",
        )
        with (self.sub / "model_log.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "part", "model", "params", "split", "n_boxes", "sec_per_img", "what_went_right", "what_went_wrong", "decision"])
            w.writerow(["r01", "2", "yolo26-m", "--conf 0.25", "pilot", "50", "0.2", "ok", "tuk-tuk thanh car img_002", "giu 6 lop"])
            w.writerow(["r02", "2", "yolo26-m", "--conf 0.5", "pilot", "30", "0.2", "sach", "sot xe xa img_011", "bo"])
            w.writerow(["r03", "2", "yoloe", "--classes three_wheeler", "pilot", "10", "0.3", "tim tuk-tuk", "box hoi long img_002", "dung cho three_wheeler"])
            w.writerow(["r04", "2", "yolo26-m+sam(sam2.1_b.pt)", "--sam sam2.1_b.pt", "custom", "12", "0.5", "mask dep", "dinh nguoi lai img_013", "chon sam2.1_b.pt"])
        with (self.sub / "time_log.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "images", "minutes", "note"])
            w.writerow(["gold_manual", "4", "22", "gán tay"])
            w.writerow(["batch_review", "24", "75", "duyệt lô"])
        with (self.sub / "qc" / "calibration_decisions.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "image", "nguon", "van_de", "quyet_dinh", "quy_tac_ghi_vao_guideline", "phien_ban_guideline"])
            w.writerow(["CAL-01", "img_013", "loi_may", "YOLO26 goi tuk-tuk la car", "luon gan three_wheeler", "muc 2", "v1.1"])
            w.writerow(["CAL-02", "img_004", "cau_hoi_mo", "xe dap dien", "bicycle neu co ban dap", "muc 2", "v1.1"])
            w.writerow(["CAL-03", "img_018", "can_review", "dam dong", "ignore_region neu < 20px", "muc 4.1.6", "v1.1"])
        for k in range(1, 4):
            Image.new("RGB", (64, 64)).save(self.sub / "masks" / f"mask_{k}.jpg")

        # Merge cached predictions to create realistic gold_manual and batch_reviewed
        self.run_script(
            "merge_prelabels",
            "data/cache/yoloe26s_three_wheeler_all_c005.coco.json",
            "data/cache/yolo26m_all_c005.coco.json",
            "--conf", "0.35", "--iou", "0.6", "--split", "batch,gold",
            "--out", "work/prelabel/merged_review/pre.coco.json",
        )
        merged = json.loads((self.root / "work" / "prelabel" / "merged_review" / "pre.coco.json").read_text(encoding="utf-8"))
        gold_names = set(self._split_names("gold"))
        gold_ids = {im["id"] for im in merged["images"] if Path(im["file_name"]).name in gold_names}
        gold_coco = {
            **merged,
            "images": [im for im in merged["images"] if im["id"] in gold_ids],
            "annotations": [an for an in merged["annotations"] if an["image_id"] in gold_ids],
        }
        (self.sub / "labels" / "gold_manual.json").write_text(json.dumps(gold_coco), encoding="utf-8")

        self.run_script(
            "threshold_sweep",
            "--pred", "data/cache/yolo26m_all_c005.coco.json", "data/cache/yoloe26s_three_wheeler_all_c005.coco.json",
            "--gold", "submission/labels/gold_manual.json",
        )
        self.run_script(
            "inject_traps",
            "--pre", "work/prelabel/merged_review/pre.coco.json",
            "--student", mshv,
        )
        imp = json.loads((self.sub / "prelabel" / "pre_traps_import.coco.json").read_text(encoding="utf-8"))
        (self.sub / "labels" / "batch_reviewed.json").write_text(json.dumps(imp), encoding="utf-8")

        self.run_script("validate_labels", "--all")
        self.run_script(
            "review_stats",
            "--pre", "submission/prelabel/pre_traps.coco.json",
            "--final", "submission/labels/batch_reviewed.json",
            "--gold", "submission/labels/gold_manual.json",
            "--key", "submission/prelabel/traps_key.json",
        )
        self.run_script("build_release")
        self.run_script("qc_sample", "--n", "5", "--seed", "42")
        with (self.sub / "qc" / "qc_review.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["error_type"] = "khong"
            row["reviewer"] = mshv
        with (self.sub / "qc" / "qc_review.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image", "object_id", "error_type", "reviewer", "note"])
            w.writeheader()
            w.writerows(rows)
        self.run_script("qc_report")

    # ------------------------------------------------------------------ BUG-01
    def test_bug01_cvat_labels_non_empty_values(self) -> None:
        self.run_script("make_cvat_labels")
        generated = json.loads((self.root / "cvat" / "labels_generated.json").read_text(encoding="utf-8"))
        for lbl in generated:
            if lbl["name"] == "ignore_region":
                self.assertEqual(lbl["attributes"], [])
                continue
            by_name = {a["name"]: a for a in lbl["attributes"]}
            self.assertEqual(by_name["conf"]["values"], [""])
            self.assertEqual(by_name["can_review"]["values"], ["false"])
            for attr in lbl["attributes"]:
                self.assertIsInstance(attr["values"], list)
                self.assertGreater(len(attr["values"]), 0, f"{lbl['name']}.{attr['name']} empty values")

        # Repo file itself must match regenerated output
        repo_labels = json.loads((SOURCE / "cvat" / "labels_generated.json").read_text(encoding="utf-8"))
        self.assertEqual(repo_labels, generated)

        # Boolean default in taxonomy should serialize to lowercase "false"/"true"
        tax = json.loads((self.sub / "taxonomy.json").read_text(encoding="utf-8"))
        tax["attributes"][1]["default"] = False
        (self.sub / "taxonomy.json").write_text(json.dumps(tax), encoding="utf-8")
        self.run_script("make_cvat_labels")
        generated_bool = json.loads((self.root / "cvat" / "labels_generated.json").read_text(encoding="utf-8"))
        self.assertEqual(generated_bool[0]["attributes"][1]["values"], ["false"])

        # Empty select attribute values must fail cleanly via die()
        tax["attributes"].append({"name": "difficulty", "type": "select", "values": [], "default": ""})
        (self.sub / "taxonomy.json").write_text(json.dumps(tax), encoding="utf-8")
        out_sel = self.run_script("make_cvat_labels", success=False)
        self.assertIn("LỖI:", out_sel)

    # ------------------------------------------------------------------ BUG-02
    def test_bug02_check_submission_catches_invalid_labels_without_pythonutf8(self) -> None:
        self._build_valid_submission()
        # Corrupt gold_manual.json: keep only 1 image and inject an invalid class
        gold_path = self.sub / "labels" / "gold_manual.json"
        bad_gold = json.loads(gold_path.read_text(encoding="utf-8"))
        bad_gold["images"] = bad_gold["images"][:1]
        bad_gold["categories"].append({"id": 99, "name": "WRONG_CLASS"})
        bad_gold["annotations"] = [
            {"id": 1, "image_id": bad_gold["images"][0]["id"], "category_id": 99, "bbox": [10, 10, 40, 40], "segmentation": []}
        ]
        gold_path.write_text(json.dumps(bad_gold), encoding="utf-8")

        out = self.run_script("check_submission", success=False)
        self.assertIn("validate_labels", out)
        self.assertIn("WRONG_CLASS", out)
        self.assertNotIn("0 lỗi", out)

        # Also if validate_labels crashes with non-zero exit without "] LỖI: " lines, check_submission must STILL fail
        (self.root / "scripts" / "validate_labels.py").write_text("import sys; sys.stderr.write('boom\\n'); sys.exit(3)\n", encoding="utf-8")
        out2 = self.run_script("check_submission", success=False)
        self.assertIn("validate_labels", out2)
        self.assertIn("3", out2)

    # ------------------------------------------------------------------ BUG-03
    def test_bug03_requirements_files_readable_with_cp1252_pip(self) -> None:
        coding_re = re.compile(rb"coding[:=]\s*([-\w.]+)")
        for fname in ("requirements.txt", "requirements-local.txt", "requirements-cvat.txt"):
            raw = (SOURCE / fname).read_bytes()
            head = b"\n".join(raw.splitlines()[:2])
            has_coding = bool(coding_re.search(head))
            if not has_coding:
                raw.decode("cp1252")  # would fail if non-cp1252 UTF-8 bytes exist without PEP 263 header
            self.assertTrue(has_coding or raw.isascii(), f"{fname} missing '# -*- coding: utf-8 -*-'")

    # ------------------------------------------------------------------ BUG-04
    def test_bug04_all_cli_scripts_safe_under_cp1252_pipe(self) -> None:
        self.run_script("make_cvat_labels")
        self.run_script("prepare_workdirs")
        self.run_script("prepare_workdirs", "--advanced")

        # check_env must not crash under cp1252 pipe even when core packages fail to import
        env = clean_cp1252_env()
        hook_dir = self.root / "_fake_site"
        hook_dir.mkdir()
        (hook_dir / "ultralytics.py").write_text("raise ImportError('simulated missing ultralytics')\n", encoding="utf-8")
        (hook_dir / "torch.py").write_text("raise ImportError('simulated missing torch')\n", encoding="utf-8")
        env["PYTHONPATH"] = str(hook_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.run(
            [sys.executable, str(self.root / "scripts" / "check_env.py")],
            cwd=self.root,
            capture_output=True,
            env=env,
        )
        stderr = proc.stderr.decode("utf-8", errors="replace")
        stdout = proc.stdout.decode("utf-8", errors="replace")
        self.assertNotIn("UnicodeEncodeError", stderr, stderr)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("không import được", stdout)

        # prelabel, download_models, and ALL 18 scripts --help must succeed under cp1252 pipe
        (self.root / "cvat" / "labels_generated.json").unlink(missing_ok=True)
        for script_name in (
            "build_release", "check_env", "check_submission", "coco_to_cvat_xml", "download_models",
            "inject_traps", "make_cvat_labels", "merge_prelabels", "prelabel", "prepare_workdirs",
            "qc_report", "qc_sample", "review_stats", "sam2_points", "self_consistency",
            "threshold_sweep", "validate_labels", "viz",
        ):
            with self.subTest(help_script=script_name):
                self.run_script(script_name, "--help")
        # make_cvat_labels --help must not write cvat/labels_generated.json
        self.assertFalse((self.root / "cvat" / "labels_generated.json").exists())

    # ------------------------------------------------------------------ BUG-05
    def test_bug05_short_and_alphanumeric_mshv_and_quoted_cli_docs(self) -> None:
        for mshv in ("001", "00123456", "HV001", "22001234"):
            with self.subTest(mshv=mshv):
                self._build_valid_submission(mshv=mshv)
                out = self.run_script("check_submission")
                self.assertIn("0 lỗi", out)

        # Placeholder <MSHV> must still fail
        (self.root / "README.md").write_text("# Day 08 Lab — <MSHV> - Nguyen Van A\n", encoding="utf-8")
        self.run_script("check_submission", success=False)

        # Unquoted comma-separated CLI flags in docs/scripts examples
        for rel_path in ("GUIDE.md", "docs/MODELS.md", "docs/ADVANCED.md", "scripts/sam2_points.py", "scripts/merge_prelabels.py"):
            txt = (SOURCE / rel_path).read_text(encoding="utf-8")
            bad = re.findall(r"--(?:classes|split|thr|pos|neg|box|prompts|map)\s+[^\s`\"'\r\n]+[,=][^\s`\"'\r\n]+", txt)
            self.assertEqual(bad, [], f"{rel_path} has unquoted comma/kv args unsafe for PowerShell: {bad}")

    # ------------------------------------------------------------------ BUG-06 & BUG-08
    def test_bug06_bug08_docs_figures_gitignore_and_colab_sync(self) -> None:
        cvat_ai = (SOURCE / "docs" / "CVAT_AI.md").read_text(encoding="utf-8")
        self.assertIn("Clean previous annotations", cvat_ai)
        self.assertNotIn("Clean old annotations", cvat_ai)
        self.assertIn("**Save**", cvat_ai)

        threshold_doc = (SOURCE / "docs" / "THRESHOLD.md").read_text(encoding="utf-8")
        self.assertIn("10 phút", threshold_doc)

        cal_doc = (SOURCE / "docs" / "CALIBRATION_QC.md").read_text(encoding="utf-8")
        self.assertIn("tối đa 3", cal_doc)
        self.assertNotIn("mọi ảnh có `can_review`", cal_doc)
        self.assertNotIn("mọi ảnh có can_review", cal_doc)

        scripts_readme = (SOURCE / "scripts" / "README.md").read_text(encoding="utf-8")
        self.assertIn("0.75", scripts_readme)
        self.assertNotIn("IoU với box gốc ≥ 0.85", scripts_readme)

        setup_doc = (SOURCE / "docs" / "SETUP.md").read_text(encoding="utf-8")
        self.assertIn("K5-L23-Day08-AI-Assisted-Labeling", setup_doc)
        self.assertNotIn("Day8-AI-Assisted-Labeling", setup_doc)

        gitignore = (SOURCE / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("models/*", gitignore)

        nb = json.loads((SOURCE / "notebooks" / "day08_colab.ipynb").read_text(encoding="utf-8"))
        nb_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        self.assertNotIn("bus=0.85", nb_src)

        make_figs = (SOURCE / "docs" / "assets" / "make_figures.py").read_text(encoding="utf-8")
        self.assertIn("pre_traps_import.coco.json", make_figs)
        self.assertIn("30'", make_figs)
        self.assertIn("10'", make_figs)
        self.assertIn("95'", make_figs)
        self.assertIn("15'", make_figs)
        self.assertIn("tuỳ chọn", make_figs)
        self.assertIn("qc_report_auto.md", make_figs)
        self.assertNotIn("QC_REPORT", make_figs)

    # ------------------------------------------------------------------ BUG-07
    def test_bug07_csv_semicolon_cp1252_and_json_utf8_bom(self) -> None:
        self._build_valid_submission()
        # Re-write qc_review.csv using semicolon delimiter and cp1252 encoding with French/extended-latin/cp1252 text
        rv = self.sub / "qc" / "qc_review.csv"
        with rv.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        lines = ["image;object_id;error_type;reviewer;note"]
        for r in rows:
            lines.append(f"{r['image']};{r['object_id']};khong;{r['reviewer']};café ok")
        rv.write_bytes("\n".join(lines).encode("cp1252"))

        out = self.run_script("qc_report")
        self.assertNotIn("không nằm trong mẫu", out)
        self.assertNotIn("CHƯA kiểm", out)

        # Re-write gold_manual.json and taxonomy.json with UTF-8 BOM
        gold = self.sub / "labels" / "gold_manual.json"
        gold.write_bytes(b"\xef\xbb\xbf" + gold.read_bytes())
        tax = self.sub / "taxonomy.json"
        tax.write_bytes(b"\xef\xbb\xbf" + tax.read_bytes())
        self.run_script("validate_labels", "--all")

        # Also support Excel's explicit "sep=;" directive line with cp1252
        rv.write_bytes(("sep=;\n" + "\n".join(lines)).encode("cp1252"))
        out_sep = self.run_script("qc_report")
        self.assertNotIn("không nằm trong mẫu", out_sep)
        self.assertNotIn("CHƯA kiểm", out_sep)

    # ------------------------------------------------------------------ BUG-09
    def test_bug09_cli_validation_and_warnings(self) -> None:
        out_m1 = self.run_script(
            "merge_prelabels",
            "data/cache/yolo26m_all_c005.coco.json",
            "--thr", "person:0.4",
            "--out", "work/tmp.json",
            success=False,
        )
        self.assertIn("LỖI:", out_m1)
        self.assertNotIn("Traceback", out_m1)

        out_m2 = self.run_script(
            "merge_prelabels",
            "data/cache/yolo26m_all_c005.coco.json",
            "--thr", "person=abc",
            "--out", "work/tmp.json",
            success=False,
        )
        self.assertIn("LỖI:", out_m2)
        self.assertNotIn("Traceback", out_m2)

        out_p = self.run_script(
            "prelabel",
            "--model", "yolo26",
            "--split", "pilot",
            "--thr", "person=abc",
            "--out", "work/tmp_p",
            success=False,
        )
        self.assertIn("LỖI:", out_p)
        self.assertNotIn("Traceback", out_p)

        # sam2_points must fail BEFORE loading ultralytics/SAM when --box or --pos is malformed
        out_s = self.run_script(
            "sam2_points",
            "--image", "img_013.jpg",
            "--box", "1,2,3",
            "--out", "work/tmp_s",
            success=False,
        )
        self.assertIn("LỖI:", out_s)
        self.assertNotIn("Traceback", out_s)

        out_s2 = self.run_script(
            "sam2_points",
            "--image", "img_013.jpg",
            "--pos", "400,480;600,110",
            "--out", "work/tmp_s",
            success=False,
        )
        self.assertIn("LỖI:", out_s2)
        self.assertNotIn("Traceback", out_s2)

        out_s3 = self.run_script(
            "sam2_points",
            "--image", "img_013.jpg",
            "--box", "200,200,100,300",
            "--out", "work/tmp_s",
            success=False,
        )
        self.assertIn("LỖI:", out_s3)
        self.assertNotIn("Traceback", out_s3)

        # validate_labels must format 19.6px as 19.6px (not rounded to 20px)
        gold_path = self.sub / "labels" / "gold_manual.json"
        gold_names = self._split_names("gold")
        coco = {
            "images": [{"id": i, "file_name": n, "width": self.img_sizes[n][0], "height": self.img_sizes[n][1]}
                       for i, n in enumerate(gold_names, 1)],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 19.6], "segmentation": []}],
        }
        gold_path.write_text(json.dumps(coco), encoding="utf-8")
        out_v = self.run_script("validate_labels", "submission/labels/gold_manual.json", "--expect", "gold")
        self.assertIn("19.6px", out_v)
        self.assertNotIn("20px < 20px", out_v)

        # inject_traps must warn when fewer than requested traps can be created
        self.run_script(
            "merge_prelabels",
            "data/cache/yoloe26s_three_wheeler_all_c005.coco.json",
            "--conf", "0.2",
            "--split", "batch,gold",
            "--out", "work/prelabel/low/pre.coco.json",
        )
        low_coco = json.loads((self.root / "work" / "prelabel" / "low" / "pre.coco.json").read_text(encoding="utf-8"))
        for an in low_coco["annotations"]:
            an["score"] = 0.25
            an["attributes"]["conf"] = "0.25"
        (self.root / "work" / "prelabel" / "low" / "pre.coco.json").write_text(json.dumps(low_coco), encoding="utf-8")
        out_i = self.run_script(
            "inject_traps",
            "--pre", "work/prelabel/low/pre.coco.json",
            "--student", "22001234",
            "--out", "work/prelabel/low_traps.coco.json",
            "--key", "work/prelabel/low_key.json",
        )
        self.assertIn("CẢNH BÁO:", out_i)

        # check_submission when review_stats.json has gold=null must warn about pre_traps.coco.json vs pre_traps_import
        self._build_valid_submission()
        rs = json.loads((self.sub / "qc" / "review_stats.json").read_text(encoding="utf-8"))
        rs["gold"] = None
        (self.sub / "qc" / "review_stats.json").write_text(json.dumps(rs), encoding="utf-8")
        out_c = self.run_script("check_submission", success=False)
        self.assertIn("pre_traps.coco.json", out_c)

    # ------------------------------------------------------------------ Negative E2E checks
    def test_negative_submission_checks(self) -> None:
        self._build_valid_submission()
        self.assertIn("0 lỗi", self.run_script("check_submission"))

        # 1. Missing SAM line in model_log.csv
        ml = self.sub / "model_log.csv"
        ml_orig = ml.read_text(encoding="utf-8")
        ml.write_text(
            ml_orig.replace("yolo26-m+sam(sam2.1_b.pt)", "gdino").replace("--sam sam2.1_b.pt", "--conf 0.3"),
            encoding="utf-8",
        )
        self.assertIn("SAM", self.run_script("check_submission", success=False))
        ml.write_text(ml_orig, encoding="utf-8")

        # 2. Tampered taxonomy rules
        tax_p = self.sub / "taxonomy.json"
        tax_orig = tax_p.read_text(encoding="utf-8")
        tax = json.loads(tax_orig)
        tax["rules"]["trap_catch_min"] = 0.1
        tax_p.write_text(json.dumps(tax), encoding="utf-8")
        self.assertIn("rules", self.run_script("check_submission", success=False))
        tax_p.write_text(tax_orig, encoding="utf-8")

        # 3. Empty error_type in qc_review.csv
        rv_p = self.sub / "qc" / "qc_review.csv"
        rv_orig = rv_p.read_text(encoding="utf-8")
        rv_p.write_text(rv_orig.replace(",khong,", ",,", 1), encoding="utf-8")
        self.assertIn("error_type", self.run_script("check_submission", success=False))
        rv_p.write_text(rv_orig, encoding="utf-8")

    def test_git_submission_checks(self) -> None:
        self._build_valid_submission()
        shutil.copyfile(SOURCE / ".gitignore", self.root / ".gitignore")
        env = clean_cp1252_env()
        subprocess.run(["git", "init"], cwd=self.root, check=True, capture_output=True, env=env)
        subprocess.run(["git", "config", "user.name", "Test Student"], cwd=self.root, check=True, capture_output=True, env=env)
        subprocess.run(["git", "config", "user.email", "student@example.invalid"], cwd=self.root, check=True, capture_output=True, env=env)
        for i in range(1, 5):
            (self.root / f"part{i}.txt").write_text(f"part {i}\n", encoding="utf-8")
            subprocess.run(["git", "add", f"part{i}.txt"], cwd=self.root, check=True, capture_output=True, env=env)
            subprocess.run(["git", "commit", "-m", f"Part {i}"], cwd=self.root, check=True, capture_output=True, env=env)

        out = self.run_script("check_submission")
        self.assertIn("0 lỗi, 0 cảnh báo", out)

        # Force-adding a model weight inside models/ must be flagged
        (self.root / "models").mkdir(exist_ok=True)
        (self.root / "models" / "fake.pt").write_bytes(b"weights")
        subprocess.run(["git", "add", "-f", "models/fake.pt"], cwd=self.root, check=True, capture_output=True, env=env)
        out_bad = self.run_script("check_submission", success=False)
        self.assertIn("models/", out_bad)

    def test_advanced_xml_and_self_consistency_cp1252(self) -> None:
        self._build_valid_submission()
        out_xml = self.run_script(
            "coco_to_cvat_xml",
            "--coco", "submission/labels/batch_reviewed.json",
            "--shape", "box",
            "--out", "work/cvat_import/batch.xml",
        )
        self.assertIn("box", out_xml)
        self.assertTrue((self.root / "work" / "cvat_import" / "batch.xml").exists())

        blind = self.sub / "labels" / "self_check_blind.json"
        shutil.copyfile(self.sub / "labels" / "batch_reviewed.json", blind)
        out_sc = self.run_script(
            "self_consistency",
            "--a", "submission/labels/batch_reviewed.json",
            "--b", "submission/labels/self_check_blind.json",
            "--images-from", "b",
        )
        self.assertIn("ĐẠT", out_sc)

    def test_day08_setup_cp1252_wizard_sync_and_no_cvat_ai_recommendation(self) -> None:
        env = clean_cp1252_env()
        setup_py_path = SOURCE / "docs" / "day08_setup.py"
        setup_py_src = setup_py_path.read_text(encoding="utf-8")

        # 1. Verify docs/day08_setup.py stays standalone (zero local repo imports)
        tree = ast.parse(setup_py_src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported & {"common", "scripts", "viz", "prelabel"}, f"Unexpected repo imports: {imported}")

        # 2. Verify no-args, --help, -h, and check --light do not crash with UnicodeEncodeError under piped cp1252
        for args in ([], ["--help"], ["-h"]):
            with self.subTest(setup_args=args):
                proc = subprocess.run(
                    [sys.executable, str(setup_py_path), *args],
                    cwd=self.root,
                    capture_output=True,
                    env=env,
                )
                stdout = proc.stdout.decode("utf-8", errors="replace")
                stderr = proc.stderr.decode("utf-8", errors="replace")
                self.assertEqual(proc.returncode, 0, stderr)
                self.assertNotIn("UnicodeEncodeError", stderr, stderr)
                self.assertNotIn("Traceback", stderr, stderr)
                self.assertIn("day08_setup.py", stdout)

        proc_light = subprocess.run(
            [sys.executable, str(setup_py_path), "check", "--light"],
            cwd=self.root,
            capture_output=True,
            env=env,
        )
        stdout_light = proc_light.stdout.decode("utf-8", errors="replace")
        stderr_light = proc_light.stderr.decode("utf-8", errors="replace")
        self.assertNotIn("UnicodeEncodeError", stderr_light, stderr_light)
        self.assertNotIn("UnicodeDecodeError", stderr_light, stderr_light)
        self.assertNotIn("Traceback", stderr_light, stderr_light)
        self.assertIn(proc_light.returncode, (0, 1), stdout_light + stderr_light)
        self.assertIn("Python ", stdout_light)
        if proc_light.returncode == 0:
            self.assertIn("SẴN SÀNG cho buổi học.", stdout_light)
        else:
            self.assertIn("Còn mục [LỖI]", stdout_light)

        # 3. Verify download skip ('[ có ]'), download error ('[LỖI]'), and check --light CVAT error under clean cp1252
        models_dir = self.root / "models"
        models_dir.mkdir(exist_ok=True)
        for name in ("yolo26n.pt", "yoloe-26s-seg.pt", "mobileclip2_b.ts", "mobile_sam.pt", "sam2.1_t.pt"):
            (models_dir / name).write_bytes(b"0" * 1_000_001)
        proc_dl_skip = subprocess.run(
            [sys.executable, str(setup_py_path), "download", "--profile", "8gb"],
            cwd=self.root,
            capture_output=True,
            env=env,
        )
        out_dl_skip = proc_dl_skip.stdout.decode("utf-8", errors="replace")
        err_dl_skip = proc_dl_skip.stderr.decode("utf-8", errors="replace")
        self.assertEqual(proc_dl_skip.returncode, 0, out_dl_skip + err_dl_skip)
        self.assertNotIn("UnicodeEncodeError", err_dl_skip, err_dl_skip)
        self.assertIn("[ có ] yolo26n.pt", out_dl_skip)

        dl_err_code = "\n".join([
            "import sys, urllib.error",
            f"sys.path.insert(0, {str(setup_py_path.parent)!r})",
            "import day08_setup",
            "def _fail(*a, **kw): raise urllib.error.URLError('offline')",
            "day08_setup.urllib.request.urlopen = _fail",
            "day08_setup.MD = day08_setup.pathlib.Path('models_missing').resolve()",
            "day08_setup.download('8gb')",
        ])
        proc_dl_err = subprocess.run([sys.executable, "-c", dl_err_code], cwd=self.root, capture_output=True, env=env)
        out_dl_err = proc_dl_err.stdout.decode("utf-8", errors="replace")
        err_dl_err = proc_dl_err.stderr.decode("utf-8", errors="replace")
        self.assertEqual(proc_dl_err.returncode, 0, out_dl_err + err_dl_err)
        self.assertNotIn("UnicodeEncodeError", err_dl_err, err_dl_err)
        self.assertIn("[LỖI] yolo26n.pt:", out_dl_err)

        cvat_err_code = "\n".join([
            "import sys, urllib.error",
            f"sys.path.insert(0, {str(setup_py_path.parent)!r})",
            "import day08_setup",
            "def _fail(*a, **kw): raise urllib.error.URLError('refused')",
            "day08_setup.urllib.request.urlopen = _fail",
            "day08_setup.check(True)",
        ])
        proc_cvat_err = subprocess.run([sys.executable, "-c", cvat_err_code], cwd=self.root, capture_output=True, env=env)
        out_cvat_err = proc_cvat_err.stdout.decode("utf-8", errors="replace")
        err_cvat_err = proc_cvat_err.stderr.decode("utf-8", errors="replace")
        self.assertEqual(proc_cvat_err.returncode, 1, out_cvat_err + err_cvat_err)
        self.assertNotIn("UnicodeEncodeError", err_cvat_err, err_cvat_err)
        self.assertIn("không kết nối được CVAT", out_cvat_err)

        # 4. Verify Python 3.13 is accepted and Python <3.10 / >=3.14 report the 3.10 - 3.13 (khuyến nghị 3.12) guidance
        for ver_tuple, expect_rc, expect_text in (
            ((3, 9, 18), 1, "cần Python 3.10 - 3.13 (khuyến nghị 3.12)"),
            ((3, 13, 0), 0, "[ OK ] Python 3.13.0"),
            ((3, 14, 0), 1, "cần Python 3.10 - 3.13 (khuyến nghị 3.12)"),
        ):
            code = "\n".join([
                "import sys, types, urllib.request",
                f"sys.path.insert(0, {str(setup_py_path.parent)!r})",
                "import day08_setup",
                "class _V(tuple):",
                "    @property",
                "    def major(self): return self[0]",
                "    @property",
                "    def minor(self): return self[1]",
                "    @property",
                "    def micro(self): return self[2]",
                f"day08_setup.sys = types.SimpleNamespace(version_info=_V({ver_tuple!r}), exit=sys.exit)",
                "class _R:",
                "    status = 200",
                "    def __enter__(self): return self",
                "    def __exit__(self, *a): return False",
                "day08_setup.urllib.request.urlopen = lambda *a, **kw: _R()",
                "day08_setup.check(True)",
            ])
            proc_v = subprocess.run([sys.executable, "-c", code], cwd=self.root, capture_output=True, env=env)
            out_v = proc_v.stdout.decode("utf-8", errors="replace")
            err_v = proc_v.stderr.decode("utf-8", errors="replace")
            self.assertNotIn("UnicodeEncodeError", err_v, err_v)
            self.assertEqual(proc_v.returncode, expect_rc, out_v + err_v)
            self.assertIn(expect_text, out_v)

        # 5. Verify embedded <script type="text/plain" id="setup-py"> matches docs/day08_setup.py identically
        wizard_html = (SOURCE / "docs" / "setup_wizard.html").read_text(encoding="utf-8")
        m = re.search(r'<script type="text/plain" id="setup-py">\r?\n(.*?)\r?\n</script>', wizard_html, re.S)
        self.assertIsNotNone(m, "Missing <script type='text/plain' id='setup-py'> in docs/setup_wizard.html")
        embedded = m.group(1).replace("\r\n", "\n")
        disk_norm = setup_py_src.replace("\r\n", "\n").rstrip("\n")
        self.assertEqual(embedded, disk_norm)
        self.assertNotRegex(wizard_html, r"3\.13\s+trở\s+lên")
        self.assertIn("3.10–3.13", wizard_html)
        self.assertIn("3.14+", wizard_html)

        # 6. Verify no cvat.ai setup recommendation or stale 'server lớp' remains anywhere in active setup docs/guides/scripts
        active_docs = (
            "README.md",
            "GUIDE.md",
            "CHECKPOINTS.md",
            "SUBMISSION.md",
            "RULES.md",
            "RUBRIC.md",
            "docs/SETUP.md",
            "docs/CVAT_AI.md",
            "docs/COLAB.md",
            "docs/MODELS.md",
            "docs/ADVANCED.md",
            "docs/THRESHOLD.md",
            "docs/CALIBRATION_QC.md",
            "docs/REVIEW_CHECKLIST.md",
            "docs/setup_wizard.html",
            "scripts/README.md",
            "scripts/check_env.py",
            "scripts/cvat_function_yolo26.py",
            "notebooks/day08_colab.ipynb",
            "K5-L23-Day08-AI-Assisted-Labeling.md",
            "K5-L23-Day08-AI-Assisted-Labeling.html",
        )
        for rel_path in active_docs:
            txt = (SOURCE / rel_path).read_text(encoding="utf-8")
            with self.subTest(doc=rel_path):
                self.assertNotRegex(txt, r"app\.cvat\.ai|server\s+(?:của\s+)?lớp", f"Stale CVAT server text in {rel_path}")
                # Allow docs.cvat.ai links and explicit negations ("không dùng ... cvat.ai"), forbid any other cvat.ai mention
                cleaned = re.sub(r"https?://docs\.cvat\.ai\S*", "", txt)
                cleaned = re.sub(r"không[^\n.]{0,45}cvat\.ai", "", cleaned, flags=re.I)
                self.assertNotIn("cvat.ai", cleaned.lower(), f"Unexpected cvat.ai recommendation in {rel_path}")

        # 7. Verify Pha 0 / CP0 docs and vlearn-lab-authoring prose word counts on Lab Markdown
        for rel_path in ("K5-L23-Day08-AI-Assisted-Labeling.md", "K5-L23-Day08-AI-Assisted-Labeling.html"):
            txt = (SOURCE / rel_path).read_text(encoding="utf-8")
            with self.subTest(lab_doc=rel_path):
                self.assertIn("http://localhost:8080", txt)
                self.assertIn("docs/setup_wizard.html", txt)
                self.assertIn("docs/day08_setup.py", txt)
                self.assertIn("--set small", txt)
                self.assertIn("--set sam1", txt)
                self.assertIn("day08_setup.py check --light", txt)

        lab_md = (SOURCE / "K5-L23-Day08-AI-Assisted-Labeling.md").read_text(encoding="utf-8")
        self.assertNotRegex(lab_md, r"<(?:details|summary|div|span|table|tr|td|p|br|img|a)\b", "Raw HTML in Lab MD")
        body = re.sub(r"^---\r?\n.*?\r?\n---\r?\n", "", lab_md, flags=re.S)
        sections = re.split(r"(?m)^## ", body)
        self.assertEqual(len(sections), 9)
        for idx, sec in enumerate(sections):
            text = "\n".join(sec.splitlines()[1:]) if idx > 0 else sec
            text = re.sub(r":::(?:code-tabs|checklist|chart|diagram|image|video|repository).*?:::", "", text, flags=re.S)
            text = re.sub(r":::callout[^\n]*\n", "", text)
            text = re.sub(r"(?m)^:::\s*$", "", text)
            text = re.sub(r"```.*?```", "", text, flags=re.S)
            text = re.sub(r"(?m)^###+ .*$", "", text)
            text = re.sub(r"(?m)^\|.*\|$", "", text)
            text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
            text = re.sub(r"https?://\S+", "", text)
            wc = len([w for w in re.split(r"\s+", text.strip()) if w])
            lo, hi = (180, 300) if idx in (0, 1) else ((220, 360) if idx == 8 else (500, 1000))
            self.assertTrue(lo <= wc <= hi, f"Section {idx} word count {wc} not in [{lo}, {hi}]")


if __name__ == "__main__":
    unittest.main()
