#!/usr/bin/env python3

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE_PATH = Path(__file__).with_name("pipeline.py")
SPEC = importlib.util.spec_from_file_location("sensor_recorder_pipeline", PIPELINE_PATH)
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class PipelineIntegrationTest(unittest.TestCase):
    def test_validate_normalize_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recording = root / "recording"
            output = root / "output"
            recording.mkdir()
            (recording / "meta.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "state": "finished",
                        "capture_mode": "arkit",
                    }
                ),
                encoding="utf-8",
            )
            (recording / "wide.mp4").write_bytes(b"test-video")

            with (recording / "arkit_pose.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                fields = [
                    "frame_index", "record_slot", "sensor_sec", "utc_sec",
                    "tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz",
                    "exposure_sec", "tracking_state", "fx_px", "fy_px",
                    "cx_px", "cy_px", "width_px", "height_px",
                ]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(5):
                    writer.writerow(
                        {
                            "frame_index": index,
                            "record_slot": index,
                            "sensor_sec": 100.0 + index * 0.1,
                            "utc_sec": 1000.0 + index * 0.1,
                            "tx_m": index * 0.01,
                            "ty_m": 0,
                            "tz_m": 0,
                            "qw": 1,
                            "qx": 0,
                            "qy": 0,
                            "qz": 0,
                            "exposure_sec": 0.01,
                            "tracking_state": "normal",
                            "fx_px": 500,
                            "fy_px": 500,
                            "cx_px": 320,
                            "cy_px": 240,
                            "width_px": 640,
                            "height_px": 480,
                        }
                    )

            for filename, measurement_fields in (
                ("accelerometer.csv", ("ax_m_s2", "ay_m_s2", "az_m_s2")),
                ("gyroscope.csv", ("gx_rad_s", "gy_rad_s", "gz_rad_s")),
            ):
                with (recording / filename).open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    fields = ["sensor_sec", "utc_sec", *measurement_fields]
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for index in range(11):
                        row = {
                            "sensor_sec": 99.95 + index * 0.05,
                            "utc_sec": 999.95 + index * 0.05,
                        }
                        row.update({field: 0 for field in measurement_fields})
                        writer.writerow(row)

            config = Path(__file__).resolve().parents[2] / "configs" / "iphone_arkit.json"
            test_config = root / "config.json"
            config_value = json.loads(config.read_text(encoding="utf-8"))
            config_value["images"]["enabled"] = False
            test_config.write_text(json.dumps(config_value), encoding="utf-8")
            arguments = [
                "single", "--data", str(recording), "--output", str(output),
                "--config", str(test_config), "--to", "normalize",
            ]
            self.assertEqual(pipeline.main(arguments), 0)
            self.assertEqual(pipeline.main(arguments), 0)

            metadata = json.loads(
                (output / "normalized" / "recording.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["counts"]["frames"], 5)
            self.assertEqual(metadata["counts"]["keyframes"], 3)
            self.assertEqual(metadata["counts"]["imu_pairs"], 11)
            self.assertTrue((output / "normalized" / "calibration.csv").is_file())

            fake_importer = root / "fake_importer.py"
            fake_importer.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "output = next(a.split('=', 1)[1] for a in sys.argv "
                "if a.startswith('--output_map='))\n"
                "path = pathlib.Path(output)\n"
                "path.mkdir(parents=True, exist_ok=True)\n"
                "(path / 'metadata').write_text('test', encoding='utf-8')\n",
                encoding="utf-8",
            )
            fake_importer.chmod(0o755)
            full_arguments = [
                "single", "--data", str(recording), "--output", str(output),
                "--config", str(test_config), "--vimap-importer", str(fake_importer),
            ]
            self.assertEqual(pipeline.main(full_arguments), 0)
            self.assertEqual(pipeline.main(full_arguments), 0)
            report = json.loads(
                (output / "maps" / "00_imported" / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["verified_counts"]["vertices"], 3)
            self.assertEqual(report["verified_counts"]["viwls_edges"], 2)
            self.assertEqual(report["verified_counts"]["raw_image_resources"], 0)


if __name__ == "__main__":
    unittest.main()
