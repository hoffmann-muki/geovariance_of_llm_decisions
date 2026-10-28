"""Behavioral checks for scientific input integrity and sample separation."""

import contextlib
import copy
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import prepare_data as prep


def records():
    rows = []
    descriptions = {
        "Protests": [
            "Health workers marched outside the hospital demanding wages and protective equipment.",
            "University students assembled at the campus gate to oppose a proposed increase in tuition.",
            "Farmers protested beside the river after a flood destroyed crops and damaged irrigation channels.",
            "Residents staged a sit-in at the electricity office over repeated power cuts in their homes.",
            "Retired soldiers rallied near the courthouse requesting payment of overdue pensions.",
            "Shopkeepers demonstrated in the market against a new municipal tax on commercial premises.",
            "Teachers gathered by the school entrance calling for smaller classes and additional textbooks.",
            "Environmental activists held banners outside a factory to demand an end to chemical dumping.",
        ],
        "Riots": [
            "An angry crowd assaulted a motorist suspected of stealing spare parts in the industrial district.",
            "Demonstrators threw stones at officers and smashed parked vehicles outside parliament.",
            "Rival football supporters fought with sticks near the railway station after their match.",
            "A mob burned a warehouse and looted its stock after a dispute over ownership of the building.",
            "Taxi drivers beat a competitor and damaged his vehicle during a confrontation at the airport.",
            "A crowd forced entry into a council meeting and struck several officials with chairs.",
            "Protesters set barricades on fire and hurled bottles at security guards near a university dormitory.",
            "Villagers attacked a group accused of cattle theft and injured several suspects with farming tools.",
        ],
    }
    for country in ("Alpha", "Beta"):
        for event_type in ("Protests", "Riots"):
            for number in range(8):
                row = {field: "" for field in sorted(prep.REQUIRED)}
                row.update(
                    event_id_cnty=f"{country}_{event_type}_{number}",
                    country=country, region="Example region", event_type=event_type,
                    sub_event_type="Peaceful protest" if event_type == "Protests" else "Violent demonstration",
                    event_date=f"2024-01-{number + 1:02d}", admin1="Example province",
                    location=f"{event_type} town {number}", actor1="Participants", inter1="Protesters",
                    notes=f"  {country} account {number}: {descriptions[event_type][number]}\nOriginal wording PRESERVED.  ",
                )
                rows.append(row)
    return rows


def config():
    return {
        "seed": 42, "main_per_cell": 3, "pilot_per_cell": 1, "reserve_per_cell": 1,
        "pilot_countries": ["Alpha"],
        "country_regions": {"Alpha": "Example region", "Beta": "Example region"},
        "country_aliases": {},
        "samples": [{"name": "protests_riots", "countries": ["Alpha", "Beta"], "event_types": ["Protests", "Riots"]}],
        "near_duplicate_similarity": .9, "near_duplicate_days": 1, "length_bins_words": [40, 80],
    }


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "raw"
        self.inputs.mkdir()
        self.config = config()
        self.rows = records()
        self.write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def write_inputs(self, rows=None):
        with (self.inputs / "events.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(prep.REQUIRED))
            writer.writeheader()
            writer.writerows(self.rows if rows is None else rows)

    def run_preparation(self, name="prepared", config=None):
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            report = prep.run(self.inputs, output, config or self.config)
        return output, report

    def csv(self, path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_end_to_end_preserves_sources_and_leaves_reviews_pending(self):
        before = prep.file_digest(self.inputs / "events.csv")
        output, report = self.run_preparation()
        self.assertEqual(report["sample_counts"], {"main": 12, "pilot": 2, "reserve": 4})
        main = self.csv(output / "candidates.csv")
        self.assertEqual(set(Counter((r["country"], r["label"]) for r in main).values()), {3})
        self.assertEqual({r["country"] for r in self.csv(output / "pilot.csv")}, {"Alpha"})
        original = {r["event_id_cnty"]: r for r in self.rows}
        for name in ("candidates.csv", "pilot.csv", "reserves.csv"):
            for row in self.csv(output / name):
                self.assertEqual(row["notes"], original[row["event_id_cnty"]]["notes"])
                self.assertEqual(row["notes"], self.rows[int(row["source_record"]) - 1]["notes"])
        self.assertEqual(report["status"], "awaiting_review")
        self.assertEqual(before, prep.file_digest(self.inputs / "events.csv"))
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["inputs"][0]["sha256"], before)
        for name, metadata in manifest["outputs"].items():
            self.assertEqual(prep.file_digest(output / name), metadata["sha256"])

    def test_reruns_are_byte_identical(self):
        first, _ = self.run_preparation("first")
        second, _ = self.run_preparation("second")
        self.assertEqual(sorted(p.name for p in first.iterdir()), sorted(p.name for p in second.iterdir()))
        for path in first.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes(), path.name)

    def test_seed_changes_selection(self):
        first, _ = self.run_preparation("first")
        changed = {**self.config, "seed": 99}
        second, _ = self.run_preparation("second", changed)
        ids = lambda path: {r["event_id_cnty"] for r in self.csv(path / "candidates.csv")}
        self.assertNotEqual(ids(first), ids(second))

    def test_pilot_main_and_reserves_share_no_event_groups(self):
        # A chain of similar reports across consecutive dates forms one group.
        for i in range(3):
            self.rows[i]["location"] = "Same town"
            self.rows[i]["notes"] = (
                "Residents held a peaceful demonstration outside the town hall demanding "
                f"improved public services. Approximately {10 + i} people attended."
            )
        self.write_inputs()
        output, _ = self.run_preparation()
        seen = []
        for name in ("candidates.csv", "pilot.csv", "reserves.csv"):
            seen.extend(r["event_group_id"] for r in self.csv(output / name))
        self.assertEqual(len(seen), len(set(seen)))
        pairs = self.csv(output / "related_event_pairs.csv")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len({r["event_group_id"] for r in pairs}), 1)

    def test_exact_duplicates_use_normalized_text_without_changing_notes(self):
        a = {**self.rows[0], "event_id_cnty": "AAA", "notes": "  A FULLWIDTH Ａ report.\n"}
        b = {**self.rows[1], "event_id_cnty": "ZZZ", "notes": "a fullwidth a REPORT."}
        self.write_inputs([a, b])
        raw, _, excluded = prep.read_sources(self.inputs, self.config)
        kept = prep.deduplicate(raw, excluded)
        self.assertEqual([r["event_id_cnty"] for r in kept], ["AAA"])
        self.assertEqual(kept[0]["notes"], a["notes"])
        self.assertEqual(excluded[0]["reason"], "duplicate_notes")
        self.assertEqual(excluded[0]["retained_event_id"], "AAA")

    def test_conflicting_ids_and_note_labels_are_excluded(self):
        a, b = copy.deepcopy(self.rows[:2])
        b["event_id_cnty"] = a["event_id_cnty"]
        self.write_inputs([a, b])
        raw, _, excluded = prep.read_sources(self.inputs, self.config)
        self.assertEqual(prep.deduplicate(raw, excluded), [])
        self.assertEqual([r["reason"] for r in excluded], ["conflicting_event_id"] * 2)
        b["event_id_cnty"] = "other_id"
        b["notes"] = a["notes"]
        b["event_type"] = "Riots"
        self.write_inputs([a, b])
        raw, _, excluded = prep.read_sources(self.inputs, self.config)
        self.assertEqual(prep.deduplicate(raw, excluded), [])
        self.assertEqual([r["reason"] for r in excluded], ["conflicting_note_labels"] * 2)

    def test_identical_id_records_retain_one_record(self):
        self.write_inputs([self.rows[0], self.rows[0]])
        raw, _, excluded = prep.read_sources(self.inputs, self.config)
        self.assertEqual(len(prep.deduplicate(raw, excluded)), 1)
        self.assertEqual(excluded[0]["reason"], "duplicate_event_id")

    def test_invalid_required_values_are_reported_and_excluded(self):
        bad = copy.deepcopy(self.rows[:4])
        bad[0]["event_date"] = "2024-02-30"
        bad[1]["notes"] = " \n "
        bad[2]["event_type"] = "Unexpected label"
        bad[3]["region"] = "Wrong region"
        self.write_inputs(bad)
        raw, _, excluded = prep.read_sources(self.inputs, self.config)
        self.assertEqual(prep.deduplicate(raw, excluded), [])
        self.assertEqual({r["reason"] for r in excluded}, {
            "invalid_event_date", "missing_notes", "unknown_event_type", "country_region_mismatch",
        })

    def test_empty_optional_actor_remains_eligible(self):
        self.rows[0]["inter1"] = ""
        self.write_inputs()
        raw, _, excluded = prep.read_sources(self.inputs, self.config)
        kept = prep.deduplicate(raw, excluded)
        self.assertEqual(len(kept), 32)
        self.assertEqual(kept[0]["actor_group"], "unknown")

    def test_shortfalls_are_explicit_and_do_not_reduce_requested_quotas(self):
        self.write_inputs(self.rows[:2])
        output, report = self.run_preparation()
        self.assertEqual(report["status"], "quota_shortfall")
        summary = self.csv(output / "sampling_summary.csv")
        self.assertEqual(len(summary), 4)
        self.assertTrue(all(r["main_requested"] == "3" for r in summary))
        self.assertGreater(report["main_quota_shortfall"], 0)

    def test_cli_returns_nonzero_for_quota_shortfalls(self):
        self.write_inputs(self.rows[:2])
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        result = subprocess.run([
            sys.executable, str(prep.ROOT / "prepare_data.py"), "--input-dir", str(self.inputs),
            "--output-dir", str(self.root / "output"), "--config", str(config_path),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertTrue((self.root / "output/quality_report.md").is_file())

    def test_existing_output_and_review_work_are_preserved(self):
        output, _ = self.run_preparation()
        sheet = output / "candidates.csv"
        sheet.write_text(sheet.read_text() + "Reviewer work must remain intact.\n")
        before = sheet.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_preparation()
        self.assertEqual(before, sheet.read_bytes())

    def test_malformed_schema_fails_before_publishing_outputs(self):
        (self.inputs / "events.csv").write_text("event_id_cnty,notes\nA,Description\n")
        with self.assertRaisesRegex(ValueError, "schema"):
            self.run_preparation()
        self.assertFalse((self.root / "prepared").exists())

    def test_aliases_and_metadata_whitespace_are_standardized(self):
        self.rows[0]["country"] = "  Alias   country  "
        self.rows[0]["event_type"] = "  protests "
        self.config["country_aliases"] = {"Alias country": "Alpha"}
        self.write_inputs()
        raw, _, excluded = prep.read_sources(self.inputs, self.config)
        self.assertEqual(raw[0]["country"], "Alpha")
        self.assertEqual(raw[0]["label"], "P")
        self.assertFalse(excluded)

    def test_similar_reports_at_different_locations_share_a_family(self):
        # Neighborhood-specific reports must remain in the same sampling family.
        left, right = copy.deepcopy(self.rows[:2])
        left.update(event_date="2025-05-28", location="City - North", notes=(
            "On 28 May 2025, overnight, rioters blocked a road with burning tires in City - North "
            "calling for the government's resignation after armed clashes in the capital."
        ))
        right.update(event_date="2025-05-28", location="City - South", notes=left["notes"].replace("North", "South"))
        self.write_inputs([left, right])
        raw, _, excluded = prep.read_sources(self.inputs, self.config)
        kept = prep.deduplicate(raw, excluded)
        pairs = prep.group_related(kept, self.config)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(kept[0]["event_group_id"], kept[1]["event_group_id"])

    def test_selected_text_audit_ignores_admin_location_and_group_ids(self):
        raw, _, _ = prep.read_sources(self.inputs, self.config)
        left, right = raw[:2]
        right.update(notes=left["notes"], _text_key=left["_text_key"], admin1="Other province", location="Other town")
        left.update(sample_role="pilot", event_group_id="different-1")
        right.update(sample_role="main", event_group_id="different-2")
        flags, audit = prep.audit_similarity({"main": [right], "pilot": [left], "reserve": []}, self.config)
        self.assertFalse(audit["passed"])
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["similarity"], 1.0)

    def test_cross_admin_conflicts_are_resolved_by_resampling(self):
        # Initial source grouping cannot link this pair; all nine Alpha/P
        # examples are selected on the first pass, forcing the final audit to act.
        self.config.update(main_per_cell=6, reserve_per_cell=2)
        extra = copy.deepcopy(self.rows[0])
        extra.update(event_id_cnty="Alpha_similar", admin1="Other province", location="Different town")
        extra["notes"] = extra["notes"].replace("hospital", "clinic")
        self.write_inputs(self.rows + [extra])
        output, report = self.run_preparation()
        self.assertGreater(len(report["similarity_audit"]["passes"]), 1)
        self.assertGreater(report["similarity_audit"]["additional_links"], 0)
        self.assertEqual(report["similarity_audit"]["flagged_pairs"], 0)
        self.assertEqual(report["main_quota_shortfall"], 0)
        selected = [r for name in ("candidates.csv", "pilot.csv", "reserves.csv") for r in self.csv(output / name)]
        chosen_ids = {r["event_id_cnty"] for r in selected}
        self.assertFalse({self.rows[0]["event_id_cnty"], "Alpha_similar"} <= chosen_ids)

    def test_comparison_uses_shared_subtypes_without_relabeling_sources(self):
        self.config["primary_subtypes"] = {"Protests": "Peaceful protest", "Riots": "Violent demonstration"}
        self.rows[0]["sub_event_type"] = "Protest with intervention"
        self.rows[16]["sub_event_type"] = "Protest with intervention"
        self.write_inputs()
        output, report = self.run_preparation()
        self.assertEqual(report["excluded_rows"], 0)
        main = self.csv(output / "candidates.csv")
        self.assertTrue(all(r["sub_event_type"] == self.config["primary_subtypes"][r["event_type"]] for r in main))
        self.assertEqual(report["eligible_rows"], len(self.rows))

    def test_word_length_allocations_are_identical_across_countries(self):
        self.config.update(match_length_bins=True, length_bins_words=[40, 80])
        # Opposing source length distributions must still yield the same quotas.
        for row in self.rows:
            number = int(row["event_id_cnty"].rsplit("_", 1)[1])
            medium = number < (5 if row["country"] == "Alpha" else 3)
            if medium:
                row["notes"] += " Additional background describes the context and the events in detail." * 3
        self.write_inputs()
        output, report = self.run_preparation()
        self.assertEqual(report["main_quota_shortfall"], 0)
        rows = self.csv(output / "candidates.csv")
        for event_type in ("Protests", "Riots"):
            by_country = {c: Counter(r["length_bin"] for r in rows if r["country"] == c and r["event_type"] == event_type)
                          for c in ("Alpha", "Beta")}
            self.assertEqual(by_country["Alpha"], by_country["Beta"])
        self.assertTrue(report["comparability_passed"])

    def test_no_common_length_support_is_not_silently_replaced(self):
        self.config["match_length_bins"] = True
        for row in self.rows:
            if row["country"] == "Beta":
                row["notes"] += " Additional background information on the event and surrounding community." * 10
        self.write_inputs()
        output, report = self.run_preparation()
        self.assertEqual(report["status"], "quota_shortfall")
        self.assertEqual(report["sample_counts"]["main"], 0)
        self.assertEqual(report["main_quota_shortfall"], 12)
        self.assertTrue(all(r["main_requested"] == "3" for r in self.csv(output / "sampling_summary.csv")))

    def test_event_class_quota_is_applied_consistently_to_all_countries(self):
        self.config["main_per_event_type"] = {"Riots": 2}
        self.config["match_length_bins"] = True
        output, report = self.run_preparation()
        main = self.csv(output / "candidates.csv")
        counts = Counter((r["country"], r["event_type"]) for r in main)
        self.assertEqual(counts, {("Alpha", "Protests"): 3, ("Beta", "Protests"): 3,
                                  ("Alpha", "Riots"): 2, ("Beta", "Riots"): 2})
        self.assertEqual(report["main_quota_shortfall"], 0)

    def test_length_assignment_minimizes_total_distance_without_reusing_events(self):
        rows = [{"event_id_cnty": str(i), "event_group_id": str(i), "word_count": n}
                for i, n in enumerate([40, 43, 80])]
        used = set()
        chosen = prep.nearest_length_sample(rows, [41, 50], used, 42, "example")
        self.assertEqual(sum(abs(r["word_difference_from_target"]) for r in chosen), 8)
        self.assertEqual({r["word_count"] for r in chosen}, {40, 43})
        self.assertEqual(len(used), 2)
        self.assertTrue(all("target_word_count" not in r for r in rows))


if __name__ == "__main__":
    unittest.main()
