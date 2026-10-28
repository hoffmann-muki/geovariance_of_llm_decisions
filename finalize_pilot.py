#!/usr/bin/env python3
"""Validate reviewed source events and export accepted pilot inputs."""

import argparse
import csv
import itertools
import json
import tempfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import prepare_data as prep

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/processed"
REVIEW = ROOT / "data/reviews/pilot_review.json"
FAMILIES = ("neutral_control", "actor_identity", "action_wording", "legitimizing_framing",
            "delegitimizing_framing", "intensity", "source_attribution")
SLOT_FIELDS = ("country", "event_type", "label", "sub_event_type", "length_bin")
CHANGE_FIELDS = ("original_event_id", "final_event_id", "country", "label", "length_bin",
                 "original_words", "final_words", "original_actor_group", "final_actor_group",
                 "original_actor1", "final_actor1", "original_month", "final_month", "reason")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_snapshot(path):
    manifest = json.loads((path / "manifest.json").read_text())
    for name, info in manifest["outputs"].items():
        require(prep.file_digest(path / name) == info["sha256"],
                f"Prepared output changed: {name}")
    return manifest


def similarity_hits(pilot, others, threshold=.90, within=True):
    """Screen both SequenceMatcher directions, with only safe upper-bound pruning."""
    normalized = {r["event_id_cnty"]: prep.text_key(r["notes"]) for r in pilot + others}
    pairs = itertools.chain(itertools.product(pilot, others),
                            itertools.combinations(pilot, 2) if within else ())
    checked, hits = 0, []
    for a, b in pairs:
        checked += 1
        ab = SequenceMatcher(None, normalized[a["event_id_cnty"]], normalized[b["event_id_cnty"]], autojunk=False)
        if ab.real_quick_ratio() < threshold or ab.quick_ratio() < threshold:
            continue
        ba = SequenceMatcher(None, normalized[b["event_id_cnty"]], normalized[a["event_id_cnty"]], autojunk=False)
        forward, reverse = ab.ratio(), ba.ratio()
        if max(forward, reverse) >= threshold:
            hits.append({"pilot_event_id": a["event_id_cnty"], "other_event_id": b["event_id_cnty"],
                         "forward_similarity": round(forward, 9), "reverse_similarity": round(reverse, 9)})
    return checked, hits


def accepted_review(row, blind, semantic):
    event_id = row["event_id_cnty"]
    require(blind["predicted_label"] == row["label"], f"Blind label disagreement: {event_id}")
    require(blind["low_ambiguity_suitability"] == "yes", f"Unresolved blind objection: {event_id}")
    require(semantic["source_label"] == row["label"] and
            semantic["decision"] in {"retain", "retain_with_restrictions"},
            f"Unresolved semantic objection: {event_id}")
    for span in [blind["evidence_span"]] + semantic["verbatim_evidence"]:
        require(bool(span) and span in row["notes"], f"Nonverbatim evidence: {event_id}")
    allowed = set(semantic["allowed_edit_families_for_construction"])
    excluded = set(semantic["excluded_edit_families_for_initial_batch"])
    require(not allowed & excluded and allowed | excluded == set(FAMILIES),
            f"Incomplete family policy: {event_id}")


def decision_digest(review):
    decisions = {key: review[key] for key in ("cases", "selection", "family_policy", "reserve_quarantine")}
    return prep.digest(json.dumps(decisions, sort_keys=True, ensure_ascii=False))


def load_review(path):
    review = json.loads(path.read_text(encoding="utf-8"))
    require(review["review_mode"] == "ai_only_adversarial_user_authorized" and
            review["human_review_performed"] is False, "Incorrect review provenance")
    methods = review["methods_review"]
    require(methods["decision"] == "approved_from_methods_perspective_for_input_preparation" and
            not methods["blocking_findings"], "Methods review has unresolved findings")
    require(methods["reviewed_decisions_sha256"] == decision_digest(review),
            "Review decisions changed after sign-off")
    cases = {r["event_id"]: r for r in review["cases"]}
    selection = review["selection"]
    require(len(cases) == len(review["cases"]), "Duplicate review case")
    require(len({r["event_id"] for r in selection}) == len(selection), "Repeated selected event")
    return review, cases


def finalize(output_dir, source_dir=SOURCE, review_path=REVIEW):
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    review, cases = load_review(review_path)
    manifest = verified_snapshot(source_dir)
    require(manifest["status"] == "awaiting_review", "Preprocessing validation did not pass")
    for name, checksum in review["methods_review"]["reviewed_selection_files"].items():
        require(prep.file_digest(source_dir / name) == checksum, f"Methods-reviewed selection changed: {name}")
    original, main, reserves, pool = [read_csv(source_dir / name) for name in
                                    ("pilot.csv", "candidates.csv", "reserves.csv", "sampling_pool.csv")]
    by_id = {r["event_id_cnty"]: r for r in pool}
    originals = {r["event_id_cnty"]: r for r in original}
    require(len(review["selection"]) == len(originals) and
            {r["original_event_id"] for r in review["selection"]} == set(originals),
            "Review does not cover the original pilot slots exactly once")
    # Review all recorded cases against the original source, including rejected alternatives.
    needed = {(r["source_file"], r["source_record"]) for r in cases.values()}
    raw = {}
    for source in manifest["inputs"]:
        path = ROOT / "conflict_data" / source["file"]
        require(prep.file_digest(path) == source["sha256"], f"Raw source changed: {path.name}")
        for ordinal, row in enumerate(read_csv(path), 1):
            key = (path.name, str(ordinal))
            if key in needed:
                raw[key] = row
    for event_id, case in cases.items():
        source = raw[case["source_file"], case["source_record"]]
        require(source["event_id_cnty"] == event_id and prep.digest(source["notes"]) == case["notes_sha256"],
                f"Reviewed source changed: {event_id}")
    used = {r["event_group_id"] for r in main + original + reserves}
    pilot, edits, changes = [], [], []
    for choice in review["selection"]:
        event_id = choice["event_id"]
        case, candidate, old = cases[event_id], by_id[event_id], originals[choice["original_event_id"]]
        source = raw[candidate["source_file"], candidate["source_record"]]
        require(candidate["source_file"] == case["source_file"] and
                candidate["source_record"] == case["source_record"], f"Review/source reference mismatch: {event_id}")
        require(all(candidate[k] == old[k] for k in SLOT_FIELDS), f"Replacement slot changed: {event_id}")
        require(all(candidate[k] == prep.clean(source[k]) for k in set(candidate) & set(source)),
                f"Raw metadata mismatch: {event_id}")
        require(int(candidate["word_count"]) == len(source["notes"].split()), f"Word count mismatch: {event_id}")
        if event_id != old["event_id_cnty"]:
            require(candidate["event_group_id"] not in used, f"Replacement family already allocated: {event_id}")
        row = {**old, **candidate, "notes": source["notes"], "sample_id": "a_pilot_" + event_id,
               "original_event_id": old["event_id_cnty"], "input_review_status": "accepted_for_variant_construction",
               "review_mode": "ai_only_adversarial", "human_review_performed": "false", "variants_approved": "false"}
        accepted_review(row, case["blind"], case["semantic"])
        pilot.append(row)
        for family in FAMILIES:
            allowed = case["semantic"]["allowed_edit_families_for_construction"]
            status = ("requires_recipe_review" if family == "intensity" else
                      "eligible_for_drafting" if family in allowed else "excluded_from_initial_batch")
            edits.append({"event_id": event_id, "family": family, "status": status})
        if event_id != old["event_id_cnty"]:
            changes.append({"original_event_id": old["event_id_cnty"], "final_event_id": event_id,
                            "country": row["country"], "label": row["label"], "length_bin": row["length_bin"],
                            "original_words": old["word_count"], "final_words": row["word_count"],
                            "original_actor_group": old["actor_group"], "final_actor_group": row["actor_group"],
                            "original_actor1": old["actor1"], "final_actor1": row["actor1"],
                            "original_month": old["year_month"], "final_month": row["year_month"],
                            "reason": choice["reason"]})
    groups = {r["event_group_id"] for r in pilot}
    require(len(groups) == len(pilot), "Repeated pilot family")
    require(not groups & {r["event_group_id"] for r in main + reserves}, "Pilot overlaps another allocated role")
    quarantine = review["reserve_quarantine"]
    quarantine_ids = {r["event_id"] for r in quarantine}
    require(quarantine_ids <= {r["event_id_cnty"] for r in reserves}, "Unknown quarantined reserve")
    active = [dict(r, eligibility_status="available_pending_case_review") for r in reserves
              if r["event_id_cnty"] not in quarantine_ids]
    checked, hits = similarity_hits(pilot, main + active)
    require(not hits, f"Unresolved active similarity conflict: {hits}")
    _, evidence = similarity_hits(pilot, [r for r in reserves if r["event_id_cnty"] in quarantine_ids], within=False)
    require({(r["pilot_event_id"], r["other_event_id"]) for r in evidence} ==
            {(r["conflicts_with_pilot"], r["event_id"]) for r in quarantine}, "Quarantine evidence mismatch")
    eligibility = Counter((r["family"], r["status"]) for r in edits)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pilot-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "export"
        stage.mkdir()
        prep.write_csv(stage / "pilot.csv", prep.SAMPLE_FIELDS +
                       ["original_event_id", "input_review_status", "review_mode", "human_review_performed", "variants_approved"], pilot)
        prep.write_csv(stage / "edit_eligibility.csv", ["event_id", "family", "status"], edits)
        prep.write_csv(stage / "cohort_changes.csv", CHANGE_FIELDS, changes)
        prep.write_csv(stage / "available_reserves.csv", prep.SAMPLE_FIELDS + ["eligibility_status"], active)
        counts = Counter((r["country"], r["label"]) for r in pilot)
        actor_before = dict(sorted(Counter(r["actor_group"] for r in original).items()))
        actor_after = dict(sorted(Counter(r["actor_group"] for r in pilot).items()))
        lines = ["# Pilot Input Review", "",
                 f"Accepted **{len(pilot)} source events** for controlled-variant construction under user-authorized AI-only review. Human review was not performed.", "",
                 f"- Replaced {len(changes)} original slots while preserving country, class, primary subtype, and length band; {len(counts)} country/class cells remain.",
                 f"- Notes-only source-label agreement and positive suitability judgments cover all accepted cases. Semantic restrictions remain binding.",
                 f"- Screened {checked:,} all-date, all-country pilot/main, pilot/active-reserve, and within-pilot pairs in both directions: zero scores at or above 0.90.",
                 f"- Active reserves: {len(active)}, still pending case review. Quarantined: {', '.join(sorted(quarantine_ids)) or 'none'}. Quarantine evidence is in the manifest.",
                 "- Actor and reporting-month changes are recorded in `cohort_changes.csv`.", "",
                 "| Actor category | Original pilot | Accepted pilot |", "| --- | ---: | ---: |"]
        for category in sorted(actor_before.keys() | actor_after.keys()):
            label = category.replace("_", " ")
            lines.append(f"| {label} | {actor_before.get(category, 0)} | {actor_after.get(category, 0)} |")
        lines += ["",
                 "| Edit family | Eligible for drafting | Excluded from initial batch | Requires recipe review |",
                 "| --- | ---: | ---: | ---: |"]
        for family in FAMILIES:
            label = family.replace("_", " ")
            lines.append(f"| {label} | {eligibility[family, 'eligible_for_drafting']} | {eligibility[family, 'excluded_from_initial_batch']} | {eligibility[family, 'requires_recipe_review']} |")
        lines += ["", "Case evidence, independent assessments, construction restrictions, and replacement reasons are in [the consolidated review](../reviews/pilot_review.json). The original notes remain unchanged. Eligibility permits drafting; every exact original/edited pair still requires review before inference.", "",
                  "Input preparation is complete. Variant construction, synthetic country-swap descriptions, and inference checks remain. No constructed variants or model results are approved by this review.", ""]
        lines += [f"- {item}" for item in review["limitations"]]
        (stage / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = {"format_version": 1, "review_date": review["review_date"], "review_mode": review["review_mode"],
                  "human_review_performed": False, "variants_approved": False, "experimental_inference_performed": False,
                  "accepted_input_count": len(pilot), "original_slots_changed": len(changes),
                  "active_reserves": len(active), "quarantined_reserves": sorted(quarantine_ids),
                  "actor_composition_before": actor_before, "actor_composition_after": actor_after,
                  "similarity_audit": {"pairs_checked": checked, "threshold": .90, "autojunk": False,
                                       "direction": "both; maximum directional score triggers a flag",
                                       "scope": "all dates/countries; pilot vs main/active reserves and within pilot",
                                       "flags": hits, "quarantine_evidence": evidence},
                  "edit_eligibility": {f: {s: n for (family, s), n in eligibility.items() if family == f} for f in FAMILIES},
                  "inputs": {"preprocessing_manifest_sha256": prep.file_digest(source_dir / "manifest.json"),
                             "review_sha256": prep.file_digest(review_path),
                             "finalizer_sha256": prep.file_digest(Path(__file__)),
                             "preprocessor_sha256": prep.file_digest(Path(prep.__file__))},
                  "outputs": {p.name: {"sha256": prep.file_digest(p)} for p in sorted(stage.iterdir())}}
        prep.write_json(stage / "manifest.json", result)
        stage.rename(output_dir)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/pilot")
    args = parser.parse_args()
    try:
        result = finalize(args.output_dir.resolve(), args.source_dir, args.review)
    except (ValueError, KeyError, OSError, csv.Error) as error:
        parser.exit(1, f"Pilot finalization failed: {error}\n")
    print(f"Accepted {result['accepted_input_count']} AI-reviewed pilot inputs: {args.output_dir}")
