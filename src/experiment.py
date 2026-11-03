"""Prepare the adopted main protocol without submitting any model requests."""

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

from . import inference, model_inputs
from .utils import (
    digest,
    file_digest,
    read_csv,
    require,
    verified_snapshot,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data/variants/main"
VALIDATION = ROOT / "data/e5_validation"
RESULTS = ROOT / "runs/e5_validation"


def encoder_input(text, protocol):
    return protocol["encoder_input_template"].format(
        instruction=protocol["encoder_instruction"], text=text
    )


def encoder_prototypes(vectors, protocol):
    """Normalize each reference, average within class, then normalize each mean."""
    references = protocol["encoder_reference_inputs"]
    require(len(vectors) == len(references) == 18, "Need eighteen reference embeddings")
    grouped = {label: [] for label in protocol["class_order"]}
    for reference, vector in zip(references, vectors):
        grouped[reference["label"]].append(inference.normalise(vector))
    result = []
    for label in protocol["class_order"]:
        require(len(grouped[label]) == 3, "Need three references per class")
        mean = [math.fsum(values) / 3 for values in zip(*grouped[label])]
        result.append(inference.normalise(mean))
    return result


def reviewed_references(protocol):
    rows = [
        r
        for r in read_csv(VALIDATION / "inputs.csv")
        if r["kind"] == "example_centroids"
    ]
    require(
        Counter(r["label"] for r in rows)
        == Counter({label: 3 for label in protocol["class_order"]}),
        "Selected reference counts changed",
    )
    reviews = json.loads((VALIDATION / "reviews.json").read_text())
    contexts = reviews["reviews"]
    require(
        len(contexts) == len({r["reviewer"] for r in contexts}) == 2,
        "Need two independent review contexts",
    )
    assessments = [
        {r["input_id"]: r for r in context["assessments"]} for context in contexts
    ]
    decisions = {r["input_id"]: r for r in reviews["adjudications"]}
    references = []
    for label in protocol["class_order"]:
        for row in sorted(
            (r for r in rows if r["label"] == label), key=lambda r: r["input_id"]
        ):
            ident = row["input_id"]
            require(row["text_sha256"] == digest(row["text"]), "Reference text changed")
            require(
                decisions[ident]["decision"] == "accept"
                and decisions[ident]["text_sha256"] == row["text_sha256"],
                "Reference not adjudicated for this text",
            )
            for mapping in assessments:
                judgment = mapping[ident]
                require(
                    judgment["decision"] == "accept"
                    and judgment["predicted_label"] == label
                    and judgment["text_sha256"] == row["text_sha256"]
                    and bool(judgment["evidence"])
                    and judgment["evidence"] in row["text"],
                    "Reference lacks concordant literal review",
                )
            references.append(
                {
                    "reference_id": ident,
                    "label": label,
                    "text": row["text"],
                    "text_sha256": row["text_sha256"],
                }
            )
    return references


def prepare(output_dir, token_counter=None):
    require(not output_dir.exists(), "Use an unused protocol directory")
    verified_snapshot(MAIN)
    validation = verified_snapshot(VALIDATION)
    results = verified_snapshot(RESULTS)
    compatibility = verified_snapshot(ROOT / "runs/compatibility")
    compatibility_inputs = verified_snapshot(ROOT / "data/compatibility")
    require(
        results["input_manifest_sha256"] == file_digest(VALIDATION / "manifest.json")
        and results["test_complete"]
        and results["selected_method"] == "example_centroids",
        "Selected validation run changed",
    )
    require(
        compatibility["technical_gate_passed"]
        and compatibility["input_manifest_sha256"]
        == file_digest(ROOT / "data/compatibility/manifest.json"),
        "Compatibility evidence changed",
    )
    selection = json.loads((RESULTS / "selection.json").read_text())
    require(
        file_digest(RESULTS / "selection.json") == results["selection_sha256"]
        and selection["selected_method"] == "example_centroids",
        "Development selection changed",
    )
    require(
        validation["inputs"]["main_manifest_sha256"]
        == file_digest(MAIN / "manifest.json"),
        "Validation targets another source release",
    )
    metrics = json.loads((RESULTS / "test_metrics.json").read_text())[
        "example_centroids"
    ]
    protocol = json.loads((MAIN / "protocol.json").read_text())
    references = reviewed_references(protocol)
    protocol.pop("prototype_tokens")
    protocol.pop("provider_compatibility_tested")
    protocol.pop("note")
    protocol.update(
        format_version=2,
        encoder_method="example_centroids",
        encoder_instruction=validation["instructions"]["short"],
        encoder_reference_inputs=references,
        encoder_classification="Embed the eighteen unprefixed reference examples in listed order; L2-normalize each vector, average the three vectors per class, and L2-normalize each mean. L2-normalize each instructed event vector and select its highest-cosine class; exact ties use class_order. Reject invalid dimensions, zero/nonfinite vectors or means. Similarities are not probabilities.",
        provider_metadata={
            model_inputs.ENCODER: validation["models"][model_inputs.ENCODER],
            model_inputs.DECODER: compatibility_inputs["models"][model_inputs.DECODER],
        },
        decision={
            "date": "2026-11-03",
            "basis": "User selected the short instruction and example-based classes after reviewing held-out results, retaining frozen E5 weights and fixed references.",
            "held_out_correct": metrics["correct"],
            "held_out_n": metrics["n"],
            "original_screening_guardrail_passed": results["guardrail_passed"],
            "decision_after_test_results": True,
            "limitations": "Small curated test; civilian attacks and strategic developments each scored 2/5. Adoption does not establish geographic invariance or operational accuracy.",
        },
        analysis_amendment={
            "registry": "data/analysis_plan.json",
            "registry_sha256": file_digest(ROOT / "data/analysis_plan.json"),
            "change": "Before main inference, replace the encoder instruction and class-reference representation. Preserve all stimuli, class meanings, 438 contrasts per model, endpoints, missingness rules, multiplicity adjustment and request order. The registry's original protocol hash remains a historical preparation binding; this protocol is the adopted model configuration.",
        },
        main_inference_performed=False,
        provider_preflight="Compatibility evidence records prior checks; recheck model availability, routing and pricing before an authorized main run. Served weight revisions are not exposed.",
    )
    count, tokenizer = (
        model_inputs.load_token_counter() if token_counter is None else token_counter
    )
    require(
        tokenizer == protocol["tokenizer"],
        "Tokenizer differs from the prepared release",
    )
    for reference in references:
        reference["encoder_tokens"] = count(reference["text"])
        require(0 < reference["encoder_tokens"] <= 512, "Reference exceeds E5 context")
    stimuli = read_csv(MAIN / "stimuli.csv")
    prior_audit = {r["input_id"]: r for r in read_csv(MAIN / "length_audit.csv")}
    require(len(prior_audit) == len(stimuli) == 10820, "Source stimulus count changed")
    audit = []
    for row in stimuli:
        previous = prior_audit[row["stimulus_id"]]
        require(
            previous["text_sha256"] == row["text_sha256"] == digest(row["text"]),
            "Stimulus text changed",
        )
        text = encoder_input(row["text"], protocol)
        tokens = count(text)
        require(
            type(tokens) is int and 0 < tokens <= 512,
            "Stimulus exceeds E5 context; truncation is forbidden",
        )
        audit.append(
            {**previous, "encoder_input_sha256": digest(text), "encoder_tokens": tokens}
        )
    lengths = {r["input_id"]: r["encoder_tokens"] for r in audit}
    originals = sorted(
        (r for r in stimuli if r["condition"] == "original"),
        key=lambda r: (lengths[r["stimulus_id"]], r["stimulus_id"]),
    )
    repeat_ids = [
        originals[index]["stimulus_id"]
        for index in (0, len(originals) // 2, len(originals) - 1)
    ]
    require(
        repeat_ids == protocol["repeat_stimulus_ids"],
        "Repeat length ranking changed; review the request-order amendment",
    )
    schedule = read_csv(MAIN / "request_order.csv")
    require(
        len(schedule) == len({r["request_id"] for r in schedule}) == 21646,
        "Request order changed",
    )
    expected = Counter(
        (model, row["stimulus_id"], "0")
        for model in (model_inputs.ENCODER, model_inputs.DECODER)
        for row in stimuli
    )
    expected.update(
        (model, ident, "1")
        for model in (model_inputs.ENCODER, model_inputs.DECODER)
        for ident in repeat_ids
    )
    require(
        Counter((r["model"], r["stimulus_id"], r["replicate"]) for r in schedule)
        == expected,
        "Request schedule no longer matches the inputs",
    )
    registry = json.loads((ROOT / "data/analysis_plan.json").read_text())
    require(
        registry["inputs"]["stimuli_sha256"] == file_digest(MAIN / "stimuli.csv")
        and len(registry["primary_contrasts"]) == 438,
        "Analysis eligibility changed",
    )
    protocol["stimuli"] = "data/variants/main/stimuli.csv"
    protocol["request_order"] = "data/variants/main/request_order.csv"
    protocol["planned_classifications"] = len(schedule)
    protocol["encoder_reference_embeddings"] = len(references)
    inputs = [
        MAIN / "manifest.json",
        MAIN / "stimuli.csv",
        MAIN / "protocol.json",
        MAIN / "request_order.csv",
        VALIDATION / "manifest.json",
        RESULTS / "manifest.json",
        ROOT / "data/analysis_plan.json",
        ROOT / "data/compatibility/manifest.json",
        ROOT / "runs/compatibility/manifest.json",
        Path(__file__),
        Path(model_inputs.__file__),
        Path(inference.__file__),
        ROOT / "src/utils.py",
    ]
    manifest = {
        "scope": "adopted_main_protocol_preparation",
        "preparation_complete": True,
        "main_inference_performed": False,
        "inputs": {
            str(path.relative_to(ROOT)): {"sha256": file_digest(path)}
            for path in inputs
        },
        "stimuli": len(stimuli),
        "planned_classifications": len(schedule),
        "encoder_reference_embeddings": len(references),
        "maximum_encoder_tokens": max(lengths.values()),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".experiment-", dir=output_dir.parent
    ) as directory:
        stage = Path(directory) / "export"
        stage.mkdir()
        write_json(stage / "protocol.json", protocol)
        write_csv(stage / "length_audit.csv", list(audit[0]), audit)
        (stage / "report.md").write_text(
            f"# Adopted Experiment Protocol\n\nE5 uses the validation-selected short instruction and eighteen reviewed, country-neutral examples: three per class, averaged after normalization and normalized again. Its weights remain frozen. Llama retains its existing prompt and CoreWeave bf16 route.\n\nThe user selected this approach on 3 November 2026 after its 23/30 held-out result. The original screening guardrail remains failed; V and S each scored 2/5. This is a documented decision after validation, not a retroactive screening pass.\n\nAll {len(stimuli):,} stimuli and the {len(schedule):,}-classification request order are reused from the immutable [source release](../variants/main/report.md). The eighteen reference embeddings replace the original six-definition embeddings. Updated E5 inputs reach at most {manifest['maximum_encoder_tokens']} of 512 tokens; truncation is disabled.\n\nThe [analysis registry](../analysis_plan.json) retains its endpoints and 438 contrasts per model. The protocol records the encoder amendment and preserves the registry's historical binding. Main inference has not run; the CLI only prepares inputs.\n"
        )
        manifest["outputs"] = {
            path.name: {"sha256": file_digest(path)} for path in stage.iterdir()
        }
        write_json(stage / "manifest.json", manifest)
        stage.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/experiment")
    args = parser.parse_args()
    manifest = prepare(args.output_dir)
    print(
        f"Prepared adopted E5 protocol for {manifest['stimuli']:,} stimuli; no inference performed."
    )


if __name__ == "__main__":
    main()
