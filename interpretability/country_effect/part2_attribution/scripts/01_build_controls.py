"""Stage 1A: build the control names and the Stage 2 baselines (PREREGISTRATION.md § 3, § 5).

Applies the token rules of configs/stimuli_controls.json with the Llama
tokenizer, draws the seeded subsets, checks that no control is a Part 1 name
and that every control renders through the country slot like the Part 1
names. Writes outputs/01_controls/controls.json and its manifest.

  cd interpretability && python -m country_effect.part2_attribution.scripts.01_build_controls
"""

import random

from country_effect.part2_attribution import common as C
from geoflip import scoring
from geoflip import study as S
from geoflip.utils import load_secret, read_json, require, write_json

OUT = C.OUTPUTS / "01_controls"
POPULATION_LIMIT = 5.0  # millions


def token_count(tokenizer, name):
    return len(tokenizer.encode(" " + name, add_special_tokens=False))


def main():
    import os

    import transformers

    spec = scoring.MODELS[C.MODEL_KEY]
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = load_secret("HF_TOKEN", scoring.ENV_FILE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec["hf_id"], revision=spec["revision"]
    )
    cfg = read_json(C.CONFIGS / "stimuli_controls.json")
    descriptions, names, tasks = C.part1()
    taken = {
        n.lower()
        for n in list(names["real"])
        + list(names["real"].values())
        + list(names["fictional"])
    }
    rng = random.Random(cfg["seed"])

    out, rejected = {}, []
    for family, fam in cfg["families"].items():
        if "population_millions" in fam:
            pool = [
                n
                for n, pop in fam["population_millions"].items()
                if pop < POPULATION_LIMIT
            ]
        else:
            pool = list(fam["candidates"])
        allowed = fam["tokens"] if isinstance(fam["tokens"], list) else [fam["tokens"]]
        eligible = []
        for name in pool:
            n = token_count(tokenizer, name)
            if n not in allowed:
                rejected.append(
                    {
                        "family": family,
                        "name": name,
                        "tokens": n,
                        "reason": f"tokens not in {allowed}",
                    }
                )
                continue
            require(name.lower() not in taken, f"{name} is already a Part 1 name")
            eligible.append(name)
        if "draw" in fam:
            require(len(eligible) >= fam["draw"], f"{family}: too few eligible names")
            eligible = sorted(rng.sample(eligible, fam["draw"]))
        for name in eligible:
            ids = tokenizer.encode(" " + name, add_special_tokens=False)
            out[name] = {"family": family, "tokens": len(ids), "token_ids": ids}
    require(len(out) == len({n.lower() for n in out}), "Duplicate control names")

    # Every control renders through the slot exactly like the Part 1 names.
    trial = {"slot": names["slot"], "real": {}, "fictional": {n: n for n in out}}
    for b in S.BOUNDARIES:
        spec_b = S.load_inputs()["descriptions"]["boundaries"][b]
        base = {
            k: v
            for k, v in C.contexts(b, "study")[0].items()
            if k in ("actors", "grievance", "place", "time") and v
        }
        findings = S.slot_findings(b, spec_b, {"base_id": "x", **base}, trial)
        require(not findings, f"Slot findings: {findings[:3]}")

    panel = C.panel()
    draw = random.Random(panel["seed"])
    baselines = {}
    for length, family in panel["baselines"]["families"].items():
        pool = sorted(n for n, v in out.items() if v["family"] == family)
        baselines[length] = draw.sample(pool, panel["baselines"]["draw"])

    OUT.mkdir(parents=True, exist_ok=True)
    result = {"names": out, "rejected": rejected, "baselines": baselines}
    write_json(OUT / "controls.json", result)
    C.write_manifest(
        OUT,
        __file__,
        [C.CONFIGS / "stimuli_controls.json", C.PART1 / "inputs" / "names.json"],
        [OUT / "controls.json"],
    )
    by_family = {}
    for n, v in out.items():
        by_family.setdefault(v["family"], []).append(f"{n}({v['tokens']})")
    for family, items in by_family.items():
        print(f"{family:18s} {len(items):2d}: {', '.join(items)}")
    print("rejected:", [(r["name"], r["tokens"]) for r in rejected])
    print("baselines:", baselines)


if __name__ == "__main__":
    main()
