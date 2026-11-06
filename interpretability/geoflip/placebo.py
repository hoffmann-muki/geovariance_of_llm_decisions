"""Placebo country names: invented, pronounceable, and matched in length to the real names.

A placebo goes through the same pipeline as a real country. How often placebo
pairs "flip" estimates how many flips arise from surface form alone (PLAN.md
§ 6). Names are built from a fixed syllable inventory with a fixed seed. Any
candidate is rejected if it is a real place name (any ACLED country, admin
area or location, or a name in WORLD_COUNTRIES), or a single token in the
model's vocabulary (a sign that it is an existing word or name). The kept names
match the character-length distribution of the real names by quantile.

  python -m geoflip.placebo --out inputs/v4/placebos.json
"""

import argparse
import json
import random
from pathlib import Path

from .acled import load_notes
from .utils import read_json

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261106
N_PLACEBOS = 20
ONSETS = [
    "b",
    "d",
    "f",
    "g",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "z",
    "br",
    "dr",
    "gr",
    "kr",
    "tr",
    "st",
]
VOWELS = ["a", "e", "i", "o", "u"]
CODAS = ["", "", "", "n", "r", "l", "s"]
ENDINGS = ["ia", "a", "an", "or", "en", "ar", "ine", "and", "esh", "ova"]

WORLD_COUNTRIES = """Afghanistan Albania Algeria Andorra Angola Antigua Argentina Armenia Australia Austria
Azerbaijan Bahamas Bahrain Bangladesh Barbados Belarus Belgium Belize Benin Bhutan Bolivia Bosnia Botswana
Brazil Brunei Bulgaria Burkina Burundi Cambodia Cameroon Canada Chad Chile China Colombia Comoros Congo Croatia
Cuba Cyprus Czechia Denmark Djibouti Dominica Ecuador Egypt Eritrea Estonia Eswatini Ethiopia Fiji Finland
France Gabon Gambia Georgia Germany Ghana Greece Grenada Guatemala Guinea Guyana Haiti Honduras Hungary Iceland
India Indonesia Iran Iraq Ireland Israel Italy Jamaica Japan Jordan Kazakhstan Kenya Kiribati Kosovo Kuwait
Kyrgyzstan Laos Latvia Lebanon Lesotho Liberia Libya Liechtenstein Lithuania Luxembourg Madagascar Malawi
Malaysia Maldives Mali Malta Mauritania Mauritius Mexico Micronesia Moldova Monaco Mongolia Montenegro Morocco
Mozambique Myanmar Namibia Nauru Nepal Netherlands Nicaragua Niger Nigeria Norway Oman Pakistan Palau Palestine
Panama Paraguay Peru Philippines Poland Portugal Qatar Romania Russia Rwanda Samoa Senegal Serbia Seychelles
Singapore Slovakia Slovenia Somalia Spain Sudan Suriname Sweden Switzerland Syria Taiwan Tajikistan Tanzania
Thailand Togo Tonga Trinidad Tunisia Turkey Turkmenistan Tuvalu Uganda Ukraine Uruguay Uzbekistan Vanuatu
Venezuela Vietnam Yemen Zambia Zimbabwe""".split()


def real_place_names(rows):
    names = set(WORLD_COUNTRIES)
    for r in rows:
        for key in ("country", "admin1", "admin2", "admin3", "location"):
            for part in r.get(key, "").replace("-", " ").split():
                names.add(part.strip("(),."))
    return {n.lower() for n in names if n}


def candidate(rng):
    syllables = rng.choice([1, 2, 2, 3])
    stem = "".join(
        rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)
        for _ in range(syllables)
    )
    return (stem + rng.choice(ENDINGS)).capitalize()


def generate(real_names, tokenizer, place_names, n=N_PLACEBOS, seed=SEED):
    rng = random.Random(seed)
    vocab = tokenizer.get_vocab()
    pool = set()
    while len(pool) < 3000:
        name = candidate(rng)
        if (
            name.lower() in place_names
            or "Ġ" + name in vocab
            or not 4 <= len(name) <= 12
        ):
            continue
        pool.add(name)
    pool = sorted(pool)
    rng.shuffle(pool)
    # Quantile matching on character length (without "the" and spaces).
    targets = sorted(
        len(name.replace("the ", "").replace(" ", "")) for name in real_names
    )
    quantiles = [targets[round(i * (len(targets) - 1) / (n - 1))] for i in range(n)]
    chosen = []
    for length in quantiles:
        best = min(
            (p for p in pool if p not in chosen),
            key=lambda p: (abs(len(p) - length), pool.index(p)),
        )
        chosen.append(best)
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--countries", type=Path, default=ROOT / "inputs" / "countries.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    from .scoring import MODELS

    spec = MODELS["llama3.2-3b"]
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    real = list(read_json(args.countries)["real"].values())
    names = generate(real, tokenizer, real_place_names(load_notes()))
    count = (
        lambda x: len(tokenizer.encode(" in " + x, add_special_tokens=False)) - 1
    )  # noqa: E731
    out = {
        "seed": SEED,
        "method": "geoflip/placebo.py: syllable names, not a real place, not a single vocabulary token, length-matched by quantile",
        "placebos": {
            name: {"chars": len(name), "tokens": count(name)} for name in names
        },
        "real_tokens": {name: count(name) for name in real},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["placebos"]))


if __name__ == "__main__":
    main()
