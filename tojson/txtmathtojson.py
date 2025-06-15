import pandas as pd
import json
from pathlib import Path
import argparse
import sys

DEFAULT_INPUT_CSV  = "../autodl-tmp/math_captcha_data.csv"
DEFAULT_OUTPUT_JSON = "math_qa.json"
QUESTION_COL       = "question"
ANSWER_COL         = "answer"


def build_arg_parser() -> argparse.ArgumentParser:

    ap = argparse.ArgumentParser(
        description="Convert CSV (question, answer) → JSON for LLM fine‑tuning"
    )
    ap.add_argument(
        "-i", "--input", default=DEFAULT_INPUT_CSV,
        help=f"CSV(default: {DEFAULT_INPUT_CSV})"
    )
    ap.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT_JSON,
        help=f"JSON(default: {DEFAULT_OUTPUT_JSON})"
    )
    ap.add_argument(
        "--qcol", default=QUESTION_COL,
        help=f"question(default: {QUESTION_COL})"
    )
    ap.add_argument(
        "--acol", default=ANSWER_COL,
        help=f"answer(default: {ANSWER_COL})"
    )
    return ap

def main():
    args = build_arg_parser().parse_args()

    in_path  = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()


    try:
        df = pd.read_csv(in_path, encoding="utf-8")
    except FileNotFoundError:
        sys.exit(f"{in_path}")
    except Exception as e:
        sys.exit(f"{e}")

    for col in (args.qcol, args.acol):
        if col not in df.columns:
            sys.exit(f"'{col}'")

    if df.empty:
        sys.exit("❌empty")


    samples = []
    for idx, row in df.iterrows():
        q = str(row[args.qcol]).strip()
        a = str(row[args.acol]).strip()

        if not q or not a:
            sys.exit(f"❌{idx+1} ")

        samples.append(
            {
                "id": f"sample_{idx+1:06d}",
                "conversations": [
                    {"from": "user",      "value": q},
                    {"from": "assistant", "value": a},
                ],
            }
        )


    try:
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(samples, fp, ensure_ascii=False, indent=2)
    except Exception as e:
        sys.exit(f"❌JSON: {e}")


if __name__ == "__main__":
    main()
