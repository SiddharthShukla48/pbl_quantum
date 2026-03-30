import argparse
from pathlib import Path
import pandas as pd


def _format_exam_rows(df: pd.DataFrame) -> str:
    """Format all exams in one slot into a compact single-cell string."""
    if df.empty:
        return ""

    lines = []
    for _, row in df.sort_values(["course_code", "exam_id"]).iterrows():
        lines.append(
            f"{row['course_code']} (id={int(row['exam_id'])}, sem={row['semester']}, enr={int(row['enrollment'])})"
        )
    return " | ".join(lines)


def build_day_slot_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert global time slots into a 2-slot-per-day matrix.

    Mapping rule:
    - day = floor(time_slot / 2) + 1
    - slot_in_day = 1 for even time_slot, 2 for odd time_slot

    Examples:
    - time_slot 0 -> day 1, slot 1
    - time_slot 1 -> day 1, slot 2
    - time_slot 2 -> day 2, slot 1
    - time_slot 3 -> day 2, slot 2
    """
    working = df.copy()
    working["day"] = (working["time_slot"] // 2) + 1
    working["slot_in_day"] = (working["time_slot"] % 2) + 1

    days = sorted(working["day"].unique().tolist())
    matrix_rows = []

    for day in days:
        day_df = working[working["day"] == day]
        slot1_df = day_df[day_df["slot_in_day"] == 1]
        slot2_df = day_df[day_df["slot_in_day"] == 2]

        matrix_rows.append(
            {
                "day": int(day),
                "slot_1": _format_exam_rows(slot1_df),
                "slot_2": _format_exam_rows(slot2_df),
            }
        )

    return pd.DataFrame(matrix_rows)


def process_timetable(input_csv: Path, output_dir: Path) -> Path:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = {"time_slot", "exam_id", "course_code", "enrollment"}
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing_required)}")

    if "semester" not in df.columns and "year" in df.columns:
        df = df.rename(columns={"year": "semester"})

    if "semester" not in df.columns:
        raise ValueError("Input CSV must contain either 'semester' or 'year' column.")

    # Keep output order deterministic.
    df = df.sort_values(["time_slot", "course_code", "exam_id"]).reset_index(drop=True)

    output_name = f"{input_csv.stem}_matrix_2slots_per_day.csv"
    matrix_path = output_dir / output_name

    matrix_df = build_day_slot_matrix(df)
    matrix_df.to_csv(matrix_path, index=False)

    return matrix_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process timetable_neal CSV: rename year->semester and build 2-slot/day matrix."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to timetable_neal.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs (default: same directory as input)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = process_timetable(input_csv=input_csv, output_dir=output_dir)

    print("Post-processing complete.")
    print(f"2-slot/day matrix:    {matrix_path}")


if __name__ == "__main__":
    main()
