"""Print one table row per finished run in results/<experiment>/<run>/summary.json."""
import json
from pathlib import Path

import pandas as pd

rows = []
for summary_path in sorted(Path("results").glob("*/*/summary.json")):
    summary = json.loads(summary_path.read_text())
    test = summary["test"]
    rows.append({
        "experiment": summary_path.parent.parent.name,
        "run": summary["name"],
        "samples/class": summary["arguments"]["samples_per_class"] or "all",
        "components": "+".join(summary["arguments"]["components"]) or "none",
        "epochs": summary["arguments"]["epochs"],
        "test top1": round(test["top1"], 4),
        "test top5": round(test["top5"], 4),
        "nested F1": round(test.get("nested_f1_micro", float("nan")), 3),
        "positioned F1": round(test.get("positioned_f1_micro", float("nan")), 3),
        "minutes": round(sum(record["seconds"] for record in summary["history"]) / 60),
    })
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False) if rows else "no finished runs")
