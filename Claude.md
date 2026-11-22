# CLAUDE.md


## Project Overview

### Notebook Generation & Data Science Guardrails
- **Flag discretionary decisions (⚠️):** Before code that applies a choice another analyst could reasonably make differently, add `⚠️ <decision>. <what it changes in the result>.` in markdown. This covers which rows count as the population (including not filtering), thresholds, time windows, handling of missing values, combining multiple values, and metric or test choice. Don't flag facts about the data format, established standards or choices the user specified.
- **Minimal Markdown:** Keep all Markdown text to an absolute minimum. Write only the essential purpose of the section. Omit obvious statements, fluff, and step-by-step explanations of standard code. Keep definitions of the analysed population, denominators, units and time windows.
- **Explicit Variable Naming:** Use full-word names for variables that live across cells and avoid abbreviations. Write `timestamp` instead of `ts` and `station_count` instead of `n_sta`. Short loop variables are fine inside a single comprehension or lambda. Keep lines under 100 characters; split long expressions into named variables within the same cell.
- **Self-Explanatory Code:** Keep inline comments to an absolute minimum. Rely on clear naming conventions and small, well-named functions to reveal your logic rather than writing comments to explain unclear code. 
- **Small, Single-Step Cells:** Limit each cell to a single logical step and split long cells instead of growing them. After any step whose result is needed to understand the next one, print that result (e.g., shape, value counts, `.head()`, key numbers) in its own dedicated cell.
- **Answer Check:** Before finishing, check that every question in the goal has a cell whose output answers it directly. Summarise the answers in your chat reply, not in the notebook.
- **Plot the main results:** Show each main result of the notebook (the numbers a reader needs to understand the outcome) as a chart.
- **No Trailing Empty Cells:** Before finishing, remove any empty cells at the end of the notebook.
- **Row counts:** print the row count before and after every filter, dedup and join that drops rows, and give n next to every share.
- **Hypotheses are not facts:** Any claim no output in the notebook supports is a hypothesis. This includes explanations of why the data looks the way it does and predictions of what a different filter, threshold or method would show. Label it "Hypothesis:" and avoid words like "almost certainly" or "clearly". If you filter or remove data based on a hypothesis, say so in the ⚠️ flag.

### Experiments
- Every experiment variant gets its own folder `results/E##_<short_name>/` with a README (question, what changed, how to rerun, results, conclusion) and a row in `EXPERIMENTS.md`. Never overwrite an earlier experiment's folder or label file.
- New labels go into a new `data/component_targets_<rule>.npz`; new model or loss variants become options of `scripts/train.py` whose defaults reproduce earlier experiments.
- Never commit, push, branch or tag; the user decides what is committed and when.
