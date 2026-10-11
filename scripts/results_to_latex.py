#!/usr/bin/env python3
"""Turn locked-run artifacts into \\input-able LaTeX for the Results chapter.

Usage:
    python scripts/results_to_latex.py OUT_DIR RUN_DIR [RUN_DIR ...]

Each RUN_DIR is a prepared/executed run directory (results/experiments/<id> or
results/smoke/<id>). Several may be given, for example one HuggingFace run for
RQ1/RQ2 and one Real-CATS run for RQ3/RQ4; artifacts are looked up across all of them.

The script writes one LaTeX file per Results table plus a scalar-macro file and a
provenance file into OUT_DIR (default target: writing/generated), and copies the
verified plots into OUT_DIR's sibling figures directory. It doubles as a
completeness check: if any run *claims* a research question in its protocol but is
missing an artifact that question needs, the script stops with an error. Research
questions that no provided run claims are written as clearly marked "pending"
placeholders so the chapter still compiles before the full run finishes.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.verify import verify_run


DELTA_LABELS = {
    300: "5 min",
    1800: "30 min",
    3600: "1 hour",
    21600: "6 hours",
    86400: "1 day",
    259200: "3 days",
    604800: "1 week",
    1209600: "2 weeks",
    2592000: "30 days",
}

HEADLINE_VARIANT = "canonical"

COMPARISON_LABELS = {
    "motifs_vs_volume": "fingerprint vs. motif volume",
    "combined_vs_baseline_composition_volume": (
        "combined vs. augmented baseline"
    ),
}

# Artifacts each research question needs, relative to a run directory. Presence of
# these is required whenever a run's protocol lists the research question.
REQUIRED = {
    "rq1": [
        "tables/rq1_hypothesis_tests.csv",
        "metrics/hf_controlled_canonical_metrics.csv",
        "metrics/hf_prevalence_canonical_metrics.csv",
        "metrics/hf_ordering_sensitivity.json",
        "tables/rq1_representation_tests.csv",
        "tables/rq1_role_diagnostics.csv",
        "tables/rq1_xgboost_robustness.csv",
        "figures/rq1_ablation_controlled.pdf",
        "figures/rq1_ablation_prevalence.pdf",
    ],
    "rq2": [
        "tables/rq2_delta_comparisons.csv",
        "metrics/hf_ordering_sensitivity.json",
        "tables/rq2_motif_coverage.csv",
        "figures/rq2_delta_performance_controlled.pdf",
        "figures/rq2_delta_performance_prevalence.pdf",
    ],
    "rq3": [
        "tables/rq3_global_results.csv",
        "tables/rq3_label_results.csv",
        "metrics/realcats_ordering_sensitivity.json",
        "figures/rq3_fingerprint_heatmap.pdf",
        "figures/rq3_confusion_matrix.pdf",
    ],
    "rq4": [
        "tables/rq4_global_results.csv",
        "tables/rq4_category_results.csv",
        "metrics/realcats_rq4_ordering_sensitivity.json",
        "figures/rq4_label_delta_curves.pdf",
    ],
    "blte": ["metrics/blte_results.json"],
    "validation": ["metrics/synthetic_validation.json"],
}


def fnum(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def fp(value) -> str:
    """Format a p-value for a table cell."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def fdelta(seconds) -> str:
    try:
        return DELTA_LABELS.get(int(seconds), f"{int(seconds)}\\,s")
    except (TypeError, ValueError):
        return "--"


def count_word(value: int) -> str:
    return {1: "one", 2: "two", 3: "three", 4: "four"}.get(value, str(value))


def latex_command(name: str, value: str) -> str:
    if not name.isalpha():
        raise ValueError(f"LaTeX command names must contain letters only: {name}")
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def tex_escape(text: object) -> str:
    out = str(text)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                 ("_", "\\_"), ("#", "\\#"), ("$", "\\$")):
        out = out.replace(a, b)
    return out


def reject_cell(flag) -> str:
    return "reject" if bool(flag) else "not rejected"


def yes_no(flag) -> str:
    return "yes" if bool(flag) else "no"


class RunSet:
    """A collection of run directories looked up as one artifact namespace."""

    def __init__(self, run_dirs: list[Path]) -> None:
        self.run_dirs = run_dirs
        self.protocols = {}
        self.verification = {}
        self.executed: list[Path] = []
        self.citable = bool(run_dirs)
        self.profiles = []
        self.run_ids = []
        self.claimed: set[str] = set()
        for run_dir in run_dirs:
            protocol_path = run_dir / "protocol.json"
            if not protocol_path.exists():
                raise SystemExit(f"No protocol.json in {run_dir}")
            protocol = json.loads(protocol_path.read_text())
            self.protocols[run_dir] = protocol
            self.run_ids.append(protocol.get("run_id", run_dir.name))
            self.profiles.append(protocol.get("execution_profile", "thesis"))
            if not (run_dir / "execution_manifest.json").exists():
                self.citable = False
                continue
            try:
                report = verify_run(run_dir, require_citable=False)
            except (OSError, RuntimeError, ValueError) as error:
                raise SystemExit(f"Run verification failed for {run_dir}: {error}") from error
            if not report.get("completed", False):
                raise SystemExit(f"Run is not complete: {run_dir}")
            self.verification[run_dir] = report
            self.executed.append(run_dir)
            self.claimed.update(protocol.get("research_questions", []))
            if not report.get("citable", False):
                self.citable = False
        self.rq_sources = {
            rq: self._choose_replications(
                [
                    run_dir
                    for run_dir in self.executed
                    if rq in self.protocols[run_dir].get("research_questions", [])
                ],
                f"research question {rq}",
            )
            for rq in self.claimed
        }
        # Keep a primary owner for single-run artifacts and backwards compatibility.
        # Replication-aware writers use sources_for() and report every selected run.
        self.owners = {
            rq: sources[0] if sources else None
            for rq, sources in self.rq_sources.items()
        }

    def _candidate_rank(self, run_dir: Path) -> tuple[int, int, int]:
        protocol = self.protocols[run_dir]
        profile_rank = {"thesis": 2, "development": 1, "smoke": 0}.get(
            protocol.get("execution_profile"), 1
        )
        return (
            int(bool(self.verification[run_dir].get("citable"))),
            profile_rank,
            -len(protocol.get("research_questions", [])),
        )

    def _choose(self, candidates: list[Path], description: str) -> Path | None:
        if not candidates:
            return None
        ranked = sorted(candidates, key=self._candidate_rank, reverse=True)
        if len(ranked) > 1 and self._candidate_rank(ranked[0]) == self._candidate_rank(ranked[1]):
            raise SystemExit(
                f"Ambiguous verified sources for {description}: "
                + ", ".join(str(path) for path in ranked)
            )
        return ranked[0]

    def _replication_index(self, run_dir: Path) -> int | None:
        value = self.protocols[run_dir].get("replication", {}).get("index")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _choose_replications(
        self, candidates: list[Path], description: str
    ) -> list[Path]:
        """Select all equally ranked, explicitly indexed thesis replications."""
        if not candidates:
            return []
        ranked = sorted(candidates, key=self._candidate_rank, reverse=True)
        best_rank = self._candidate_rank(ranked[0])
        selected = [path for path in ranked if self._candidate_rank(path) == best_rank]
        if len(selected) == 1:
            return selected

        indices = [self._replication_index(path) for path in selected]
        if any(index is None for index in indices) or len(set(indices)) != len(indices):
            raise SystemExit(
                f"Ambiguous verified sources for {description}: "
                + ", ".join(str(path) for path in selected)
                + ". Multiple equal-ranked runs require distinct locked replication indices."
            )
        return sorted(
            selected,
            key=lambda path: (self._replication_index(path), path.name),
        )

    def sources_for(self, rq: str) -> list[Path]:
        return list(self.rq_sources.get(rq, []))

    def latest_source_for(self, rq: str) -> Path | None:
        """Return the highest indexed replication selected for an RQ."""
        sources = self.sources_for(rq)
        if not sources:
            return None
        return max(
            sources,
            key=lambda path: (
                self._replication_index(path)
                if self._replication_index(path) is not None
                else -1,
                path.name,
            ),
        )

    def csv_from(self, run_dir: Path, relative: str) -> pd.DataFrame:
        path = run_dir / relative
        if not path.is_file():
            raise SystemExit(f"Expected artifact '{relative}' is missing from {run_dir}")
        return pd.read_csv(path)

    def json_from(self, run_dir: Path, relative: str) -> dict:
        path = run_dir / relative
        if not path.is_file():
            raise SystemExit(f"Expected artifact '{relative}' is missing from {run_dir}")
        return json.loads(path.read_text())

    def find(self, relative: str) -> Path | None:
        candidates = [
            run_dir for run_dir in self.executed if (run_dir / relative).is_file()
        ]
        owner = self._choose(candidates, relative)
        return None if owner is None else owner / relative

    def path_for(self, rq: str, relative: str) -> Path | None:
        owner = self.owners.get(rq)
        if owner is None:
            return None
        candidate = owner / relative
        return candidate if candidate.is_file() else None

    def require_for(self, rq: str, relative: str) -> Path:
        path = self.path_for(rq, relative)
        if path is None:
            owner = self.owners.get(rq)
            raise SystemExit(
                f"Expected artifact '{relative}' for {rq} is missing from {owner}"
            )
        return path

    def csv_for(self, rq: str, relative: str) -> pd.DataFrame:
        return pd.read_csv(self.require_for(rq, relative))

    def dataset(self, name: str, rq: str | None = None) -> dict | None:
        if rq is not None:
            owner = self.owners.get(rq)
            if owner is None:
                return None
            return self.protocols[owner].get("datasets", {}).get(name)
        candidates = [
            run_dir
            for run_dir in self.executed
            if name in self.protocols[run_dir].get("datasets", {})
        ]
        owner = self._choose(candidates, f"dataset {name}")
        return None if owner is None else self.protocols[owner]["datasets"][name]

    def require(self, relative: str) -> Path:
        path = self.find(relative)
        if path is None:
            raise SystemExit(
                f"Expected artifact '{relative}' not found in any of: "
                + ", ".join(str(d) for d in self.run_dirs)
            )
        return path

    def csv(self, relative: str) -> pd.DataFrame:
        return pd.read_csv(self.require(relative))

    def json(self, relative: str) -> dict:
        return json.loads(self.require(relative).read_text())

    def check_completeness(self) -> None:
        """An executed run must contain every artifact its claimed RQs need.

        A prepared-only run (no execution_manifest.json) is exempt because its
        RQ artifacts are legitimately absent until execution. It is not used as
        a data or result source.
        """
        for run_dir, protocol in self.protocols.items():
            if not (run_dir / "execution_manifest.json").exists():
                continue
            for rq in protocol.get("research_questions", []):
                for relative in REQUIRED.get(rq, []):
                    if not (run_dir / relative).exists():
                        raise SystemExit(
                            f"Executed run {run_dir.name} claims '{rq}' but artifact "
                            f"'{relative}' is missing. The run is incomplete; do not "
                            "build the chapter from it."
                        )


PENDING = "\\emph{Pending the full run (see Section~\\ref{sec:results_provenance}).}"


def pending_table(caption: str, label: str) -> str:
    return (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{PENDING}\n\\end{{table}}\n"
    )


def write_provenance(runs: RunSet, out: Path) -> None:
    ids = ", ".join(
        tex_escape(runs.protocols[path].get("run_id", path.name))
        for path in runs.executed
    ) or "none"
    lines = ["% Auto-generated by scripts/results_to_latex.py. Do not edit by hand."]
    lines.append(f"\\newcommand{{\\ResRunIds}}{{{ids}}}")
    lines.append(
        "\\newcommand{\\ResCitable}{" + ("true" if runs.citable else "false") + "}"
    )
    if runs.citable:
        lines.append("\\newcommand{\\ResPreliminaryBanner}{}")
    else:
        lines.append(
            "\\newcommand{\\ResPreliminaryBanner}{\\begin{center}\\fbox{"
            "\\parbox{0.9\\linewidth}{\\centering\\textbf{Preliminary, non-citable numbers.} "
            "At least one supplied source is non-citable or incomplete. The figures "
            "and tables are placeholders for fully verified thesis-profile runs.}}\\end{center}}"
        )
    (out / "results_provenance.tex").write_text("\n".join(lines) + "\n")


def write_figures(runs: RunSet, out: Path) -> None:
    """Copy verified result plots beside the generated LaTeX directory."""
    figure_out = out.parent / "figures"
    figure_out.mkdir(parents=True, exist_ok=True)
    for source in runs.sources_for("rq1"):
        replication = runs._replication_index(source)
        suffix = f"_rep{replication}" if replication is not None else ""
        for stem in ("rq1_ablation_controlled", "rq1_ablation_prevalence"):
            shutil.copy2(
                source / "figures" / f"{stem}.pdf",
                figure_out / f"{stem}{suffix}.pdf",
            )
    for source in runs.sources_for("rq2"):
        replication = runs._replication_index(source)
        suffix = f"_rep{replication}" if replication is not None else ""
        for stem in (
            "rq2_delta_performance_controlled",
            "rq2_delta_performance_prevalence",
        ):
            shutil.copy2(
                source / "figures" / f"{stem}.pdf",
                figure_out / f"{stem}{suffix}.pdf",
            )
    latest_hf = runs.latest_source_for("rq1")
    if latest_hf is not None:
        replication = runs._replication_index(latest_hf)
        suffix = f"_rep{replication}" if replication is not None else ""
        for stem in (
            "rq1_representation_ablation_controlled",
            "rq1_representation_ablation_prevalence",
            "rq2_motif_coverage_controlled",
            "rq2_motif_coverage_prevalence",
        ):
            source = latest_hf / "figures" / f"{stem}.pdf"
            if source.is_file():
                shutil.copy2(source, figure_out / f"{stem}{suffix}.pdf")
    for rq, stems in {
        "rq3": ("rq3_fingerprint_heatmap", "rq3_confusion_matrix"),
        "rq4": ("rq4_label_delta_curves",),
    }.items():
        sources = runs.sources_for(rq)
        if not sources:
            continue
        for stem in stems:
            shutil.copy2(
                sources[0] / "figures" / f"{stem}.pdf",
                figure_out / f"{stem}.pdf",
            )


def write_macros(runs: RunSet, out: Path) -> None:
    lines = ["% Auto-generated scalar macros. Do not edit by hand."]

    def cmd(name: str, value: str) -> None:
        lines.append(latex_command(name, value))

    hf_owner_rq = "rq1" if "rq1" in runs.claimed else "rq2"
    hf_source = runs.latest_source_for(hf_owner_rq)
    hf = (
        runs.protocols[hf_source].get("datasets", {}).get("huggingface")
        if hf_source is not None
        else None
    )
    if hf:
        attr = hf.get("attrition", {})
        cmd("ResCohortLabelled", f"{attr.get('labelled_targets', 0):,}")
        cmd("ResCohortInactive", f"{attr.get('inactive_targets', 0):,}")
        cmd("ResCohortEligible", f"{attr.get('eligible_active_targets', 0):,}")
        cmd("ResPrevalenceN", f"{attr.get('prevalence_targets', 0):,}")
        cmd("ResControlledN", f"{attr.get('controlled_targets', 0):,}")
        extraction = hf.get("edge_extraction", {})
        incident = extraction.get("incident", extraction)
        cmd("ResHFSourceEdges", f"{incident.get('source_edges', 0):,}")
        cmd("ResHFSelectedEdges", f"{incident.get('selected_edges', 0):,}")
    else:
        for name in (
            "ResCohortLabelled",
            "ResCohortInactive",
            "ResCohortEligible",
            "ResPrevalenceN",
            "ResControlledN",
            "ResHFSourceEdges",
            "ResHFSelectedEdges",
        ):
            cmd(name, "--")

    latest_rq1 = runs.latest_source_for("rq1")
    if latest_rq1 is not None and (latest_rq1 / REQUIRED["rq1"][0]).is_file():
        hyp = runs.csv_from(latest_rq1, "tables/rq1_hypothesis_tests.csv")
        canon = hyp[hyp["ordering_variant"] == HEADLINE_VARIANT]
        for _, row in canon.iterrows():
            key = str(row["cohort"]).capitalize()
            cmd(f"ResRQOne{key}DeltaAP", fnum(row["delta_average_precision"]))
            cmd(f"ResRQOne{key}Reject", reject_cell(row["reject_h0"]))
    else:
        for cohort in ("Controlled", "Prevalence"):
            cmd(f"ResRQOne{cohort}DeltaAP", "--")
            cmd(f"ResRQOne{cohort}Reject", "pending")
    sens_rq = "rq1" if "rq1" in runs.claimed else "rq2"
    latest_sens = runs.latest_source_for(sens_rq)
    sens = (
        latest_sens / "metrics/hf_ordering_sensitivity.json"
        if latest_sens is not None
        else None
    )
    if sens is not None and not sens.is_file():
        sens = None
    if sens:
        general = json.loads(sens.read_text()).get("general_signal")
        cmd("ResHFGeneralSignal", "yes" if general else "no")
    else:
        cmd("ResHFGeneralSignal", "pending")
    if "rq3" in runs.claimed and runs.path_for("rq3", REQUIRED["rq3"][0]):
        glob = runs.csv_for("rq3", "tables/rq3_global_results.csv")
        canon = glob[glob["ordering_variant"] == HEADLINE_VARIANT]
        if len(canon):
            cmd("ResRQThreeMacroAP", fnum(canon.iloc[0]["macro_average_precision"]))
            cmd("ResRQThreePermP", fp(canon.iloc[0]["permutation_p_value"]))
        else:
            cmd("ResRQThreeMacroAP", "--")
            cmd("ResRQThreePermP", "--")
    else:
        cmd("ResRQThreeMacroAP", "--")
        cmd("ResRQThreePermP", "--")
    if "rq4" in runs.claimed and runs.path_for("rq4", REQUIRED["rq4"][0]):
        glob = runs.csv_for("rq4", "tables/rq4_global_results.csv")
        canon = glob[glob["ordering_variant"] == HEADLINE_VARIANT]
        if len(canon):
            cmd(
                "ResRQFourDeltaMacroAP",
                fnum(canon.iloc[0]["observed_delta_macro_ap"]),
            )
            cmd("ResRQFourP", fp(canon.iloc[0]["p_value"]))
        else:
            cmd("ResRQFourDeltaMacroAP", "--")
            cmd("ResRQFourP", "--")
    else:
        cmd("ResRQFourDeltaMacroAP", "--")
        cmd("ResRQFourP", "--")

    (out / "results_macros.tex").write_text("\n".join(lines) + "\n")


def write_cohorts(runs: RunSet, out: Path) -> None:
    hf_owner_rq = "rq1" if "rq1" in runs.claimed else "rq2"
    sources = runs.sources_for(hf_owner_rq)
    if not sources:
        (out / "results_cohorts.tex").write_text(
            pending_table("HuggingFace cohort construction.", "tab:res_cohorts")
        )
        return
    rows = []
    for source in sources:
        protocol = runs.protocols[source]
        hf = protocol.get("datasets", {}).get("huggingface", {})
        attr = hf.get("attrition", {})
        extraction = hf.get("edge_extraction", {})
        incident = extraction.get("incident", extraction)
        replication = runs._replication_index(source)
        selected = int(attr.get("prevalence_targets", 0)) + int(
            attr.get("controlled_targets", 0)
        )
        rows.append(
            f"{replication if replication is not None else '--'} & "
            f"{int(attr.get('available_after_prior_run_exclusions', 0)):,} & "
            f"{int(attr.get('prevalence_targets', 0)):,} & "
            f"{int(attr.get('controlled_targets', 0)):,} & {selected:,} & "
            f"{int(incident.get('selected_edges', 0)):,} \\\\"
        )
    first_hf = runs.protocols[sources[0]]["datasets"]["huggingface"]
    first_attr = first_hf.get("attrition", {})
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{The {count_word(len(sources))} non-overlapping HuggingFace samples. "
        f"The source contains {int(first_attr.get('labelled_targets', 0)):,} labelled targets. "
        f"{int(first_attr.get('eligible_active_targets', 0)):,} pass the common "
        "activity and upper-tail rules before previously selected samples are excluded.}\n"
        "\\label{tab:res_cohorts}\n\\footnotesize\n"
        "\\begin{tabular}{@{}r r r r r r@{}}\n\\toprule\n"
        "\\textbf{Sample} & \\textbf{Available} & \\textbf{Prevalence} & "
        "\\textbf{Controlled} & \\textbf{Total} & \\textbf{Incident edges} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_cohorts.tex").write_text(table)


def write_rq1(runs: RunSet, out: Path) -> None:
    sources = runs.sources_for("rq1")
    if not sources:
        (out / "results_rq1.tex").write_text(
            pending_table("RQ1: motif contribution over the baseline.", "tab:res_rq1")
        )
        return
    rows = []
    for source in sources:
        hyp = runs.csv_from(source, "tables/rq1_hypothesis_tests.csv")
        canon = hyp[hyp["ordering_variant"] == HEADLINE_VARIANT]
        replication = runs._replication_index(source)
        for _, row in canon.iterrows():
            cohort = str(row["cohort"])
            metrics = runs.csv_from(
                source, f"metrics/hf_{cohort}_{HEADLINE_VARIANT}_metrics.csv"
            )
            sel = int(row["selected_delta"])
            at = metrics[metrics["delta"] == sel].set_index("arm")["average_precision"]
            rows.append(
                f"{replication if replication is not None else '--'} & "
                f"{cohort.capitalize()} & {fdelta(sel)} & {fnum(at.get('baseline'))} & "
                f"{fnum(at.get('motifs'))} & {fnum(at.get('combined'))} & "
                f"{fnum(row['delta_average_precision'])} & "
                f"[{fnum(row['ci_lower'])}, {fnum(row['ci_upper'])}] & "
                f"{fp(row['p_value'])} & {yes_no(row['reject_h0'])} \\\\"
            )
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{RQ1: average precision of each feature view at the selected window "
        f"for {count_word(len(sources))} non-overlapping samples using the default "
        "timestamp ordering. The table also reports the combined-minus-baseline "
        "difference with its 95\\,\\% confidence interval, the permutation $p$-value, "
        "and the decision.}\n"
        "\\label{tab:res_rq1}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}r l l r r r r c c l@{}}\n\\toprule\n"
        "\\textbf{Sample} & \\textbf{Cohort} & \\textbf{$\\delta$} & \\textbf{Base} & \\textbf{Motifs} & "
        "\\textbf{Comb.} & \\textbf{$\\Delta$AP} & \\textbf{95\\,\\% CI} & "
        "\\textbf{$p$} & \\textbf{Significant?} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_rq1.tex").write_text(table)


def write_rq1_diagnostics(runs: RunSet, out: Path) -> None:
    sources = runs.sources_for("rq1")
    if not sources:
        (out / "results_rq1_representations.tex").write_text(
            pending_table(
                "RQ1 secondary representation comparisons.",
                "tab:res_rq1_representations",
            )
        )
        (out / "results_rq1_incoming_roles.tex").write_text(
            pending_table(
                "RQ1 prespecified all-incoming role diagnostics.",
                "tab:res_rq1_incoming",
            )
        )
        return

    comparisons = pd.concat(
        [
            runs.csv_from(source, "tables/rq1_representation_tests.csv")
            for source in sources
        ],
        ignore_index=True,
    )
    comparisons = comparisons.loc[
        (comparisons["ordering_variant"] == HEADLINE_VARIANT)
        & comparisons["comparison"].isin(
            {"motifs_vs_volume", "combined_vs_baseline_composition_volume"}
        )
    ].sort_values(["replication_index", "cohort", "comparison"])
    rows = [
        f"{int(row['replication_index'])} & {str(row['cohort']).capitalize()} & "
        f"{tex_escape(COMPARISON_LABELS.get(row['comparison'], row['comparison']))} & "
        f"{fnum(row['reference_average_precision'])} & "
        f"{fnum(row['alternative_average_precision'])} & "
        f"{fnum(row['observed_delta_ap'])} & "
        f"[{fnum(row['ci_lower'])}, {fnum(row['ci_upper'])}] & "
        f"{fp(row['p_value_bh'])} & {yes_no(row['reject_h0_bh'])} \\\\"
        for _, row in comparisons.iterrows()
    ]
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Secondary RQ1 representation comparisons at the "
        "development-selected window using the default timestamp ordering. The first comparison "
        "tests fingerprint shape beyond total motif count. The second tests the "
        "combined model against a baseline augmented with motif composition and "
        "volume. P-values use Benjamini--Hochberg correction within the "
        "comparison family for each sample.}\n"
        "\\label{tab:res_rq1_representations}\n\\scriptsize\n"
        "\\begin{tabular}{@{}r l p{3.4cm} r r r c r l@{}}\n\\toprule\n"
        "\\textbf{Sample} & \\textbf{Cohort} & \\textbf{Comparison} & \\textbf{Ref.} & "
        "\\textbf{Alt.} & \\textbf{$\\Delta$AP} & \\textbf{95\\,\\% CI} & "
        "\\textbf{$p_{\\mathrm{BH}}$} & \\textbf{Significant?} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_rq1_representations.tex").write_text(table)

    roles = pd.concat(
        [runs.csv_from(source, "tables/rq1_role_diagnostics.csv") for source in sources],
        ignore_index=True,
    )
    incoming = {"star_pre_iii", "star_mid_iii", "star_post_iii"}
    roles = roles.loc[
        roles["role"].isin(incoming) & (roles["scope"] == "test")
    ].sort_values(
        ["replication_index", "cohort", "role"]
    )
    role_rows = [
        f"{int(row['replication_index'])} & {str(row['cohort']).capitalize()} & "
        f"{tex_escape(row['role'])} & {fnum(row['roc_auc_direct_count'])} & "
        f"{fnum(row['average_precision_direct_count'])} & "
        f"{fnum(row['nonzero_coverage'])} \\\\"
        for _, row in roles.iterrows()
    ]
    role_table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Descriptive held-out diagnostics for the three "
        "prespecified all-incoming star roles. Counts are used directly as scores. "
        "These values are not fitted-model performance.}\n"
        "\\label{tab:res_rq1_incoming}\n\\footnotesize\n"
        "\\begin{tabular}{@{}r l l r r r@{}}\n\\toprule\n"
        "\\textbf{Sample} & \\textbf{Cohort} & \\textbf{Role} & "
        "\\textbf{ROC-AUC} & \\textbf{AP} & \\textbf{Coverage} \\\\\n\\midrule\n"
        + "\n".join(role_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_rq1_incoming_roles.tex").write_text(role_table)


def write_rq1_final_extensions(runs: RunSet, out: Path) -> None:
    """Write triangle and external replication tables when present."""
    latest_hf = runs.latest_source_for("rq1")
    triangle_path = (
        latest_hf / "tables/rq1_triangle_ablation.csv"
        if latest_hf is not None
        else None
    )
    if triangle_path is None or not triangle_path.is_file():
        (out / "results_rq1_triangles.tex").write_text(
            pending_table("RQ1 triangle-role ablation.", "tab:res_rq1_triangles")
        )
    else:
        triangles = pd.read_csv(triangle_path)
        triangles = triangles[
            triangles["ordering_variant"].eq(HEADLINE_VARIANT)
        ].sort_values(["cohort", "comparison"])
        comparison_labels = {
            "baseline_triangles_vs_baseline": "triangles added to baseline",
            "combined_vs_combined_without_triangles": "40 roles versus 32 roles",
        }
        rows = [
            f"{str(row['cohort']).capitalize()} & "
            f"{comparison_labels.get(row['comparison'], row['comparison'])} & "
            f"{fnum(row['reference_average_precision'])} & "
            f"{fnum(row['alternative_average_precision'])} & "
            f"{fnum(row['observed_delta_ap'])} & "
            f"[{fnum(row['ci_lower'])}, {fnum(row['ci_upper'])}] & "
            f"{fp(row['p_value_bh'])} & {yes_no(row['reject_h0_bh'])} \\\\"
            for _, row in triangles.iterrows()
        ]
        table = (
            "\\begin{table}[ht]\n\\centering\n"
            "\\caption{RQ1 triangle-role comparison in the 40-role HuggingFace analysis at the selected "
            "30-day window using the default timestamp ordering. Corrected $p$-values "
            "account for the representation comparisons.}\n"
            "\\label{tab:res_rq1_triangles}\n\\footnotesize\n"
            "\\begin{tabular}{@{}l l r r r c r l@{}}\n\\toprule\n"
            "\\textbf{Cohort} & \\textbf{Comparison} & \\textbf{Ref.} & "
            "\\textbf{Alt.} & \\textbf{$\\Delta$AP} & \\textbf{95\\,\\% CI} & "
            "\\textbf{$p_{\\mathrm{BH}}$} & \\textbf{Significant?} \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )
        (out / "results_rq1_triangles.tex").write_text(table)

    latest_realcats = runs.latest_source_for("rq3")
    external_path = (
        latest_realcats / "tables/realcats_binary_rq1_results.csv"
        if latest_realcats is not None
        else None
    )
    if external_path is None or not external_path.is_file():
        (out / "results_realcats_binary.tex").write_text(
            pending_table(
                "Binary Real-CATS comparison.",
                "tab:res_realcats_binary",
            )
        )
        return

    external = pd.read_csv(external_path)
    canonical = external[
        external["ordering_variant"].eq(HEADLINE_VARIANT)
    ].iloc[0]
    metrics = pd.read_csv(
        latest_realcats / "metrics/realcats_binary_canonical_metrics.csv"
    )
    selected = int(canonical["selected_delta"])
    at = metrics[metrics["delta"].eq(selected)].set_index("arm")
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Binary comparison on the activity-matched Real-CATS sample "
        "using the default timestamp ordering.}\n"
        "\\label{tab:res_realcats_binary}\n\\footnotesize\n"
        "\\begin{tabular}{@{}l r@{}}\n\\toprule\n"
        "\\textbf{Quantity} & \\textbf{Value} \\\\\n\\midrule\n"
        "Addresses & 6,134 \\\\\n"
        f"Selected window & {fdelta(selected)} \\\\\n"
        f"One-standard-error window & {fdelta(canonical['one_se_delta'])} \\\\\n"
        f"Baseline AP & {fnum(at.loc['baseline', 'average_precision'])} \\\\\n"
        f"Motif-only AP & {fnum(at.loc['motifs', 'average_precision'])} \\\\\n"
        f"Combined AP & {fnum(at.loc['combined', 'average_precision'])} \\\\\n"
        f"$\\Delta$AP & {fnum(canonical['observed_delta_ap'])} \\\\\n"
        f"95\\,\\% CI & [{fnum(canonical['ci_lower'])}, {fnum(canonical['ci_upper'])}] \\\\\n"
        f"Permutation $p$-value & {fp(canonical['p_value'])} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_realcats_binary.tex").write_text(table)


def write_rq2(runs: RunSet, out: Path) -> None:
    sources = runs.sources_for("rq2")
    if not sources:
        (out / "results_rq2.tex").write_text(
            pending_table("RQ2: selected window against alternatives.", "tab:res_rq2")
        )
        return
    rows = []
    for source in sources:
        comp = runs.csv_from(source, "tables/rq2_delta_comparisons.csv")
        canon = comp[comp["ordering_variant"] == HEADLINE_VARIANT].drop_duplicates(
            ["cohort", "selected_delta", "one_se_delta"]
        )
        sensitivity = runs.json_from(source, "metrics/hf_ordering_sensitivity.json")
        replication = runs._replication_index(source)
        for _, row in canon.sort_values("cohort").iterrows():
            cohort = str(row["cohort"])
            unique = bool(sensitivity[cohort]["rq2_unique_optimum_in_all_variants"])
            rows.append(
                f"{replication if replication is not None else '--'} & "
                f"{cohort.capitalize()} & {fdelta(row['selected_delta'])} & "
                f"{fdelta(row['one_se_delta'])} & {'yes' if unique else 'no'} \\\\"
            )
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{RQ2: development-CV window selections for {count_word(len(sources))} non-overlapping "
        "samples. The final column states whether the selected maximum was better "
        "than every other tested window after correction under all seven timestamp orderings.}\n"
        "\\label{tab:res_rq2}\n\\small\n"
        "\\begin{tabular}{@{}r l l l l@{}}\n\\toprule\n"
        "\\textbf{Sample} & \\textbf{Cohort} & \\textbf{Maximum $\\delta$} & "
        "\\textbf{1-SE $\\delta$} & \\textbf{Unique maximum} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_rq2.tex").write_text(table)


def write_rq3(runs: RunSet, out: Path) -> None:
    if not ("rq3" in runs.claimed and runs.path_for("rq3", REQUIRED["rq3"][0])):
        (out / "results_rq3_global.tex").write_text(
            pending_table("RQ3: overall category separability.", "tab:res_rq3_global")
        )
        (out / "results_rq3_labels.tex").write_text(
            pending_table(
                "RQ3: per-category fingerprint separability.",
                "tab:res_rq3_labels",
            )
        )
        return
    glob = runs.csv_for("rq3", "tables/rq3_global_results.csv")
    canon = glob[glob["ordering_variant"] == HEADLINE_VARIANT].iloc[0]
    global_table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{RQ3: overall separability of the criminal categories from their "
        "motif fingerprints using the default timestamp ordering.}\n\\label{tab:res_rq3_global}\n\\small\n"
        "\\begin{tabular}{@{}l r@{}}\n\\toprule\n"
        "\\textbf{Quantity} & \\textbf{Value} \\\\\n\\midrule\n"
        f"Global window & {fdelta(canon['global_delta'])} \\\\\n"
        f"Macro average precision & {fnum(canon['macro_average_precision'])} \\\\\n"
        f"Macro F1 & {fnum(canon['macro_f1'])} \\\\\n"
        f"Permutation $p$-value & {fp(canon['permutation_p_value'])} \\\\\n"
        f"Category signal significant? & {yes_no(canon['reject_h0'])} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_rq3_global.tex").write_text(global_table)

    labels = runs.csv_for("rq3", "tables/rq3_label_results.csv")
    labels = labels[labels["ordering_variant"] == HEADLINE_VARIANT]
    realcats = runs.dataset("realcats", "rq3") or {}
    attrition = realcats.get("attrition", {})
    counts = attrition.get("exact_label_counts", {})
    inferential = attrition.get("inferential_labels", [])
    total = sum(int(counts.get(label, 0)) for label in inferential)
    rows = [
        f"{tex_escape(row['label'])} & {int(counts.get(row['label'], 0)):,} & "
        f"{fnum(int(counts.get(row['label'], 0)) / total if total else None)} & "
        f"{fnum(row['average_precision'])} \\\\"
        for _, row in labels.sort_values("average_precision", ascending=False).iterrows()
    ]
    label_table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{RQ3: one-vs-rest average precision for each behaviour category "
        "at the globally selected window using the default timestamp ordering. The address count "
        "and class share are included because average precision depends on class "
        "prevalence.}\n"
        "\\label{tab:res_rq3_labels}\n"
        "\\small\n\\begin{tabular}{@{}l r r r@{}}\n\\toprule\n"
        "\\textbf{Category} & \\textbf{$n$} & \\textbf{Share} & \\textbf{AP} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_rq3_labels.tex").write_text(label_table)


def write_rq4(runs: RunSet, out: Path) -> None:
    if not ("rq4" in runs.claimed and runs.path_for("rq4", REQUIRED["rq4"][0])):
        (out / "results_rq4_global.tex").write_text(
            pending_table(
                "RQ4: category-specific versus shared temporal windows.",
                "tab:res_rq4_global",
            )
        )
        (out / "results_rq4_categories.tex").write_text(
            pending_table(
                "RQ4: per-category temporal-window comparisons.",
                "tab:res_rq4_categories",
            )
        )
        return

    global_results = runs.csv_for("rq4", "tables/rq4_global_results.csv")
    canonical = global_results[
        global_results["ordering_variant"] == HEADLINE_VARIANT
    ].iloc[0]
    global_table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{RQ4: held-out comparison of development-selected category-specific "
        "windows with one shared window using the default timestamp ordering.}\n"
        "\\label{tab:res_rq4_global}\n\\small\n"
        "\\begin{tabular}{@{}l r@{}}\n\\toprule\n"
        "\\textbf{Quantity} & \\textbf{Value} \\\\\n\\midrule\n"
        f"Shared window & {fdelta(canonical['global_delta'])} \\\\\n"
        f"Distinct category windows & {int(canonical['unique_category_deltas'])} \\\\\n"
        f"$\\Delta$ macro AP & {fnum(canonical['observed_delta_macro_ap'])} \\\\\n"
        f"95\\% CI & [{fnum(canonical['ci_lower'])}, {fnum(canonical['ci_upper'])}] \\\\\n"
        f"Bootstrap $p$-value & {fp(canonical['p_value'])} \\\\\n"
        f"Improvement significant? & {yes_no(canonical['reject_h0'])} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_rq4_global.tex").write_text(global_table)

    categories = runs.csv_for("rq4", "tables/rq4_category_results.csv")
    categories = categories[
        categories["ordering_variant"] == HEADLINE_VARIANT
    ].sort_values("label")
    rows = [
        f"{tex_escape(row['label'])} & {fdelta(row['category_delta'])} & "
        f"{fnum(row['observed_delta_ap'])} & "
        f"[{fnum(row['ci_lower'])}, {fnum(row['ci_upper'])}] & "
        f"{fp(row['p_value_bh'])} & {yes_no(row['reject_h0'])} \\\\"
        for _, row in categories.iterrows()
    ]
    category_table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{RQ4: category-specific windows compared with the shared window "
        "using the default timestamp ordering. P-values use Benjamini--Hochberg correction.}\n"
        "\\label{tab:res_rq4_categories}\n\\footnotesize\n"
        "\\begin{tabular}{@{}l l r l r l@{}}\n\\toprule\n"
        "\\textbf{Category} & \\textbf{$\\delta$} & \\textbf{$\\Delta$AP} & "
        "\\textbf{95\\% CI} & \\textbf{$p_{\\mathrm{BH}}$} & "
        "\\textbf{Significant?} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_rq4_categories.tex").write_text(category_table)


def write_validation(runs: RunSet, out: Path) -> None:
    values = {
        "ResSyntheticOutcome": "pending",
        "ResSyntheticRoles": "0",
        "ResSyntheticBoundary": "pending",
        "ResBLTEDelta": "--",
        "ResBLTEBaselineAP": "--",
        "ResBLTEMotifsAP": "--",
        "ResBLTECombinedAP": "--",
    }
    synth = next(
        (
            run_dir / "metrics/synthetic_validation.json"
            for run_dir in runs.executed
            if (run_dir / "metrics/synthetic_validation.json").is_file()
        ),
        None,
    )
    if synth:
        data = json.loads(synth.read_text())
        values["ResSyntheticOutcome"] = "passed" if data.get("passed") else "failed"
        values["ResSyntheticRoles"] = str(len(data.get("role_checks", [])))
        values["ResSyntheticBoundary"] = (
            "passed" if data.get("delta_boundary", {}).get("passed") else "failed"
        )
    blte = runs.path_for("blte", "metrics/blte_results.json")
    if blte:
        data = json.loads(blte.read_text())
        m = data.get("metrics", {})
        values["ResBLTEDelta"] = fdelta(data.get("delta"))
        values["ResBLTEBaselineAP"] = fnum(
            m.get("baseline", {}).get("average_precision")
        )
        values["ResBLTEMotifsAP"] = fnum(
            m.get("motifs", {}).get("average_precision")
        )
        values["ResBLTECombinedAP"] = fnum(
            m.get("combined", {}).get("average_precision")
        )
    header = "% Auto-generated validation macros. Do not edit by hand."
    commands = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()]
    (out / "results_validation.tex").write_text(
        header + "\n" + "\n".join(commands) + "\n"
    )


def write_hypothesis(runs: RunSet, out: Path) -> None:
    rows = []
    latest_hf = runs.latest_source_for("rq1")
    if latest_hf is not None:
        sens = runs.json_from(latest_hf, "metrics/hf_ordering_sensitivity.json")
        positive = all(
            all(sens[cohort].get("effect_directions_positive", []))
            for cohort in ("controlled", "prevalence")
        )
        rows.append(
            "RQ1 & small, inconsistent AP increase & "
            "no reliable improvement over the baseline & "
            f"positive direction in all seven orderings: {'yes' if positive else 'no'} \\\\"
        )

        selections = {
            delta
            for cohort in ("controlled", "prevalence")
            for delta in sens[cohort].get("selected_deltas", [])
        }
        selected_text = (
            fdelta(next(iter(selections)))
            if len(selections) == 1
            else "multiple windows"
        )
        unique = all(
            sens[cohort].get("rq2_unique_optimum_in_all_variants", False)
            for cohort in ("controlled", "prevalence")
        )
        rows.append(
            "RQ2 & AP generally increases across the tested window range & "
            f"{selected_text} selected in the 40-role analysis, "
            f"{'with' if unique else 'without'} a unique optimum & "
            f"same selected window in all seven orderings: {'yes' if len(selections) == 1 else 'no'} \\\\"
        )

    latest_realcats = runs.latest_source_for("rq3")
    rq3_sensitivity = (
        latest_realcats / "metrics/realcats_ordering_sensitivity.json"
        if latest_realcats is not None
        else None
    )
    if rq3_sensitivity is not None and rq3_sensitivity.is_file():
        rsens = json.loads(rq3_sensitivity.read_text())
        rows.append(
            "RQ3 & weak but detectable fingerprint differences & "
            f"category information {'found' if rsens.get('ordering_robust_global_separability') else 'not established'}, "
            "with no conventional-feature comparison & "
            f"category signal in all seven orderings: {'yes' if rsens.get('ordering_robust_global_separability') else 'no'} \\\\"
        )
    rq4_sensitivity = (
        latest_realcats / "metrics/realcats_rq4_ordering_sensitivity.json"
        if latest_realcats is not None
        else None
    )
    if rq4_sensitivity is not None and rq4_sensitivity.is_file():
        rq4 = json.loads(rq4_sensitivity.read_text())
        robust = rq4.get("ordering_robust_category_specific_windows")
        rows.append(
            "RQ4 & category-specific versus shared windows & "
            f"improvement on held-out addresses: {'yes' if robust else 'no'} & "
            "no significant improvement in any ordering. Selected windows vary \\\\"
        )
    if not rows:
        (out / "results_hypothesis.tex").write_text(
            pending_table("Summary of hypothesis decisions.", "tab:res_hypothesis")
        )
        return
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Summary of the answers to the four research questions.}\n"
        "\\label{tab:res_hypothesis}\n\\footnotesize\n"
        "\\begin{tabular}{@{}l p{3.6cm} p{4.0cm} p{4.0cm}@{}}\n\\toprule\n"
        "\\textbf{Question} & \\textbf{Finding} & \\textbf{Answer} & "
        "\\textbf{Other orderings} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_hypothesis.tex").write_text(table)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: results_to_latex.py OUT_DIR RUN_DIR [RUN_DIR ...]")
    out = Path(sys.argv[1])
    run_dirs = [Path(p) for p in sys.argv[2:]]
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"Not a directory: {run_dir}")
    out.mkdir(parents=True, exist_ok=True)

    runs = RunSet(run_dirs)
    runs.check_completeness()

    write_provenance(runs, out)
    write_figures(runs, out)
    write_macros(runs, out)
    write_cohorts(runs, out)
    write_rq1(runs, out)
    write_rq1_diagnostics(runs, out)
    write_rq1_final_extensions(runs, out)
    write_rq2(runs, out)
    write_rq3(runs, out)
    write_rq4(runs, out)
    write_validation(runs, out)
    write_hypothesis(runs, out)

    status = "citable" if runs.citable else "NON-CITABLE"
    print(f"Wrote LaTeX to {out} from {len(run_dirs)} run(s): {', '.join(runs.run_ids)}")
    print(f"Claimed research questions: {sorted(runs.claimed) or 'none'}")
    print(f"Provenance: {status}")


if __name__ == "__main__":
    main()
