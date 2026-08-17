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


def tex_escape(text: object) -> str:
    out = str(text)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                 ("_", "\\_"), ("#", "\\#"), ("$", "\\$")):
        out = out.replace(a, b)
    return out


def reject_cell(flag) -> str:
    return "reject" if bool(flag) else "not rejected"


def support_cell(flag) -> str:
    return "supported" if bool(flag) else "not supported"


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
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    hf_owner_rq = "rq1" if "rq1" in runs.claimed else "rq2"
    hf = runs.dataset("huggingface", hf_owner_rq)
    if hf:
        attr = hf.get("attrition", {})
        cmd("ResCohortLabelled", f"{attr.get('labelled_targets', 0):,}")
        cmd("ResCohortInactive", f"{attr.get('inactive_targets', 0):,}")
        cmd("ResCohortEligible", f"{attr.get('eligible_active_targets', 0):,}")
        cmd("ResPrevalenceN", f"{attr.get('prevalence_targets', 0):,}")
        cmd("ResControlledN", f"{attr.get('controlled_targets', 0):,}")
        extraction = hf.get("edge_extraction", {})
        cmd("ResHFSourceEdges", f"{extraction.get('source_edges', 0):,}")
        cmd("ResHFSelectedEdges", f"{extraction.get('selected_edges', 0):,}")
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

    if "rq1" in runs.claimed and runs.path_for("rq1", REQUIRED["rq1"][0]):
        hyp = runs.csv_for("rq1", "tables/rq1_hypothesis_tests.csv")
        canon = hyp[hyp["ordering_variant"] == HEADLINE_VARIANT]
        for _, row in canon.iterrows():
            key = str(row["cohort"]).capitalize()
            cmd(f"ResRQ1{key}DeltaAP", fnum(row["delta_average_precision"]))
            cmd(f"ResRQ1{key}Reject", reject_cell(row["reject_h0"]))
    else:
        for cohort in ("Controlled", "Prevalence"):
            cmd(f"ResRQ1{cohort}DeltaAP", "--")
            cmd(f"ResRQ1{cohort}Reject", "pending")
    sens_rq = "rq1" if "rq1" in runs.claimed else "rq2"
    sens = runs.path_for(sens_rq, "metrics/hf_ordering_sensitivity.json")
    if sens:
        general = json.loads(sens.read_text()).get("general_signal")
        cmd("ResHFGeneralSignal", "yes" if general else "no")
    else:
        cmd("ResHFGeneralSignal", "pending")
    if "rq3" in runs.claimed and runs.path_for("rq3", REQUIRED["rq3"][0]):
        glob = runs.csv_for("rq3", "tables/rq3_global_results.csv")
        canon = glob[glob["ordering_variant"] == HEADLINE_VARIANT]
        if len(canon):
            cmd("ResRQ3MacroAP", fnum(canon.iloc[0]["macro_average_precision"]))
            cmd("ResRQ3PermP", fp(canon.iloc[0]["permutation_p_value"]))
        else:
            cmd("ResRQ3MacroAP", "--")
            cmd("ResRQ3PermP", "--")
    else:
        cmd("ResRQ3MacroAP", "--")
        cmd("ResRQ3PermP", "--")
    if "rq4" in runs.claimed and runs.path_for("rq4", REQUIRED["rq4"][0]):
        glob = runs.csv_for("rq4", "tables/rq4_global_results.csv")
        canon = glob[glob["ordering_variant"] == HEADLINE_VARIANT]
        if len(canon):
            cmd(
                "ResRQ4DeltaMacroAP",
                fnum(canon.iloc[0]["observed_delta_macro_ap"]),
            )
            cmd("ResRQ4P", fp(canon.iloc[0]["p_value"]))
        else:
            cmd("ResRQ4DeltaMacroAP", "--")
            cmd("ResRQ4P", "--")
    else:
        cmd("ResRQ4DeltaMacroAP", "--")
        cmd("ResRQ4P", "--")

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
        replication = runs._replication_index(source)
        selected = int(attr.get("prevalence_targets", 0)) + int(
            attr.get("controlled_targets", 0)
        )
        rows.append(
            f"{replication if replication is not None else '--'} & "
            f"{int(attr.get('available_after_prior_run_exclusions', 0)):,} & "
            f"{int(attr.get('prevalence_targets', 0)):,} & "
            f"{int(attr.get('controlled_targets', 0)):,} & {selected:,} & "
            f"{int(extraction.get('selected_edges', 0)):,} \\\\"
        )
    first_hf = runs.protocols[sources[0]]["datasets"]["huggingface"]
    first_attr = first_hf.get("attrition", {})
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{The two disjoint HuggingFace confirmation samples. The source "
        f"contains {int(first_attr.get('labelled_targets', 0)):,} labelled targets; "
        f"{int(first_attr.get('eligible_active_targets', 0)):,} pass the common "
        "activity and upper-tail rules before prior-run exclusions.}\n"
        "\\label{tab:res_cohorts}\n\\footnotesize\n"
        "\\begin{tabular}{@{}r r r r r r@{}}\n\\toprule\n"
        "\\textbf{Rep.} & \\textbf{Available} & \\textbf{Prevalence} & "
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
                f"{fp(row['p_value'])} & {reject_cell(row['reject_h0'])} \\\\"
            )
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{RQ1: average precision of each feature view at the selected window "
        "for both disjoint confirmations (canonical ordering), the combined-minus-baseline "
        "difference with its 95\\,\\% confidence interval, the permutation $p$-value, "
        "and the decision.}\n"
        "\\label{tab:res_rq1}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}r l l r r r r c c l@{}}\n\\toprule\n"
        "\\textbf{Rep.} & \\textbf{Cohort} & \\textbf{$\\delta$} & \\textbf{Base} & \\textbf{Motifs} & "
        "\\textbf{Comb.} & \\textbf{$\\Delta$AP} & \\textbf{95\\,\\% CI} & "
        "\\textbf{$p$} & \\textbf{Decision} \\\\\n\\midrule\n"
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
        f"{fp(row['p_value_bh'])} & {reject_cell(row['reject_h0_bh'])} \\\\"
        for _, row in comparisons.iterrows()
    ]
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Prespecified secondary RQ1 representation comparisons at the "
        "development-selected window under canonical ordering. The first contrast "
        "tests fingerprint shape beyond total motif count; the second tests the "
        "combined model against a baseline augmented with motif composition and "
        "volume. P-values use Benjamini--Hochberg correction across all eight "
        "pre-specified contrasts per cohort.}\n"
        "\\label{tab:res_rq1_representations}\n\\scriptsize\n"
        "\\begin{tabular}{@{}r l p{3.4cm} r r r c r l@{}}\n\\toprule\n"
        "\\textbf{Rep.} & \\textbf{Cohort} & \\textbf{Comparison} & \\textbf{Ref.} & "
        "\\textbf{Alt.} & \\textbf{$\\Delta$AP} & \\textbf{95\\,\\% CI} & "
        "\\textbf{$p_{\\mathrm{BH}}$} & \\textbf{Decision} \\\\\n\\midrule\n"
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
        "prespecified all-incoming star roles. Counts are used directly as scores; "
        "these values are not fitted-model performance.}\n"
        "\\label{tab:res_rq1_incoming}\n\\footnotesize\n"
        "\\begin{tabular}{@{}r l l r r r@{}}\n\\toprule\n"
        "\\textbf{Rep.} & \\textbf{Cohort} & \\textbf{Role} & "
        "\\textbf{ROC-AUC} & \\textbf{AP} & \\textbf{Coverage} \\\\\n\\midrule\n"
        + "\n".join(role_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out / "results_rq1_incoming_roles.tex").write_text(role_table)


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
        "\\caption{RQ2: development-CV window selections for both disjoint "
        "confirmations. The final column states whether the selected maximum was "
        "uniquely superior after corrected comparisons under all seven orderings.}\n"
        "\\label{tab:res_rq2}\n\\small\n"
        "\\begin{tabular}{@{}r l l l l@{}}\n\\toprule\n"
        "\\textbf{Rep.} & \\textbf{Cohort} & \\textbf{Maximum $\\delta$} & "
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
        "motif fingerprints (canonical ordering).}\n\\label{tab:res_rq3_global}\n\\small\n"
        "\\begin{tabular}{@{}l r@{}}\n\\toprule\n"
        "\\textbf{Quantity} & \\textbf{Value} \\\\\n\\midrule\n"
        f"Global window & {fdelta(canon['global_delta'])} \\\\\n"
        f"Macro average precision & {fnum(canon['macro_average_precision'])} \\\\\n"
        f"Macro F1 & {fnum(canon['macro_f1'])} \\\\\n"
        f"Permutation $p$-value & {fp(canon['permutation_p_value'])} \\\\\n"
        f"Decision & {reject_cell(canon['reject_h0'])} \\\\\n"
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
        "at the globally selected window (canonical ordering). The address count "
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
                "RQ4: category-specific versus global temporal windows.",
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
        "windows with one global window (canonical ordering).}\n"
        "\\label{tab:res_rq4_global}\n\\small\n"
        "\\begin{tabular}{@{}l r@{}}\n\\toprule\n"
        "\\textbf{Quantity} & \\textbf{Value} \\\\\n\\midrule\n"
        f"Global window & {fdelta(canonical['global_delta'])} \\\\\n"
        f"Distinct category windows & {int(canonical['unique_category_deltas'])} \\\\\n"
        f"$\\Delta$ macro AP & {fnum(canonical['observed_delta_macro_ap'])} \\\\\n"
        f"95\\% CI & [{fnum(canonical['ci_lower'])}, {fnum(canonical['ci_upper'])}] \\\\\n"
        f"Bootstrap $p$-value & {fp(canonical['p_value'])} \\\\\n"
        f"Decision & {reject_cell(canonical['reject_h0'])} \\\\\n"
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
        f"{fp(row['p_value_bh'])} & {reject_cell(row['reject_h0'])} \\\\"
        for _, row in categories.iterrows()
    ]
    category_table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{RQ4: category-specific windows compared with the global window "
        "(canonical ordering). P-values use Benjamini--Hochberg correction.}\n"
        "\\label{tab:res_rq4_categories}\n\\footnotesize\n"
        "\\begin{tabular}{@{}l l r l r l@{}}\n\\toprule\n"
        "\\textbf{Category} & \\textbf{$\\delta$} & \\textbf{$\\Delta$AP} & "
        "\\textbf{95\\% CI} & \\textbf{$p_{\\mathrm{BH}}$} & "
        "\\textbf{Decision} \\\\\n\\midrule\n"
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
    hf_rq = "rq1" if "rq1" in runs.claimed else "rq2"
    for source in runs.sources_for(hf_rq):
        sensitivity_path = source / "metrics/hf_ordering_sensitivity.json"
        if not sensitivity_path.is_file():
            continue
        sens = json.loads(sensitivity_path.read_text())
        replication = runs._replication_index(source)
        rep_label = replication if replication is not None else "--"
        for cohort in [c for c in sens if c != "general_signal"]:
            detail = sens[cohort]
            rows.append(
                f"RQ1 (rep. {rep_label}, {cohort}) & motifs add signal & "
                f"{support_cell(detail.get('rq1_rejected_in_all_variants'))} & "
                f"robust: {'yes' if detail.get('ordering_robust') else 'no'} \\\\"
            )
            rows.append(
                f"RQ2 (rep. {rep_label}, {cohort}) & a single best window & "
                f"{support_cell(detail.get('rq2_unique_optimum_in_all_variants'))} & "
                "-- \\\\"
            )
    rq3_sensitivity = runs.path_for("rq3", "metrics/realcats_ordering_sensitivity.json")
    if rq3_sensitivity:
        rsens = json.loads(rq3_sensitivity.read_text())
        rows.append(
            "RQ3 & categories are separable & "
            f"{support_cell(rsens.get('ordering_robust_global_separability'))} & "
            f"robust: {'yes' if rsens.get('ordering_robust_global_separability') else 'no'} \\\\"
        )
    rq4_sensitivity = runs.path_for(
        "rq4",
        "metrics/realcats_rq4_ordering_sensitivity.json",
    )
    if rq4_sensitivity:
        rq4 = json.loads(rq4_sensitivity.read_text())
        robust = rq4.get("ordering_robust_category_specific_windows")
        rows.append(
            "RQ4 & category-specific windows outperform one global window & "
            f"{support_cell(robust)} & robust: {'yes' if robust else 'no'} \\\\"
        )
    if not rows:
        (out / "results_hypothesis.tex").write_text(
            pending_table("Summary of hypothesis decisions.", "tab:res_hypothesis")
        )
        return
    table = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Summary of the hypothesis decisions across all seven orderings.}\n"
        "\\label{tab:res_hypothesis}\n\\small\n"
        "\\begin{tabular}{@{}l l l l@{}}\n\\toprule\n"
        "\\textbf{Question} & \\textbf{Tested claim} & \\textbf{Conclusion} & "
        "\\textbf{Ordering} \\\\\n\\midrule\n"
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
