# Professor feedback: Overleaf replacements

The current thesis has been revised in place. Every changed passage in
`thesis.tex` is inside a numbered comment block:

```tex
% BEGIN PROFESSOR FEEDBACK PF01: Introduction
...
% END PROFESSOR FEEDBACK PF01
```

These comments do not appear in the compiled document. There are 35 active blocks.
Each block also has its own ready-to-copy file in `professor_feedback/`.
The snippets are replacements, not additional sections to append.

This guide and the cumulative archive include both feedback rounds. If you
already transferred the first round, use
[PROFESSOR_FEEDBACK_ROUND2.md](PROFESSOR_FEEDBACK_ROUND2.md) and
`PROFESSOR_FEEDBACK_ROUND2.zip` for only the additional changes.
PF33 replaces the whole Related Work chapter and supersedes PF02 and PF03.
The old PF02/PF03 files are retained locally as earlier copies, but are not
included in the current archive and must not be pasted after PF33.

## Transfer to Overleaf

1. Use the table below to replace the corresponding passages in your Overleaf
   `thesis.tex`. Copy the complete snippet, including its section heading when
   present. The comment markers can stay.
2. For PF15, replace both the existing **Project structure and entry points**
   subsection and the following **Engineering challenges** subsection, stopping
   immediately before **Reproducibility**. Engineering challenges now appears
   in PF32 as an appendix chapter.
3. PF32 replaces the existing `\input{text/appendix}` line with the appendix
   boundary and the new engineering chapter. Keep the contents of your existing
   `text/appendix.tex`. If an `\appendix` command already immediately precedes
   this location, use it once.
4. Add the entry in
   [hellinger_reference.bib](professor_feedback/hellinger_reference.bib)
   to your existing `thesis.bib`. Do not replace the whole bibliography.
   Also append the three entries in
   [round2_new_references.bib](professor_feedback/round2_new_references.bib),
   and replace the four matching entries using
   [round2_corrected_references.bib](professor_feedback/round2_corrected_references.bib).
   Do not create duplicate bibliography keys.
5. Replace `generated/results_hypothesis.tex` with the revised file.
6. Upload the 11 PDFs listed below into Overleaf's `figures/` folder, replacing
   the files with the same names. Text changes alone do not enlarge the fonts
   inside a PDF.
7. Recompile. The moved terminology table and engineering appendix, and the
   additional RQ3 figure, will update their numbering automatically.

The archive `PROFESSOR_FEEDBACK_OVERLEAF.zip` contains this guide, the 35
active snippets, the bibliography additions and corrections, the revised summary table, and the
11 replacement PDFs. It does not require replacing the complete thesis file.

## Text blocks

| Block | Passage | Replacement boundary |
|---|---|---|
| [PF01](professor_feedback/PF01.tex) | Introduction | Whole chapter, before Related Work |
| [PF33](professor_feedback/PF33.tex) | Expanded Related Work with 21 references | Whole chapter, replacing old PF02/PF03 and the other Related Work text |
| [PF34](professor_feedback/PF34.tex) | Theoretical Foundations chapter label | Chapter heading and one-sentence introduction |
| [PF04](professor_feedback/PF04.tex) | Blockchain and Ethereum fundamentals | Whole section |
| [PF35](professor_feedback/PF35.tex) | Graph-theory foundations citation corrections | Whole section |
| [PF05](professor_feedback/PF05.tex) | Temporal networks | Whole section |
| [PF06](professor_feedback/PF06.tex) | Temporal motifs | Whole section, including counting and fingerprint subsections, before Blockchain Relevance |
| [PF36](professor_feedback/PF36.tex) | Relation to earlier temporal modelling | Only the paragraph beginning “For this thesis, this has a direct methodological consequence” |
| [PF07](professor_feedback/PF07.tex) | Types of Coordinated Malicious Behaviour on Ethereum | Whole subsection with its figures and table, before Summary |
| [PF08](professor_feedback/PF08.tex) | Method: System overview | Whole section |
| [PF09](professor_feedback/PF09.tex) | Method: Data acquisition | Whole section, before Temporal graph construction |
| [PF10](professor_feedback/PF10.tex) | Temporal graph construction | Whole section |
| [PF11](professor_feedback/PF11.tex) | Temporal motif extraction | Whole section, before Detection methods |
| [PF12](professor_feedback/PF12.tex) | Detection methods | Whole section |
| [PF13](professor_feedback/PF13.tex) | Evaluation | Whole section |
| [PF14](professor_feedback/PF14.tex) | Implementation details: introduction and Language and libraries | From Implementation details through the end of Language and libraries |
| [PF15](professor_feedback/PF15.tex) | Project structure and entry points, including removal of Engineering challenges | Stop before Reproducibility |
| [PF16](professor_feedback/PF16.tex) | Reproducibility | Whole subsection |
| [PF17](professor_feedback/PF17.tex) | Result artifacts | Whole subsection, before Results |
| [PF18](professor_feedback/PF18.tex) | Results introduction | Chapter heading and introduction, before Evaluation Samples |
| [PF19](professor_feedback/PF19.tex) | RQ1 results | Whole section, including the revised heading |
| [PF20](professor_feedback/PF20.tex) | RQ2 results | Whole section |
| [PF21](professor_feedback/PF21.tex) | RQ3 results | Whole section, including both separate figures |
| [PF22](professor_feedback/PF22.tex) | RQ4 results | Whole section |
| [PF23](professor_feedback/PF23.tex) | Additional Checks | Whole section |
| [PF24](professor_feedback/PF24.tex) | Discussion introduction | Chapter heading and introduction, before Interpretation of the Main Findings |
| [PF25](professor_feedback/PF25.tex) | Detection value of temporal motifs | Whole subsection |
| [PF26](professor_feedback/PF26.tex) | Choice of the temporal window | Whole subsection |
| [PF27](professor_feedback/PF27.tex) | Differences between behaviour categories | Whole subsection |
| [PF28](professor_feedback/PF28.tex) | Limitations heading and new network-significance paragraph | Replace only the Limitations heading with this block, leaving Labels and unit of analysis and the later paragraphs in place |
| [PF29](professor_feedback/PF29.tex) | Cohort composition limitation | Only this paragraph, before Time representation |
| [PF37](professor_feedback/PF37.tex) | Conclusion & Outlook chapter label | Only the chapter heading |
| [PF30](professor_feedback/PF30.tex) | Conclusion | Whole section, before Outlook |
| [PF31](professor_feedback/PF31.tex) | Outlook: temporal ordering and a network null model | Only the paragraph beginning “A further experiment should separate temporal order” |
| [PF32](professor_feedback/PF32.tex) | Appendix boundary and Engineering challenges | Replace the existing appendix input line as described above |

PF06 moves `tab:motif_terminology` into the theoretical foundations and corrects
the fingerprint definition. PF11 removes the old copy of that table.
Both replacements are needed to avoid duplicate labels.

## Replacement figures

| File in `figures/` | Change |
|---|---|
| `05-method-steps.pdf` | Larger grey-box text and a compact two-column layout |
| `01-pipeline.pdf` | Larger grey-box text, readable module names, and a vertical layout |
| `rq1_ablation_controlled_rep3.pdf` | Larger axis, category, and value labels |
| `rq1_ablation_prevalence_rep3.pdf` | Larger axis, category, and value labels |
| `rq2_delta_performance_controlled_rep3.pdf` | Larger labels and legend, with the original bootstrap procedure retained |
| `rq2_delta_performance_prevalence_rep3.pdf` | Larger labels and legend, with the original bootstrap procedure retained |
| `rq2_motif_coverage_controlled_rep3.pdf` | Larger labels and distinct line styles |
| `rq2_motif_coverage_prevalence_rep3.pdf` | Larger labels and distinct line styles |
| `rq3_fingerprint_heatmap.pdf` | The same 40 selected components in two panels of 20, with larger labels |
| `rq3_confusion_matrix.pdf` | Larger labels and cell counts, now placed at full width in its own figure |
| `rq4_label_delta_curves.pdf` | Larger labels and an external legend, now placed at full width |

The diagram body text is 11.5 points at its native size, with 12.5-point box
headings. Plot labels are generally 11.5 to 12 points at their native size.
The smaller RQ1 and RQ2 canvases also avoid the heavy reduction of text that
occurred when large original plots were placed side by side.

The pipeline diagrams now show nine windows rather than the outdated seven.
The method text specifies their range from five minutes to 30 days.

## Scientific changes and source checks

- Abbreviations are expanded at their first use in the thesis body.
- The chapter overview is complete and all numbered chapters have labels.
- Related Work now cites 21 distinct works, including a direct predecessor on
  temporal-motif features for Ethereum phishing detection. The contribution
  statements acknowledge this earlier work. See the
  [source checks](professor_feedback/ROUND2_SOURCE_CHECKS.md).
- Narrative citations use `\citet{}` and parenthetical citations use `\citep{}`
  throughout. The bibliography style remains unchanged.
- The cited [Real-CATS paper](https://arxiv.org/html/2501.15553v1) expands its
  name as “Real-world dataset of Cryptocurrency Addresses with Transaction
  profileS”. This published expansion is used instead of “Real-world Criminal
  Account Transaction Set”, which was in the draft and feedback.
- Synthetic fixtures are now called validation templates. Their known
  directions, timestamps, three transfers, and two- or three-node structures
  are explained. Their role is separate from real-world evaluation data.
- The text distinguishes global candidate patterns, a focal node, its local
  role, structural classes, and the complete fingerprint. Forty refers to
  local role counts. Real-CATS uses 32 roles in each of five layers.
- Counting correctness, predictive significance, and network overrepresentation
  are distinguished. The completed experiment has no random-network null model.
  The revision states that limitation explicitly rather than claiming that the
  counted candidates are statistically established network motifs.
- There is no separate “rapid” threshold. The permitted span is the selected
  delta, between five minutes and 30 days.
- Equal-timestamp ordering can change the assigned candidate pattern.
  Only the order between distinct timestamps is preserved.
- The Hellinger transform now cites
  [Legendre and Gallagher (2001)](https://doi.org/10.1007/s004420100716).
- RQ1 tests the fingerprint's added predictive value over 17 conventional
  features. It does not isolate the contribution of temporal ordering.
- RQ2 asks how the window affects usefulness. The text does not claim a unique
  optimum within or beyond the tested grid.
- RQ3 tests category separability with fingerprints alone. It does not claim
  superiority over conventional behavioural features for category classification.
- Campaign groups are defined from the three description-prefix rules in the
  implementation. The permutation swaps model scores within a group. The
  bootstrap samples whole groups within each class and allows unequal group
  sizes, so address counts and class proportions can vary across resamples.
- Explanations of the results remain in the Discussion, with the Results
  focused on measurements and statistical comparisons.

Regenerating the plots exposed an existing coverage mismatch. The saved
`thesis_final_v8_hf` table gives the following values for the canonical
ordering, all addresses, and all 40 retained roles:

| Cohort | Five-minute coverage | 30-day coverage | Zero fingerprint at 30 days |
|---|---:|---:|---:|
| Controlled | 12.2% | 77.7% | 22.3% |
| Prevalence | 11.1% | 75.9% | 24.1% |

The Results, Discussion, and Conclusion now use these values. Predictive scores
and hypothesis decisions were not changed. The category count representation
is also correctly described as log-transformed counts, matching the code.

## Verification and remaining template limitations

The thesis was read in full before editing. All changed thesis lines are inside
the 35 active marked blocks. A punctuation check found no prose semicolons or em dashes.
Semicolons required by TikZ and the LaTeX math spacing command `\;` remain.

The changed exporter passes its six existing tests. The new figures are built
from the saved final metrics, counts, and predictions. Models were not retrained
and the locked experiment directories were not edited. The plotted values and
component selection are recorded in `figures/professor_feedback_sources.json`
in the repository.

A local layout check uses a temporary book class with a 14.5 cm text width.
The original `thesisclass.cls`, title page, existing appendix file, and
`motifs_paranjape` image are not present in this repository, so the exact
Overleaf document cannot be compiled locally. The temporary check substitutes
those missing elements without changing the submitted source. The latest local
check resolves all chapter references and citations. The Introduction's empty
reference and Related Work draft notes are now fixed. Unrelated draft text,
including the Abstract placeholder and “blabla” in Temporal networks, remains
outside this feedback revision and still needs attention before submission.

To reproduce the figure exports in this repository after running the normal
table exporter:

```sh
venv/bin/python scripts/render_thesis_feedback_figures.py
```

The rendering script follows the PDF and data visualization skill guidance by
checking the output visually and sizing the type for its placement in the thesis.
