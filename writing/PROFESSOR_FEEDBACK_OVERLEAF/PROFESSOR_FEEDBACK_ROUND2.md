# Additional professor feedback: chapter overview, Related Work, and citations

These changes are already applied to `thesis.tex` and `thesis.bib`.
Use `PROFESSOR_FEEDBACK_ROUND2.zip` if you have already transferred the first
feedback package to Overleaf. No figure uploads are needed for this round.
Earlier extracted copies of the first archive are not updated automatically.

## What changed

- The Introduction now ends with one complete paragraph outlining Chapters 2
  through 7 and the engineering appendix. All seven numbered main chapters and
  the engineering appendix have labels.
- Related Work now contains 21 distinct references, comprising 20 research
  papers and the dataset release. The chapter is organised around graph-based
  detection, time-aware learning, datasets, temporal motifs, and the position
  of this thesis. Its draft notes have been removed.
- Narrative citations use `\citet{key}`, producing author names as part of
  the sentence. Supporting citations use `\citep{key}`. Bare `\cite{}` and
  manually assembled author-plus-`\citeyear{}` citations have been removed
  throughout the thesis. The existing bibliography style is unchanged.
- Earlier work already applies temporal motif features to Ethereum phishing.
  The Introduction, Related Work, theoretical connection, and Discussion now
  acknowledge this and describe the contribution as the present evaluation.
- Three references were added. Four existing entries have corrected metadata.
  The existing keys are retained, even where the first author's name or the
  version year differs from the key.

## Text replacements

Copy the complete content of each file below over the corresponding existing
passage. These are replacements, not additional text to append.
The numbered comment markers may stay in Overleaf.

| Snippet | Replace this passage |
|---|---|
| [PF01](professor_feedback/PF01.tex) | Entire Introduction, stopping before Related Work |
| [PF33](professor_feedback/PF33.tex) | Entire Related Work chapter, including its old draft notes and old PF02/PF03 blocks, stopping before Theoretical Foundations |
| [PF34](professor_feedback/PF34.tex) | Theoretical Foundations chapter heading and its one-sentence introduction |
| [PF04](professor_feedback/PF04.tex) | Blockchain and Ethereum fundamentals section |
| [PF35](professor_feedback/PF35.tex) | Graph-theory foundations section, stopping before Temporal networks |
| [PF05](professor_feedback/PF05.tex) | Temporal networks section |
| [PF06](professor_feedback/PF06.tex) | Temporal motifs section, including Counting algorithm and Motif fingerprint, stopping before Blockchain Relevance |
| [PF36](professor_feedback/PF36.tex) | Only the Blockchain Relevance paragraph beginning “For this thesis, this has a direct methodological consequence” |
| [PF11](professor_feedback/PF11.tex) | Temporal motif extraction section in Method |
| [PF13](professor_feedback/PF13.tex) | Evaluation section in Method |
| [PF14](professor_feedback/PF14.tex) | Implementation details heading, introduction, and Language and libraries subsection, stopping before Project structure and entry points |
| [PF25](professor_feedback/PF25.tex) | Detection value of temporal motifs subsection in Discussion |
| [PF37](professor_feedback/PF37.tex) | Only the Conclusion & Outlook chapter heading, not its sections |

PF33 supersedes the old PF02 and PF03 snippets. Do not paste them again after
PF33. The new small snippets PF34 and PF37 add chapter labels without replacing
the following sections. PF13 also restores the first-use definition of the F1
score, which previously appeared in the old Related Work text.

## Bibliography changes

1. Append [round2_new_references.bib](professor_feedback/round2_new_references.bib)
   to your bibliography. It contains `weber2019bitcoin`, `wu2022mixing`, and
   `wang2023temporal`.
2. Use [round2_corrected_references.bib](professor_feedback/round2_corrected_references.bib)
   to replace the four entries with matching keys in your bibliography.
   Do not append duplicate entries.

| Existing key | Correction |
|---|---|
| `McDonnell2014` | Ömer Nebil Yaveroğlu and Brett A. Schmerl's names |
| `wu2020temporal` | Correct first author Dan Lin, full author list, and DOI |
| `lee2020ethereum` | Page range 155 to 166 |
| `mo2022motifgcn` | Correct author list beginning with Song Li, and the title and year of the revised 2023 preprint |

The older key names remain valid in the LaTeX source. The rendered author names
and years come from the corrected bibliography entries.

## Source and validation notes

[ROUND2_SOURCE_CHECKS.md](professor_feedback/ROUND2_SOURCE_CHECKS.md) records
the sources used for the expanded chapter and the corrections.
The description of Wang and colleagues' paper is deliberately limited to its
documented task and representation. No performance numbers or detailed
experimental settings were added for that paper.

The citation check covers commands throughout the thesis. It is not a new
line-by-line factual audit of every claim in the unchanged chapters.
Unrelated draft placeholders, including the Abstract and “blabla” in the
temporal-networks section, remain and need attention before submission.

The source contains no empty references or missing citation keys. A local
LaTeX and BibTeX check uses the same temporary substitute class as the first
round, since the Overleaf class, title page, existing appendix, and original
motif-catalogue image are not present locally. The actual Overleaf document
still needs to be recompiled there after the replacements.

The revised text keeps the no-em-dash and no-prose-semicolon convention.
Required TikZ syntax and citation punctuation generated by the bibliography
style are left unchanged.
