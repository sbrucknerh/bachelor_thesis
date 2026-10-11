Final submission review, 8 October 2026

I reviewed the complete current `thesis.tex`, its bibliography and included
result tables, and all 80 pages of `Bruckner_Hrubesch (7).pdf`, including the
appendix. There were issues worth correcting before submission. The corrections
below are now in the local source. The existing September PDF predates these
corrections and some of the current table captions.

The revised source passes a local review build with no unresolved citations or
cross-references, no overfull boxes, and no BibTeX warnings. The separate list of
aids also compiles cleanly and has been visually checked. A final build in the
complete Overleaf project remains necessary: the exact project class, original
title page, appendix source, and motif-catalogue image are missing from `writing/`.

**Files to use**

| File | What to do |
| --- | --- |
| `thesis.tex` | Replace the main thesis source in Overleaf. |
| `thesis.bib` | Replace the bibliography and rebuild it. |
| `generated/results_rq1_representations.tex` | Replace this table; its column spacing was reduced to keep it inside the text area. |
| `generated/results_hypothesis.tex` | Replace this table; its caption now identifies which dataset the RQ1/RQ2 ordering statements refer to. |
| `generated/results_cohorts.tex` and `generated/results_rq1_triangles.tex` | Replace these tables; the spelling of Hugging Face in their captions was made consistent. |
| `titlepage_final.tex` | Copy its contents into the existing Overleaf `titlepage.tex`. It uses the existing metadata commands and `logos/WULogo.pdf`. |
| `hilfsmittelverzeichnis.tex` | Completed version of `hilfsmittelverzeichnis_original.tex`; upload and include as described below. |
| `hilfsmittelverzeichnis_standalone.tex` | Optional wrapper for compiling the list as a separate document. Do not input this wrapper into the thesis. |

These result-table edits change presentation and explanatory captions only.
Regenerating the tables with the existing export script would overwrite them.
The experimental code, data, results, and numerical table entries were not changed.
`final_submission_review.patch` records this review's changes to the existing files,
separately from changes that were already present in the working tree.

**What was corrected**

- The original PDF has a date-only page ii. The supplied title-page replacement
  uses flexible vertical space and keeps the title, author, advisor, and date
  on one page. It was checked with the locally available WU template.
- Contents entries for the lists of figures and tables were written after the
  lists, so they pointed to their final pages. They now point to their starts.
  The bibliography entry was also anchored to the preceding appendix page;
  a page break now places that entry on the actual references page.
- The contents listed a List of Abbreviations although no such list was
  generated. That empty entry was removed. The abstract now has a contents entry.
- The date notation was made consistent. The existing completion date,
  **23 September 2026**, was retained. The dummy submission date was cleared.
- Green hyperlink rectangles were hidden; the links remain clickable.
- `amsmath` is now explicitly loaded for `\text{...}` in the equations.
  Microtypography resolves small lines protruding into the margin, and `xurl`
  permits long URLs to wrap.
- The wide RQ1 representation table now fits the text area without reducing its
  font size further. This was a visible margin problem in the supplied PDF.
- British spelling in the body and heading capitalisation were made consistent.
  Examples include `behaviour`, `modelled`, `neighbourhood`, `formalises`, and
  `summarises`. The thesis title retains its existing spelling, **Behavior**,
  because it may be the registered title.
- Several awkward phrases were corrected, including “what a Blockchain is,”
  “well applicable for this thesis,” and the singular “Bitcoin and Ethereum
  ecosystem.” “Hugging Face” is now consistently written as the product name.
- Three small explanatory corrections were made: gas usage is no longer
  equated with the transaction fee; temporal order is not presented as proof of
  coordination; and the three-incident-transfer activity threshold is not
  presented as a mathematical necessity for every local motif role. A sentence
  claiming raw feature magnitude would dominate the random forest was removed.
- The research-question summary now makes clear that its RQ1/RQ2 ordering
  statements refer to the 40-role Hugging Face analysis. This avoids appearing
  to contradict the Real-CATS result.

**Bibliography corrections and checks**

- Two `note = {VERIFY: confirm this is the intended reference}` fields would
  print in a new bibliography. They were removed from the two Chen entries.
- `pohlmann2019anomaly` had the wrong conference and an incomplete author list.
  It now identifies Wittek, Wittek, Ioniță, and Pohlmann's paper in the
  *Blockchain Autumn School 2020* proceedings, pp. 106–110. The publisher
  records completion in 2020 and repository release in January 2021. The old
  citation key is retained to avoid breaking citations.
  [Publisher record](https://monami.hs-mittweida.de/frontdoor/index/index/docId/11872)
  and [full text](https://monami.hs-mittweida.de/files/11872/Druckversion_Wittek_paper.pdf).
- `temporal_sampling_ethereum` previously attributed the paper to Alomari and
  Cheng in 2023. The publicly indexed OpenReview manuscript instead gives
  anonymous authors, the title *Probabilistic Temporal Sampling for Anomaly
  Detection in Ethereum Networks*, and a NeurIPS 2025 submission notice. The
  entry now describes that anonymous manuscript without implying acceptance.
  If you used a different, named version, replace this entry with that version's
  actual metadata. [Public manuscript](https://openreview.net/pdf?id=AhyjbXSmUN).
- The Uniswap paper is now a journal article rather than a conference paper;
  this removes the BibTeX volume/number warning. The SSRN working paper uses
  an appropriate entry type. Proper names are protected from lowercasing.
- The Hugging Face dataset's access date records this review's actual access
  on 8 October 2026. The source revision and numerical dataset description
  used in the experiments were not altered.
- The quoted Hornuf figures (19 categories, 5,644 addresses, about $1.65 billion,
  approximately 60% Ponzi share, and the FTC comparison) also occur in the
  journal version; the old reminder to check them was removed.
  [Journal article copy](https://www.econstor.eu/bitstream/10419/312431/1/191888756X.pdf).

There are 35 distinct citation keys used by the main text. None is missing from
the bibliography, and there are no duplicate bibliography keys or duplicate
labels. This was a final editorial and consistency review, with targeted checks
of problematic references; it was not an independent reproduction of the
experiments or a verification of every claim against every cited paper.

**Adding the Hilfsmittelverzeichnis**

After `\bibliography{thesis}` and before `\end{document}`, insert:

```latex
\input{hilfsmittelverzeichnis}
```

If you place the file in the Overleaf `text/` folder, use
`\input{text/hilfsmittelverzeichnis}` instead.

The include file needs no additional packages or preamble changes. It uses
ordinary `tabular` tables from the supplied
`hilfsmittelverzeichnis_original.tex`, retaining the WU heading, title and
author fields, all four table headings, the declaration, and the date/signature
fields. The original template is preserved unchanged. The entries are split
into two ordinary tables with a page break; there is no `longtable`, `array`,
or `hyperref` dependency. It does not automatically add a contents entry. The separate wrapper uses the
template's A4 article format, 12-point base font, and one-inch margins.

It covers Claude, ChatGPT, Perplexity, Overleaf's writing tools, this final Codex
review, and the analysis software documented in Chapter 4. Claude's contribution
to the Python code and underlying tool is explicitly disclosed. No model
versions, invented prompts, shared chats, or claims that all work was performed
without assistance have been inserted.

WU's policy asks for the tool, purpose, affected parts, and documentation of
prompt histories **if available**, and covers non-AI software too. No shared
chat links were supplied here. Add any that you have to the documentation cells;
references to source files identify outputs but do not replace existing prompt records.
The scope is described as individual passages and tables across multiple
chapters; narrow it if your actual use was confined to particular chapters.
[WU guidance](https://www.wu.ac.at/studierende/mein-studium/bachelorguide/bachelorguide-fuer-deutschsprachige-programme/lvs-pruefungen-schriftliche-arbeiten/hilfsmittelverzeichnis)
and `WUPOL_Hilfsmittelverzeichnis.pdf`, pp. 3–4.

WU specifies that the form for a thesis is submitted with the work in agreement
with the supervisor. Use the included form or the separate PDF according to
that arrangement. Date and sign the completed form; those fields are left
blank for you. The list does not replace citations or the methods chapter.

**Before submitting**

1. Apply the files above in the complete Overleaf project, add the include line,
   and recompile from scratch. Inspect the title page, contents links, RQ1 table,
   references, and two list-of-aids pages in the resulting PDF.
2. Check that **23 September 2026** is the intended completion date. If your
   submission template uses `\submissiontime`, enter the actual submission date
   in that command; it is currently blank rather than guessed.
3. Check the disclosure against your actual tool use, attach any available prompt
   records, date and sign the form, and follow the submission format agreed with
   your supervisor.

**Limits of the local build**

The review used `thesisclass.cls` from the WU template in Downloads. Its legacy
`algorithmic` package was omitted only in the temporary build because the thesis
uses `algpseudocode`. The motif image was recovered from the supplied PDF for
layout checking. The missing appendix source was represented by a labelled stub;
the real appendix was read and inspected in the supplied PDF. Consequently, the
temporary full-thesis build is not a submission artifact. The checked title-page
replacement and standalone list-of-aids PDF are usable outputs. Legacy class
warnings and some underfull spacing warnings remain in the review build; no
missing citations, broken references, or overflowing lines remain.
