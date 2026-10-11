# Final draft cleanup: Overleaf replacements

These are only the three changes from the current request. They are already
applied in writing/thesis.tex and marked with BEGIN FINAL TEXT FIX FT01 through
FT03. The comments do not appear in the compiled thesis.

Use the three snippets in [final_text_fixes](final_text_fixes/). Replace the
existing passages, rather than appending the snippets. You do not need to copy
the earlier feedback packages again.

| Snippet | Replace in Overleaf |
| --- | --- |
| [FT01_dates.tex](final_text_fixes/FT01_dates.tex) | The definitions of \timeend and \submissiontime, directly after \timestart. Leave \timestart unchanged. |
| [FT02_abstract.tex](final_text_fixes/FT02_abstract.tex) | Everything from \chapter*{Abstract} through the draft notes, stopping before \chapter{Introduction}. |
| [FT03_temporal_networks.tex](final_text_fixes/FT03_temporal_networks.tex) | Both paragraphs starting at \paragraph{Time-respecting paths and edge ordering.} and ending immediately before \paragraph{What static projection loses.}. This includes replacing the old time-window paragraph. |

## Date

The completion and submission dates have not been supplied. Both fields are
intentionally blank, so the dummy dates are no longer printed. Once the dates
are fixed, enter them inside the final braces of the two definitions.
Do not replace them with \today, which changes with the compilation date.
The original start date is unchanged.

The separate Overleaf title-page template is not present locally. Clearing the
date does not remove any page that the template creates for it.

## Text

The Abstract is 201 words and reports the scope, method, and main findings
without claiming direct campaign detection or a consistently significant
improvement. No new abbreviations are introduced.

The temporal-networks replacement explains time-respecting paths, equal-time
ordering, the distinction between paths and connected candidate patterns,
overlapping sliding windows, and the total-span meaning of \delta. The
window choice is linked to the development-data selection protocol.

It uses the existing bibliography keys kovanen2011temporal and
paranjape2017motifs. No bibliography upload is needed. The primary sources
checked for the definitions are
[Kovanen et al., Section 4](https://arxiv.org/html/1107.5646v2) and
[Paranjape et al., Sections 3 and 4.1](https://arxiv.org/pdf/1612.09259).

## Left unchanged

The \section{Summary} and its - 5-8 pages note are unchanged, as requested.
The appendix, figures, code listings, and earlier corrections are also unchanged.

For this thesis, a separate multi-page summary at the end of Theoretical
Foundations would be repetitive. Either omit the section or use one short
paragraph to connect the foundations to the method, unless your supervisor or
programme specifically requires more. This is distinct from the thesis-wide
Conclusion and Outlook chapter, which should remain.

## Check after pasting

A temporary book-class build compiled without unresolved citations or
cross-references. The changed text was checked for layout and has no overfull
lines in that build. The Abstract text fits within half a page.

Recompile in Overleaf and inspect the Abstract, the expanded Temporal networks
section, and the date page. The added text may move later content to different
pages. The exact template layout still needs an Overleaf check because
thesisclass.cls, titlepage.tex, and the original motif-catalogue figure are not
available locally. The temporary check is not a replacement thesis PDF.

