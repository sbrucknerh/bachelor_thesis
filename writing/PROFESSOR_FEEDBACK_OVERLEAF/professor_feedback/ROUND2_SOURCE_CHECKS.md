# Related Work source checks

The revised chapter cites 21 distinct works. Twenty are research papers and one
is the dataset release. The links below identify primary papers, author copies,
publisher records, or the dataset creator's documentation used for this revision.
Descriptions focus on the task, representation, and relationship to this thesis.
They do not imply that all methods were reproduced or compared experimentally.

| Citation key | Role in the chapter | Source |
|---|---|---|
| `lee2020ethereum` | Ethereum interaction-network representations | [Author paper](https://souravsengupta.com/publications/2020_www.pdf) |
| `ostapowicz2019detecting` | Supervised address-feature baselines | [Publisher](https://link.springer.com/chapter/10.1007/978-3-030-34223-4_2) |
| `chen2022phishing` | Engineered address features and random forests | [Full paper](https://www.techscience.com/JCS/v4n3/51402/html) |
| `sanjalawe2023atd` | Multidigraph features and semi-supervised detection | [Author paper](https://services.uop.edu.jo/InstructorProfile/files/Publication202391620515.pdf) |
| `wu2020phishers` | Timestamp-aware trans2vec embeddings | [Authors' preprint](https://arxiv.org/abs/1911.09259) |
| `wu2020temporal` | Temporal weighted walks and link prediction | [Authors' record, with journal DOI](https://arxiv.org/abs/2012.15462) |
| `weber2019bitcoin` | Elliptic and graph-versus-feature baselines | [Authors' preprint](https://arxiv.org/abs/1908.02591) |
| `ghoshcatalog2025` | Local and global temporal dependence in phishing detection | [Authors' project](https://github.com/Medhasree93/CATALOG), [conference proceedings](https://archives.iw3c2.org/www2025/www2025-proceedings.pdf) |
| `alemari2021labeled` | BLTE benchmark dataset | [Publisher](https://link.springer.com/chapter/10.1007/978-981-33-6835-4_5), [authors' release](https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network) |
| `shi2025realcats` | Address profiles, histories, and abuse labels | [Authors' preprint](https://arxiv.org/abs/2501.15553) |
| `fesevu2025ethfraud` | Source transaction histories and address labels | [Dataset card](https://huggingface.co/datasets/fesevu/ethereum_fraud_dataset_by_activity) |
| `milo2002network` | Motifs as statistically overrepresented patterns | [Paper](https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/Milo-2002-motifs.pdf) |
| `McDonnell2014` | Local motif-role fingerprints | [Full paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0114503) |
| `kovanen2011temporal` | Event order in temporal pattern definitions | [Authors' preprint](https://arxiv.org/abs/1107.5646) |
| `paranjape2017motifs` | Bounded-span temporal pattern counting | [Authors' preprint](https://arxiv.org/abs/1612.09259) |
| `lal2021money` | Temporal cycles in Ethereum money trails | [Authors' preprint](https://arxiv.org/abs/2108.11818) |
| `wu2022mixing` | Hybrid motif features for Bitcoin mixing detection | [Authors' preprint and journal DOI](https://arxiv.org/abs/2001.05233) |
| `wang2023temporal` | Direct precedent for motif-based Ethereum phishing detection | [Publisher record](https://ieeexplore.ieee.org/document/10218023), [conference contents](https://www.proceedings.com/content/070/070355webtoc.pdf) |
| `arnold2024motifs` | Local patterns, null references, and tied timestamps | [Full paper](https://www.nature.com/articles/s41598-024-75348-7) |
| `liu2023temporal` | Fraud detection outside blockchain networks | [Authors' preprint](https://arxiv.org/abs/2301.07791) |
| `mo2022motifgcn` | Snapshot motifs in signed Bitcoin trust networks | [Revised paper](https://arxiv.org/html/2211.13123v2) |

## Corrections and limits

- The earlier claim that no labelled blockchain detector uses temporal motif
  features has been removed. Wang and colleagues' 2023 conference paper is a
  direct precedent. Full publisher text was not accessible through the tools,
  so the revision does not state its scores, sampling parameters, exact motif
  catalogue, or window-selection procedure.
- The Lin paper's author list and DOI were wrong in the existing bibliography.
  The key `wu2020temporal` remains to avoid breaking manual Overleaf updates.
- The McDonnell paper's publisher lists Ömer Nebil Yaveroğlu and Brett A. Schmerl.
  These names replace the incorrect given names in the draft bibliography.
- The Lee paper occupies pages 155 to 166, not 662 to 672.
- The trust-network paper is by Song Li and colleagues. The revised text uses
  the 2023 second version, which describes signed-link classification. The
  arXiv landing page retains the older title, so the revised document itself
  is used for the title and task. The legacy key `mo2022motifgcn` remains.
- The dataset's page is a dataset release, not an additional peer-reviewed
  detection study. It is counted once among the 21 references.
- Source scores are reported only in their original study context. No ranking
  across incompatible datasets or direct comparison with this thesis is claimed.

The citation-command audit applies throughout the thesis. Bibliographic and
factual checks in this round concentrate on the expanded chapter and the
inconsistencies it exposed. Existing verification notes for other bibliography
entries are not silently treated as resolved.
