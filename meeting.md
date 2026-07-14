what is the final goal (for theoretical and practical)

research questions
Can GNN models detect coordinated manipulation better than traditional heuristics?
Can GNNs effectively distinguish between coordinated and organic user behavior?

do I need to explain everything (graph theory etc.) in detail

project structure:

1. Load + clean transactions (+ per-address labels, if any)
2. Build NetworkX graph → per-node structural features                  ← `--mode graph` stops here
3. Count per-node temporal motif fingerprints (selected backend)        ← `--mode motifs` stops here
4. null model testing - significance testing
--4. Assemble feature matrix per --features {aggregates, motifs, full}
5. Train + evaluate  (supervised)  OR  detect + cluster  (unsupervised)
6. Visualize (skipped with --no-plot)



-----
* sliding window -> 3600 and difference until 

* patterns of scamming -> if scam is on the account or on the transaction

* 2 groups scam/noscam

* multithreading for counting motifs

* null model generation for synthetic dataset and hypothesis testing/pvalue

* confirm if 5 counts are the same for each motif