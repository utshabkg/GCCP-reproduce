# Draft email to original authors

**To:** longkehan15@nudt.edu.cn (Kehan Long), shashali@nudt.edu.cn (Shasha Li, corresponding), tangjintao@nudt.edu.cn (Jintao Tang, corresponding), tingwang@nudt.edu.cn (Ting Wang, corresponding), xuchen@nudt.edu.cn (Chen Xu)

**Subject:** Reproducibility study of GCCP/PAGC (SIGIR 2025) — three clarifications

Dear Dr. Long, Dr. Li, Dr. Xu, Dr. Tang, and Dr. Wang,

I am leading a reproducibility study of your SIGIR 2025 paper *"Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information"* as a course project at Missouri University of Science and Technology, with the goal of submitting the resulting paper to a reproducibility track. Our re-implementation, built first from the paper text and then audited against your `ChainsawM/GCCP` repository, reproduces your reported NDCG@10 within 2--4% on average across TREC DL 19/20 and the eight BEIR subsets at all three Flan model scales. Code, per-query scores, run logs, and statistical tests are at https://github.com/utshabkg/GCCP-reproduce.

While preparing the manuscript I found three points where I could not reconcile our reproduction with the paper from the released code alone, and I would be very grateful for your input.

**(1) Sentence segmentation in spectral MDS.** Your `MultiDocSummarizer.extract_sentences` keeps a sentence iff `current_doc_length + len(clean_sent) <= max_doc_length OR current_doc_length < 128`, with `max_doc_length=200` and `min_words=3`. Our faithful port closes about 1.2 NDCG@10 points of GCCP on DL20/Flan-T5-Large but ~85% of the original 5.7% PAGC gap on DL20/Flan-T5-Large remains unexplained. The gap shrinks to 2% with Flan-UL2. Are there additional preprocessing details (e.g. document-level normalization, query filtering specific to DL20, special handling of `define:` or other MS MARCO query patterns) that we may be missing?

**(2) Anchor ablation on TREC DL.** Your Table 5 reports GCCP+Top = 0.6099 vs GCCP+Spectral = 0.6076 (averaged DL19+DL20), and your text notes the "Top" exception explicitly. Our reproduction replicates this direction but at ~4× the magnitude (+0.008 vs +0.002 for GCCP, and +0.011 PAGC for the Borda rank-aggregation we tested in addition to your Linear). I would like to confirm: in your Table 5, when you compute the "Top" anchor variant with `#LLM calls = 100`, is "Top" the *single* top-1 BM25 document used as the entire anchor text, or some other operationalization? And is "Random" "one randomly drawn passage from top-10" or "one passage from the full top-100"?

**(3) Hyperparameters not stated in the paper.** Our audit found seven undocumented details that determine whether reproduction succeeds: (a) `decoder_input_text='<pad> '` for RG-YN; (b) `'<pad> Passage '` for GCCP; (c) target tokens lowercase `'yes'/'no'`; (d) target tokens uppercase `'A'/'B'`; (e) spectral threshold $\theta = 0.2$; (f) BM25 $k_1=0.9, b=0.4$; (g) document truncation at 128 tokens. We reconstructed all seven from your code. Is there a chance any of these were set differently in the runs reported in Tables 1, 2, or 5? I ask because the gap between our DL19 PAGC at Flan-T5-XL (0.7030) and yours (0.7281) is ~3.5%, and even reproducing point (a) is enough to swing NDCG@10 from 0.24 to 0.55 in our hands.

If it is convenient, a one-line confirmation of "Top" definition for question (2) and a pointer to the run script you used for Table 1 would be very helpful. I am of course happy to share any of our intermediate artifacts if useful, and we will cite your responses in the paper if you would like (or omit them entirely if you prefer).

Thank you very much for releasing the code; the reproduction would have been considerably harder without it.

Best regards,
Utshab Kumar Ghosh
PhD student, Department of Computer Science
Missouri University of Science and Technology
u.ghosh@mst.edu
https://github.com/utshabkg/GCCP-reproduce
