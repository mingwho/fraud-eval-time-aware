# Literature review notes — "Random Splits Overstate Fraud Detection" (AICCC 2026)

Compiled 2026-10-04/05. Companion file: `refs_verified.bib` (same folder). BibTeX keys below refer to that file.

**How things were checked.** Bibliographic fields come from the Crossref REST API, the publisher or proceedings page, JMLR, the NeurIPS proceedings site, the arXiv API, or Kaggle's citation endpoint. Where a live publisher page blocked automated access (USENIX, ACM DL, MDPI, Springer), an Internet Archive snapshot of the same page was read instead. DBLP was not usable (bot wall). Every quote in section 2 was string-matched against the full text I downloaded; none comes from an abstract or a search-engine summary.

---

## 2. Evidence that the criticised practice is real

### 2.1 Papers that state the practice (passage read in full text)

Practice codes: **(a)** random / shuffled / stratified split or k-fold with no temporal ordering; **(b)** resampling applied to the whole dataset before the train/test split.

| # | Paper (BibTeX key) | Dataset | Practice | Verbatim quote | Headline metric reported | Where I read it |
|---|---|---|---|---|---|---|
| 1 | Taha & Malebary, IEEE Access 8, 2020 (`taha2020intelligent`), doi:10.1109/ACCESS.2020.2971354 | ULB + UCSD-FICO | (a) random 5-fold CV | "Each data set is divided randomly into five separate subsets of equal size." | Accuracy 98.40%, AUC 92.88%, precision 97.34%, F1 56.95% | https://ieeexplore.ieee.org/ielx7/6287639/8948470/08979331.pdf |
| 2 | Ileberi, Sun & Wang, IEEE Access 9, 2021 (`ileberi2021performance`), doi:10.1109/ACCESS.2021.3134330 | ULB (+ a synthetic set) | (b) SMOTE, then split | "In the first step, the credit card fraud (CCF) dataset is loaded through the SMOTE block. In the second step, the CCF dataset is divided into a training and a test set." | ET-AdaBoost and XGB-AdaBoost accuracy 99.98% on ULB | Accepted-manuscript PDF from IEEE, archived: https://web.archive.org/web/20231115232541/https://ieeexplore.ieee.org/ielx7/6287639/6514899/09651991.pdf |
| 3 | Esenogho et al., IEEE Access 10, 2022 (`esenogho2022neural`), doi:10.1109/ACCESS.2022.3148298 | ULB | (a) stratified 10-fold CV | "we utilized the stratified 10-fold cross-validation technique to evaluate the performance of the models." | Sensitivity 0.996, specificity 0.998, AUC 0.990 (with SMOTE-ENN) | https://ieeexplore.ieee.org/ielx7/6287639/9668973/09698195.pdf |
| 4 | Alfaiz & Fati, Electronics 11(4):662, 2022 (`alfaiz2022enhanced`), doi:10.3390/electronics11040662 | ULB | (a) stratified k-fold CV | "A real-world credit card fraud detection dataset of European cardholders is used in each model along with stratified K-fold cross-validation." | AUC 97.94%, recall 95.91%, F1 87.40% | https://web.archive.org/web/20260829050803/https://www.mdpi.com/2079-9292/11/4/662 |
| 5 | Padhi et al., Sensors 22(23):9321, 2022 (`padhi2022rhsofs`), doi:10.3390/s22239321 | ULB | (a) shuffled stratified 10-fold | "Initially, the entire original dataset has been randomly shuffled; The randomly shuffled dataset is split into k folds (we have set k = 10)" | No single headline figure in the passages I read (per-classifier tables) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9739875/ |
| 6 | Ahmad et al., Int. J. Inf. Technol. 15(1), 2023 (`ahmad2023class`), doi:10.1007/s41870-022-00987-w | ULB | (b) variant: class balancing before the split (under-sampling, not SMOTE) | "After the dataset is balanced, the balanced dataset is divided into two portions: 70% for the training set, and 30% for the testing set" | Accuracy up to 0.943 (ANN) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9209320/ |
| 7 | Chung & Lee, Sensors 23(18):7788, 2023 (`chung2023credit`), doi:10.3390/s23187788 | IEEE-CIS + PaySim + Sparkov + one other Kaggle set | (a) 80/20 partition + stratified 5-fold | "we employed Stratified K-Fold cross-validation with a fold value of 5." | Recall 1.0000, 0.9701, 1.0000, 0.9362 across the four datasets | https://pmc.ncbi.nlm.nih.gov/articles/PMC10535547/ |
| 8 | Khalid et al., Big Data Cogn. Comput. 8(1):6, 2024 (`khalid2024enhancing`), doi:10.3390/bdcc8010006 | ULB | (b) under-sampling / SMOTE, then 80/20 split | "After the sampling process, the subsequent step involved splitting the data into training and testing samples." | "roughly 100%" accuracy on the testing samples (SMOTE data) | https://web.archive.org/web/20251230153528/https://www.mdpi.com/2504-2289/8/1/6 |
| 9 | Azim Mim, Majadi & Mazumder, Heliyon 10(3):e25466, 2024 (`mim2024soft`), doi:10.1016/j.heliyon.2024.e25466 | ULB | (a) shuffled random 80/20 | "the dataset is shuffled, and 80% of the instances are randomly assigned to the training dataset, while the remaining 20% form the test/validation dataset." | Precision 0.9870, recall 0.9694, AUROC 0.9936 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10850588/ |
| 10 | Du et al., PLOS ONE 19(3):e0294537, 2024 (`du2024novel`), doi:10.1371/journal.pone.0294537 | ULB | (a) stratified 6:2:2 | "we employed stratified sampling to divide the preprocessed dataset into three subsets: training, validation, and test sets, with a ratio of 6:2:2." | ACC "consistently high at above 0.9995" | https://pmc.ncbi.nlm.nih.gov/articles/PMC10917329/ |
| 11 | Becerra-Suarez et al., Front. Artif. Intell. 9:1871972, 2026 (`becerrasuarez2026conservative`), doi:10.3389/frai.2026.1871972 | ULB + IEEE-CIS | (a) stratified 60/20/20 | "stratified sampling was applied to three mutually exclusive subsets: 60% for training, 20% for calibration, and 20% for final testing" | Accuracy > 0.999 for all models on ULB; best F1 0.855 | https://pmc.ncbi.nlm.nih.gov/articles/PMC13424206/ |

Notes on individual rows:

- **Row 2 (Ileberi 2021)** — I read IEEE's accepted-manuscript PDF (early-access version), not the final typeset article. The pipeline description is in the framework section; check the final version's wording before quoting.
- **Row 3 (Esenogho 2022)** — only (a) is explicit. Whether SMOTE-ENN was applied inside the folds or to the whole dataset is not stated in what I read. Hayat & Magnier (below) read it as whole-dataset balancing; I could not confirm that from the paper's own words, so do not cite it for (b).
- **Row 6 (Ahmad 2023)** — this is balancing by under-sampling before the split, so the test set is not the natural class distribution. It is not synthetic-sample leakage in the SMOTE sense. Use it only if your text covers "resampling before splitting" generally.
- **Row 7 (Chung & Lee 2023)** — the 80/20 partition is not described as random; the explicit statement is the stratified 5-fold. I did not establish which of the four recall figures belongs to IEEE-CIS.
- **Row 11 (Becerra-Suarez 2026)** — published 2026, outside your preferred 2019–2025 window. Oversampling is done correctly (training set only); the only issue is the non-temporal split. The authors themselves list temporal validation as future work ("Future research should extend the evaluation to more datasets, incorporate temporal validation…").
- **Georgiev et al. 2026** (`georgiev2026modeling`, section 3) also belongs here: its main protocol is "Stratified random 80/20 holdout via StratifiedShuffleSplit" on ULB.
- Rows 5, 6, 7, 9, 10, 11 were read as Europe PMC full-text XML of the listed PMCID; the PMC link is the human-readable equivalent.

**Coverage gap.** Only two peer-reviewed rows involve IEEE-CIS (7 and 11), and neither is a (b) case. I did not find an open-access, peer-reviewed IEEE-CIS paper that explicitly applies SMOTE before the split. One preprint states a random split on IEEE-CIS outright — Han & Wu, arXiv:2606.10393 (2026): "The split is stratified random rather than temporal, although TransactionDT would support time-based testing." It is not peer-reviewed and is not in the .bib.

**Candidates I checked and rejected** (no explicit statement in the text, or wrong dataset): Alarfaj et al. 2022 (IEEE Access), Benchaji et al. 2021 and Ileberi et al. 2022 (J. Big Data), Mienye & Sun 2023 (IEEE Access), Siam et al. 2025 (PLOS ONE; SMOTE is applied after the split), and two Scientific Reports papers that use a different "2023 European cardholders" Kaggle dataset or describe their pipeline inconsistently.

### 2.2 Is there systematic evidence of how common this is?

**No. I found no survey or audit that counts how many fraud-detection papers use random splits or apply oversampling before splitting.** What exists is qualitative:

- **Hayat & Magnier 2025** (`hayat2025data`; Mathematics 13(16):2563, doi:10.3390/math13162563; read as arXiv:2506.02703v3). A critical review of 20 deep-learning and ML methods evaluated on the ULB dataset (their Table 1), plus a deliberately leaky MLP experiment showing that SMOTE-before-split alone yields 99.9% recall. They report no count or percentage. Their summary wording: "Numerous studies perform critical preprocessing steps (normalization, SMOTE etc.) before train-test splitting" and "Most approaches fail to account for the time-dependent nature of transaction data, neglecting temporal splitting which is essential for real-world deployment." By my own tally of their text, roughly 9–11 of the 20 are flagged for confirmed or suspected pre-split balancing or normalisation; that number is mine, not theirs, so do not attribute it to them. They run no temporal-split experiment.
- **Breskuvienė & Dzemyda 2024** (`breskuviene2024enhancing`): "In many cases, studies addressing the fraud-detection problem in credit card transactions unrealistically evaluate the performance of the proposed method by splitting the dataset into train and test using random split". Assertion with two example citations, no count.
- **Zuberi et al. 2026** (`zuberi2026robust`): "the majority of published studies employ random train-test splitting and produce evaluation metrics that overestimate operational performance on future transactions". The supporting citation is Lucas et al. 2020 (Future Gener. Comput. Syst. 102), a feature-engineering paper. I did not read that paper and cannot confirm it supports a "majority" claim; treat this as an unsupported assertion.
- **Kapoor & Narayanan** (`kapoor2023leakage`) is cross-disciplinary, not fraud-specific. The arXiv v1 abstract (2207.07048) says "17 fields where errors have been found, collectively affecting 329 papers". I did not fetch the Patterns abstract; the published count may differ, so check it before quoting a number.
- Not read: Meena, Wadhwani & Rasool, "A Comparative Review of Fraud Detection Techniques Using the IEEE-CIS Dataset" (Springer LNNS chapter, 2026, doi:10.1007/978-3-032-10940-8_17, pp. 207–223 per Crossref). It reviews IEEE-CIS studies and might tabulate split practice; it is paywalled and I could not open it.

A small audit of your own (e.g. coding the split and resampling order of N papers per dataset) would be a genuine contribution, since nobody appears to have published one.

---

## 3. Closest prior work (random vs. chronological splits for payment/card fraud)

**Prior work exists; the comparison is not new, but nobody I found has made it the subject of a paper on both datasets with classical models and alert-budget metrics.** In order of closeness:

1. **Grover et al., "Fraud Dataset Benchmark and Applications"** (`grover2022fraud`; arXiv:2208.14417v3, preprint, Amazon). Closest on datasets: covers both `ccfraud` (the ULB data) and `ieeecis` with standardised out-of-time test splits. Their Table 8 compares each AutoML tool's local validation AUC with out-of-time holdout AUC: "the bias is high with AutoGluon and H2O (11.4% and 6.3% in AUC respectively) on ieeecis. This is likely because the k-fold bagging and cross validation are not done out-of-time. On ccfraud, the bias is low in all models." So the IEEE-CIS gap is already documented, and they found little gap on ULB — which bears directly on your ULB claim. Differences from your paper: AutoML tools rather than a controlled split comparison, ROC-AUC only, no SMOTE-ordering experiment, no alert-budget metrics.

2. **Georgiev, Markova, Mihova & Todorov 2026** (`georgiev2026modeling`; Mathematics 14(9):1496). Runs both protocols on ULB, but only for one model (a CNN) in an appendix: "The main difference appears in recall, which decreases from 0.8776 under the random split to 0.8000 under the chronological protocol." Under the chronological split the CNN scores precision 0.594, recall 0.800, F1 0.682, MCC 0.689; they describe precision as nearly unchanged. They keep the random stratified split as their main protocol.

3. **Breskuvienė & Dzemyda 2024** (`breskuviene2024enhancing`; J. Big Data 11:182). A feature-selection paper that adopts a time-based split (earliest 80% for training) on ULB and two synthetic datasets, argues against random splitting, and reports results under both a time-based and a stratified random 80/20 split on ULB so it can compare with earlier papers (their Tables 9 and 10). Also states "Applying sampling methods before splitting the dataset into train and test sets leads to deceptively high results." The two tables did not survive text extraction, so I cannot report the size of the gap they found.

4. **Kabane 2024** (`kabane2024impact`; arXiv:2412.07437, preprint). Covers the SMOTE half of your claim on ULB with XGBoost: sampling before vs. after the split vs. none. The pre-split figures (accuracy 99.969%, recall 100%) appear to be taken from a cited earlier study rather than re-run; the post-split experiments are the author's own (e.g. baseline 99.96% accuracy, 91.50% recall). No temporal split.

5. **Hayat & Magnier 2025** (`hayat2025data`). Demonstrates on ULB that a leaky pipeline with a trivial MLP beats published models; names missing temporal validation as a flaw but does not test it.

6. **Zuberi et al. 2026** (`zuberi2026robust`; Sci. Rep. 16:27776). Chronological 70/15/15 split on ULB with train-only preprocessing and explicit drift measurement across 85 model–sampler combinations. No random-split arm, so no measured gap. They note the dataset spans only 48 hours, so drift is "short-horizon, mostly intra-day variation" — a limitation that applies equally to any ULB temporal split, including yours.

7. **Hsin et al. 2022** (`hsin2022feature`; IEEE Access 10). Adjacent domain: fund-transfer fraud on proprietary bank data, not public card data. A direct chronological-vs-random comparison (their Table 9): chronological sampling gives recall between 1.49% and 70.59% depending on training size, while random partitioning lets the model "foresee the rapid changes in future modi operandi from the training set, which clearly is impossible in real-world applications, as reflected in the unrealistically high detection performance."

8. **Le Borgne et al. handbook** (`leborgne2022fraud`). Prescribes chronological train/test sets separated by a delay period, and prequential validation, on simulated data. On the two pages I read (baseline modelling; validation strategies) there is no random-vs-chronological head-to-head.

**What this means for your novelty claim.** Safe to claim: a controlled, like-for-like comparison of random vs. chronological splits (and pre- vs. post-split SMOTE) across both ULB and IEEE-CIS with classical models and alert-budget metrics. Not safe to claim: being first to show random splits inflate fraud scores, or first to show it on these datasets (items 1–3). Expect a smaller effect on ULB than on IEEE-CIS; items 1, 2 and 6 all point that way, and a reviewer who knows FDB will ask.

Non-peer-reviewed GitHub and blog write-ups making the same point on IEEE-CIS turned up in search; I did not verify or include them.

---

## 4. Could not verify, or differed from your description

### Differences from what you listed

| # | Item | What differed |
|---|---|---|
| 3 | Dal Pozzolo et al. 2015 | Crossref gives the venue as "2015 IEEE Symposium Series on Computational Intelligence", pp. 159–166. Some papers cite it as CIDM (a symposium within SSCI); SSCI is the registered container. |
| 8 | Bahnsen et al. 2016 | First author's family name is deposited as "Correa Bahnsen". |
| 9 | Whitrow et al. | Online 30 July 2008; issue is 18(1), February 2009. 2009 is right for the issue. |
| 10 | PaySim | No DOI. Pages 249–255 confirmed from the PDF footers. The proceedings title on the PDF is "Proceedings of the European Modeling and Simulation Symposium, 2016"; I did not add an ordinal or location. |
| 11 | Reviews | Cherif 2023 and Hilal 2022 both verified as you described. I also verified Hafez et al. 2025 (J. Big Data 12:6), which is card-specific and newer. Suggested pair: Cherif 2023 + Hafez 2025 if you want card-specific; Hilal 2022 covers financial fraud anomaly detection more broadly. I checked metadata only and did not read any of the three for content. |
| 12 | IEEE-CIS Kaggle citation | Kaggle's citation lists eight authors, including "Catherine Huang" and three user handles ("inversion", "Lynn@Vesta", "Marcus2010"). Entry reproduces Kaggle's BibTeX. |
| 14 | Kaufman et al. | Two versions exist: KDD 2011 (three authors, no Stitelman) and TKDD 2012 (four authors, Article 15). The .bib has the TKDD one, as you asked. |
| 17 | Santos et al. 2018 | Title is "Imbalanced Datasets" (one word), and Crossref appends "[Research Frontier]". Crossref spells the third author "Pedro Henrigues Abreu", almost certainly a typo; other Crossref records for the same person read "Henriques", which the .bib uses. Confirm against the PDF. |
| 18 | Gama et al. 2014 | Article 44, 37 pages; online March 2014, issue April 2014. |
| 22 | Roberts et al. 2017 | 14 authors; online March 2017, issue 40(8) August 2017. |
| 31 | Grinsztajn et al. | NeurIPS 2022 Datasets and Benchmarks track; pp. 507–520; DOI 10.52202/068431-0037. |
| 6 | Lucas & Jurgovsky | Still arXiv-only (arXiv comment: "To be published"). |

Everything else in Part 1 (items 1, 2, 4, 5, 7, 13, 15, 16, 19, 20, 21, 23, 24, 26, 27, 28, 29) matched your description; exact volume, pages and DOI are in the .bib.

### Not verified

- **Page numbers for LightGBM (#25) and PyTorch (#30).** The official NeurIPS BibTeX leaves `pages` empty and there is no DOI. Both entries are otherwise verified and carry the proceedings URL. If the ACM template needs pages, take them from the printed proceedings.
- **The ULB dataset's own Kaggle page** (kaggle.com/datasets/mlg-ulb/creditcardfraud) was not fetched; no separate dataset entry is in the .bib. Dal Pozzolo 2015 is the citation.
- **Hayat & Magnier** — metadata verified for the Mathematics version, but the text I read is arXiv v3. Quotes may differ slightly in the published version.
- **Ileberi et al. 2021** — quote taken from the accepted manuscript (see 2.1).
- **Breskuvienė & Dzemyda Tables 9–10** and **Hsin et al. Table 9 cell values** — not read beyond what the running text reports.
- **Lucas et al. 2020 (FGCS)** and **Meena et al. 2026 (Springer chapter)** — not read; see 2.2.
- **Kapoor & Narayanan paper count** — from arXiv v1 only; see 2.2.
- **DBLP** was never consulted (blocked), so nothing here is DBLP-verified.
