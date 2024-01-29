---
title: " One Model to Rule Them All: Generalizable, Fast, and Accurate Deep-QSPR with `fastprop`"
date: "January 2024"
author: "Jackson Burns, MIT CCSE"
geometry: margin=1in
note: Compile this paper with "pandoc --citeproc --bibliography=paper.bib -s paper.md -o paper.pdf"
---

# Current Project Status:
 - `fastprop` has all of the training features that it needs; the prediction side still needs be written
 - all but one of the benchmarks I want to run with `fastprop` have been run, though running some more comparison studies with chemprop might be worthwhile (the other benchmark is a 'shrinking training size' benchmark for QM8)
 - the paper is in bullets with a good overall structure, results needed for writing are included in this document inline

# Outline
 - **Problem**: Quantitative Structure-Property/Activity Relationship studies, often referred to interchangeably as QS(P/A)R, seek to establish a mapping between molecular structure and an arbitrary Quantity of Interest (QOI).
 - **Historical Approach**: Historically this was done on a QOI-by-QOI basis with new descriptors being devised by researchers to _specifically_ map to their QOI (see the sub-field of transition metal-mediated cross-coupling reactions, where molecular descriptors specific to ligands such as cone angle, sterimol parameters, and buried volume are used).
 - This continued for years and culminated in packages like (sort these chronologically) DRAGON, Mordred, PaDEL-descriptor, E-DRAGON, etc. Other packages like molfeat and AIMSim aggregate these other descriptor generation tools

 - **Historical Approach, Today**: Break these two ideas out in much greater detail in the body of the text:
 - The logical combination of bulk molecular-level descriptors with deep learning as a regression technique (as stated in the chemprop paper) has not seen the same success as learned representations, like UniMol [@unimol], Communicative Message Passing Neural Networks [@cmpnn], and especially Chemprop [@chemprop_theory; @chemprop_software].
 - Despite past failures, this paper establishes that by combining a cogent set of molecular descriptors with deep learning, generalizable deep-QSPR is realized.

# QSPR and `fastprop`
 - in the same pattern as the outline in subsections, establish the idea of QSPR, the historical focus on linear methods with bespoke descriptors, the failure of these methods to work generally (I claim because the descriptor sets were not sufficient and the regression techniques could not learn the complex functions, i.e. non-linearity), the move into learned representations to counter this, and then the idea that modern software finally has good enough sets of descriptors to facilitate the previous approach and by going back and not 'starting' from zero we can go faster more interpretably and on smaller datasets

## Historical Approaches
See this outstanding review by Muratov et. al (https://doi.org/10.1039/D0CS00098A), qsar was historically linear methods (nowadays also called ML), especially random forest, and has only recently tried DL; DL has been unsuccessful primarily because of data limitations, I will prove that it still can be successful.

## Shift to Learned Representations
 - in the search for generalizable QSPR, we shifted away from molecular descriptors and fingerprints toward learned representations
 - exact paper is hard to pin, but the only general-purpose property prediction models on the 'leaderboards' today follow this line of thinking:
Mention UniMol, Chemprop, CMPNN, and  GSL-MPP (https://arxiv.org/pdf/2312.16855.pdf) and SGGRL (https://arxiv.org/pdf/2401.03369.pdf).
And also MHNN: https://github.com/schwallergroup/mhnn

## Previous Deep-QSPR and the `fastprop` Approach
 - as we moved toward learned representations, we lost interpretablity, sacrificed speed, and **lost the ability to correlate small datasets with a target** ([@chemprop_theory] states that <1000 entries, fingerprint models are competitive with Chemprop again) **because we are starting from near-zero everytime**
 - This review talks about wanting to fit on low data and the advanced ML techinques needed to do it when using these more involved models.
https://chemrxiv.org/engage/chemrxiv/article-details/65b154a166c13817292fad82 they want low data DL, here it is! None of this involved stuff
**by starting from such an informed initialization (100 years of descriptor design) we circumvent the need for advanced training techniques and progressively slower, complex, and uninterpretable models**

`fastprop` is simply the `mordred` molecular descriptor calculator connected to an FNN with research software engineering best practices applied at the same level as the best alternative in the literature Chemprop [@chemprop_software].
See the `fastprop` logo in Figure \ref{logo}.
This contradicts the leading review on deep-QSPR [@ma_deep_qsar] which says that these are ineffective, and that fingerprints should be used instead.

![fastprop logo.\label{logo}](../fastprop_logo.png){ width=2in }

This is a simple idea, here are some that have tried it before and failed:
 - In the Fuel property prediction world, this review by Comesana and coauthors [@fuels_qsar_method] establishes that (1) methods using huge numbers of descriptors are not the most successful (2) instead introduces a method to downsample
 - Esaki and coauthors [@fubrain] took the mordred descriptors for a dataset of small molecules but then they did a huge pre-processing pipeline that removed all but 53 descs, and then only did linear methods. DeepDelta then fit on this dataset and improved the accuracy, but I do better here in the Benchmarks section.

Add more examples to the above after writing the benchmarks section (basically want to provide a short summary of each of the qspr-style benchmarks here, to get the point across).

But of course, `fastprop` stands on the shoulder of giants, see this next section which talks about the evolution of molecular descriptor software.

### Maturation of Descriptor Generation Software
 - part of the failure of this method may be the lack of descriptor generation software with a sufficient collection of descriptors, which `mordred` isThis continued for - expand on the line form the intro: DRAGON, Mordred, PaDEL-descriptor, E-DRAGON, etc. Other packages like molfeat and AIMSim aggregate these other descriptor generation tools
 - from here forward `mordred` will be referred to as such, though the actual implementation of `fastprop` uses a community-maintained fork of the original (which is no longer maintained) aptly named `mordredcommunity` (cite the Zenodo)

# Benchmarks
Start by establishing difference between small and large datasets:
 - There are a number of established molecular property prediction benchmarks in the cheminfomatics community, especially those standardized by MoleculeNet.
 - Some of these, namely QM8/9, are perceived as the 'gold standard' for property prediction.
 - While this task is important, the enormous size of these datsets in particular means that many model architectures are highly accurate given the rich coverage of chemical space (inherent in the design of the combinatorial datasets) i.e. these datasets are so comprehensive, and model worth its salt should be able to learn.

Talk about the real world datasets to be worked on here:
 - Real world datasets, particularly those common in QSPR studies, often number in the hundreds.
 - To demonstrate the applicability of `fastprop` to these regimes, many smaller datasets are selected, including some from the QSPR literature that are not established benchmarks.
 - As explained in each subsection, these studies relied on **more complex and slow modeling techniques (the ARA dataset) or design of a bespoke descriptor (the PAH dataset)** and **have not yeen come to rely on learned representations as a go-to tool (whereas QM9 etc. are dominated by CMPNN, UniMol, Chemprop)**; **we believe because in these data-limited environments the 'start from almost-nothing' approach of Chemprop leads to overfitting or requires more data** and that the 'start from 100 years of QSPR' approach of `fastprop` circumvents this.

The authors of Chemprop have even suggested on GitHub issues in the past that molecular-level features be added to Chemprop learned representations (see https://github.com/chemprop/chemprop/issues/146#issuecomment-784310581) to avoid overfitting.

To emphasize this point further, the benchmarks are presented in order of size descending (for first regression and then classification).
Consistently `fastprop` is able to match the performance of learned representations on large (O(10,000)+ entries) datasets and compete with or exceed their performance on small ($\leq$O(1,000) entries).

All of these `fastprop` benchmarks are reproducible, and complete instructions for installation, data retrieval and preparation, and training are publicly available on the `fastprop` GitHub repository [github.com/jacksonburns/fastprop](https://github.com/jacksonburns/fastprop).

**TODO: re-run all of these (with timing, too) for a first complete draft**

## Methods
All of these use 80% of the dataset for training (selected randomly, unless otherwise stated), 10% for validation, and holdout the remaining 10% for testing (unless otherwise stated).
Sampling is performed using the `astartes` package [@astartes], which implements a variety of sampling algorithms and is highly reproducible.

### Reporting of Error Bounds
 - Results for `fastprop` are reported with repetitions (explain) rather than with cross-validation (CV) (also explain).
 - When CV is performed without a holdout set, occasionally seen in literature, it allows the model to learn the validation dataset (the accuracy on which is reported as the 'result') during training (i.e. data leakage).
 - Even when CV is done corectly, i.e. with a holdout set, for small datasets especially this can lead to misreprentation of accuracy. The holdout set, when initially randomly selected, may contain only 'easy' samples; regardless of the number of folds within the training/validation data, the reported performance will be overly optimistic.

Rant part that needs to be refined:
The solution is to either _nest_ cross validation (i.e. repeat cross validation multiple times with different holdout sets (as was done in https://pubs.acs.org/doi/10.1021/acsomega.1c02156, for example)) or to just simply do repeats (like the MHNN paper)(more practical for deep learning, which is slow compared to linear methods **and, if anything, only makes the model performance not as good as it could since I don't optimize on each possible subset (so statistcally I'm not guaranteed a good result and NN should not be a steep function of this training choice) be so really I'm being super-duper extra honest here**).

 - By simply repeating a random selection of train/validation/testing subsets, we reduce the impact of biased holdout set selections while also avoiding data leakage.
 - For this study, the number of repetitions vary per-benchmark and are determined by first setting the number of repetitions to two, and then increasing until the result of a 2-sided T-test between the validation and testing set accuracy yielded a p-value greater than 0.05 at 95% confidence.
 - At that point it is asserted that a reasonable approximation of the accuracy as a function of the holdout set has been achieved.

### Choice of Evaluation Metric
The evaluation metric used in each of these metrics (L1, L2, AUROC, etc.) are presented here because that is what the reference papers (and the literature at large) use, especially the metrics established in moleculenet.
It is my opinion that this choice of metric 'buries the lede' by requiring the reader to understand the relative magnitude of the target variables (for regression) or understand the nuances of classification quantification (ARUOC).
On the HOPV15 subset, for example, an MAE of 1.1 seems exceptional in the literature, except that the WMAPE is nearly N%.
That's right - when weighted by their magnitude, the best performing model is still usually off by about N%.
For classification, especially multilabel, AUROC scores near one are impressive despite accuracy being low.

The best metrics to use are those that are actually interpretable: accuracy for classification, and mape (or wmape) for regression.
Pertinent Metrics are included on a per-study basis to emphasize when a 'good looking' scale-dependent metric is also intuitively good.

### Timing Results
Execution time is as reported by the unix `time` command using Chemprop version 1.6.1 on Python 3.8 and `fastprop` 0.0.0b0 on Python 3.11 and include the complete invocation of their respective program, i.e. `time chemprop_train` and `time fastprop train`.
The insignificant time spent manually collating Chemprop results (Chemprop does not natively support repetitions) is excluded.
The latter version of Python is known to be broadly faster (as well as the more updated versions of some dependencies that come with it), but the overall speed trends still stand.

TODO: report the timing results per-epoch rather than the overall time, which removes the annoyance of different numbers of epochs to converge.

also TODO: break out and specifically mention the time to generate features for `fastprop`

## Regression Datasets
See Table \ref{regression_results_table} for a summary of all the regression dataset results.
Especially noteworthy is the performance on the ESOL and PAH datasets, which dramatically surpass literature best and have only 55 datapoints, respectively.

Table: Summary of regression benchmark results. \label{regression_results_table}

+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|   Benchmark   | Samples (k)        |   Metric    |          Literature Best          | `fastprop` |        Chemprop         |   Speedup   |
+===============+====================+=============+===================================+============+=========================+=============+
|QM9            |~130                |L1           |0.0047$^a$                         |0.0063      |0.0081$^a$               |      ~      |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|OCELOTv1       |~25                 |L1           |0.128$^b$                          |0.148       |0.140$^b$                |      ~      |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|QM8            |~22                 |L1           |0.016$^a$                          |0.013       |0.019$^a$                |      ~      |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|ESOL           |~1.1                |L2           |0.55$^c$                           |0.57        |0.67$^c$                 |      ~      |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|FreeSolv       |~0.6                |L2           |1.29$^d$                           |1.06        |1.37$^d$                 |      ~      |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|Flash          |~0.6                |L2           |13.2$^e$                           |13.5        |21.2*                    | 5m43s/1m20s |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|YSI            |~0.4                |L1           |See $^f$                           |8.3/20.2    |21.8*                    | 4m3s/2m15s  |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|HOPV15$^g$     |~0.3                |L1           |1.32$^g$                           |1.44        |WIP                      |     WIP     |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|Fubrain        |~0.3                |L2           |0.44$^h$                           |0.19        |0.22*                    |  5m11s/54s  |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|PAH            |~0.06               |R2           |0.96$^i$                           |0.96        |0.75*                    |  36s/2m12s  |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+

a [@unimol] b [@mhnn] (summary result is geometric mean across tasks) c [@cmpnn] d [@deepdelta] e [@flash] f [@ysi] (original study did not report accuracy across entire dataset) g [@hopv15_subset] (uses a subset of the complete HOPV15 dataset) h [@fubrain] i [@pah] * These results were generated for this study.

### QM9
 - ~130k points
 - Originally described in Scientific Data [@qm9] and perhaps _the_ most established property prediction benchmark; quantum mechanics derived descriptors for all small molecules containing one to nine heavy atoms.
 - Data was retrieved from MoleculeNet [@moleculenet] in a more readily usable format.
 - For comparison, performance metrics are retrieved from the paper presenting the UniMol architecture [@unimol], a derivative of Chemprop that also encodes 3D information.
 - In this study they trained on only three especially difficult targets (homo, lumo, and gap) using 80/10/10 scaffold-based splitting.
 - `fastprop` gets an L1 average across the three tasks of 0.0062, which beats everything except the CMPNN which is also encoding 3D information.
<!-- [01/17/2024 02:59:17 PM fastprop.fastprop_core] INFO: Displaying validation results:
                              count      mean       std       min       25%       50%       75%       max
validation_mse_loss             2.0  0.062624  0.002877  0.060589  0.061606  0.062624  0.063641  0.064658
validation_mean_wmape           2.0  0.031053  0.000352  0.030804  0.030928  0.031053  0.031177  0.031302
validation_wmape_output_homo    2.0  0.022841  0.001046  0.022102  0.022472  0.022841  0.023211  0.023581
validation_wmape_output_lumo    2.0  0.037646  0.000530  0.037271  0.037459  0.037646  0.037834  0.038021
validation_wmape_output_gap     2.0  0.032671  0.000540  0.032289  0.032480  0.032671  0.032862  0.033053
validation_l1_avg               2.0  0.006285  0.000167  0.006167  0.006226  0.006285  0.006344  0.006403
validation_l1_output_homo       2.0  0.005193  0.000270  0.005003  0.005098  0.005193  0.005289  0.005384
validation_l1_output_lumo       2.0  0.005782  0.000033  0.005759  0.005770  0.005782  0.005794  0.005805
validation_l1_output_gap        2.0  0.007879  0.000264  0.007692  0.007786  0.007879  0.007973  0.008066
validation_rmse_avg             2.0  0.008689  0.000094  0.008623  0.008656  0.008689  0.008723  0.008756
validation_rmse_output_homo     2.0  0.007231  0.000269  0.007041  0.007136  0.007231  0.007326  0.007421
validation_rmse_output_lumo     2.0  0.008019  0.000250  0.007842  0.007931  0.008019  0.008107  0.008195
validation_rmse_output_gap      2.0  0.010819  0.000262  0.010634  0.010726  0.010819  0.010912  0.011004
[01/17/2024 02:59:17 PM fastprop.fastprop_core] INFO: Displaying testing results:
                        count      mean       std       min       25%       50%       75%       max
test_mse_loss             2.0  0.062043  0.003316  0.059698  0.060870  0.062043  0.063215  0.064388
test_mean_wmape           2.0 -0.017467  0.066770 -0.064680 -0.041073 -0.017467  0.006140  0.029747
test_wmape_output_homo    2.0  0.021891  0.000724  0.021379  0.021635  0.021891  0.022147  0.022403
test_wmape_output_lumo    2.0 -0.105799  0.203817 -0.249920 -0.177860 -0.105799 -0.033739  0.038321
test_wmape_output_gap     2.0  0.031509  0.002782  0.029542  0.030525  0.031509  0.032492  0.033476
test_l1_avg               2.0  0.006247  0.000357  0.005995  0.006121  0.006247  0.006374  0.006500
test_l1_output_homo       2.0  0.005246  0.000161  0.005132  0.005189  0.005246  0.005302  0.005359
test_l1_output_lumo       2.0  0.005687  0.000452  0.005367  0.005527  0.005687  0.005847  0.006007
test_l1_output_gap        2.0  0.007810  0.000457  0.007487  0.007648  0.007810  0.007971  0.008133
test_rmse_avg             2.0  0.008664  0.000429  0.008361  0.008513  0.008664  0.008816  0.008967
test_rmse_output_homo     2.0  0.007164  0.000041  0.007135  0.007150  0.007164  0.007179  0.007193
test_rmse_output_lumo     2.0  0.007932  0.000676  0.007454  0.007693  0.007932  0.008171  0.008410
test_rmse_output_gap      2.0  0.010896  0.000651  0.010436  0.010666  0.010896  0.011126  0.011356 -->


### OCELOTv1
 - ~25k points
 - Originally described by Bhat et al. [@ocelot]; quantum mechanics descriptors and optoelectronic properties of chromophoric small molecules.
 - Literature best model is the Molecular Hypergraph Neural Network (MHNN) [@mhnn] which specializes in the prediction of optoelectronic properties, and they have comparisons to a lot of other models.
 - Uses a 70/10/20 random split with three repetitions, final performance reported is the average across those three repetitions.
 - Performance for each metric is shown in Table \label{ocelot_results_table}.
 - `fastprop` 'trades places' with Chemprop, outperforming on some metrics and not on others. Overall geometrics mean of performance is ~6% lower. Both are worse than the MHNN across the board, though it is of course limited to this application.
 - TODO: add training time for Chemprop and possibly also the MHNN.

Table: Per-task OCELOT dataset results. \label{ocelot_results_table}

+--------+----------+----------+-----------+-------+-----------+
| Target | fastprop | Chemprop | fast/chem | MHNN  | fast/mhnn |
+========+==========+==========+===========+=======+===========+
| HOMO   | 0.326    | 0.330    | -0.013    | 0.306 | 0.064     |
+--------+----------+----------+-----------+-------+-----------+
| LUMO   | 0.283    | 0.289    | -0.022    | 0.258 | 0.096     |
+--------+----------+----------+-----------+-------+-----------+
| H-L    | 0.554    | 0.548    | 0.011     | 0.519 | 0.067     |
+--------+----------+----------+-----------+-------+-----------+
| VIE    | 0.205    | 0.191    | 0.075     | 0.178 | 0.153     |
+--------+----------+----------+-----------+-------+-----------+
| AIE    | 0.196    | 0.173    | 0.135     | 0.162 | 0.212     |
+--------+----------+----------+-----------+-------+-----------+
| CR1    | 0.056    | 0.055    | 0.011     | 0.053 | 0.049     |
+--------+----------+----------+-----------+-------+-----------+
| CR2    | 0.055    | 0.053    | 0.047     | 0.052 | 0.067     |
+--------+----------+----------+-----------+-------+-----------+
| HR     | 0.106    | 0.133    | -0.204    | 0.099 | 0.070     |
+--------+----------+----------+-----------+-------+-----------+
| VEA    | 0.190    | 0.157    | 0.211     | 0.138 | 0.377     |
+--------+----------+----------+-----------+-------+-----------+
| AEA    | 0.183    | 0.154    | 0.189     | 0.124 | 0.477     |
+--------+----------+----------+-----------+-------+-----------+
| AR1    | 0.054    | 0.051    | 0.066     | 0.050 | 0.088     |
+--------+----------+----------+-----------+-------+-----------+
| AR2    | 0.049    | 0.052    | -0.062    | 0.046 | 0.061     |
+--------+----------+----------+-----------+-------+-----------+
| ER     | 0.099    | 0.098    | 0.006     | 0.092 | 0.072     |
+--------+----------+----------+-----------+-------+-----------+
| S0S1   | 0.281    | 0.249    | 0.130     | 0.241 | 0.167     |
+--------+----------+----------+-----------+-------+-----------+
| S0T1   | 0.217    | 0.150    | 0.449     | 0.145 | 0.499     |
+--------+----------+----------+-----------+-------+-----------+
| G-Mean | 0.148    | 0.140    | 0.059     | 0.128 | 0.159     |
+--------+----------+----------+-----------+-------+-----------+

<!-- [01/27/2024 01:34:22 PM fastprop.fastprop_core] INFO: Displaying validation results:
                              count         mean           std       min       25%       50%           75%           max
validation_mse_loss             3.0     0.437923      0.031080  0.408489  0.421674  0.434859      0.452641      0.470422
validation_mean_mape            3.0   457.692561    791.853958  0.488916  0.515466  0.542015    686.294384   1372.046753
validation_mape_output_homo     3.0     0.043952      0.000813  0.043013  0.043713  0.044413      0.044421      0.044430
validation_mape_output_lumo     3.0  6860.512874  11878.665632  1.704524  2.362084  3.019645  10289.917049  20576.814453
validation_mape_output_hl       3.0     0.081019      0.000771  0.080131  0.080766  0.081401      0.081462      0.081524
validation_mape_output_vie      3.0     0.027475      0.000456  0.027014  0.027251  0.027488      0.027706      0.027924
validation_mape_output_aie      3.0     0.027116      0.000250  0.026878  0.026985  0.027092      0.027235      0.027377
validation_mape_output_cr1      3.0     0.320258      0.025930  0.290674  0.310865  0.331057      0.335050      0.339042
validation_mape_output_cr2      3.0     0.289624      0.006519  0.284585  0.285943  0.287301      0.292144      0.296986
validation_mape_output_hr       3.0     0.279492      0.007925  0.274156  0.274938  0.275721      0.282160      0.288599
validation_mape_output_vea      3.0     1.565336      0.746833  1.087140  1.135040  1.182939      1.804434      2.425929
validation_mape_output_aea      3.0     1.378040      0.607123  0.910523  1.034958  1.159394      1.611798      2.064202
validation_mape_output_ar1      3.0     0.232608      0.004380  0.228735  0.230231  0.231728      0.234544      0.237361
validation_mape_output_ar2      3.0     0.217496      0.005116  0.214423  0.214543  0.214664      0.219033      0.223402
validation_mape_output_er       3.0     0.216211      0.003481  0.213963  0.214207  0.214451      0.217336      0.220221
validation_mape_output_s0s1     3.0     0.083662      0.000929  0.082665  0.083241  0.083818      0.084160      0.084503
validation_mape_output_s0t1     3.0     0.113278      0.005188  0.108385  0.110557  0.112730      0.115724      0.118718
validation_mean_wmape           3.0     0.158738      0.001945  0.156521  0.158031  0.159542      0.159847      0.160153
validation_wmape_output_homo    3.0     0.044866      0.001004  0.043731  0.044480  0.045228      0.045434      0.045639
validation_wmape_output_lumo    3.0     0.220602      0.017266  0.207375  0.210835  0.214295      0.227215      0.240135
validation_wmape_output_hl      3.0     0.083119      0.001707  0.081269  0.082362  0.083455      0.084044      0.084633
validation_wmape_output_vie     3.0     0.027440      0.000501  0.026888  0.027226  0.027564      0.027716      0.027867
validation_wmape_output_aie     3.0     0.026985      0.000299  0.026651  0.026864  0.027078      0.027152      0.027226
validation_wmape_output_cr1     3.0     0.269902      0.005773  0.264228  0.266967  0.269707      0.272738      0.275770
validation_wmape_output_cr2     3.0     0.264344      0.003076  0.261077  0.262924  0.264771      0.265978      0.267184
validation_wmape_output_hr      3.0     0.253679      0.003357  0.251126  0.251778  0.252430      0.254956      0.257481
validation_wmape_output_vea     3.0     0.215136      0.011052  0.208607  0.208755  0.208904      0.218400      0.227897
validation_wmape_output_aea     3.0     0.181748      0.006029  0.177335  0.178313  0.179291      0.183954      0.188617
validation_wmape_output_ar1     3.0     0.218091      0.003922  0.215752  0.215827  0.215902      0.219260      0.222618
validation_wmape_output_ar2     3.0     0.205097      0.002949  0.202737  0.203444  0.204151      0.206277      0.208402
validation_wmape_output_er      3.0     0.202633      0.003069  0.200151  0.200917  0.201683      0.203874      0.206065
validation_wmape_output_s0s1    3.0     0.078706      0.001999  0.076408  0.078035  0.079661      0.079855      0.080049
validation_wmape_output_s0t1    3.0     0.088727      0.000619  0.088063  0.088446  0.088829      0.089059      0.089289
validation_l1_avg               3.0     0.190971      0.002980  0.187569  0.189898  0.192228      0.192672      0.193117
validation_l1_output_homo       3.0     0.327876      0.008082  0.318807  0.324656  0.330505      0.332411      0.334316
validation_l1_output_lumo       3.0     0.282298      0.003973  0.278997  0.280093  0.281188      0.283948      0.286708
validation_l1_output_hl         3.0     0.556048      0.014018  0.541137  0.549592  0.558047      0.563503      0.568959
validation_l1_output_vie        3.0     0.206442      0.004058  0.201922  0.204778  0.207634      0.208702      0.209771
validation_l1_output_aie        3.0     0.197372      0.002464  0.194591  0.196417  0.198242      0.198762      0.199282
validation_l1_output_cr1        3.0     0.056464      0.001262  0.055018  0.056025  0.057032      0.057187      0.057342
validation_l1_output_cr2        3.0     0.056339      0.000187  0.056125  0.056274  0.056423      0.056446      0.056469
validation_l1_output_hr         3.0     0.107138      0.001400  0.105522  0.106718  0.107914      0.107946      0.107978
validation_l1_output_vea        3.0     0.189927      0.000915  0.189228  0.189409  0.189591      0.190277      0.190962
validation_l1_output_aea        3.0     0.183977      0.001184  0.182842  0.183362  0.183882      0.184544      0.185205
validation_l1_output_ar1        3.0     0.054081      0.001246  0.052930  0.053419  0.053908      0.054656      0.055404
validation_l1_output_ar2        3.0     0.048891      0.000793  0.048298  0.048441  0.048584      0.049188      0.049792
validation_l1_output_er         3.0     0.098550      0.001752  0.097158  0.097566  0.097974      0.099246      0.100518
validation_l1_output_s0s1       3.0     0.280686      0.007707  0.271787  0.278418  0.285048      0.285135      0.285223
validation_l1_output_s0t1       3.0     0.218480      0.001132  0.217562  0.217847  0.218133      0.218939      0.219745
validation_rmse_avg             3.0     0.262428      0.007821  0.254018  0.258902  0.263785      0.266633      0.269481
validation_rmse_output_homo     3.0     0.449266      0.016604  0.430230  0.443520  0.456810      0.458784      0.460759
validation_rmse_output_lumo     3.0     0.381361      0.011412  0.369203  0.376120  0.383037      0.387439      0.391841
validation_rmse_output_hl       3.0     0.762951      0.034178  0.723505  0.752551  0.781597      0.782674      0.783751
validation_rmse_output_vie      3.0     0.279988      0.008204  0.270847  0.276628  0.282410      0.284559      0.286709
validation_rmse_output_aie      3.0     0.266181      0.004701  0.260797  0.264535  0.268272      0.268873      0.269473
validation_rmse_output_cr1      3.0     0.080651      0.003373  0.076759  0.079615  0.082471      0.082597      0.082722
validation_rmse_output_cr2      3.0     0.080876      0.000333  0.080521  0.080723  0.080926      0.081054      0.081182
validation_rmse_output_hr       3.0     0.148035      0.002827  0.144893  0.146867  0.148841      0.149606      0.150372
validation_rmse_output_vea      3.0     0.263131      0.007303  0.255435  0.259715  0.263994      0.266979      0.269964
validation_rmse_output_aea      3.0     0.252015      0.002363  0.249485  0.250940  0.252394      0.253280      0.254165
validation_rmse_output_ar1      3.0     0.076508      0.004346  0.073287  0.074036  0.074786      0.078119      0.081452
validation_rmse_output_ar2      3.0     0.070932      0.005591  0.067537  0.067706  0.067874      0.072630      0.077386
validation_rmse_output_er       3.0     0.136559      0.010008  0.130081  0.130795  0.131509      0.139798      0.148086
validation_rmse_output_s0s1     3.0     0.382504      0.015409  0.365091  0.376566  0.388041      0.391210      0.394379
validation_rmse_output_s0t1     3.0     0.305466      0.007914  0.300469  0.300903  0.301338      0.307964      0.314590
[01/27/2024 01:34:22 PM fastprop.fastprop_core] INFO: Displaying testing results:
                        count         mean          std       min       25%       50%          75%          max
test_mse_loss             3.0     0.414413     0.005850  0.408153  0.411748  0.415344     0.417542     0.419741
test_mean_mape            3.0    92.198119   158.913946  0.427868  0.449110  0.470352   138.083244   275.696136
test_mape_output_homo     3.0     0.043621     0.000397  0.043206  0.043434  0.043662     0.043829     0.043996
test_mape_output_lumo     3.0  1378.096812  2383.949202  1.659294  1.723099  1.786904  2066.315571  4130.844238
test_mape_output_hl       3.0     0.080582     0.001628  0.078841  0.079840  0.080839     0.081453     0.082067
test_mape_output_vie      3.0     0.027308     0.000335  0.027082  0.027116  0.027150     0.027421     0.027693
test_mape_output_aie      3.0     0.026981     0.000402  0.026579  0.026780  0.026980     0.027181     0.027383
test_mape_output_cr1      3.0     0.299770     0.002900  0.296461  0.298721  0.300980     0.301424     0.301869
test_mape_output_cr2      3.0     0.298791     0.017050  0.285199  0.289225  0.293251     0.305586     0.317922
test_mape_output_hr       3.0     0.281513     0.004548  0.276411  0.279697  0.282984     0.284063     0.285143
test_mape_output_vea      3.0     1.653483     0.619840  1.284069  1.295681  1.307293     1.838190     2.369087
test_mape_output_aea      3.0     1.294828     0.186082  1.085695  1.221190  1.356684     1.399395     1.442105
test_mape_output_ar1      3.0     0.236608     0.004589  0.232684  0.234085  0.235487     0.238570     0.241654
test_mape_output_ar2      3.0     0.220259     0.000343  0.219870  0.220129  0.220389     0.220453     0.220518
test_mape_output_er       3.0     0.218587     0.001987  0.216829  0.217509  0.218188     0.219465     0.220743
test_mape_output_s0s1     3.0     0.083439     0.001196  0.082061  0.083058  0.084056     0.084129     0.084202
test_mape_output_s0t1     3.0     0.109259     0.003797  0.106142  0.107144  0.108147     0.110817     0.113487
test_mean_wmape           3.0     0.158096     0.000491  0.157581  0.157864  0.158147     0.158353     0.158560
test_wmape_output_homo    3.0     0.044575     0.000320  0.044220  0.044442  0.044664     0.044752     0.044840
test_wmape_output_lumo    3.0     0.227521     0.006413  0.220539  0.224706  0.228873     0.231012     0.233150
test_wmape_output_hl      3.0     0.082861     0.001325  0.081355  0.082367  0.083380     0.083614     0.083848
test_wmape_output_vie     3.0     0.027293     0.000341  0.027040  0.027098  0.027157     0.027419     0.027681
test_wmape_output_aie     3.0     0.026852     0.000428  0.026414  0.026642  0.026870     0.027070     0.027271
test_wmape_output_cr1     3.0     0.266528     0.001639  0.264637  0.266014  0.267392     0.267473     0.267555
test_wmape_output_cr2     3.0     0.261581     0.002490  0.259954  0.260148  0.260342     0.262395     0.264448
test_wmape_output_hr      3.0     0.251739     0.002153  0.250158  0.250513  0.250869     0.252530     0.254192
test_wmape_output_vea     3.0     0.207057     0.003082  0.204025  0.205491  0.206957     0.208572     0.210188
test_wmape_output_aea     3.0     0.178846     0.001928  0.176941  0.177871  0.178801     0.179798     0.180796
test_wmape_output_ar1     3.0     0.220005     0.002169  0.218079  0.218830  0.219581     0.220968     0.222354
test_wmape_output_ar2     3.0     0.205686     0.000692  0.205180  0.205291  0.205402     0.205939     0.206475
test_wmape_output_er      3.0     0.203604     0.000542  0.203230  0.203294  0.203357     0.203792     0.204226
test_wmape_output_s0s1    3.0     0.078924     0.001010  0.077961  0.078399  0.078837     0.079406     0.079975
test_wmape_output_s0t1    3.0     0.088367     0.000982  0.087239  0.088036  0.088834     0.088932     0.089029
test_l1_avg               3.0     0.190300     0.002143  0.188212  0.189202  0.190192     0.191343     0.192494
test_l1_output_homo       3.0     0.325678     0.002570  0.322784  0.324673  0.326561     0.327126     0.327690
test_l1_output_lumo       3.0     0.282688     0.007269  0.274615  0.279675  0.284736     0.286725     0.288714
test_l1_output_hl         3.0     0.553871     0.008389  0.544205  0.551179  0.558153     0.558704     0.559254
test_l1_output_vie        3.0     0.205291     0.002621  0.203425  0.203793  0.204161     0.206224     0.208288
test_l1_output_aie        3.0     0.196372     0.003120  0.193229  0.194823  0.196417     0.197943     0.199469
test_l1_output_cr1        3.0     0.055605     0.000020  0.055587  0.055594  0.055600     0.055613     0.055627
test_l1_output_cr2        3.0     0.055465     0.000419  0.055011  0.055278  0.055545     0.055691     0.055838
test_l1_output_hr         3.0     0.105898     0.000687  0.105162  0.105585  0.106009     0.106265     0.106522
test_l1_output_vea        3.0     0.190082     0.004800  0.185522  0.187577  0.189632     0.192362     0.195091
test_l1_output_aea        3.0     0.183167     0.003304  0.180091  0.181421  0.182751     0.184705     0.186660
test_l1_output_ar1        3.0     0.054388     0.000906  0.053429  0.053968  0.054506     0.054868     0.055229
test_l1_output_ar2        3.0     0.048789     0.000344  0.048499  0.048599  0.048699     0.048934     0.049169
test_l1_output_er         3.0     0.098627     0.000701  0.097829  0.098367  0.098905     0.099026     0.099146
test_l1_output_s0s1       3.0     0.281265     0.003726  0.278538  0.279142  0.279746     0.282628     0.285510
test_l1_output_s0t1       3.0     0.217308     0.002805  0.214077  0.216407  0.218737     0.218924     0.219111
test_rmse_avg             3.0     0.256080     0.002886  0.253468  0.254531  0.255595     0.257387     0.259179
test_rmse_output_homo     3.0     0.443481     0.001835  0.442378  0.442422  0.442465     0.444032     0.445599
test_rmse_output_lumo     3.0     0.368955     0.007154  0.361032  0.365962  0.370893     0.372917     0.374941
test_rmse_output_hl       3.0     0.746982     0.005715  0.740997  0.744283  0.747569     0.749975     0.752382
test_rmse_output_vie      3.0     0.278618     0.005604  0.275241  0.275384  0.275527     0.280307     0.285087
test_rmse_output_aie      3.0     0.265085     0.005893  0.261201  0.261694  0.262187     0.267027     0.271866
test_rmse_output_cr1      3.0     0.079453     0.000489  0.079168  0.079170  0.079173     0.079595     0.080018
test_rmse_output_cr2      3.0     0.078827     0.000752  0.078236  0.078403  0.078570     0.079122     0.079674
test_rmse_output_hr       3.0     0.144776     0.000900  0.144002  0.144282  0.144563     0.145163     0.145763
test_rmse_output_vea      3.0     0.252127     0.006967  0.245065  0.248693  0.252320     0.255658     0.258995
test_rmse_output_aea      3.0     0.244011     0.004844  0.239268  0.241541  0.243814     0.246383     0.248951
test_rmse_output_ar1      3.0     0.074700     0.002408  0.072075  0.073646  0.075216     0.076012     0.076808
test_rmse_output_ar2      3.0     0.067193     0.000779  0.066514  0.066768  0.067022     0.067533     0.068043
test_rmse_output_er       3.0     0.131601     0.001616  0.129736  0.131116  0.132496     0.132534     0.132572
test_rmse_output_s0s1     3.0     0.371995     0.003422  0.369494  0.370046  0.370597     0.373246     0.375895
test_rmse_output_s0t1     3.0     0.293400     0.002714  0.290960  0.291938  0.292916     0.294620     0.296323
[01/27/2024 01:34:22 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.258>0.05. -->

### QM8
 - ~22k points
 - Predecessor to QM9 first described in 2015 [@qm8], follows the same procedure but includes only up to eight heavy atoms. Again used the data as prepared by MoleculeNet [@moleculenet].
 - Again comparing to the UniMol study [@unimol].
 - UniMol got 0.00156, `fastprop` got 0.0166, and Chemprop got 0.0190.
 - Of note is that this looks good, but by looking at the weighted mean absolute percentage error (wMAPE) (see Table \label{qm8_results_table}) we see a different story. Despite being among the 'most accurate' of existing literature models, `fastprop` has large errors.

Table: Per-task QM8 dataset results. \label{qm8_results_table}

+---------+-------+
| Metric  | wMAPE |
+=========+=======+
| E1-CC2  | 3.7%  |
+---------+-------+
| E2-CC2  | 3.4%  |
+---------+-------+
| f1-CC2  | 73.2% |
+---------+-------+
| f2-CC2  | 81.0% |
+---------+-------+
| E1-PBE0 | 3.6%  |
+---------+-------+
| E2-PBE0 | 3.4%  |
+---------+-------+
| f1-PBE0 | 68.6% |
+---------+-------+
| F2-PBE0 | 79.0% |
+---------+-------+
| E1-CAM  | 3.4%  |
+---------+-------+
| E2-CAM  | 3.1%  |
+---------+-------+
| f1-CAM  | 69.4% |
+---------+-------+
| f2-CAM  | 71.1% |
+---------+-------+
| Average | 38.6% |
+---------+-------+

<!-- [01/16/2024 10:23:52 AM fastprop.fastprop_core] INFO: Displaying validation results:
                                         count      mean       std       min       25%       50%       75%       max
validation_mse_loss                        2.0  0.200573  0.005814  0.196462  0.198518  0.200573  0.202629  0.204684
unitful_validation_mean_wmape              2.0  0.313240  0.007304  0.308075  0.310657  0.313240  0.315822  0.318404
unitful_validation_wmape_output_E1-CC2     2.0  0.028929  0.002025  0.027497  0.028213  0.028929  0.029645  0.030360
unitful_validation_wmape_output_E2-CC2     2.0  0.027545  0.001336  0.026601  0.027073  0.027545  0.028017  0.028490
unitful_validation_wmape_output_f1-CC2     2.0  0.583290  0.009481  0.576585  0.579938  0.583290  0.586642  0.589994
unitful_validation_wmape_output_f2-CC2     2.0  0.670674  0.021927  0.655169  0.662922  0.670674  0.678427  0.686179
unitful_validation_wmape_output_E1-PBE0    2.0  0.029330  0.002092  0.027850  0.028590  0.029330  0.030070  0.030810
unitful_validation_wmape_output_E2-PBE0    2.0  0.027064  0.001307  0.026140  0.026602  0.027064  0.027527  0.027989
unitful_validation_wmape_output_f1-PBE0    2.0  0.544427  0.000538  0.544047  0.544237  0.544427  0.544617  0.544807
unitful_validation_wmape_output_f2-PBE0    2.0  0.663062  0.014573  0.652758  0.657910  0.663062  0.668214  0.673366
unitful_validation_wmape_output_E1-CAM     2.0  0.027375  0.001852  0.026066  0.026721  0.027375  0.028030  0.028684
unitful_validation_wmape_output_E2-CAM     2.0  0.024817  0.001599  0.023686  0.024252  0.024817  0.025382  0.025948
unitful_validation_wmape_output_f1-CAM     2.0  0.524005  0.033400  0.500388  0.512197  0.524005  0.535814  0.547623
unitful_validation_wmape_output_f2-CAM     2.0  0.608355  0.027743  0.588738  0.598547  0.608355  0.618164  0.627972
unitful_validation_l1_avg                  2.0  0.012361  0.000345  0.012116  0.012239  0.012361  0.012483  0.012605
unitful_validation_l1_output_E1-CC2        2.0  0.006372  0.000434  0.006065  0.006219  0.006372  0.006526  0.006679
unitful_validation_l1_output_E2-CC2        2.0  0.006857  0.000319  0.006632  0.006744  0.006857  0.006970  0.007082
unitful_validation_l1_output_f1-CC2        2.0  0.013471  0.000090  0.013407  0.013439  0.013471  0.013502  0.013534
unitful_validation_l1_output_f2-CC2        2.0  0.027845  0.000268  0.027656  0.027750  0.027845  0.027940  0.028035
unitful_validation_l1_output_E1-PBE0       2.0  0.006366  0.000447  0.006050  0.006208  0.006366  0.006524  0.006682
unitful_validation_l1_output_E2-PBE0       2.0  0.006561  0.000306  0.006345  0.006453  0.006561  0.006669  0.006778
unitful_validation_l1_output_f1-PBE0       2.0  0.011277  0.000167  0.011159  0.011218  0.011277  0.011336  0.011395
unitful_validation_l1_output_f2-PBE0       2.0  0.021629  0.000048  0.021595  0.021612  0.021629  0.021646  0.021663
unitful_validation_l1_output_E1-CAM        2.0  0.005944  0.000393  0.005666  0.005805  0.005944  0.006083  0.006222
unitful_validation_l1_output_E2-CAM        2.0  0.006071  0.000387  0.005798  0.005934  0.006071  0.006208  0.006345
unitful_validation_l1_output_f1-CAM        2.0  0.012031  0.000483  0.011690  0.011861  0.012031  0.012202  0.012373
unitful_validation_l1_output_f2-CAM        2.0  0.023903  0.001135  0.023101  0.023502  0.023903  0.024305  0.024706
unitful_validation_rmse_avg                2.0  0.022372  0.000161  0.022258  0.022315  0.022372  0.022429  0.022486
unitful_validation_rmse_output_E1-CC2      2.0  0.008769  0.000628  0.008325  0.008547  0.008769  0.008991  0.009213
unitful_validation_rmse_output_E2-CC2      2.0  0.009572  0.000393  0.009294  0.009433  0.009572  0.009711  0.009851
unitful_validation_rmse_output_f1-CC2      2.0  0.030954  0.000267  0.030765  0.030859  0.030954  0.031048  0.031143
unitful_validation_rmse_output_f2-CC2      2.0  0.047789  0.001568  0.046680  0.047235  0.047789  0.048343  0.048897
unitful_validation_rmse_output_E1-PBE0     2.0  0.008498  0.000638  0.008047  0.008273  0.008498  0.008724  0.008950
unitful_validation_rmse_output_E2-PBE0     2.0  0.008659  0.000346  0.008414  0.008537  0.008659  0.008782  0.008904
unitful_validation_rmse_output_f1-PBE0     2.0  0.027808  0.002149  0.026288  0.027048  0.027808  0.028567  0.029327
unitful_validation_rmse_output_f2-PBE0     2.0  0.038761  0.000817  0.038183  0.038472  0.038761  0.039050  0.039338
unitful_validation_rmse_output_E1-CAM      2.0  0.008049  0.000568  0.007648  0.007849  0.008049  0.008250  0.008451
unitful_validation_rmse_output_E2-CAM      2.0  0.008016  0.000489  0.007670  0.007843  0.008016  0.008189  0.008362
unitful_validation_rmse_output_f1-CAM      2.0  0.029131  0.000161  0.029017  0.029074  0.029131  0.029188  0.029245
unitful_validation_rmse_output_f2-CAM      2.0  0.042460  0.000357  0.042208  0.042334  0.042460  0.042586  0.042713
[01/16/2024 10:23:52 AM fastprop.fastprop_core] INFO: Displaying testing results:
                                   count      mean       std       min       25%       50%       75%       max
test_mse_loss                        2.0  0.221585  0.001861  0.220269  0.220927  0.221585  0.222243  0.222901
unitful_test_mean_wmape              2.0  0.308316  0.006083  0.304015  0.306166  0.308316  0.310467  0.312617
unitful_test_wmape_output_E1-CC2     2.0  0.028781  0.001292  0.027868  0.028325  0.028781  0.029238  0.029695
unitful_test_wmape_output_E2-CC2     2.0  0.027396  0.000766  0.026854  0.027125  0.027396  0.027666  0.027937
unitful_test_wmape_output_f1-CC2     2.0  0.580761  0.012467  0.571945  0.576353  0.580761  0.585169  0.589577
unitful_test_wmape_output_f2-CC2     2.0  0.683063  0.023970  0.666113  0.674588  0.683063  0.691538  0.700013
unitful_test_wmape_output_E1-PBE0    2.0  0.029097  0.001363  0.028133  0.028615  0.029097  0.029579  0.030061
unitful_test_wmape_output_E2-PBE0    2.0  0.026556  0.001226  0.025689  0.026122  0.026556  0.026989  0.027422
unitful_test_wmape_output_f1-PBE0    2.0  0.501698  0.009261  0.495150  0.498424  0.501698  0.504972  0.508246
unitful_test_wmape_output_f2-PBE0    2.0  0.666396  0.035308  0.641429  0.653912  0.666396  0.678879  0.691363
unitful_test_wmape_output_E1-CAM     2.0  0.027505  0.001035  0.026774  0.027140  0.027505  0.027871  0.028237
unitful_test_wmape_output_E2-CAM     2.0  0.024964  0.001391  0.023980  0.024472  0.024964  0.025455  0.025947
unitful_test_wmape_output_f1-CAM     2.0  0.485728  0.031679  0.463328  0.474528  0.485728  0.496928  0.508129
unitful_test_wmape_output_f2-CAM     2.0  0.617848  0.003311  0.615507  0.616677  0.617848  0.619019  0.620189
unitful_test_l1_avg                  2.0  0.012821  0.000111  0.012742  0.012781  0.012821  0.012860  0.012899
unitful_test_l1_output_E1-CC2        2.0  0.006333  0.000271  0.006141  0.006237  0.006333  0.006429  0.006525
unitful_test_l1_output_E2-CC2        2.0  0.006813  0.000192  0.006678  0.006746  0.006813  0.006881  0.006949
unitful_test_l1_output_f1-CC2        2.0  0.013935  0.000044  0.013904  0.013920  0.013935  0.013951  0.013966
unitful_test_l1_output_f2-CC2        2.0  0.030965  0.001010  0.030252  0.030608  0.030965  0.031322  0.031679
unitful_test_l1_output_E1-PBE0       2.0  0.006308  0.000280  0.006110  0.006209  0.006308  0.006407  0.006507
unitful_test_l1_output_E2-PBE0       2.0  0.006435  0.000287  0.006232  0.006333  0.006435  0.006536  0.006638
unitful_test_l1_output_f1-PBE0       2.0  0.010951  0.000118  0.010867  0.010909  0.010951  0.010992  0.011034
unitful_test_l1_output_f2-PBE0       2.0  0.022884  0.000532  0.022508  0.022696  0.022884  0.023072  0.023260
unitful_test_l1_output_E1-CAM        2.0  0.005963  0.000206  0.005817  0.005890  0.005963  0.006036  0.006108
unitful_test_l1_output_E2-CAM        2.0  0.006104  0.000336  0.005866  0.005985  0.006104  0.006222  0.006341
unitful_test_l1_output_f1-CAM        2.0  0.012307  0.000473  0.011973  0.012140  0.012307  0.012474  0.012641
unitful_test_l1_output_f2-CAM        2.0  0.024849  0.000393  0.024571  0.024710  0.024849  0.024988  0.025127
unitful_test_rmse_avg                2.0  0.023458  0.000069  0.023409  0.023433  0.023458  0.023482  0.023507
unitful_test_rmse_output_E1-CC2      2.0  0.009032  0.000668  0.008560  0.008796  0.009032  0.009268  0.009504
unitful_test_rmse_output_E2-CC2      2.0  0.009409  0.000341  0.009168  0.009289  0.009409  0.009530  0.009651
unitful_test_rmse_output_f1-CC2      2.0  0.033309  0.000596  0.032888  0.033098  0.033309  0.033519  0.033730
unitful_test_rmse_output_f2-CC2      2.0  0.053217  0.001793  0.051949  0.052583  0.053217  0.053851  0.054485
unitful_test_rmse_output_E1-PBE0     2.0  0.008815  0.000729  0.008299  0.008557  0.008815  0.009073  0.009330
unitful_test_rmse_output_E2-PBE0     2.0  0.008670  0.000515  0.008306  0.008488  0.008670  0.008852  0.009034
unitful_test_rmse_output_f1-PBE0     2.0  0.026180  0.000683  0.025697  0.025939  0.026180  0.026421  0.026663
unitful_test_rmse_output_f2-PBE0     2.0  0.041940  0.001727  0.040718  0.041329  0.041940  0.042550  0.043161
unitful_test_rmse_output_E1-CAM      2.0  0.008418  0.000657  0.007953  0.008186  0.008418  0.008650  0.008883
unitful_test_rmse_output_E2-CAM      2.0  0.008094  0.000492  0.007746  0.007920  0.008094  0.008268  0.008442
unitful_test_rmse_output_f1-CAM      2.0  0.030481  0.001667  0.029303  0.029892  0.030481  0.031071  0.031660
unitful_test_rmse_output_f2-CAM      2.0  0.043930  0.001995  0.042519  0.043224  0.043930  0.044635  0.045340 -->

### ESOL
 - ~1.1k points
 - First described in 2004 [@esol] and has since become a critically important benchmark for QSPR/molecular proeprty prediction studies; includes molecular structure for some simple organic molecules are their corresponding experimentally measured free energy of solvation.
 - Reference against the CMPNN paper [@cmpnn], but speficially the amended results shown on their GitHub page: https://github.com/SY575/CMPNN/blob/b647df22ec8fde81785c5a86138ac1efd9ccf9c1/README.md and TODO: run both Chemprop and CMPNN on this dataset again with repeitions (and withold data, unlike in the CMPNN study)
 - `fastprop` achieves RMSE of 0.566 kcal/mol on an 80/10/10 random split, on a different split the CMPNN and Chemprop get 0.547 and 0.665, respectively.
<!-- [01/16/2024 11:55:48 AM fastprop.fastprop_core] INFO: Displaying validation results:
                          count      mean       std       min       25%       50%       75%       max
validation_mse_loss         2.0  0.072686  0.000135  0.072591  0.072639  0.072686  0.072734  0.072781
unitful_validation_wmape    2.0  0.119234  0.017337  0.106975  0.113104  0.119234  0.125363  0.131493
unitful_validation_l1       2.0  0.415738  0.031417  0.393523  0.404630  0.415738  0.426846  0.437953
unitful_validation_rmse     2.0  0.564958  0.000524  0.564587  0.564773  0.564958  0.565143  0.565328
[01/16/2024 11:55:48 AM fastprop.fastprop_core] INFO: Displaying testing results:
                    count      mean       std       min       25%       50%       75%       max
test_mse_loss         2.0  0.072983  0.001132  0.072182  0.072583  0.072983  0.073383  0.073783
unitful_test_wmape    2.0  0.117443  0.011269  0.109475  0.113459  0.117443  0.121428  0.125412
unitful_test_l1       2.0  0.427814  0.005198  0.424139  0.425976  0.427814  0.429652  0.431489
unitful_test_rmse     2.0  0.566101  0.004390  0.562997  0.564549  0.566101  0.567653  0.569205
[01/16/2024 11:55:48 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing L1 yielded p value of p=0.645>0.05. -->

### FreeSolv
 - ~0.6k points
 - Another classical benchmark, but among the smallest recognized.
 - Reference results are from the paper presenting the DeepDelta architecture [@deepdelta], which also includes results for other architectures. They do not use a validation set, and report the performance on 10 fold cross-validation, meaning each time the model trains on 90% of the data, has no validation data for early stopping, and then is evaluated based on the performance on the remaining 10%. It is not explained in the paper, but is shown in the repository that the smaller set is witheld.
 - Chemprop got 1.372 RMSE, DeepDelta got 1.290, and `fastprop` gets 1.06, which is approaching the conventionally agreed upon irreducible error of this dataset.

<!-- [01/14/2024 02:25:31 PM fastprop.fastprop_core] INFO: Displaying validation results:
                          count      mean       std       min       25%       50%       75%       max
validation_mse_loss         8.0  0.077811  0.021239  0.045162  0.066696  0.076949  0.082898  0.117327
unitful_validation_wmape    8.0  0.150100  0.028615  0.105746  0.126448  0.156406  0.172733  0.184089
unitful_validation_l1       8.0  0.689524  0.069588  0.607875  0.634752  0.682801  0.723501  0.816781
unitful_validation_rmse     8.0  1.063753  0.146103  0.817076  0.992939  1.066451  1.106214  1.316967
[01/14/2024 02:25:31 PM fastprop.fastprop_core] INFO: Displaying testing results:
                    count      mean       std       min       25%       50%       75%       max
test_mse_loss         8.0  0.078353  0.028153  0.048043  0.052906  0.074801  0.101876  0.114290
unitful_test_wmape    8.0  0.134703  0.010428  0.120989  0.126518  0.135097  0.141009  0.151088
unitful_test_l1       8.0  0.657234  0.073684  0.559451  0.595808  0.666151  0.720486  0.737936
unitful_test_rmse     8.0  1.060747  0.194462  0.842738  0.884339  1.045691  1.226791  1.299813 -->

### Flash
 - ~0.6k points (total dataset is ~1k, but only a subset have the flash point)
 - Assembled and fitted to by Saldana and coauthors [@flash], includes primarily alkanes and some oxygen containing compounds and their literature-reported flash point.
 - The reference study reports the performance on only one fold but manuualy confirms that the distribution of poitns in the three splits follow the parent dataset. Used a 70/20/10 random split, as is done here.
 - Reference study achieved an RMSE of 13.2, an MAE of 8.4, and an MAPE of 2.5% using a complex multi-model ensembling method.
 - `fastprop` achieves 13.5, 9.0, and 2.7% with a total execution time of 1m20s (9s for descriptor calculation).
 - Chemprop achieves an RMSE of 21.2 and an MAE of 13.8 (Chemprop does not report MAPE), taking 5m44s to do so.

<!-- fastprop:
[01/24/2024 03:05:17 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count       mean       std        min        25%        50%        75%        max
validation_mse_loss    4.0   0.087003  0.024714   0.058950   0.073606   0.085720   0.099118   0.117622
validation_mape        4.0   0.027867  0.004092   0.024257   0.025059   0.026918   0.029725   0.033375
validation_wmape       4.0   0.029241  0.003804   0.026191   0.026319   0.028269   0.031191   0.034236
validation_l1          4.0   9.516333  1.362214   8.333776   8.489315   9.234716  10.261735  11.262125
validation_mdae        4.0   5.761031  0.973835   5.056880   5.269190   5.393213   5.885053   7.200817
validation_rmse        4.0  16.437566  2.365327  13.635116  15.208911  16.427479  17.656134  19.260189
[01/24/2024 03:05:17 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count       mean       std        min        25%        50%        75%        max
test_mse_loss    4.0   0.058840  0.018530   0.037408   0.047539   0.059112   0.070412   0.079728
test_mape        4.0   0.026840  0.005151   0.022082   0.023411   0.025781   0.029210   0.033716
test_wmape       4.0   0.027709  0.004336   0.023308   0.024553   0.027416   0.030573   0.032696
test_l1          4.0   8.959434  1.304854   7.811850   7.894664   8.784337   9.849107  10.457213
test_mdae        4.0   5.940652  1.709536   4.338257   4.868059   5.583856   6.656449   8.256640
test_rmse        4.0  13.490086  2.186714  10.861741  12.219429  13.620798  14.891455  15.857007
[01/24/2024 03:05:17 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.117>0.05.

Took 9 seconds to calculate descriptors

real    1m19.858s
user    2m0.425s
sys     0m30.173s


chemprop:
real    5m43.666s
user    5m38.506s
sys     0m16.395s

Task,Mean mae,Standard deviation mae,Fold 0 mae,Mean rmse,Standard deviation rmse,Fold 0 rmse
flash,11.623691196044842,0.0,11.623691196044842,19.88638811200749,0.0,19.88638811200749
flash,15.65630420287718,0.0,15.65630420287718,23.552562784757406,0.0,23.552562784757406
flash,15.43755266509213,0.0,15.43755266509213,22.464650219929577,0.0,22.464650219929577
flash,12.421265601066294,0.0,12.421265601066294,18.784380177702204,0.0,18.784380177702204

for MAE:
Mean	13.784703416270112
Standard Error	1.031331527510691

for RMSE:
Mean	21.17199532359917
Standard Error	1.1064790292313673 -->

### YSI
 - ~0.4k compounds
 - Assembled by Das an coauthors [@ysi] from a collection of other smaller datasets, maps molecular structure to a unified-scale Yield Sooting Index, a molecular property of interest to the combustion community.
 - Reference study performs leave-one-out cross validation, allowing the model to train on all but one of the data points (effective training size >99%) and reported accuracy is the average performance on all individual points (which is also data leakage). They report Median Absolute Deviation (MdAE) on the "high scale" and "low scale" molecules (see the paper for more details) of 28.6 and 2.35, respectively.
 - `fastprop`, using a more typical 80/10/10 random split and 4 repeitions, achieves an MdAE of 8.29 across the entire dataset, an RMSE of 36.5, and an MAE of 20.2 in 2m15s (13 seconds for feature generation).
 - Chemprop on the same splitting approach achieves an MAE of 21.8 and an RMSE of 48.4 in 4m3s.
 - While an "apples-to-apples" comparison is difficult owing to the lack of standard error metric reporting, the MdAE of `fastprop` is seemingly competitive with the bespoke model in the study. MAE is similar between `fastprop` and Chemprop, though RMSE shows that Chemprop is significantly less accurate on 'outlier' compounds than `fastprop`.

<!-- chemprop:
Task,Mean mae,Standard deviation mae,Fold 0 mae,Mean rmse,Standard deviation rmse,Fold 0 rmse
YSI,25.01130964102828,0.0,25.01130964102828,53.687524865608594,0.0,53.687524865608594
YSI,22.742283113235967,0.0,22.742283113235967,67.50147227703528,0.0,67.50147227703528
YSI,19.103583744925203,0.0,19.103583744925203,40.036306419637036,0.0,40.036306419637036
YSI,20.190894554613426,0.0,20.190894554613426,32.30321545955367,0.0,32.30321545955367

for mae: 
Mean	21.762017763450718
Standard Error	1.3245916801629232

for rmse:
Mean	48.38212975545864
Standard Error	7.756076799409034

real    4m3.257s
user    3m56.623s
sys     0m16.604s

fastprop:
feature generation took 13 seconds
[01/24/2024 02:07:47 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count       mean        std        min        25%        50%        75%        max
validation_mse_loss    4.0   0.020897   0.017674   0.008690   0.008998   0.014284   0.026183   0.046330
validation_mape        4.0   0.186855   0.049963   0.131290   0.151384   0.192094   0.227565   0.231940
validation_wmape       4.0   0.108513   0.032333   0.074570   0.084766   0.108441   0.132188   0.142600
validation_l1          4.0  18.396958   5.283186  14.691376  14.714375  16.492038  20.174621  25.912382
validation_mdae        4.0   7.258557   1.945397   5.755852   6.053782   6.610182   7.814957  10.058012
validation_rmse        4.0  37.921621  15.931041  26.023083  26.477633  32.788874  44.232862  60.085655
[01/24/2024 02:07:47 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count       mean       std        min        25%        50%        75%        max
test_mse_loss    4.0   0.017690  0.007267   0.008951   0.013776   0.017897   0.021811   0.026017
test_mape        4.0   0.206968  0.049201   0.142847   0.182325   0.215643   0.240286   0.253739
test_wmape       4.0   0.102447  0.009450   0.093839   0.097580   0.100054   0.104921   0.115840
test_l1          4.0  20.241003  4.845866  15.046371  16.646652  20.728768  24.323120  24.460104
test_mdae        4.0   8.287672  1.886135   6.668498   6.888764   7.866325   9.265233  10.749538
test_rmse        4.0  36.485026  7.946628  26.409878  32.570827  37.251917  41.166116  45.026394
[01/24/2024 02:07:47 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.877>0.05.

real    2m14.937s
user    1m52.823s
sys     1m1.971s -->

### HOPV15 Subset
 - ~0.3k Organic Photovoltaic Compounds
 - This study [@hopv15_subset] trained on a subset of HOPV15 [cite original hopv15 study] matching some criteria described in the paper, and they saw SVR give an accuracy 1.32 L1 averaged across 5 randomly selected folds of 60/20/20 data splits.
 - Due to the very small size of this dataset and as a point of comparison to the reference study, 5 repetitions are used.
 - `fastprop` reports an average L1 of 1.44 across the same number of repetitions, though the standard deviation is 0.18 and accuracy across the repeitions ranged from as low as 1.12 to as high as 1.57.
 - WIP on detailed comparison with Chemprop.

<!-- [01/23/2024 11:22:14 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_mse_loss    5.0  0.769138  0.193359  0.524239  0.630217  0.783427  0.923952  0.983856
validation_wmape       5.0  0.468045  0.060016  0.395862  0.422238  0.481899  0.492871  0.547358
validation_l1          5.0  1.619807  0.196770  1.342793  1.510417  1.649071  1.770630  1.826126
validation_rmse        5.0  1.948286  0.251272  1.619144  1.775276  1.979340  2.149541  2.218128
[01/23/2024 11:22:14 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_mse_loss    5.0  0.636690  0.123323  0.419931  0.652266  0.695075  0.706868  0.709310
test_wmape       5.0  0.409631  0.074432  0.297398  0.393970  0.409640  0.449684  0.497463
test_l1          5.0  1.443413  0.184273  1.120585  1.486124  1.494168  1.536834  1.579354
test_rmse        5.0  1.776624  0.185695  1.449138  1.806064  1.864390  1.880140  1.883385
[01/23/2024 11:22:14 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.254>0.05. -->

### Fubrain
 - ~0.3k small molecule drugs
 - First described by Esaki and coauthors, the fubrain dataset is a collection of small molecule drugs and their corresponding unbound fraction in the brain, a critical metric for drug development [@fubrain].
 - The study that first generated this dataset got 0.44 RMSE with 10-fold CV (using Mordred descriptors, like `fastprop`, but with a different model architecture), and then DeepDelta reported 0.830+/-0.023 RMSE using the same approach.
 - In both studies 10-fold cross validation using the default settings in scikit-learn (CITE, then cite specific KFold page) will result in a 90/10 train/test split, and a separate holdout set is identified afterward for reporting performance.
 - OOB performance is twice as good as the reference study and dramatically better than DeepDelta, which also suffers from scaling issues: `fastprop` RMSE 0.19 (54s to train, 30s descriptor generation), Chemprop RMSE 0.22 (5m11s to train).
<!-- [01/23/2024 12:18:36 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_mse_loss    4.0  0.459971  0.195559  0.198392  0.394382  0.486529  0.552118  0.668435
validation_wmape       4.0  0.625896  0.158742  0.455302  0.528967  0.612142  0.709071  0.823997
validation_l1          4.0  0.121224  0.025237  0.086006  0.112100  0.127828  0.136952  0.143232
validation_rmse        4.0  0.173373  0.041073  0.116233  0.161759  0.181953  0.193566  0.213353
[01/23/2024 12:18:36 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_mse_loss    4.0  0.528621  0.127916  0.411545  0.422581  0.522131  0.628171  0.658675
test_wmape       4.0  0.696168  0.032418  0.668985  0.678049  0.686482  0.704602  0.742725
test_l1          4.0  0.126220  0.012108  0.111144  0.119120  0.128339  0.135438  0.137059
test_rmse        4.0  0.188680  0.023042  0.167408  0.169634  0.187761  0.206807  0.211790
[01/23/2024 12:18:36 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.540>0.05. -->

<!-- Chemprop:
RMSE:
	Column 1
Mean	0.22348309487917928
Standard Error	0.01820014751019536
Median	0.2153173006272221
Mode	#N/A
Standard Deviation	0.03640029502039072
Sample Variance	0.0013249814775714815
Kurtosis	2.2720557401483568
Skewness	1.2428089327791678
Range	0.08589779625109051
Minimum	0.1886999910055912
Maximum	0.2745977872566817
Sum	0.8939323795167171
Count	4

MAE:
	Column 1
Mean	0.1580851177817399
Standard Error	0.014966396991876716
Median	0.15069989368176856
Mode	#N/A
Standard Deviation	0.029932793983753432
Sample Variance	0.0008959721556738256
Kurtosis	2.567106902165488
Skewness	1.3599759527202175
Range	0.07027859859775615
Minimum	0.13033104258283315
Maximum	0.2006096411805893
Sum	0.6323404711269596
Count	4
real    5m11.395s
user    5m8.992s
sys     0m15.378s


Timing for fastprop: descriptor generation was 30 seconds, total was 54s. -->

 - Chemprop's performance on this dataset is still much better than the reference dataset, but worse than `fastprop`. More importantly, Chemprop also significantly overfits to the validation/training data, as can be seen by the severly diminished performance on the witheld testing set.
 - As stated earlier, this is likely to blame on the 'start from almost-nothing' strategy in Chemprop, where the initial representation is only a combination of simple atomic descrtiptors based on the connectivity and not the human-designed high-level molecular descriptors used in `fastprop`.
This will also become clear in the later ARA classification dataset (see [ARA](#ara)).


### PAHs
 - Only 55 molecules.
 - Originally compiled by Arockiaraj et al. [@pah], contains water/octanol partition coefficients (logP, among some other properties for some molecules) for polyclicic aromatic hydrocarbons ranging from napthalene up to circumcoronene.
 - Reference study designed a new set of molecular descriptors that show a strong correlation to logP, with correlation coefficients between 96 to 99%.
 - `fastprop`, using a 4-repetition 80/10/10 random split, is able to achieve a correlation coefficient of 0.960 +/- 0.027 in 2m12s (17s descriptor calculation). This is an L1 error of 0.230 and an MAPE of 3.1%.
 - Chemprop using the same scheme achieves only 0.746 +/- 0.085 in 36s. This is an L1 error 1.07 +/1 0.20.
 - `fastprop` is able to correlate structure directly to property without expert knowledge in only minutes while having only 44 training points.

<!-- fastprop: about 17 seconds calculating descriptors

[01/25/2024 11:26:04 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_mse_loss    4.0  0.010593  0.006000  0.003780  0.008306  0.010096  0.012382  0.018399
validation_r2          4.0  0.967020  0.044817  0.900350  0.961918  0.985614  0.990716  0.996501
validation_mape        4.0  0.023903  0.008261  0.013881  0.018817  0.025077  0.030163  0.031579
validation_wmape       4.0  0.022955  0.006833  0.014001  0.019613  0.024176  0.027519  0.029467
validation_l1          4.0  0.175690  0.051109  0.104058  0.165468  0.186700  0.196922  0.225304
validation_mdae        4.0  0.146728  0.045694  0.086612  0.123915  0.156485  0.179298  0.187331
validation_rmse        4.0  0.213124  0.064931  0.131663  0.192047  0.215169  0.236246  0.290493
[01/25/2024 11:26:04 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_mse_loss    4.0  0.022717  0.015894  0.009908  0.011239  0.018316  0.029795  0.044327
test_r2          4.0  0.960160  0.027347  0.922591  0.948588  0.968076  0.979649  0.981897
test_mape        4.0  0.031277  0.012971  0.023626  0.024545  0.025400  0.032131  0.050680
test_wmape       4.0  0.027624  0.004221  0.023829  0.024298  0.027011  0.030337  0.032647
test_l1          4.0  0.229951  0.052396  0.177245  0.192995  0.224651  0.261607  0.293254
test_mdae        4.0  0.174251  0.047216  0.130138  0.141541  0.165600  0.198310  0.235666
test_rmse        4.0  0.308459  0.109823  0.213177  0.226902  0.284881  0.366438  0.450899
[01/25/2024 11:26:04 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.186>0.05.

real    2m11.969s
user    3m50.216s
sys     0m14.508s

chemprop

Task,Mean mae,Standard deviation mae,Fold 0 mae,Mean rmse,Standard deviation rmse,Fold 0 rmse,Mean r2,Standard deviation r2,Fold 0 r2
log_p,1.583810856149771,0.0,1.583810856149771,2.2727142211004185,0.0,2.2727142211004185,0.502462086273753,0.0,0.502462086273753
log_p,1.1298430017715362,0.0,1.1298430017715362,1.5680174445057777,0.0,1.5680174445057777,0.7729733835836915,0.0,0.7729733835836915
log_p,0.9642980585445118,0.0,0.9642980585445118,1.0899792588299095,0.0,1.0899792588299095,0.8170510411737727,0.0,0.8170510411737727
log_p,0.5977449950615908,0.0,0.5977449950615908,0.749184197908931,0.0,0.749184197908931,0.8928415401318607,0.0,0.8928415401318607

mae:
Mean	1.0689242278818525
Standard Error	0.20448631611361848

rmse:
Mean	1.4199737805862591
Standard Error	0.33014368512470244

r2:
Mean	0.7463320127907694
Standard Error	0.08497476403531372

real    0m36.336s
user    0m32.538s
sys     0m15.281s -->

## Classification Datasets
See Table \ref{classification_results_table} for a summary of all the classification dataset results.
Especially noteworthy is the performance on QuantumScents dataset, which outperforms the best literature result.

Table: Summary of regression benchmark results. \label{classification_results_table}

+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|Benchmark              |Samples (k)         |Metric      |Literature Best             |`fastprop`  |Chemprop                    |   Speedup   |
+=======================+====================+============+============================+============+============================+=============+
|binary-HIV             | ~41                |AUROC       |0.81 $^a$                   |0.81        |0.77 $^a$                   |      ~      |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|ternary-HIV            |~41                 |AUROC       |~                           |0.83        |WIP                         |      ~      |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|QuantumScents          |~3.5                |AUROC       |0.88 $^b$                   |0.91        |0.85 $^b$                   |      ~      |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|SIDER                  |~1.4                |AUROC       |0.67 $^c$                   |0.66        |0.57 $^c$                   |      ~      |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|Pgp                    |~1.3                |AUROC       |WIP                         |0.93        |WIP                         |      ~      |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|ARA                    |~0.8                |AUROC       |0.95 $^d$                   |0.95        |0.90*                       | 16m54s/2m7s |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+

a [@unimol] b [@quantumscents] c [@cmpnn] d [@ara] * These results were generated for this study.

### HIV Inhibition
~41k points

This dataset is unique in that the labels in the original study include three possible classes (a _multiclass_) regression problem whereas the most common reported metric is instead lumping positive and semi-positive labels into a single class to reduce the task to _binary_ classification.
Both metrics are reported here.

#### Multiclass
OOB:
<!-- [01/18/2024 09:50:50 AM fastprop.fastprop_core] INFO: Displaying validation results:
                       count      mean       std       min       25%       50%       75%       max
validation_kldiv_loss    2.0  0.154615  0.019345  0.140936  0.147776  0.154615  0.161455  0.168295
validation_auroc         2.0  0.798816  0.006996  0.793869  0.796342  0.798816  0.801289  0.803762
[01/18/2024 09:50:50 AM fastprop.fastprop_core] INFO: Displaying testing results:
                 count      mean       std       min       25%       50%       75%       max
test_kldiv_loss    2.0  0.136559  0.041120  0.107482  0.122021  0.136559  0.151097  0.165635
test_auroc         2.0  0.828546  0.010771  0.820930  0.824738  0.828546  0.832354  0.836163
[01/18/2024 09:50:50 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.082>0.05. -->

#### Binary
<!-- [01/18/2024 10:54:11 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    2.0  0.135136  0.016572  0.123418  0.129277  0.135136  0.140995  0.146854
validation_f1          2.0  0.293810  0.004932  0.290323  0.292066  0.293810  0.295554  0.297297
validation_auroc       2.0  0.772978  0.010357  0.765655  0.769316  0.772978  0.776640  0.780301
[01/18/2024 10:54:11 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    2.0  0.116358  0.026022  0.097957  0.107157  0.116358  0.125558  0.134758
test_f1          2.0  0.307614  0.160427  0.194175  0.250894  0.307614  0.364333  0.421053
test_auroc       2.0  0.814166  0.053693  0.776199  0.795183  0.814166  0.833149  0.852133
[01/18/2024 10:54:11 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.398>0.05. -->

Results range from as high as the best reported results and lower.
Others do not report using folds.

### QuantumScents
Unique in that this is QSOR.
~3.5k points.
[@quantumscents]
<!-- [01/19/2024 07:44:19 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    3.0  0.075712  0.001303  0.074208  0.075307  0.076405  0.076463  0.076522
validation_auroc       3.0  0.915910  0.004798  0.913018  0.913141  0.913264  0.917356  0.921448
[01/19/2024 07:44:19 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    3.0  0.076284  0.000804  0.075691  0.075826  0.075961  0.076580  0.077200
test_auroc       3.0  0.915418  0.005326  0.909292  0.913653  0.918015  0.918481  0.918947
[01/19/2024 07:44:19 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.911>0.05.

[01/19/2024 08:15:04 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    3.0  0.075457  0.001448  0.073790  0.074988  0.076186  0.076291  0.076396
validation_auroc       3.0  0.915995  0.006418  0.911147  0.912357  0.913566  0.918420  0.923273
[01/19/2024 08:15:04 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    3.0  0.076526  0.000616  0.075960  0.076197  0.076435  0.076808  0.077182
test_auroc       3.0  0.914514  0.004367  0.909671  0.912695  0.915720  0.916935  0.918150
[01/19/2024 08:15:04 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.758>0.05. -->

### SIDER
TODO: double check scaffold vs random for this in the reference study (unimol).
Matches CMPNN performance OOB:
<!-- [01/16/2024 04:17:32 PM fastprop.fastprop_core] INFO: Displaying validation results:
                    count      mean       std       min       25%       50%       75%       max
validation_ce_loss    2.0  0.510756  0.001610  0.509618  0.510187  0.510756  0.511325  0.511895
validation_auroc      2.0  0.651896  0.033294  0.628353  0.640125  0.651896  0.663667  0.675439
[01/16/2024 04:17:32 PM fastprop.fastprop_core] INFO: Displaying testing results:
              count      mean       std       min       25%       50%       75%       max
test_ce_loss    2.0  0.486880  0.005306  0.483128  0.485004  0.486880  0.488756  0.490632
test_auroc      2.0  0.658325  0.008519  0.652302  0.655314  0.658325  0.661337  0.664349
[01/16/2024 04:17:32 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.816>0.05. -->

### Pgp
Among the higest performers OOB:
<!-- 
[01/17/2024 11:09:47 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    4.0  0.358711  0.114111  0.286426  0.296073  0.309923  0.372561  0.528569
validation_f1          4.0  0.808379  0.128135  0.622642  0.774410  0.854218  0.888186  0.902439
validation_auroc       4.0  0.915684  0.025065  0.881954  0.907542  0.919325  0.927467  0.942130
[01/17/2024 11:09:47 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    4.0  0.379170  0.028873  0.342636  0.362729  0.385658  0.402100  0.402729
test_f1          4.0  0.861819  0.030854  0.816327  0.855806  0.874916  0.880930  0.881119
test_auroc       4.0  0.914512  0.012382  0.902293  0.904942  0.913952  0.923522  0.927851
[01/17/2024 11:09:47 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.936>0.05. -->

Slight improvement with hopt:

<!-- [01/17/2024 11:51:38 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    4.0  0.387118  0.069688  0.310836  0.352300  0.379792  0.414611  0.478053
validation_f1          4.0  0.851747  0.037130  0.800000  0.837500  0.862500  0.876747  0.881988
validation_auroc       4.0  0.917608  0.018889  0.904749  0.907524  0.910028  0.920112  0.945628
[01/17/2024 11:51:38 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    4.0  0.387347  0.032653  0.358048  0.359642  0.386927  0.414632  0.417484
test_f1          4.0  0.874467  0.008665  0.862903  0.870271  0.876923  0.881119  0.881119
test_auroc       4.0  0.926062  0.009703  0.914280  0.921085  0.926352  0.931329  0.937263
[01/17/2024 11:51:38 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.456>0.05. -->

### ARA
`fastprop`:

<!-- [01/23/2024 09:46:32 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    4.0  0.324625  0.078561  0.248426  0.270645  0.312653  0.366633  0.424765
validation_accuracy    4.0  0.872024  0.017857  0.857143  0.857143  0.869048  0.883929  0.892857
validation_f1          4.0  0.874281  0.019900  0.853659  0.861142  0.872294  0.885433  0.898876
validation_auroc       4.0  0.937131  0.025501  0.903977  0.923698  0.942977  0.956410  0.958593
[01/23/2024 09:46:32 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    4.0  0.286704  0.060484  0.204559  0.263829  0.297927  0.320802  0.346404
test_accuracy    4.0  0.882353  0.025415  0.858824  0.867647  0.876471  0.891176  0.917647
test_f1          4.0  0.878230  0.029198  0.857143  0.862934  0.867215  0.882511  0.921348
test_auroc       4.0  0.950813  0.018066  0.932553  0.942168  0.947544  0.956188  0.975610
[01/23/2024 09:46:32 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing accuracy yielded p value of p=0.531>0.05.

real	2m7.199s
user	13m19.752s
sys	0m40.241s

Chemprop: 
real	16m54.734s
user	15m53.029s
sys	1m4.654s

Task	Mean auc	Standard deviation auc	Fold 0 auc	Mean accuracy	Standard deviation accuracy	Fold 0 accuracy
Activity	0.8572281959378732	0.0	0.8572281959378732	0.788235294117647	0.0	0.788235294117647
Activity	0.9312638580931263	0.0	0.9312638580931263	0.8588235294117647	0.0	0.8588235294117647
Activity	0.9438888888888888	0.0	0.9438888888888888	0.8588235294117647	0.0	0.8588235294117647
Activity	0.8600891861761426	0.0	0.8600891861761426	0.788235294117647	0.0	0.788235294117647

From gnumeric, for accuracy:
Mean	0.8235294117647058
Standard Error	0.020377068324339723
Median	0.8235294117647058
Mode	0.788235294117647
Standard Deviation	0.04075413664867945
Sample Variance	0.0016608996539792375

and for auroc:
	Column 1
Mean	0.8981175322740077
Standard Error	0.022934306425495557
Median	0.8956765221346344
Mode	#N/A
Standard Deviation	0.045868612850991114
Sample Variance	0.0021039296448741073
Kurtosis	-5.605963785311696
Skewness	0.06182539892223904
Range	0.08666069295101553
Minimum	0.8572281959378732
Maximum	0.9438888888888888
Sum	3.592470129096031
Count	4 -->

# Limitations and Future Work
## Combination with Learned Representations
This seems like an intuitive next step, but we shouldn't do it.
_Not_ just slamming all the mordred features into Chemprop, which would definitely give good results but would take out the `fast` part (FNN would also have to be increased in size).
`fastprop` is good enough on its own.

## Stereochemical Descriptors
In its current state, the underlying `mordredcommunity` featurization engine does not include any connectivity based-descriptors that reflect the presence or absence of stereocenters.
While some of the 3D descriptors it implements will inherently reflects these differences somewhat, more explicit descriptors like the Stero Signature Molecular Descriptor (see https://doi.org/10.1021/ci300584r) may prove helpful in the future.

This is really just one named domain - any domain where there are bespoke descriptors could be added to `mordredcommunity`.

## Inference Time
Slow on inference, especially on virtual libaries which may number in the millions of compounds.
Thankfully descriptor calclulation is embarassingly parallel, and in practice the number of descriptors needed to be calculated can be reduced once those which are relevant to the neural network are selected based on their weights.

## Incoporating 3D Descriptors
 - already supported by `mordredcommunity` just need to find a good way to ingest 3D data or embed 3D conformers (cite some lit. like quantumscents for the latter point)

# Using `fastprop`
`fastprop` is open source under the terms of a permissive license (MIT) and hosted on GitHub.
Detailed instructions for various installation methods can be found at the repository, as well as an issue tracker, etc.

## Example Usage

As an example, this is what the input file looks like for running QM9 with `fastprop` (see https://github.com/JacksonBurns/fastprop/blob/main/benchmarks/qm9/qm9.yml for more):
```
# qm9.yml

# architecture
# optimize: True
hidden_size: 1900
fnn_layers: 3

# generic
output_directory: qm9
input_file: qm9/benchmark_data.csv
target_columns: homo lumo gap
smiles_column: smiles
number_repeats: 2

# featurization
precomputed: qm9/precomputed.csv

# training
random_seed: 27
batch_size: 163072
number_epochs: 600
patience: 30

# data splitting
train_size: 0.8
val_size: 0.1
test_size: 0.1
sampler: scaffold
```

## Choice of Descriptors
 - multiple descriptor generation tools were tried, none of which yielded the same performance as `mordred`
 - users can walk up with their own descriptors, and as shown in the reference code for the QuantumScents benchmark do so eithe rprogrammatically or through the command line interface
 - on-th-fly descriptor generation can be pared down to only those which are often highly weighted, which would speed up inference and training time

# Cited Works