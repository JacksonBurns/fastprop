---
title: " One Model to Rule Them All: Generalizable, Fast, and Accurate Deep-QSPR with `fastprop`"
date: "Januray 2024"
author: "Jackson Burns, MIT CCSE"
geometry: margin=1in
note: Compile this paper with "pandoc --citeproc --bibliography=paper.bib -s paper.md -o paper.pdf"
---

# Outline

Quantitative Structure-Property/Activity Relationship studies, often referred to interchangeably as QS(P/A)R, seek to establish a mapping between molecular structure and an arbitrary Quantity of Interest (QOI).
Historically this was done on a QOI-by-QOI basis with new descriptors being devised by researchers to _specifically_ map to their QOI (see the sub-field of transition metal-mediated cross-coupling reactions, where molecular descriptors specific to ligands such as cone angle, sterimol parameters, and buried volume are used).
This continued for years and culminated in packages like (sort these chronologically) DRAGON, Mordred, PaDEL-descriptor, E-DRAGON, etc.
Other packages like molfeat and AIMSim aggregate these other descriptor generation tools.\
As these packages evolved, so too did those used for deep learning.
The logical combination of bulk molecular-level descriptors with deep learning as a regression technique has not seen the same success as learned representations, like UniMol, CMPNN, and especially Chemprop.
Despite past failures, this paper established that by combining a cogent set of molecular descriptors with deep learning, generalizable deep-QSPR is realized.

# `fastprop` Approach & Previous Efforts
`fastprop` is simply the `mordred` molecular descriptor calculator connected to an FNN with research software engineering best practices applied at the same level as the best alternative in the literature Chemprop.
This contradicts the leading review on deep-QSPR (ma-et-al-2015-deep-neural-nets-as-a-method-for-quantitative-structure-activity-relationships) which says that these are ineffective, and that fingerprints should be used instead.

This is a simple idea, here are some that have tried it before and failed:
 - fuels paper
 - https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00180 they did a huge pre-processing pipeline that removed all but 53 descs, and then only did linear methods. DeepDelta then fit on this dataset and improved the accuracy.

## Descriptor Generation Software
 - part of the failure of this method may be the lack of descriptor generation software with a suggicient collection of descrtiptors, which `mordred` is
 - from here forward `mordred` will be referred to as such, though the actual implementaiton of `fastprop` uses a community-maintained fork of the original (which is no longer mainted) aptly named `mordredcommunity` (cite the Zenodo)


# Benchmarks

There are a number of established molecular property prediction benchmarks in the cheminfomatics community, especially those standardized by MoleculeNet.
Some of these, namely QM8/9, are perceived as the 'gold standard' for property prediction.
While this task is important, the enormous size of these datsets in particular means that many model architectures are highly accurate given the rich coverage of chemical space (inherent in the design of the combinatorial datasets).
i.e. these datasets are so comprehensive, and model worth its salt should be able to learn.

Real world datasets, particularly those common in QSPR studies, often number in the hundreds.
To demonstrate the applicability of `fastprop` to these regimes, many smaller datasets are selected, including some from the QSPR literature that are not established benchmarks.
These studies relied on **more complex and slow modeling techniques (see the fuels study, the ARA dataset) or design of a bespoke descriptor (find these)** and **have not yeen come to rely on learned representations as a go-to tool**; **we believe because in these data-limited environments the 'start from almost-nothing' approach of Chemprop leads to overfitting or requires more data** and that the 'start from 100 years of QSPR' approach of `fastprop` circumvents this.

The authors of Chemprop have even suggested on GitHub issues in the past that molecular-level features be added to Chemprop learned representations (see https://github.com/chemprop/chemprop/issues/146#issuecomment-784310581) to avoid overfitting.

To emphasize this point further, the benchmarks are presented in order of size descending (for first regression and then classification).
Consistently `fastprop` is able to match the performance of learned representations on large (O(10,000)+ entries) datasets and compete with or exceed their performance on small ($\leq$O(1,000) entries).

**TODO: re-run all of these since the training metric is now using mean reduction and also with timing added**

## Methods
All of these use 80% of the dataset for training (selected randomly, unless otherwise stated), 10% for validation, and holdout the remaining 10% for testing (unless otherwise stated).

### Reporting of Error Bars
Results for `fastprop` are reported with repetitions rather than with cross-validation (CV).
In general and as is seen oin literatre, CV is performed without a holdout set which allows the model to learn the validation dataset (the accuracy on which is reported as the 'result') during training (i.e. data leakage).
While literature precedent if typically to use CV, for small datasets especially this can lead to misreprentation of accuracy.
The holdout set, when initially randomly selected, may contain only 'easy' samples; regardless of the number of folds within the training/validation data, the reported performance will be overly optimistic.
The solution is to either _nest_ cross validation (i.e. repeat cross validation multiple times with different holdout sets (as was done in https://pubs.acs.org/doi/10.1021/acsomega.1c02156, for example)) or to just simply do repeats (more practical for deep learning, which is slow compared to linear methods **and, if anything, only makes the model performance not as good as it could since I don't optimize on each possible subset (so statistcally I'm not guaranteed a good result and NN should not be a steep function of this training choice) be so really I'm being super-duper extra honest here**).

By simply repeating a random selection of train/validation/testing subsets, we reduce the impact of biased holdout set selections while also avoiding data leakage.
For this study, the number of repetitions vary per-benchmark and are determined by first setting the number of repetitions to two, and then increasing until the result of a 2-sided T-test between the validation and testing set accuracy yielded a p-value greater than 0.05 at 95% confidence.
At that point it is asserted that a reasonable approximation of the accuracy as a function of the holdout set has been achieved.

### Choice of Evaluation Metric
The evluation metric used in each of these metrics (L1, L2, AUROC, etc.) are presented here because that is what the reference papers (and the literature at large) use.
It is my opinion that this choice of metric 'buries the lede' by requiring the reader to understand the relative magnitude of the target variables (for regression) or understand the nuances of classiciation quantification (ARUOC).
On the HOPV15 dataset, for example, an MAE of 1.1 seems exceptional in the literature, except that the WMAPE is nearly 30%.
That's right - when weighted by their magnitude, the best performing model is still usually off by about 30%.
For classifiction, espeically multilabel, AUROC scores near one are impressive despite accuracy being low.

Pertinent Metrics are included on a per-study basis.

### Timing Results
Execution time is as reported by the unix `time` command using Chemprop version 1.6.1 on Python 3.8 and `fastprop` 0.0.0b0 on Python 3.11 and include the complete invocation of their respective program, i.e. `time chemprop_train` and `time fastprop train`.
The insigifnicant time spent manually collating Chemprop results (Chemprop does not natively support repetitions) is excluded.
The latter version of Python is known to be broadly faster (as well as the more updated versions of some dependencies that come with it), but the overall speed trends still stand.

TODO: report the timing results per-epoch rather than the overall time, which removes the annoyance of different numbers of epochs to converge.

also TODO: break out and specifically mention the time to generate features for `fastprop`

## Regression Datasets

<!-- Copied this table from the README using an online md->rst converter -->

+---------------+--------------------+--------+-----------------------------+------------+-----------------------+-----------+
|   Benchmark   | Number Samples (k) | Metric |       Literature Best       | `fastprop` |       Chemprop        |  Speedup  |
+===============+====================+========+=============================+============+=======================+===========+
|      QM9      |        ~130        |   L1   |    0.0047  [ref: unimol]    |   0.0063   | 0.0081 [ref: unimol]  |           |
+---------------+--------------------+--------+-----------------------------+------------+-----------------------+-----------+
|      QM8      |        ~22         |   L1   |     0.016 [ref: unimol]     |   0.016    |  0.019 [ref: unimol]  |           |
+---------------+--------------------+--------+-----------------------------+------------+-----------------------+-----------+
|     ESOL      |        ~1.1        |   L2   |      0.55 [ref: cmpnn]      |    0.57    |   0.67 [ref: cmpnn]   |           |
+---------------+--------------------+--------+-----------------------------+------------+-----------------------+-----------+
|   FreeSolv    |        ~0.6        |   L2   |    1.29 [ref: DeepDelta]    |    1.06    | 1.37 [ref: DeepDelta] |           |
+---------------+--------------------+--------+-----------------------------+------------+-----------------------+-----------+
| HOPV15 Subset |        ~0.3        |   L1   | 1.32 [ref: the kraft paper] |    1.44    |          WIP          |           |
+---------------+--------------------+--------+-----------------------------+------------+-----------------------+-----------+
|    Fubrain    |        ~0.3        |   L2   |  0.44 [ref: fubrain paper]  |    0.19    | 0.22 [ref: this repo] | 5m11s/54s |
+---------------+--------------------+--------+-----------------------------+------------+-----------------------+-----------+


### QM9

~130k points
This is training only on three targest (the three hardest) using 80/10/10 sacaffold as done in the UniMol paper (see the benchmarks directory).
Great OOB:
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

Small improvement with hopt:
<!-- [01/18/2024 09:15:08 AM fastprop.fastprop_core] INFO: Displaying validation results:
                              count      mean       std       min       25%       50%       75%       max
validation_mse_loss             2.0  0.063158  0.004941  0.059664  0.061411  0.063158  0.064905  0.066652
validation_mean_wmape           2.0  0.030519  0.002153  0.028997  0.029758  0.030519  0.031281  0.032042
validation_wmape_output_homo    2.0  0.023095  0.001381  0.022118  0.022606  0.023095  0.023583  0.024071
validation_wmape_output_lumo    2.0  0.035917  0.003607  0.033366  0.034641  0.035917  0.037192  0.038467
validation_wmape_output_gap     2.0  0.032547  0.001472  0.031506  0.032026  0.032547  0.033068  0.033588
validation_l1_avg               2.0  0.006292  0.000313  0.006070  0.006181  0.006292  0.006403  0.006513
validation_l1_output_homo       2.0  0.005251  0.000346  0.005006  0.005129  0.005251  0.005374  0.005496
validation_l1_output_lumo       2.0  0.005773  0.000105  0.005699  0.005736  0.005773  0.005811  0.005848
validation_l1_output_gap        2.0  0.007851  0.000488  0.007506  0.007679  0.007851  0.008024  0.008196
validation_rmse_avg             2.0  0.008691  0.000256  0.008510  0.008601  0.008691  0.008782  0.008872
validation_rmse_output_homo     2.0  0.007301  0.000360  0.007047  0.007174  0.007301  0.007429  0.007556
validation_rmse_output_lumo     2.0  0.007999  0.000087  0.007937  0.007968  0.007999  0.008029  0.008060
validation_rmse_output_gap      2.0  0.010774  0.000495  0.010424  0.010599  0.010774  0.010949  0.011123
[01/18/2024 09:15:08 AM fastprop.fastprop_core] INFO: Displaying testing results:
                        count      mean       std       min       25%       50%       75%       max
test_mse_loss             2.0  0.063456  0.005917  0.059272  0.061364  0.063456  0.065548  0.067640
test_mean_wmape           2.0 -0.017396  0.063493 -0.062293 -0.039845 -0.017396  0.005052  0.027500
test_wmape_output_homo    2.0  0.022185  0.001192  0.021342  0.021764  0.022185  0.022607  0.023028
test_wmape_output_lumo    2.0 -0.106038  0.195473 -0.244258 -0.175148 -0.106038 -0.036928  0.032182
test_wmape_output_gap     2.0  0.031664  0.003800  0.028976  0.030320  0.031664  0.033007  0.034351
test_l1_avg               2.0  0.006299  0.000544  0.005914  0.006107  0.006299  0.006492  0.006684
test_l1_output_homo       2.0  0.005316  0.000273  0.005123  0.005219  0.005316  0.005412  0.005509
test_l1_output_lumo       2.0  0.005738  0.000652  0.005276  0.005507  0.005738  0.005968  0.006199
test_l1_output_gap        2.0  0.007844  0.000708  0.007344  0.007594  0.007844  0.008095  0.008345
test_rmse_avg             2.0  0.008725  0.000634  0.008276  0.008501  0.008725  0.008949  0.009173
test_rmse_output_homo     2.0  0.007286  0.000076  0.007232  0.007259  0.007286  0.007313  0.007340
test_rmse_output_lumo     2.0  0.007978  0.000898  0.007343  0.007660  0.007978  0.008295  0.008613
test_rmse_output_gap      2.0  0.010911  0.000929  0.010255  0.010583  0.010911  0.011240  0.011568
[01/18/2024 09:15:08 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.951>0.05. -->

### QM8
~22k points
Sampling techniques and tons of repetitions are not required here since the dataset is so large that nearly any random sample should cover a significant amount of space just by chance.

Without any optimization, get:
<!-- [01/14/2024 03:54:22 PM fastprop.fastprop_core] INFO: Displaying validation results:
                                         count      mean       std       min       25%       50%       75%       max
validation_mse_loss                        2.0  0.192557  0.008195  0.186763  0.189660  0.192557  0.195454  0.198352
unitful_validation_mean_wmape              2.0  0.344357  0.004437  0.341220  0.342788  0.344357  0.345925  0.347494
unitful_validation_wmape_output_E1-CC2     2.0  0.027181  0.000462  0.026854  0.027017  0.027181  0.027344  0.027507
unitful_validation_wmape_output_E2-CC2     2.0  0.026339  0.000465  0.026010  0.026174  0.026339  0.026503  0.026667
unitful_validation_wmape_output_f1-CC2     2.0  0.657772  0.018779  0.644494  0.651133  0.657772  0.664412  0.671051
unitful_validation_wmape_output_f2-CC2     2.0  0.735917  0.014783  0.725464  0.730690  0.735917  0.741144  0.746370
unitful_validation_wmape_output_E1-PBE0    2.0  0.027052  0.000147  0.026948  0.027000  0.027052  0.027104  0.027156
unitful_validation_wmape_output_E2-PBE0    2.0  0.025541  0.000693  0.025050  0.025296  0.025541  0.025786  0.026031
unitful_validation_wmape_output_f1-PBE0    2.0  0.620125  0.022971  0.603882  0.612003  0.620125  0.628246  0.636368
unitful_validation_wmape_output_f2-PBE0    2.0  0.710052  0.023048  0.693755  0.701903  0.710052  0.718200  0.726349
unitful_validation_wmape_output_E1-CAM     2.0  0.025539  0.000283  0.025339  0.025439  0.025539  0.025639  0.025739
unitful_validation_wmape_output_E2-CAM     2.0  0.023580  0.000461  0.023254  0.023417  0.023580  0.023743  0.023905
unitful_validation_wmape_output_f1-CAM     2.0  0.595597  0.028180  0.575671  0.585634  0.595597  0.605560  0.615523
unitful_validation_wmape_output_f2-CAM     2.0  0.657588  0.007699  0.652145  0.654867  0.657588  0.660310  0.663032
unitful_validation_l1_avg                  2.0  0.013121  0.000017  0.013109  0.013115  0.013121  0.013128  0.013134
unitful_validation_l1_output_E1-CC2        2.0  0.005987  0.000091  0.005923  0.005955  0.005987  0.006019  0.006052
unitful_validation_l1_output_E2-CC2        2.0  0.006557  0.000129  0.006466  0.006512  0.006557  0.006603  0.006648
unitful_validation_l1_output_f1-CC2        2.0  0.015217  0.000263  0.015031  0.015124  0.015217  0.015310  0.015403
unitful_validation_l1_output_f2-CC2        2.0  0.030554  0.000082  0.030496  0.030525  0.030554  0.030583  0.030612
unitful_validation_l1_output_E1-PBE0       2.0  0.005872  0.000025  0.005854  0.005863  0.005872  0.005881  0.005890
unitful_validation_l1_output_E2-PBE0       2.0  0.006192  0.000178  0.006066  0.006129  0.006192  0.006255  0.006318
unitful_validation_l1_output_f1-PBE0       2.0  0.012841  0.000298  0.012630  0.012736  0.012841  0.012947  0.013052
unitful_validation_l1_output_f2-PBE0       2.0  0.023159  0.000191  0.023024  0.023091  0.023159  0.023226  0.023294
unitful_validation_l1_output_E1-CAM        2.0  0.005545  0.000053  0.005508  0.005527  0.005545  0.005564  0.005583
unitful_validation_l1_output_E2-CAM        2.0  0.005769  0.000117  0.005686  0.005727  0.005769  0.005810  0.005851
unitful_validation_l1_output_f1-CAM        2.0  0.013678  0.000325  0.013448  0.013563  0.013678  0.013793  0.013907
unitful_validation_l1_output_f2-CAM        2.0  0.026084  0.000149  0.025979  0.026031  0.026084  0.026137  0.026190
unitful_validation_rmse_avg                2.0  0.021851  0.000435  0.021544  0.021698  0.021851  0.022005  0.022159
unitful_validation_rmse_output_E1-CC2      2.0  0.008178  0.000147  0.008074  0.008126  0.008178  0.008230  0.008282
unitful_validation_rmse_output_E2-CC2      2.0  0.009193  0.000151  0.009087  0.009140  0.009193  0.009247  0.009300
unitful_validation_rmse_output_f1-CC2      2.0  0.030104  0.000481  0.029764  0.029934  0.030104  0.030274  0.030445
unitful_validation_rmse_output_f2-CC2      2.0  0.048713  0.000484  0.048370  0.048541  0.048713  0.048884  0.049055
unitful_validation_rmse_output_E1-PBE0     2.0  0.007875  0.000002  0.007874  0.007874  0.007875  0.007875  0.007876
unitful_validation_rmse_output_E2-PBE0     2.0  0.008199  0.000320  0.007973  0.008086  0.008199  0.008313  0.008426
unitful_validation_rmse_output_f1-PBE0     2.0  0.026837  0.001426  0.025829  0.026333  0.026837  0.027342  0.027846
unitful_validation_rmse_output_f2-PBE0     2.0  0.037850  0.000370  0.037589  0.037719  0.037850  0.037981  0.038112
unitful_validation_rmse_output_E1-CAM      2.0  0.007449  0.000043  0.007419  0.007434  0.007449  0.007464  0.007479
unitful_validation_rmse_output_E2-CAM      2.0  0.007679  0.000187  0.007546  0.007612  0.007679  0.007745  0.007811
unitful_validation_rmse_output_f1-CAM      2.0  0.027930  0.000595  0.027509  0.027720  0.027930  0.028141  0.028351
unitful_validation_rmse_output_f2-CAM      2.0  0.042209  0.001392  0.041225  0.041717  0.042209  0.042702  0.043194
[01/14/2024 03:54:22 PM fastprop.fastprop_core] INFO: Displaying testing results:
                                   count      mean       std       min       25%       50%       75%       max
test_mse_loss                        2.0  0.215020  0.012603  0.206108  0.210564  0.215020  0.219476  0.223931
unitful_test_mean_wmape              2.0  0.337677  0.000919  0.337027  0.337352  0.337677  0.338002  0.338327
unitful_test_wmape_output_E1-CC2     2.0  0.027267  0.000454  0.026946  0.027107  0.027267  0.027428  0.027589
unitful_test_wmape_output_E2-CC2     2.0  0.025745  0.000439  0.025435  0.025590  0.025745  0.025901  0.026056
unitful_test_wmape_output_f1-CC2     2.0  0.641459  0.001794  0.640191  0.640825  0.641459  0.642094  0.642728
unitful_test_wmape_output_f2-CC2     2.0  0.723625  0.005894  0.719457  0.721541  0.723625  0.725709  0.727793
unitful_test_wmape_output_E1-PBE0    2.0  0.027102  0.000382  0.026832  0.026967  0.027102  0.027237  0.027372
unitful_test_wmape_output_E2-PBE0    2.0  0.024828  0.000102  0.024756  0.024792  0.024828  0.024864  0.024901
unitful_test_wmape_output_f1-PBE0    2.0  0.587282  0.010816  0.579634  0.583458  0.587282  0.591106  0.594930
unitful_test_wmape_output_f2-PBE0    2.0  0.719084  0.003830  0.716376  0.717730  0.719084  0.720438  0.721792
unitful_test_wmape_output_E1-CAM     2.0  0.025809  0.000675  0.025331  0.025570  0.025809  0.026047  0.026286
unitful_test_wmape_output_E2-CAM     2.0  0.022945  0.000009  0.022939  0.022942  0.022945  0.022948  0.022952
unitful_test_wmape_output_f1-CAM     2.0  0.552909  0.036638  0.527002  0.539956  0.552909  0.565863  0.578816
unitful_test_wmape_output_f2-CAM     2.0  0.674068  0.030505  0.652498  0.663283  0.674068  0.684853  0.695639
unitful_test_l1_avg                  2.0  0.013568  0.000317  0.013344  0.013456  0.013568  0.013680  0.013792
unitful_test_l1_output_E1-CC2        2.0  0.006000  0.000112  0.005921  0.005960  0.006000  0.006040  0.006080
unitful_test_l1_output_E2-CC2        2.0  0.006403  0.000108  0.006327  0.006365  0.006403  0.006441  0.006479
unitful_test_l1_output_f1-CC2        2.0  0.015477  0.000360  0.015222  0.015350  0.015477  0.015604  0.015731
unitful_test_l1_output_f2-CC2        2.0  0.032834  0.002454  0.031099  0.031966  0.032834  0.033701  0.034569
unitful_test_l1_output_E1-PBE0       2.0  0.005876  0.000097  0.005808  0.005842  0.005876  0.005911  0.005945
unitful_test_l1_output_E2-PBE0       2.0  0.006017  0.000034  0.005992  0.006004  0.006017  0.006029  0.006041
unitful_test_l1_output_f1-PBE0       2.0  0.012826  0.000611  0.012394  0.012610  0.012826  0.013042  0.013258
unitful_test_l1_output_f2-PBE0       2.0  0.024711  0.000604  0.024284  0.024497  0.024711  0.024924  0.025137
unitful_test_l1_output_E1-CAM        2.0  0.005595  0.000164  0.005480  0.005538  0.005595  0.005653  0.005711
unitful_test_l1_output_E2-CAM        2.0  0.005610  0.000006  0.005606  0.005608  0.005610  0.005613  0.005615
unitful_test_l1_output_f1-CAM        2.0  0.014009  0.000553  0.013618  0.013814  0.014009  0.014205  0.014400
unitful_test_l1_output_f2-CAM        2.0  0.027460  0.001751  0.026222  0.026841  0.027460  0.028079  0.028698
unitful_test_rmse_avg                2.0  0.022982  0.000599  0.022558  0.022770  0.022982  0.023193  0.023405
unitful_test_rmse_output_E1-CC2      2.0  0.008406  0.000052  0.008369  0.008387  0.008406  0.008424  0.008443
unitful_test_rmse_output_E2-CC2      2.0  0.008824  0.000165  0.008707  0.008766  0.008824  0.008882  0.008941
unitful_test_rmse_output_f1-CC2      2.0  0.032339  0.000282  0.032139  0.032239  0.032339  0.032439  0.032539
unitful_test_rmse_output_f2-CC2      2.0  0.053334  0.003692  0.050723  0.052028  0.053334  0.054639  0.055944
unitful_test_rmse_output_E1-PBE0     2.0  0.008136  0.000059  0.008095  0.008115  0.008136  0.008157  0.008178
unitful_test_rmse_output_E2-PBE0     2.0  0.007986  0.000040  0.007958  0.007972  0.007986  0.008000  0.008014
unitful_test_rmse_output_f1-PBE0     2.0  0.026221  0.000817  0.025643  0.025932  0.026221  0.026510  0.026799
unitful_test_rmse_output_f2-PBE0     2.0  0.041446  0.002795  0.039469  0.040457  0.041446  0.042434  0.043422
unitful_test_rmse_output_E1-CAM      2.0  0.007811  0.000024  0.007794  0.007803  0.007811  0.007820  0.007828
unitful_test_rmse_output_E2-CAM      2.0  0.007445  0.000002  0.007444  0.007445  0.007445  0.007446  0.007447
unitful_test_rmse_output_f1-CAM      2.0  0.029692  0.001111  0.028907  0.029300  0.029692  0.030085  0.030478
unitful_test_rmse_output_f2-CAM      2.0  0.044138  0.002698  0.042230  0.043184  0.044138  0.045092  0.046046 -->

With modest hyperparameter optimization, achieve:
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
~1.1k
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
~0.6k

After hyperparameter optimization:
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

Totally crushes DeepDelta: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00769-x/tables/2 (and the other architectures shown there!).


### HOPV15 Subset
~0.3k
This study (https://pubs.acs.org/doi/10.1021/acsomega.1c02156) trained on a subset of HOPV15 matching some criteria described in the paper, and they saw SVR give an accuracy 1.32 L1 averaged across 5 randomly selected folds of 60/20/20 data splits.
Due to the very small size of this dataset and as a point of comparison to the reference study, 5 repetitions are used.

[01/23/2024 11:22:14 AM fastprop.fastprop_core] INFO: Displaying validation results:
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
[01/23/2024 11:22:14 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.254>0.05.

### Fubrain
~0.3k

The study that first generated this dataset got 0.44 RMSE https://doi.org/10.1021/acs.jcim.9b00180 with 10-fold CV and then DeepDelta reported 0.830+/-0.023 RMSE using the same approach.
In both studies 10-fold cross validation using the default settings in scikit-learn (CITE, then cite specific KFold page) will result in a 90/10 train/test split, and a separate holdout set is identified afterward for reporting performance.

OOB performance is twice as good as the reference study and dramatically better than DeepDelta, which also suffers from scaling issues.
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

Chemprop:
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


Timing for fastprop: descriptor generation was 30 seconds, total was 54s.

Chemprop's performance on this dataset is still much better than the reference dataset, but worse than `fastprop`. More importantly, Chemprop also significantly overfits to the validation/training data, as can be seen by the severly diminished performance on the witheld testing set.
As stated earlier, this is likely to blame on the 'start from almost-nothing' strategy in Chemprop, where the initial representation is only a combination of simple atomic descrtiptors based on the connectivity and not the human-designed high-level molecular descriptors used in `fastprop`.
This will also become clear in the later ARA classification dataset (see [ARA](#ara)).



# WIPs
## PAHs
https://doi.org/10.1080/1062936X.2023.2239149

fastprop: about 17 seconds calculating descriptors

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
sys     0m15.281s


## Flash
https://pubs.acs.org/doi/10.1021/ef200795j

This is perhaps the best point of comparison:
 - holds out 10% of the data
 - only reports one repeat but verified that the distribution of targest was similar in test/val/train
Only a little over 1000 datapoints.

fastprop:
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
Standard Error	1.1064790292313673



## H2S
[01/24/2024 12:53:11 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_mse_loss    2.0  0.118679  0.021531  0.103454  0.111067  0.118679  0.126291  0.133904
validation_mape        2.0  0.410330  0.012879  0.401223  0.405776  0.410330  0.414883  0.419436
validation_wmape       2.0  0.185141  0.021952  0.169619  0.177380  0.185141  0.192902  0.200663
validation_l1          2.0  0.043572  0.001372  0.042602  0.043087  0.043572  0.044057  0.044542
validation_rmse        2.0  0.058517  0.005330  0.054748  0.056633  0.058517  0.060402  0.062287
[01/24/2024 12:53:11 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_mse_loss    2.0  0.124909  0.003943  0.122121  0.123515  0.124909  0.126303  0.127697
test_mape        2.0  0.341950  0.048751  0.307477  0.324713  0.341950  0.359186  0.376422
test_wmape       2.0  0.185302  0.009566  0.178537  0.181920  0.185302  0.188684  0.192066
test_l1          2.0  0.043859  0.000828  0.043274  0.043566  0.043859  0.044152  0.044444
test_rmse        2.0  0.060154  0.000950  0.059483  0.059819  0.060154  0.060490  0.060826
[01/24/2024 12:53:11 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.711>0.05.

real    6m26.113s
user    2m35.630s
sys     2m46.584s

chemprop:
Task,Mean rmse,Standard deviation rmse,Fold 0 rmse
solubility,0.05662203358114606,0.0,0.05662203358114606
solubility,0.07003554233895928,0.0,0.07003554233895928

real    7m18.991s
user    6m40.047s
sys     0m43.251s


## YSI
chemprop:
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
sys     1m1.971s


## Classification Datasets

+---------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|   Benchmark   | Number Samples (k) |   Metric   |      Literature Best       | `fastprop` |          Chemprop          |   Speedup   |
+===============+====================+============+============================+============+============================+=============+
| HIV (binary)  |        ~41         |   AUROC    |     0.81 [ref: unimol]     |    0.81    |     0.77 [ref: unimol]     |             |
+---------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
| HIV (ternary) |        ~41         |   AUROC    |                            |    0.83    |            WIP             |             |
+---------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
| QuantumScents |        ~3.5        |   AUROC    | 0.88 [ref: quantumscents]  |    0.91    | 0.85 [ref: quantumscents]  |             |
+---------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|     SIDER     |        ~1.4        |   AUROC    |     0.67 [ref: cmpnn]      |    0.66    |     0.57 [ref: cmpnn]      |             |
+---------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|      Pgp      |        ~1.3        |   AUROC    |            WIP             |    0.93    |            WIP             |             |
+---------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+
|      ARA      |        ~0.8        | Acc./AUROC | 0.91/0.95 [ref: ara paper] | 0.88/0.95  | 0.82/0.90 [ref: this repo] | 16m54s/2m7s |
+---------------+--------------------+------------+----------------------------+------------+----------------------------+-------------+


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

[01/23/2024 09:46:32 AM fastprop.fastprop_core] INFO: Displaying validation results:
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
Count	4

# Limitations

## Stereochemical Descriptors
In its current state, the underlying `mordredcommunity` featurization engine does not include any connectivity based-descriptors that reflect the presence or absence of stereocenters.
While some of the 3D descriptors it implements will inherently reflects these differences somewhat, more explicit descriptors like the Stero Signature Molecular Descriptor (see https://doi.org/10.1021/ci300584r) may prove helpful in the future.

## Inference Time
Slow on inference, especially on virtual libaries which may number in the millions of compounds.
Thankfully descriptor calclulation is embarassingly parallel, and in practice the number of descriptors needed to be calculated can be reduced once those which are relevant to the neural network are selected based on their weights.

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