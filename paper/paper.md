---
title: "Generalizable, Fast, and Accurate Deep-QSPR with `fastprop`"
subtitle: "Part 1: Framework and Benchmarks"
author: 
  - name: Jackson W. Burns \orcidlink{0000-0002-0657-9426}
    affil-id: 1
  - name: William H. Green \orcidlink{0000-0003-2603-9694}
    affil-id: 1,*
affiliations:
  - id: 1
    name: Massachusetts Institute of Technology, Cambridge, MA
  - id: "*"
    name: "Corresponding: whgreen@mit.edu"
date: January 31, 2024
geometry: margin=1in
bibliography: paper.bib
citation-style: journal-of-cheminformatics
note: |
 This paper can be compiled to other formats (pdf, word, etc.) using pandoc.

 To show the affiliations and ORCIDs, a special setup has been used. To recreate it locally,
 checkout this StackOverflow answer: https://stackoverflow.com/a/76630771 (this will require you
 to download a new default.latex and edit it as described in the post) _or_
 make sure that the version of pandoc you have installed is _exactly_ 3.1.6

 You can then compile this paper with:
   pandoc --citeproc -s paper.md -o paper.pdf --template default.latex
 from the paper directory in fastprop.

 To compile _without_ doing all that annoying setup (so, on _any_ pandoc version),
 you can just leave off '--template default.latex' i.e.:
  pandoc --citeproc -s paper.md -o paper.pdf
 This won't render the author block correctly, but everything else should work fine.
---

<!-- Graphical Abstract Goes Here -->

# Abstract
Quantitative Structure-Property/Activity Relationship studies, often referred to interchangeably as QS(P/A)R, seek to establish a mapping between molecular structure and an arbitrary Quantity of Interest (QOI).
Since its inception this was done on a QOI-by-QOI basis with new descriptors being devised by researchers to _specifically_ map to their QOI.
This continued for years and culminated in packages like DRAGON (later E-dragon), PaDEL-descriptor (and padelpy),  Mordred, and many others.
The sheer number of different packages resulted in the creation of 'meta-packages' which served only to aggregate these other calculators, including tools like molfeat, ChemDes,  Parameter Client, and AIMSim.
Despite the creation of these aggregation tools, QSPR studies have continued to focus on QOI-by-QOI studies rather than attempting to create a generalizable approach which is capable of modeling across chemical domains.

One contributing factor to this is that QSPR studies have relied almost exclusively on linear methods for regression.
Efforts to incorporate Deep Learning (DL) as a regression technique (Deep-QSPR), which would be capable of capturing the non-linear behavior of arbitrary QOIs, have instead focused on using molecular fingerprints as inputs.
The combination of bulk molecular-level descriptors with DL has remained largely unexplored, in significant part due to the orthogonal domain expertise required to combine the two.
Generalizable QSPR has turned to learned representations primarily via message passing graph neural networks.
This approach has proved remarkably effective but is not without drawbacks.
Learning a representation requires large datasets to avoid over-fitting or even learn at all, loses interpretability since an embedding's meaning can only be induced retroactively, and needs significant execution time given the complexity of the underlying message passing algorithm.
This paper introduces `fastprop`, a software package and general Deep-QSPR framework that combines a cogent set of molecular descriptors with DL to achieve state-of-the-art performance on datasets ranging from tens to tens of thousands of molecules.
`fastprop` is designed with Research Software Engineering best practices and is free and open source, hosted at github.com/jacksonburns/fastprop.

## Scientific Contribution
<!-- use a maximum of 3 sentences to specifically highlight the scientific contributions that advance the field and what differentiates your contribution from prior work on this topic. -->
`fastprop` is a QSPR framework that achieves state-of-the-art accuracy on datasets of all sizes without sacrificing speed or interpretability.
As a software package `fastprop` emphasizes Research Software Engineering best practices, reproducibility, and ease of use for experts across domains.

# Keywords
 - QSPR
 - Learned Representations
 - Deep Learning
 - Molecular Descriptors

# Introduction
Chemists have long sought a method to relate only the connectivity of a molecule to its corresponding molecular properties.
The Quantitative Structure-Property Relationship (QSPR) would effectively solve the forward problem of molecular engineering and enable rapid development.
Reviews on the topic are numerous and cover an enormous range of scientific subdomains; a comprehensive review of the literature is beyond the scope of this publication, though the work of Muratov and coauthors [@muratov_qsar] provides an excellent starting point for further review.
An abridged version of the history behind QSPR is presented here to contextualize the approach of `fastprop`.

## Historical Approaches
Early in the history of computing, limited computational power meant that significant human expertise was required to guide QSPR models toward effectiveness.
This materialized in the form of bespoke molecular descriptors: the Wiener Index in 1947 [@wiener_index], Atom-Bond Connectivity indices in 1998 [@estrada_abc], and _thousands_ of others.
To this day descriptors are still being developed - the geometric-harmonic-Zagreb degree based descriptors were proposed by Arockiaraj et al. in 2023 [@pah].
In each case, domain experts devised an algorithm which mapped a molecular structure to some scalar value.
This algorithm would take into account features of the molecule which that expert deduced were relevant to the property at hand.
This time consuming technique is of course highly effective but the dispersed nature of this chemical knowledge means that these descriptors are spread out throughout many journals and domains with no single source to compute them all.

The range of regression techniques applied to these descriptors has also been limited.
As explained by Muratov et. al [@muratov_qsar] QSPR uses linear methods ("machine learning" in modern vocabulary) almost exclusively.
The over-reliance on this category of approaches may be due to simply priorities; domain experts seek interpretability in their work, especially given that the inputs are physically meaningful descriptors, and linear method lend themselves well to this approach.
Practice may also have been a limitation, since historically training and deploying neural networks required more computer science expertise than linear methods.

All of this is not to say that DL has _never_ been applied to QSPR.
Applications of DL to QSPR, i.e. DeepQSPR, were attempted throughout this time period but focused on the use of molecular fingerprints rather than descriptors.
This may be at least partially attributed to knowledge overlap between deep learning experts and this sub-class of descriptors.
Molecular fingerprints are bit vectors which encode the presence or absence of human-chosen sub-structures in an analogous manner to the "bag of words" featurization strategy common to natural language processing.
It is reasonable to assume a DL expert may have bridged this gap to open this subdomain, and its effectiveness proved worthwhile.
In the review of DL for QSPR by Ma and coauthors [@ma_deep_qsar], they state that combinations of fingerprint descriptors are more effective than molecular-level descriptors, either matching our outperforming linear methods across a number of ADME-related datasets.
This study will later refute that suggestion.

Despite their differences, both classical- and Deep-QSPR shared a lack of generalizability.
Beyond the domains of chemistry where many of the descriptors had been originally devised, models were either unsuccessful or more likely simply never evaluated.
As interest began to shift toward the prediction of molecular properties which were themselves descriptors (i.e. derived from quantum mechanics simulations) - to which none of the devised molecular descriptors were designed to be correlated - learned representations (LRs) emerged.

## Shift to Learned Representations
The exact timing of the transition from descriptors (molecular-level or fingerprints) to LRs is difficult to ascertain.
Among the most cited at least is the work of Yang and coworkers in 2019 [@chemprop_theory] which laid the groundwork for applying LRs to "Property Prediction" - QSPR by another name.
In short, the basic idea is to initialize a molecular graph with only information about its bonds and atoms such as order, aromaticity, atomic number, etc.
Then via a Message Passing Neural Network (MPNN) architecture, which is able to aggregate these atom- and bond-level features into a vector in a manner which can be updated, the 'best' representation of the molecule is found during training.
This method proved highly accurate _and_ achieved the generalizability apparently lacking in descriptor-based modeling.
The corresponding software package Chemprop (later described in [@chemprop_software]) has become the de-facto standard for property prediction, partially because of the significant development and maintenance effort surrounding the software itself.

Following the initial success of Chemprop numerous representation learning frameworks have been devised, all of slightly improve performance.
The Communicative-MPNN (CMPNN) framework is a modified version of Chemprop with a different message passing scheme to increase the interactions between node and edge features [@cmpnn].
Uni-Mol incorporates 3D information and relies extensively on transformers [@unimol].
In a "full circle moment" architectures like the Molecular Hypergraph Neural Network (MHNN) have been devised to learn representations for specific subsets of chemistry, in that case optoelectronic properties [@mhnn].
Myriad others exist including GSL-MPP (accounts for intra-dataset molecular similarity) [@gslmpp], SGGRL (trains three representations simultaneously using different input formats) [@sggrl], and MOCO (multiple representations and contrastive pretraining) [@moco].

### Limitations
Despite the continuous incremental performance improvements, this area of research has had serious drawbacks.
A thru-theme in these frameworks is the increasing complexity of DL techniques and consequent un-interpretability.
This also means that actually _using_ these methods to do research on real-world dataset requires varying amounts of DL expertise, creating a rift between QSPR experts and these methods.
Perhaps the most significant failure is the inability to achieve good predictions on small [^1] datasets.
This is a long-standing limitation, with the original Chemprop paper stating that datasets with fewer than 1000 entries see fingerprint-based linear on par with Chemprop [@chemprop_theory].

This limitation is especially challenging because it is a _fundamental_ drawback of the LR approach.
Without the use of advanced DL techniques like pre-training or transfer learning, the model is essentially starting from near-zero information every time is trained.
This inherently requires larger datasets to allow the model to effectively 're-learn' the chemical intuition which was built in to descriptor- and fingerprint-based representations.

Efforts are of course underway to address this limitation, though none are broadly successful.
One simple but incredibly computationally expensive approach is to use delta learning, which artificially increases dataset size by generating all possible _pairs_ of molecules from the available data (thus squaring the size of the dataset).
This was attempted by Nalini et al. [@deepdelta], who use an unmodified version of Chemprop referred to as 'DeepDelta' to predict _differences_ in molecular properties for _pairs_ of molecules.
They achieve increased performance over standard LR approaches but _lost_ the ability to train on large datasets due to simple runtime limitations.
Other increasingly complex approaches are discussed in the outstanding review by van Tilborg et al. [@low_data_review], though such techniques are furthering the consequences of complexity mentioned above.

While iterations on LRs and novel approaches to low-data regimes have been in development, the classical QSPR community has continued their work.
A turning point in this domain was the release of `mordred`, a fast and well-developed package capable of calculating more than 1600 molecular descriptors.
Critically this package was fully open source and written in Python, allowing it to readily interoperate with the world-class Python DL software ecosystem that greatly benefitted the LR community.
Now despite previous evidence that molecular descriptors _cannot_ achieve generalizable QSPR in combination with DL, the opposite is shown.

[^1]: What constitutes a 'small' dataset is decidedly _not_ agreed upon by researchers.
For the purposes of this study, it will be used to refer to datasets with ~1000 samples or less, which the authors believe better reflects the size of real-world datasets.

# Implementation
At its core the `fastprop` 'architecture' is simply the `mordred` molecular descriptor calculator [^2] [@mordred] connected to a Feedforward Neural Network (FNN) implemented in PyTorch Lightning [@lightning] (Figure \ref{logo}).
The user simply specifies a set of SMILES [@smiles], a linear textual encoding of molecules, and their corresponding properties.
`fastprop` automatically calculates and caches the corresponding molecular descriptors with `mordred`, re-scales both the descriptors and the targets appropriately, and then trains an FNN with to predict the indicated target properties.
By default this FNN is two hidden layers with 1800 neurons each connected by ReLU activation functions, though the configuration can be readily changed via the command line interface or configuration file.
`fastprop` owes its success to the cogent set of descriptors assembled by the developers of `mordred`, the ease of training FNNs with modern software like PyTorch Lightning, and the careful application of Research Software Engineering best practices that make it as user friendly as the best-maintained alternatives.

![`fastprop` logo.\label{logo}](../fastprop_logo.png){ width=2in }

This trivially simple idea has been alluded to in previous published work but neither described in detail nor lauded for its generalizability or accuracy.
Comesana and coauthors, based on a review of the biofuels property prediction landscape, established that methods (DL or otherwise) using large numbers of molecular descriptors were unsuccessful, instead proposing a feature selection method [@fuels_qsar_method].
As a baseline in a study of photovoltaic property prediction, Wu et al. reported using the `mordred` descriptors in combination with both a Random Forest and an Artificial Neural Network, though the performance is worse than their bespoke model and no code is available for inspection [@wu_photovoltaic].

Others have also incorporated `mordred` descriptors into their modeling efforts, though none with a simple FNN as described above.
Esaki and coauthors started a QSPR study with `mordred` descriptors for a dataset of small molecules, but via an enormously complex modeling pipeline (using only linear methods) removed all but 53 [@fubrain].
Yalamanchi and coauthors used DL on `mordred` descriptors as part of a two-headed representation, but their network architecture was sequential hidden layers _decreasing_ in size to only 12 features [@yalamanchi] as opposed to the constant 1800 in `fastprop`.

The reason `fastprop` stands out from these studies and contradicts previous reports is for the simple reason that it works.
As discussed at length in the [Results & Discussion](#results--discussion) section, this approach matches the performance of leading LR approaches on common benchmark datasets and bespoke QSPR models on small real-world datasets.
`fastprop` also overcomes the limitations of LRs discussed above.
Because all inputs to the FNN are physically meaningful molecular descriptors, intermediate representations in the FNN are also physically meaningful and can be directly interpreted.
The simplicity of the framework enables domain experts to apply it easily and makes model training dramatically faster than LRs.
Most importantly this approach is successful on the _smallest_ of real-world datasets.
By starting from such an informed initialization the FNN can be readily trained on datasets with as few as _forty_ training examples (see [PAHs](#pahs)).

[^2]: The original `mordred` package is no longer maintained. `fastprop` uses a fork of `mordred` called `mordredcommunity` that is maintained by community-contributed patches (see github.com/JacksonBurns/mordred-community).
Multiple descriptor calculators from the very thorough review by McGibbon et al. [@representation_review] could be used instead, though none are as readily interoperable as `mordred`.

## Example Usage
`fastprop` is built with ease of use at the forefront of design.
To that end, input data is accepted in the immensely popular Comma-Separated Values (CSV) format, editable with all modern spreadsheet editors and completely platform independent.
An example specify some properties for benzene is shown below, including its SMILES string:

```
compound,smiles,log_p,retention_index,boiling_point_c,acentric_factor
Benzene,C1=CC=CC=C1,2.04,979,80,0.21
```

`fastprop` itself is accessed via the command line interface, with configuration parameters passed as either command line arguments or in an easily editable configuration file:

```
# pah.yml
# generic args
output_directory: pah
random_seed: 55
problem_type: regression
# featurization
input_file: pah/arockiaraj_pah_data.csv
target_columns: log_p
smiles_column: smiles
descriptors: all
# preprocessing
zero_variance_drop: False
colinear_drop: False
# training
number_repeats: 4
number_epochs: 100
batch_size: 64
patience: 15
train_size: 0.8
val_size: 0.1
test_size: 0.1
sampler: random
```

Training, prediction, and feature importance and then readily accessible via the commands `fastprop train`, `fastprop predict`, or `fastprop shap`, respectively.
The `fastprop` GitHub repository contains a Jupyter notebook runnable from the browser via Google colab which allows users to actually execute the above example, which is also discussed at length in the [PAHs section](#pahs), as well as further details about each configurable option.

# Results & Discussion
There are a number of established molecular property prediction benchmarks commonly used in LR studies, especially those standardized by MoleculeNet [@moleculenet].
Principal among them are QM8 [@qm8] and QM9 [@qm9], often regarded as _the_ standard benchmark for property prediction.
These are important benchmarks and are included here for completeness, though the enormous size and rich coverage of chemical space (inherent in the design of the combinatorial datasets) means that nearly all model architectures are highly accurate.

Real world datasets, particularly those common in QSPR studies, often number in the hundreds.
To demonstrate the applicability of `fastprop` to these regimes, many smaller datasets are selected including some from the QSPR literature that are not established benchmarks.
These studies relied on more complex and slow modeling techniques ([ARA](#ara)) or the design of a bespoke descriptor ([PAHs](#pahs)) and have not yet come to rely on learned representations as a go-to tool.
In these data-limited regimes where LRs sometimes struggle, `fastprop` and its intuition-loaded initialization are highly powerful.
To emphasize this point further, the benchmarks are presented in order of size, descending, for first regression and then classification tasks.
Consistently `fastprop` is able to match the performance of LRs on large datasets (O(10,000)+ entries) and compete with or exceed their performance on small datasets ($\leq$O(1,000) entries).

<!-- The authors of Chemprop have even suggested on GitHub issues in the past that molecular-level features be added to Chemprop learned representations (see https://github.com/chemprop/chemprop/issues/146#issuecomment-784310581) to avoid over-fitting. -->

All of these `fastprop` benchmarks are reproducible, and complete instructions for installation, data retrieval and preparation, and training are publicly available on the `fastprop` GitHub repository at [github.com/jacksonburns/fastprop](https://github.com/jacksonburns/fastprop).

## Benchmark Methods
All benchmarks use 80% of the dataset for training (selected randomly, unless otherwise stated), 10% for validation, and holdout the remaining 10% for testing (unless otherwise stated).
Sampling is performed using the `astartes` package [@astartes] which implements a variety of sampling algorithms and is highly reproducible.

Results for `fastprop` are reported as the average value of a metric and its standard deviation across a number of repetitions (repeated re-sampling of the dataset).
The number of repetitions is chosen to either match referenced literature studies or else increased from two until the performance no longer meaningfully changes.
Note that this is _not_ the same as cross-validation.

For performance metrics retrieved from literature it is assumed that the authors optimized their respective models to achieve the best possible results; therefore, `fastprop` metrics are reported after model optimization using the `fastprop train ... --optimize` option.
When results are generated for this study using Chemprop, the default settings are used except that the number of epochs is increased to allow the model to converge and batch size is increased to match dataset size and speed up training.

When reported, execution time is as given by the unix `time` command using Chemprop version 1.6.1 on Python 3.8 and includes the complete invocation of Chemprop, i.e. `time chemprop_train ...`.
The insignificant time spent manually collating Chemprop results (Chemprop does not natively support repetitions) is excluded.
`fastprop` is run on version 1.0.0b2 using Python 3.11 and timing values are reported according to its internal time measurement  which was verified to be nearly identical to the Unix `time` command.
The coarse comparison of the two packages is intended to emphasize the scaling of LRs and Deep-QSPR and that `fastprop` is, generally speaking, much faster.
All models trained for this study were run on a Dell Precision series laptop with an NVIDIA Quadro RTX 4000 GPU and Intel Xeon E-2286M CPU.

### Performance Metrics
The evaluation metrics used in each of these benchmarks are chosen to match literature precedent, particularly as established by MoleculeNet [@moleculenet], where available.
It is common to use scale-dependent metrics that require readers to understand the relative magnitude of the target variables.
The authors prefer more readily interpretable metrics such as (Weighted) Mean Absolute Percentage Error (W/MAPE) and are included where relevant.

All metrics are defined according to their typical formulae which are readily available online and are implemented in common software packages.
Those presented here are summarized below, first for regression:
 - Mean Absolute Error (MAE): Absolute difference between predictions and ground truth averaged across dataset; scale-dependent.
 - Root Mean Squared Error (RMSE): Absolute differences _squared_ and then averaged; scale-dependent.
 - Mean Absolute Percentage Error (MAPE): MAE except that differences are relative (i.e. divided by the ground truth); scale-independent, range 0 (best) and up.
 - Weighted Mean Absolute Percentage Error (WMAPE): MAPE except the average is a weighted average, where the weight is the magnitude of the ground truth; scale-independent, range 0 (best) and up.
 - Coefficient of Determination (R2): Proportion of variance explained; scale-independent, range 0 (worst) to 1 (best).
and classification:
 - Area Under the Receiver Operating Curve (AUROC, AUC, or ROC-AUC): Summary statistic combining all possible classification errors; scale-independent, range 0.5 (worst, random guessing) to 1.0 (perfect classifier).
 - Accuracy: Fraction of correct classifications, expressed as either a percentage or a number; scale-independent, range 0 (worst) to 1 (perfect classifier).

## Regression Datasets
See Table \ref{regression_results_table} for a summary of all the regression dataset results.
Especially noteworthy is the performance on the ESOL and PAH datasets, which dramatically surpass literature best and have only 55 datapoints, respectively.
Citations for the datasets themselves are included in the sub-sections of this section.

Table: Summary of regression benchmark results. \label{regression_results_table}

+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|   Benchmark   | Samples (k)        |   Metric    |          Literature Best          | `fastprop` |        Chemprop         |
+===============+====================+=============+===================================+============+=========================+
|QM9            |~134                |MAE          |0.0047$^a$                         |0.0069      |0.0081$^a$               |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|OCELOTv1       |~25                 |MAE          |0.128$^b$                          |0.158       |0.140$^b$                |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|QM8            |~22                 |MAE          |0.016$^a$                          |0.018       |0.019$^a$                |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|ESOL           |~1.1                |RMSE         |0.55$^c$                           |0.64        |0.67$^c$                 |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|FreeSolv       |~0.6                |RMSE         |0.82$^c$                           |1.33        |1.26$^c$                 |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|Flash          |~0.6                |RMSE         |13.2$^e$                           |13.3        |21.2*                    |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|YSI            |~0.4                |MAE          |22.3$^f$                           |13.6        |28.9*                    |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|HOPV15$^g$     |~0.3                |MAE          |1.32$^g$                           |1.55        |1.60                     |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|Fubrain        |~0.3                |RMSE         |0.44$^h$/0.83$^d$                  |0.19/0.74   |0.22*/0.97$^d$           |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+
|PAH            |~0.06               |R2           |0.96$^i$                           |0.98        |0.59*                    |
+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+

a [@unimol] b [@mhnn] (summary result is geometric mean across tasks) c [@cmpnn] d [@deepdelta] (delta-learning instead of direct prediction) e [@flash] f [@ysi] g [@hopv15_subset] (uses a subset of the complete HOPV15 dataset) h [@fubrain] i [@pah] * These results were generated for this study.

### QM9
Originally described in Scientific Data [@qm9] and perhaps the most established property prediction benchmark, Quantum Machine 9 (QM9) provides quantum mechanics derived descriptors for all small molecules containing one to nine heavy atoms, totaling ~134k.
The data was retrieved from MoleculeNet [@moleculenet] in a readily usable format.
As a point of comparison, performance metrics are retrieved from the paper presenting the UniMol architecture [@unimol] previously mentioned.
In that study they trained on only three especially difficult QOIs (homo, lumo, and gap) using scaffold-based splitting (a more challenging alternative to random splitting), reporting mean and standard deviation across 3 repetitions.

`fastprop` achieves 0.0069 $\pm$ 0.0002 mean absolute error, whereas Chemprop achieves 0.00814 $\pm$ 0.00001 and the UniMol framework manages 0.00467 $\pm$ 0.00004.
This places the `fastprop` framework ahead of previous learned representation approaches but still trailing UniMol.
This is not completely unexpected since UniMol encodes 3D information from the dataset whereas Chemprop and `fastprop` use only 2D.
Future work could evaluate the use of 3D-based descriptors to improve `fastprop` performance in the same manner that UniMol has with LRs.
<!-- [02/22/2024 02:00:05 PM fastprop.fastprop_core] INFO: Displaying validation results:
                              count          mean           std          min           25%           50%           75%           max
validation_mse_loss             3.0      0.074331      0.004045     0.071092      0.072064      0.073036      0.075951      0.078866
validation_mean_mape            3.0   8916.004822   8791.351180   685.089172   4285.491608   7885.894043  13031.462646  18177.031250
validation_mape_output_homo     3.0      0.024943      0.000986     0.024281      0.024377      0.024472      0.025274      0.026076
validation_mape_output_lumo     3.0  26747.952230  26374.052969  2055.206299  12856.412720  23657.619141  39094.325195  54531.031250
validation_mape_output_gap      3.0      0.038382      0.001576     0.036933      0.037543      0.038154      0.039107      0.040060
validation_mean_wmape           3.0      0.029807      0.001175     0.028462      0.029392      0.030322      0.030480      0.030637
validation_wmape_output_homo    3.0      0.024363      0.001001     0.023704      0.023787      0.023871      0.024693      0.025515
validation_wmape_output_lumo    3.0      0.029821      0.006076     0.022880      0.027640      0.032400      0.033291      0.034182
validation_wmape_output_gap     3.0      0.035238      0.001599     0.033860      0.034361      0.034862      0.035927      0.036991
validation_l1_avg               3.0      0.006832      0.000228     0.006605      0.006717      0.006830      0.006945      0.007061
validation_l1_output_homo       3.0      0.005640      0.000169     0.005450      0.005572      0.005693      0.005734      0.005775
validation_l1_output_lumo       3.0      0.006222      0.000330     0.005971      0.006036      0.006101      0.006348      0.006595
validation_l1_output_gap        3.0      0.008634      0.000321     0.008263      0.008538      0.008813      0.008819      0.008825
validation_rmse_avg             3.0      0.009365      0.000316     0.009006      0.009248      0.009489      0.009545      0.009600
validation_rmse_output_homo     3.0      0.007635      0.000120     0.007498      0.007592      0.007686      0.007703      0.007720
validation_rmse_output_lumo     3.0      0.008663      0.000514     0.008278      0.008372      0.008465      0.008856      0.009247
validation_rmse_output_gap      3.0      0.011798      0.000523     0.011243      0.011556      0.011869      0.012075      0.012282
[02/22/2024 02:00:05 PM fastprop.fastprop_core] INFO: Displaying testing results:
                        count          mean           std           min           25%           50%           75%            max
test_mse_loss             3.0      0.073060      0.000311      0.072757      0.072901      0.073045      0.073212       0.073378
test_mean_mape            3.0  20154.996419  19143.980541   6330.125000   9229.730957  12129.336914  27067.432129   42005.527344
test_mape_output_homo     3.0      0.023933      0.000574      0.023577      0.023601      0.023626      0.024110       0.024595
test_mape_output_lumo     3.0  60464.930990  57431.941419  18990.316406  27689.134766  36387.953125  81202.238281  126016.523438
test_mape_output_gap      3.0      0.038007      0.002316      0.035378      0.037136      0.038893      0.039321       0.039748
test_mean_wmape           3.0      0.057512      0.050808      0.023704      0.028298      0.032892      0.074416       0.115940
test_wmape_output_homo    3.0      0.023486      0.000632      0.023037      0.023124      0.023211      0.023710       0.024209
test_wmape_output_lumo    3.0      0.114208      0.152918      0.010358      0.026408      0.042458      0.166133       0.289807
test_wmape_output_gap     3.0      0.034843      0.001773      0.033007      0.033992      0.034976      0.035761       0.036547
test_l1_avg               3.0      0.006873      0.000226      0.006668      0.006751      0.006834      0.006975       0.007115
test_l1_output_homo       3.0      0.005614      0.000129      0.005511      0.005541      0.005571      0.005665       0.005759
test_l1_output_lumo       3.0      0.006342      0.000237      0.006069      0.006265      0.006461      0.006478       0.006495
test_l1_output_gap        3.0      0.008663      0.000406      0.008365      0.008431      0.008497      0.008811       0.009126
test_rmse_avg             3.0      0.009353      0.000259      0.009071      0.009240      0.009410      0.009494       0.009579
test_rmse_output_homo     3.0      0.007488      0.000155      0.007322      0.007419      0.007517      0.007572       0.007627
test_rmse_output_lumo     3.0      0.008727      0.000372      0.008318      0.008566      0.008814      0.008931       0.009047
test_rmse_output_gap      3.0      0.011845      0.000460      0.011377      0.011619      0.011861      0.012079       0.012296
[02/22/2024 02:00:05 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.962>0.05.
[02/22/2024 02:00:06 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/22/2024 02:00:06 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:56:27.500738 -->

### OCELOTv1
The Organic Crystals in Electronic and Light-Oriented Technologies (OCELOTv1) dataset, originally described by Bhat et al. [@ocelot], maps 15 quantum mechanics descriptors and optoelectronic properties to ~25k chromophoric small molecules.
The literature best model is the Molecular Hypergraph Neural Network (MHNN) [@mhnn] which specializes in learned representations for optoelectronic properties and also includes Chemprop as a baseline for comparison.
They used a 70/10/20 random split with three repetitions and final performance reported is the average across those three repetitions.

As done in the reference study, the MAE for each task is shown in Table \ref{ocelot_results_table}.
Meanings for each abbreviation are the same as in the original database publication [@ocelot].
The geometric mean across all tasks, which accounts for the different scales of the target values better than the arithmetic mean, is also included as a summary statistic.
Note also that the relative percentage performance difference between fastprop and chemprop (`fast/chem`) and fastprop and MHNN (`fast/MHNN`) are also included.

Table: Per-task OCELOT dataset results. MHNN and Chemprop results are retrieved from the literature [@mhnn]. \label{ocelot_results_table}

+--------+----------+----------+-----------+-------+-----------+
| Target | fastprop | Chemprop | fast/chem | MHNN  | fast/mhnn |
+========+==========+==========+===========+=======+===========+
| HOMO   | 0.328    | 0.330    | -0.7%     | 0.306 | 7.1%      |
+--------+----------+----------+-----------+-------+-----------+
| LUMO   | 0.285    | 0.289    | -1.5%     | 0.258 | 10.4%     |
+--------+----------+----------+-----------+-------+-----------+
| H-L    | 0.555    | 0.548    | 1.4%      | 0.519 | 7.0%      |
+--------+----------+----------+-----------+-------+-----------+
| VIE    | 0.210    | 0.191    | 9.9%      | 0.178 | 18.0%     |
+--------+----------+----------+-----------+-------+-----------+
| AIE    | 0.201    | 0.173    | 16.2%     | 0.162 | 24.0%     |
+--------+----------+----------+-----------+-------+-----------+
| CR1    | 0.057    | 0.055    | 3.3%      | 0.053 | 7.2%      |
+--------+----------+----------+-----------+-------+-----------+
| CR2    | 0.057    | 0.053    | 7.2%      | 0.052 | 9.3%      |
+--------+----------+----------+-----------+-------+-----------+
| HR     | 0.108    | 0.133    | -18.5%    | 0.099 | 9.5%      |
+--------+----------+----------+-----------+-------+-----------+
| VEA    | 0.194    | 0.157    | 23.5%     | 0.138 | 40.5%     |
+--------+----------+----------+-----------+-------+-----------+
| AEA    | 0.188    | 0.154    | 21.8%     | 0.124 | 51.3%     |
+--------+----------+----------+-----------+-------+-----------+
| AR1    | 0.055    | 0.051    | 8.2%      | 0.050 | 10.4%     |
+--------+----------+----------+-----------+-------+-----------+
| AR2    | 0.049    | 0.052    | -5.1%     | 0.046 | 7.3%      |
+--------+----------+----------+-----------+-------+-----------+
| ER     | 0.100    | 0.098    | 1.9%      | 0.092 | 8.5%      |
+--------+----------+----------+-----------+-------+-----------+
| S0S1   | 0.284    | 0.249    | 14.3%     | 0.241 | 18.0%     |
+--------+----------+----------+-----------+-------+-----------+
| S0T1   | 0.222    | 0.150    | 48.3%     | 0.145 | 53.4%     |
+--------+----------+----------+-----------+-------+-----------+
| G-Mean | 0.151    | 0.140    | 7.7%      | 0.128 | 17.9%     |
+--------+----------+----------+-----------+-------+-----------+

`fastprop` 'trades places with Chemprop, outperforming on four of the metrics (LUMO, HR, AR2) and under-performing on others.
Overall the geometric mean of MAE across all the tasks is ~8% higher, though this result may not be statistically significant.
Both `fastprop` and Chemprop are outperformed by the bespoke MHNN model, which is not itself evaluated on any other common property prediction benchmarks.

Although `fastprop` is not able to reach state-of-the-art accuracy on this dataset this result is still promising.
None of the descriptors implemented in `mordred` were designed to specifically correlate to these QM-derived targets, yet the FNN is able to learn a representation which is nearly as informative as Chemprop.
The fact that a bespoke modeling approach is the most performant is not surprising and instead demonstrates the continued importance of expert input on certain domains.
Were some of this expertise to be oriented toward the descriptor generation software `mordredcommunity`, new descriptors could be added to address this apparent shortcoming.
<!-- [02/22/2024 08:13:00 PM fastprop.fastprop_core] INFO: Displaying validation results:
                              count         mean          std       min       25%       50%          75%           max
validation_mse_loss             3.0     0.458501     0.057164  0.413317  0.426369  0.439421     0.481093      0.522765
validation_mean_mape            3.0   343.158319   593.544873  0.458272  0.475027  0.491782   514.508342   1028.524902
validation_mape_output_homo     3.0     0.044271     0.000460  0.043919  0.044011  0.044103     0.044447      0.044791
validation_mape_output_lumo     3.0  5142.822895  8903.641326  1.706457  2.303189  2.899922  7713.381114  15423.862305
validation_mape_output_hl       3.0     0.081335     0.000831  0.080375  0.081091  0.081806     0.081815      0.081824
validation_mape_output_vie      3.0     0.028175     0.000733  0.027661  0.027756  0.027850     0.028432      0.029014
validation_mape_output_aie      3.0     0.027848     0.000672  0.027443  0.027460  0.027477     0.028050      0.028623
validation_mape_output_cr1      3.0     0.327836     0.011577  0.314549  0.323875  0.333202     0.334479      0.335757
validation_mape_output_cr2      3.0     0.299127     0.009663  0.290125  0.294022  0.297918     0.303628      0.309337
validation_mape_output_hr       3.0     0.289242     0.009344  0.278568  0.285891  0.293213     0.294579      0.295945
validation_mape_output_vea      3.0     1.507243     0.574370  1.087102  1.179996  1.272890     1.717313      2.161736
validation_mape_output_aea      3.0     1.059045     0.125033  0.942062  0.993160  1.044258     1.117536      1.190814
validation_mape_output_ar1      3.0     0.240626     0.004368  0.237916  0.238106  0.238296     0.241980      0.245664
validation_mape_output_ar2      3.0     0.223688     0.005050  0.219446  0.220895  0.222344     0.225809      0.229274
validation_mape_output_er       3.0     0.223191     0.004302  0.219137  0.220935  0.222733     0.225218      0.227704
validation_mape_output_s0s1     3.0     0.084802     0.002831  0.082518  0.083219  0.083920     0.085945      0.087969
validation_mape_output_s0t1     3.0     0.115699     0.000624  0.115008  0.115439  0.115869     0.116045      0.116221
validation_mean_wmape           3.0     0.161395     0.003184  0.157825  0.160122  0.162420     0.163180      0.163941
validation_wmape_output_homo    3.0     0.045143     0.000701  0.044689  0.044739  0.044789     0.045370      0.045951
validation_wmape_output_lumo    3.0     0.220110     0.025429  0.205398  0.205428  0.205458     0.227466      0.249473
validation_wmape_output_hl      3.0     0.083272     0.001417  0.082091  0.082486  0.082882     0.083863      0.084844
validation_wmape_output_vie     3.0     0.028110     0.000687  0.027695  0.027713  0.027732     0.028317      0.028903
validation_wmape_output_aie     3.0     0.027690     0.000632  0.027246  0.027329  0.027412     0.027913      0.028414
validation_wmape_output_cr1     3.0     0.275331     0.004325  0.272819  0.272834  0.272849     0.276587      0.280325
validation_wmape_output_cr2     3.0     0.268745     0.004291  0.264348  0.266657  0.268965     0.270944      0.272922
validation_wmape_output_hr      3.0     0.258327     0.003611  0.254850  0.256460  0.258070     0.260065      0.262059
validation_wmape_output_vea     3.0     0.216599     0.013251  0.207212  0.209020  0.210828     0.221292      0.231756
validation_wmape_output_aea     3.0     0.185877     0.006514  0.178391  0.183690  0.188988     0.189620      0.190251
validation_wmape_output_ar1     3.0     0.223806     0.005709  0.220145  0.220517  0.220889     0.225637      0.230384
validation_wmape_output_ar2     3.0     0.209507     0.003840  0.206244  0.207391  0.208538     0.211138      0.213738
validation_wmape_output_er      3.0     0.207492     0.004985  0.203313  0.204733  0.206153     0.209581      0.213009
validation_wmape_output_s0s1    3.0     0.079778     0.003081  0.076763  0.078207  0.079650     0.081286      0.082922
validation_wmape_output_s0t1    3.0     0.091143     0.003796  0.088802  0.088954  0.089106     0.092314      0.095523
validation_l1_avg               3.0     0.193970     0.002999  0.190706  0.192653  0.194599     0.195602      0.196605
validation_l1_output_homo       3.0     0.329893     0.005807  0.326516  0.326540  0.326565     0.331581      0.336597
validation_l1_output_lumo       3.0     0.285641     0.003469  0.283326  0.283647  0.283969     0.286799      0.289630
validation_l1_output_hl         3.0     0.557060     0.011627  0.548925  0.550401  0.551876     0.561127      0.570377
validation_l1_output_vie        3.0     0.211479     0.005273  0.208258  0.208436  0.208615     0.213089      0.217564
validation_l1_output_aie        3.0     0.202533     0.004793  0.198934  0.199813  0.200692     0.204333      0.207973
validation_l1_output_cr1        3.0     0.057598     0.000743  0.056813  0.057252  0.057690     0.057990      0.058290
validation_l1_output_cr2        3.0     0.057276     0.000534  0.056799  0.056988  0.057177     0.057515      0.057852
validation_l1_output_hr         3.0     0.109095     0.000700  0.108440  0.108727  0.109013     0.109423      0.109832
validation_l1_output_vea        3.0     0.194558     0.005324  0.191008  0.191497  0.191985     0.196332      0.200680
validation_l1_output_aea        3.0     0.189130     0.006161  0.183656  0.185794  0.187932     0.191867      0.195801
validation_l1_output_ar1        3.0     0.055498     0.001648  0.054153  0.054579  0.055006     0.056171      0.057337
validation_l1_output_ar2        3.0     0.049942     0.000975  0.049336  0.049380  0.049424     0.050245      0.051067
validation_l1_output_er         3.0     0.100913     0.002594  0.099312  0.099417  0.099522     0.101714      0.103905
validation_l1_output_s0s1       3.0     0.284504     0.011130  0.273050  0.279117  0.285184     0.290232      0.295279
validation_l1_output_s0t1       3.0     0.224430     0.009287  0.218065  0.219101  0.220138     0.227612      0.235087
validation_rmse_avg             3.0     0.268082     0.015400  0.254445  0.259731  0.265016     0.274901      0.284785
validation_rmse_output_homo     3.0     0.450869     0.014164  0.435441  0.444660  0.453879     0.458583      0.463286
validation_rmse_output_lumo     3.0     0.389335     0.023207  0.368810  0.376743  0.384677     0.399598      0.414518
validation_rmse_output_hl       3.0     0.767300     0.036090  0.726387  0.753642  0.780898     0.787757      0.794616
validation_rmse_output_vie      3.0     0.286842     0.010691  0.277076  0.281130  0.285184     0.291724      0.298265
validation_rmse_output_aie      3.0     0.271062     0.003749  0.267076  0.269334  0.271593     0.273055      0.274517
validation_rmse_output_cr1      3.0     0.081833     0.003779  0.077840  0.080072  0.082305     0.083830      0.085354
validation_rmse_output_cr2      3.0     0.081802     0.000914  0.081066  0.081290  0.081515     0.082170      0.082825
validation_rmse_output_hr       3.0     0.150002     0.004057  0.146468  0.147787  0.149106     0.151769      0.154432
validation_rmse_output_vea      3.0     0.275262     0.031631  0.256818  0.256999  0.257181     0.284483      0.311786
validation_rmse_output_aea      3.0     0.262731     0.020956  0.247612  0.250771  0.253929     0.270291      0.286653
validation_rmse_output_ar1      3.0     0.078577     0.006309  0.074131  0.074967  0.075803     0.080800      0.085797
validation_rmse_output_ar2      3.0     0.072570     0.007629  0.067899  0.068169  0.068438     0.074906      0.081374
validation_rmse_output_er       3.0     0.141838     0.017033  0.131141  0.132017  0.132893     0.147187      0.161481
validation_rmse_output_s0s1     3.0     0.392055     0.033613  0.362017  0.373903  0.385789     0.407075      0.428360
validation_rmse_output_s0t1     3.0     0.319151     0.033692  0.296891  0.299770  0.302648     0.330281      0.357913
[02/22/2024 08:13:00 PM fastprop.fastprop_core] INFO: Displaying testing results:
                        count         mean           std       min       25%       50%          75%           max
test_mse_loss             3.0     0.426169      0.009204  0.417342  0.421399  0.425456     0.430582      0.435709
test_mean_mape            3.0   395.660539    684.517903  0.451032  0.453944  0.456855   593.265293   1186.073730
test_mape_output_homo     3.0     0.043939      0.000667  0.043280  0.043602  0.043923     0.044268      0.044614
test_mape_output_lumo     3.0  5929.894403  10267.922058  1.603237  1.706839  1.810442  8894.039987  17786.269531
test_mape_output_hl       3.0     0.081036      0.002200  0.079238  0.079809  0.080381     0.081935      0.083490
test_mape_output_vie      3.0     0.027956      0.000631  0.027577  0.027591  0.027606     0.028145      0.028684
test_mape_output_aie      3.0     0.027617      0.000926  0.027077  0.027083  0.027089     0.027888      0.028687
test_mape_output_cr1      3.0     0.312857      0.012791  0.302067  0.305792  0.309517     0.318252      0.326987
test_mape_output_cr2      3.0     0.310919      0.008263  0.305318  0.306174  0.307031     0.313720      0.320409
test_mape_output_hr       3.0     0.293555      0.005269  0.287525  0.291697  0.295868     0.296571      0.297273
test_mape_output_vea      3.0     1.621847      0.517986  1.265081  1.324780  1.384480     1.800230      2.215980
test_mape_output_aea      3.0     1.405634      0.298437  1.061155  1.315473  1.569791     1.577873      1.585954
test_mape_output_ar1      3.0     0.242464      0.004575  0.238820  0.239897  0.240974     0.244286      0.247599
test_mape_output_ar2      3.0     0.225263      0.001945  0.223330  0.224285  0.225239     0.226229      0.227219
test_mape_output_er       3.0     0.223549      0.002041  0.221204  0.222861  0.224518     0.224722      0.224925
test_mape_output_s0s1     3.0     0.084606      0.001663  0.082705  0.084010  0.085315     0.085556      0.085796
test_mape_output_s0t1     3.0     0.112468      0.003794  0.108290  0.110854  0.113417     0.114557      0.115697
test_mean_wmape           3.0     0.160358      0.001505  0.159404  0.159491  0.159578     0.160835      0.162093
test_wmape_output_homo    3.0     0.044842      0.000560  0.044256  0.044576  0.044897     0.045134      0.045372
test_wmape_output_lumo    3.0     0.227117      0.014630  0.215973  0.218833  0.221693     0.232689      0.243685
test_wmape_output_hl      3.0     0.083099      0.001810  0.081631  0.082088  0.082545     0.083833      0.085122
test_wmape_output_vie     3.0     0.027916      0.000611  0.027529  0.027564  0.027598     0.028109      0.028620
test_wmape_output_aie     3.0     0.027478      0.000902  0.026934  0.026957  0.026980     0.027750      0.028519
test_wmape_output_cr1     3.0     0.272248      0.002985  0.269788  0.270588  0.271387     0.273478      0.275568
test_wmape_output_cr2     3.0     0.268046      0.003807  0.263727  0.266612  0.269497     0.270206      0.270915
test_wmape_output_hr      3.0     0.257790      0.002427  0.255081  0.256802  0.258522     0.259144      0.259766
test_wmape_output_vea     3.0     0.208417      0.000352  0.208042  0.208254  0.208465     0.208604      0.208742
test_wmape_output_aea     3.0     0.180743      0.006029  0.174713  0.177729  0.180745     0.183758      0.186771
test_wmape_output_ar1     3.0     0.223203      0.001243  0.221771  0.222809  0.223847     0.223920      0.223992
test_wmape_output_ar2     3.0     0.208085      0.001777  0.206370  0.207168  0.207966     0.208942      0.209918
test_wmape_output_er      3.0     0.206107      0.001264  0.205288  0.205379  0.205470     0.206516      0.207563
test_wmape_output_s0s1    3.0     0.079831      0.001651  0.078031  0.079109  0.080188     0.080731      0.081274
test_wmape_output_s0t1    3.0     0.090454      0.003366  0.088019  0.088533  0.089047     0.091671      0.094295
test_l1_avg               3.0     0.192914      0.002772  0.190113  0.191543  0.192972     0.194314      0.195656
test_l1_output_homo       3.0     0.327626      0.004000  0.323577  0.325651  0.327725     0.329650      0.331576
test_l1_output_lumo       3.0     0.284796      0.005949  0.277983  0.282713  0.287444     0.288202      0.288961
test_l1_output_hl         3.0     0.555456      0.011025  0.546451  0.549308  0.552165     0.559958      0.567752
test_l1_output_vie        3.0     0.209976      0.004657  0.207102  0.207290  0.207478     0.211414      0.215349
test_l1_output_aie        3.0     0.200949      0.006628  0.197027  0.197123  0.197219     0.202911      0.208602
test_l1_output_cr1        3.0     0.056798      0.000443  0.056424  0.056553  0.056682     0.056985      0.057287
test_l1_output_cr2        3.0     0.056833      0.000437  0.056351  0.056648  0.056945     0.057074      0.057204
test_l1_output_hr         3.0     0.108441      0.000386  0.108096  0.108233  0.108370     0.108614      0.108858
test_l1_output_vea        3.0     0.193955      0.010548  0.185665  0.188019  0.190373     0.198100      0.205828
test_l1_output_aea        3.0     0.187564      0.009439  0.180720  0.182180  0.183640     0.190986      0.198332
test_l1_output_ar1        3.0     0.055178      0.000731  0.054334  0.054967  0.055600     0.055600      0.055601
test_l1_output_ar2        3.0     0.049358      0.000558  0.048928  0.049043  0.049157     0.049573      0.049989
test_l1_output_er         3.0     0.099840      0.001068  0.098820  0.099285  0.099750     0.100350      0.100951
test_l1_output_s0s1       3.0     0.284490      0.005678  0.278788  0.281663  0.284539     0.287342      0.290145
test_l1_output_s0t1       3.0     0.222443      0.008583  0.215991  0.217572  0.219154     0.225669      0.232184
test_rmse_avg             3.0     0.259388      0.004270  0.256027  0.256986  0.257945     0.261069      0.264193
test_rmse_output_homo     3.0     0.445763      0.001143  0.444444  0.445430  0.446415     0.446423      0.446431
test_rmse_output_lumo     3.0     0.373126      0.006550  0.366760  0.369766  0.372772     0.376309      0.379846
test_rmse_output_hl       3.0     0.749384      0.004661  0.745422  0.746816  0.748211     0.751365      0.754520
test_rmse_output_vie      3.0     0.284846      0.008854  0.279273  0.279742  0.280210     0.287633      0.295055
test_rmse_output_aie      3.0     0.270991      0.010721  0.263813  0.264829  0.265846     0.274580      0.283314
test_rmse_output_cr1      3.0     0.080434      0.000323  0.080063  0.080326  0.080589     0.080620      0.080651
test_rmse_output_cr2      3.0     0.080343      0.001093  0.079166  0.079852  0.080539     0.080932      0.081325
test_rmse_output_hr       3.0     0.147213      0.001475  0.145690  0.146502  0.147314     0.147975      0.148635
test_rmse_output_vea      3.0     0.257247      0.012767  0.247803  0.249985  0.252166     0.261969      0.271772
test_rmse_output_aea      3.0     0.249757      0.011438  0.241501  0.243229  0.244957     0.253885      0.262814
test_rmse_output_ar1      3.0     0.075746      0.002504  0.072856  0.074991  0.077125     0.077191      0.077256
test_rmse_output_ar2      3.0     0.067676      0.000842  0.067045  0.067198  0.067351     0.067992      0.068632
test_rmse_output_er       3.0     0.132936      0.002416  0.130385  0.131809  0.133232     0.134211      0.135190
test_rmse_output_s0s1     3.0     0.376503      0.006149  0.371012  0.373181  0.375351     0.379249      0.383147
test_rmse_output_s0t1     3.0     0.298857      0.008914  0.292517  0.293760  0.295003     0.302026      0.309050
[02/22/2024 08:13:00 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.399>0.05.
[02/22/2024 08:13:00 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/22/2024 08:13:00 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:06:22.415257 -->

### QM8
Quantum Machine 8 (QM8) is the predecessor to QM9 first described in 2015 [@qm8].
It follows the same generation procedure as QM9 but includes only up to eight heavy atoms for a total of approximately 22k molecules.
Again, this study used the dataset as prepared by MoleculeNet [@moleculenet] and compares to the UniMol [@unimol] set of benchmarks as a reference point, wherein they used the same data splitting procedure described previously but regressed all 12 targets in QM8.

UniMol achieved an average MAE across all tasks of 0.00156 $\pm$ 0.0001, `fastprop` approaches that performance with 0.0178 $\pm$ 0.003, and Chemprop trails both frameworks with 0.0190 $\pm$ 0.0001.
Much like with QM9 `fastprop` outperforms LR frameworks until 3D information is encoded with UniMol.
As previously stated this is achieved despite the targets being predicted not being directly intended for correlation with the `mordred` descriptors.

Of note is that even though `fastprop` is approaching the leading performance on this benchmark, other performance metrics cast doubt on the model performance.
The weighted mean absolute percentage error (wMAPE) on a per-task basis is shown in Table \ref{qm8_results_table}.

Table: Per-task QM8 dataset results. \label{qm8_results_table}

+---------+-------+
| Metric  | wMAPE |
+=========+=======+
| E1-CC2  | 3.6%  |
+---------+-------+
| E2-CC2  | 3.3%  |
+---------+-------+
| f1-CC2  | 83.0% |
+---------+-------+
| f2-CC2  | 86.7% |
+---------+-------+
| E1-PBE0 | 3.6%  |
+---------+-------+
| E2-PBE0 | 3.3%  |
+---------+-------+
| f1-PBE0 | 80.6% |
+---------+-------+
| F2-PBE0 | 86.4% |
+---------+-------+
| E1-CAM  | 3.4%  |
+---------+-------+
| E2-CAM  | 3.0%  |
+---------+-------+
| f1-CAM  | 77.3% |
+---------+-------+
| f2-CAM  | 81.2% |
+---------+-------+
| Average | 43.0% |
+---------+-------+

At each level of theory (CC2, PBE0, and CAM) `fastprop` is reaching the limit of chemical accuracy on excitation energies (E1 and E2) but is significantly less accurate on oscillator strengths (f1 and f2).
This can at least partially be attributed to dataset itself.
Manual analysis reveals that nearly 90% of the molecules in the dataset fall within only 10% of the total range of f1 values, which is highly imbalanced.
Additionally that 90% of molecules actual f1 values are all near-zero or zero, which are intentionally less represented in the wMAPE metric.
Future literature studies should take this observation into account and perhaps move away from this splitting approach toward one which accounts for this imbalance.

<!-- [02/22/2024 08:17:34 PM fastprop.fastprop_core] INFO: Displaying validation results:
                                 count          mean           std           min           25%           50%           75%           max
validation_mse_loss                3.0  2.961224e-01  2.147881e-02  2.737574e-01  2.858891e-01  2.980208e-01  3.073049e-01  3.165890e-01
validation_mean_mape               3.0  3.726341e+05  2.042732e+05  1.627118e+05  2.735783e+05  3.844448e+05  4.775952e+05  5.707458e+05
validation_mape_output_E1-CC2      3.0  3.319096e-02  9.908976e-04  3.218177e-02  3.270519e-02  3.322861e-02  3.369555e-02  3.416249e-02
validation_mape_output_E2-CC2      3.0  3.095054e-02  1.307501e-03  2.995400e-02  3.021030e-02  3.046659e-02  3.144881e-02  3.243102e-02
validation_mape_output_f1-CC2      3.0  2.465183e+05  2.729970e+05  5.374423e+04  9.032484e+04  1.269055e+05  3.429054e+05  5.589053e+05
validation_mape_output_f2-CC2      3.0  2.003174e+05  2.170272e+05  3.900190e+03  8.382240e+04  1.637446e+05  2.985260e+05  4.333074e+05
validation_mape_output_E1-PBE0     3.0  3.399180e-02  7.291666e-04  3.355667e-02  3.357089e-02  3.358512e-02  3.420936e-02  3.483361e-02
validation_mape_output_E2-PBE0     3.0  3.149488e-02  2.512730e-03  2.946717e-02  3.008932e-02  3.071148e-02  3.250873e-02  3.430598e-02
validation_mape_output_f1-PBE0     3.0  2.126848e+04  2.123202e+03  1.901504e+04  2.028694e+04  2.155884e+04  2.239520e+04  2.323156e+04
validation_mape_output_f2-PBE0     3.0  1.153059e+05  1.098323e+05  6.536243e+03  5.987344e+04  1.132106e+05  1.696908e+05  2.261709e+05
validation_mape_output_E1-CAM      3.0  3.279775e-02  8.746607e-04  3.227762e-02  3.229284e-02  3.230806e-02  3.305782e-02  3.380757e-02
validation_mape_output_E2-CAM      3.0  2.926853e-02  1.722292e-03  2.743344e-02  2.847786e-02  2.952229e-02  3.018608e-02  3.084986e-02
validation_mape_output_f1-CAM      3.0  1.377005e+06  3.397459e+05  1.116488e+06  1.184867e+06  1.253245e+06  1.507264e+06  1.761282e+06
validation_mape_output_f2-CAM      3.0  2.511194e+06  1.899370e+06  6.754806e+05  1.532596e+06  2.389712e+06  3.429051e+06  4.468390e+06
validation_mean_wmape              3.0  4.475308e-01  4.980988e-02  3.906429e-01  4.296396e-01  4.686364e-01  4.759747e-01  4.833130e-01
validation_wmape_output_E1-CC2     3.0  3.152936e-02  4.848671e-04  3.097065e-02  3.137404e-02  3.177742e-02  3.180872e-02  3.184001e-02
validation_wmape_output_E2-CC2     3.0  2.929318e-02  8.359605e-04  2.863467e-02  2.882293e-02  2.901120e-02  2.962243e-02  3.023366e-02
validation_wmape_output_f1-CC2     3.0  8.989161e-01  2.181237e-01  6.487983e-01  8.235534e-01  9.983085e-01  1.023975e+00  1.049641e+00
validation_wmape_output_f2-CC2     3.0  8.460621e-01  1.648189e-02  8.294763e-01  8.378741e-01  8.462719e-01  8.543550e-01  8.624381e-01
validation_wmape_output_E1-PBE0    3.0  3.218072e-02  3.220693e-04  3.196875e-02  3.199541e-02  3.202206e-02  3.228670e-02  3.255133e-02
validation_wmape_output_E2-PBE0    3.0  2.950394e-02  1.974154e-03  2.791693e-02  2.839861e-02  2.888028e-02  3.029744e-02  3.171460e-02
validation_wmape_output_f1-PBE0    3.0  9.767324e-01  2.569331e-01  6.813912e-01  8.907032e-01  1.100015e+00  1.124403e+00  1.148791e+00
validation_wmape_output_f2-PBE0    3.0  7.976509e-01  7.779241e-02  7.323164e-01  7.546236e-01  7.769308e-01  8.303182e-01  8.837056e-01
validation_wmape_output_E1-CAM     3.0  3.121153e-02  4.874624e-04  3.080885e-02  3.094056e-02  3.107227e-02  3.141287e-02  3.175347e-02
validation_wmape_output_E2-CAM     3.0  2.774681e-02  1.256585e-03  2.636963e-02  2.720469e-02  2.803975e-02  2.843540e-02  2.883104e-02
validation_wmape_output_f1-CAM     3.0  8.246486e-01  2.343411e-01  5.753399e-01  7.167703e-01  8.582007e-01  9.493030e-01  1.040405e+00
validation_wmape_output_f2-CAM     3.0  8.448935e-01  5.543766e-02  7.834639e-01  8.217396e-01  8.600153e-01  8.756082e-01  8.912012e-01
validation_l1_avg                  3.0  1.629429e-02  2.897021e-04  1.596688e-02  1.618273e-02  1.639857e-02  1.645799e-02  1.651742e-02
validation_l1_output_E1-CC2        3.0  7.100401e-03  7.142299e-05  7.028310e-03  7.065033e-03  7.101756e-03  7.136446e-03  7.171136e-03
validation_l1_output_E2-CC2        3.0  7.434885e-03  1.783032e-04  7.275512e-03  7.338601e-03  7.401689e-03  7.514571e-03  7.627453e-03
validation_l1_output_f1-CC2        3.0  1.741747e-02  8.539097e-05  1.731925e-02  1.738915e-02  1.745906e-02  1.746657e-02  1.747409e-02
validation_l1_output_f2-CC2        3.0  3.718752e-02  1.168312e-03  3.585434e-02  3.676488e-02  3.767541e-02  3.785411e-02  3.803280e-02
validation_l1_output_E1-PBE0       3.0  7.158640e-03  2.485383e-06  7.155819e-03  7.157707e-03  7.159595e-03  7.160051e-03  7.160507e-03
validation_l1_output_E2-PBE0       3.0  7.303289e-03  4.086059e-04  6.984135e-03  7.073031e-03  7.161927e-03  7.462866e-03  7.763805e-03
validation_l1_output_f1-PBE0       3.0  1.673976e-02  1.718111e-03  1.486630e-02  1.598876e-02  1.711122e-02  1.767649e-02  1.824175e-02
validation_l1_output_f2-PBE0       3.0  3.034832e-02  3.612462e-04  2.998723e-02  3.016761e-02  3.034799e-02  3.052886e-02  3.070972e-02
validation_l1_output_E1-CAM        3.0  6.896892e-03  5.910892e-05  6.829844e-03  6.874599e-03  6.919354e-03  6.930415e-03  6.941477e-03
validation_l1_output_E2-CAM        3.0  6.903027e-03  2.249583e-04  6.652694e-03  6.810417e-03  6.968140e-03  7.028193e-03  7.088246e-03
validation_l1_output_f1-CAM        3.0  1.652058e-02  6.021240e-04  1.583881e-02  1.629109e-02  1.674338e-02  1.686147e-02  1.697955e-02
validation_l1_output_f2-CAM        3.0  3.452070e-02  1.085045e-03  3.354906e-02  3.393527e-02  3.432149e-02  3.500652e-02  3.569154e-02
validation_rmse_avg                3.0  2.675085e-02  1.974986e-04  2.658495e-02  2.664162e-02  2.669829e-02  2.683380e-02  2.696931e-02
validation_rmse_output_E1-CC2      3.0  9.345020e-03  2.487710e-04  9.074190e-03  9.235854e-03  9.397519e-03  9.480435e-03  9.563352e-03
validation_rmse_output_E2-CC2      3.0  9.925613e-03  3.957831e-04  9.574250e-03  9.711229e-03  9.848208e-03  1.010129e-02  1.035438e-02
validation_rmse_output_f1-CC2      3.0  3.576659e-02  1.570254e-03  3.408876e-02  3.504949e-02  3.601022e-02  3.660551e-02  3.720079e-02
validation_rmse_output_f2-CC2      3.0  5.798666e-02  9.846731e-04  5.737245e-02  5.741878e-02  5.746512e-02  5.829376e-02  5.912240e-02
validation_rmse_output_E1-PBE0     3.0  9.352506e-03  5.767230e-05  9.314917e-03  9.319306e-03  9.323695e-03  9.371301e-03  9.418908e-03
validation_rmse_output_E2-PBE0     3.0  9.616460e-03  5.632452e-04  9.188814e-03  9.297367e-03  9.405920e-03  9.830283e-03  1.025465e-02
validation_rmse_output_f1-PBE0     3.0  3.341870e-02  3.574904e-03  2.959571e-02  3.178869e-02  3.398167e-02  3.533019e-02  3.667871e-02
validation_rmse_output_f2-PBE0     3.0  5.026540e-02  1.736561e-03  4.888441e-02  4.929061e-02  4.969682e-02  5.095590e-02  5.221498e-02
validation_rmse_output_E1-CAM      3.0  8.984338e-03  1.736023e-04  8.793610e-03  8.909941e-03  9.026271e-03  9.079702e-03  9.133133e-03
validation_rmse_output_E2-CAM      3.0  9.135396e-03  4.365519e-04  8.658140e-03  8.945819e-03  9.233499e-03  9.374024e-03  9.514550e-03
validation_rmse_output_f1-CAM      3.0  3.322607e-02  2.386448e-03  3.184022e-02  3.184826e-02  3.185629e-02  3.391899e-02  3.598168e-02
validation_rmse_output_f2-CAM      3.0  5.398743e-02  6.594950e-04  5.323720e-02  5.374329e-02  5.424938e-02  5.436254e-02  5.447570e-02
[02/22/2024 08:17:34 PM fastprop.fastprop_core] INFO: Displaying testing results:
                           count          mean           std            min           25%           50%           75%           max
test_mse_loss                3.0  4.381792e-01  2.735942e-01       0.184084  2.933683e-01  4.026529e-01  5.652271e-01  7.278012e-01
test_mean_mape               3.0  4.687784e+05  3.344268e+05  113314.453125  3.145762e+05  5.158379e+05  6.465104e+05  7.771829e+05
test_mape_output_E1-CC2      3.0  3.856048e-02  3.650709e-03       0.034786  3.680392e-02  3.882163e-02  4.044762e-02  4.207360e-02
test_mape_output_E2-CC2      3.0  3.435893e-02  2.072672e-03       0.032024  3.354772e-02  3.507146e-02  3.552641e-02  3.598135e-02
test_mape_output_f1-CC2      3.0  9.124813e+04  4.736281e+04   47827.578125  6.599484e+04  8.416209e+04  1.129584e+05  1.417547e+05
test_mape_output_f2-CC2      3.0  4.789280e+05  5.331655e+05  159846.453125  1.711743e+05  1.825022e+05  6.384688e+05  1.094436e+06
test_mape_output_E1-PBE0     3.0  3.868797e-02  2.537570e-03       0.035766  3.786128e-02  3.995628e-02  4.014881e-02  4.034134e-02
test_mape_output_E2-PBE0     3.0  3.448371e-02  1.759188e-03       0.032453  3.395644e-02  3.546001e-02  3.549913e-02  3.553824e-02
test_mape_output_f1-PBE0     3.0  4.193003e+04  3.497132e+04   15974.560547  2.204593e+04  2.811730e+04  5.490776e+04  8.169822e+04
test_mape_output_f2-PBE0     3.0  1.524934e+05  1.516542e+05   49095.953125  6.544622e+04  8.179648e+04  2.041922e+05  3.265879e+05
test_mape_output_E1-CAM      3.0  3.672770e-02  3.258540e-03       0.033062  3.544453e-02  3.782748e-02  3.856077e-02  3.929405e-02
test_mape_output_E2-CAM      3.0  3.142536e-02  8.549778e-04       0.030506  3.104016e-02  3.157481e-02  3.188528e-02  3.219575e-02
test_mape_output_f1-CAM      3.0  1.659797e+06  9.323249e+05  586385.687500  1.355839e+06  2.125293e+06  2.196503e+06  2.267712e+06
test_mape_output_f2-CAM      3.0  3.200944e+06  2.781384e+06  398584.843750  1.820982e+06  3.243380e+06  4.602124e+06  5.960868e+06
test_mean_wmape              3.0  4.299038e-01  2.291268e-02       0.403447  4.232573e-01  4.430680e-01  4.431324e-01  4.431967e-01
test_wmape_output_E1-CC2     3.0  3.617709e-02  3.006597e-03       0.032971  3.479847e-02  3.662544e-02  3.777988e-02  3.893432e-02
test_wmape_output_E2-CC2     3.0  3.289112e-02  1.902959e-03       0.030732  3.217480e-02  3.361768e-02  3.397072e-02  3.432377e-02
test_wmape_output_f1-CC2     3.0  8.303049e-01  1.196323e-01       0.693788  7.870345e-01  8.802810e-01  8.985633e-01  9.168456e-01
test_wmape_output_f2-CC2     3.0  8.666274e-01  2.331281e-02       0.845110  8.542436e-01  8.633767e-01  8.773858e-01  8.913949e-01
test_wmape_output_E1-PBE0    3.0  3.601028e-02  2.221194e-03       0.033509  3.514003e-02  3.677150e-02  3.726114e-02  3.775077e-02
test_wmape_output_E2-PBE0    3.0  3.251999e-02  1.746795e-03       0.030506  3.196550e-02  3.342459e-02  3.352678e-02  3.362897e-02
test_wmape_output_f1-PBE0    3.0  8.062086e-01  2.029557e-01       0.579097  7.243990e-01  8.697007e-01  9.197643e-01  9.698279e-01
test_wmape_output_f2-PBE0    3.0  8.636080e-01  7.747852e-02       0.790110  8.231459e-01  8.561819e-01  9.003570e-01  9.445322e-01
test_wmape_output_E1-CAM     3.0  3.447643e-02  2.755010e-03       0.031315  3.353421e-02  3.575383e-02  3.605735e-02  3.636087e-02
test_wmape_output_E2-CAM     3.0  3.014137e-02  8.420717e-04       0.029265  2.973993e-02  3.021491e-02  3.057959e-02  3.094426e-02
test_wmape_output_f1-CAM     3.0  7.729732e-01  9.299355e-02       0.666509  7.402989e-01  8.140889e-01  8.262054e-01  8.383218e-01
test_wmape_output_f2-CAM     3.0  8.169068e-01  4.658062e-02       0.765002  7.978232e-01  8.306439e-01  8.428590e-01  8.550741e-01
test_l1_avg                  3.0  1.778473e-02  3.294640e-03       0.014050  1.653818e-02  1.902676e-02  1.965229e-02  2.027782e-02
test_l1_output_E1-CC2        3.0  8.082521e-03  4.642318e-04       0.007579  7.876624e-03  8.173829e-03  8.334071e-03  8.494314e-03
test_l1_output_E2-CC2        3.0  8.154445e-03  2.499915e-04       0.007867  8.073411e-03  8.280284e-03  8.298398e-03  8.316512e-03
test_l1_output_f1-CC2        3.0  2.258022e-02  6.679852e-03       0.015156  1.981858e-02  2.448124e-02  2.629237e-02  2.810350e-02
test_l1_output_f2-CC2        3.0  3.735737e-02  6.311071e-03       0.030143  3.510876e-02  4.007474e-02  4.096466e-02  4.185458e-02
test_l1_output_E1-PBE0       3.0  7.995282e-03  3.462088e-04       0.007678  7.810767e-03  7.943894e-03  8.154104e-03  8.364313e-03
test_l1_output_E2-PBE0       3.0  7.965509e-03  2.191819e-04       0.007721  7.875546e-03  8.029693e-03  8.087563e-03  8.145433e-03
test_l1_output_f1-PBE0       3.0  1.934730e-02  5.816009e-03       0.012760  1.713414e-02  2.150811e-02  2.264087e-02  2.377363e-02
test_l1_output_f2-PBE0       3.0  3.059776e-02  6.344825e-03       0.023330  2.838006e-02  3.343004e-02  3.423160e-02  3.503316e-02
test_l1_output_E1-CAM        3.0  7.648244e-03  4.291877e-04       0.007155  7.503542e-03  7.851947e-03  7.894798e-03  7.937648e-03
test_l1_output_E2-CAM        3.0  7.428072e-03  1.471595e-05       0.007411  7.423786e-03  7.436492e-03  7.436568e-03  7.436645e-03
test_l1_output_f1-CAM        3.0  2.155573e-02  5.545887e-03       0.015279  1.943762e-02  2.359668e-02  2.469431e-02  2.579194e-02
test_l1_output_f2-CAM        3.0  3.470427e-02  7.302494e-03       0.026491  3.182419e-02  3.715729e-02  3.881085e-02  4.046442e-02
test_rmse_avg                3.0  3.185575e-02  1.031976e-02       0.022384  2.635676e-02  3.032929e-02  3.659150e-02  4.285372e-02
test_rmse_output_E1-CC2      3.0  1.487295e-02  6.926924e-03       0.010234  1.089178e-02  1.154994e-02  1.719262e-02  2.283529e-02
test_rmse_output_E2-CC2      3.0  1.216436e-02  2.578497e-03       0.010253  1.069805e-02  1.114347e-02  1.312023e-02  1.509700e-02
test_rmse_output_f1-CC2      3.0  4.361295e-02  1.590910e-02       0.027542  3.574183e-02  4.394157e-02  5.164838e-02  5.935519e-02
test_rmse_output_f2-CC2      3.0  5.992159e-02  1.121479e-02       0.048110  5.466965e-02  6.122885e-02  6.582715e-02  7.042545e-02
test_rmse_output_E1-PBE0     3.0  1.701016e-02  1.132822e-02       0.010158  1.047232e-02  1.078662e-02  2.043622e-02  3.008583e-02
test_rmse_output_E2-PBE0     3.0  1.464896e-02  7.345711e-03       0.010035  1.041355e-02  1.079200e-02  1.695590e-02  2.311979e-02
test_rmse_output_f1-PBE0     3.0  3.739808e-02  1.205368e-02       0.023940  3.249615e-02  4.105211e-02  4.412702e-02  4.720194e-02
test_rmse_output_f2-PBE0     3.0  5.742946e-02  2.286472e-02       0.037340  4.498903e-02  5.263777e-02  6.747403e-02  8.231030e-02
test_rmse_output_E1-CAM      3.0  1.804341e-02  1.395300e-02       0.009402  9.994914e-03  1.058777e-02  2.236408e-02  3.414039e-02
test_rmse_output_E2-CAM      3.0  1.024229e-02  9.905677e-04       0.009568  9.673655e-03  9.779445e-03  1.057950e-02  1.137956e-02
test_rmse_output_f1-CAM      3.0  3.997855e-02  9.811658e-03       0.028876  3.622661e-02  4.357764e-02  4.553003e-02  4.748242e-02
test_rmse_output_f2-CAM      3.0  5.694622e-02  1.382938e-02       0.043153  5.001361e-02  5.687426e-02  6.384285e-02  7.081144e-02
[02/22/2024 08:17:34 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.440>0.05.
[02/22/2024 08:17:34 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/22/2024 08:17:34 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:10:48.739306 -->

### ESOL
First described in 2004 [@esol] and has since become a critically important benchmark for QSPR/molecular property prediction studies.
The dataset includes molecular structure for approximately 1.1k simple organic molecules are their corresponding experimentally measured free energy of solvation.
This property is a classic target of QSPR studies and is especially well suited for `fastprop`.

The CMPNN [@cmpnn] model, a derivative of Chemprop, is used for comparison.
The performance values are _not_ those from the original paper but instead the _corrected_ results shown on the CMPNN GitHub page [@cmpnn_amended_results].
These numbers are the average and standard deviation across 5 repetitions, each including 5-fold cross validation (60/20/20 random split), for a total of 25 models.
`fastprop` performance is reported using the same split sizes across 8 repetitions _without_ cross validation; increasing the number of repetitions further did not meaningfully change model performance.

`fastprop` achieves an RMSE of 0.643 $\pm$ 0.048 trailing the CMPNN at 0.547 $\pm$ 0.011 but matching Chemprop at 0.665 $\pm$ 0.052.
The same pattern from previous benchmarks is repeated - `fastprop` matches the performance of generic learned representation approaches but is outperformed by bespoke modeling techniques.
In this case the CMPNN has been designed to perform better on these benchmark datasets specifically and at an increased cost in both execution time and complexity.
<!-- [02/22/2024 10:40:29 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count         mean           std       min       25%       50%         75%           max
validation_mse_loss    8.0     0.095556      0.021735  0.071144  0.084887  0.090992    0.098194      0.144221
validation_r2          8.0     0.906880      0.019380  0.863877  0.905810  0.912962    0.919242      0.921702
validation_mape        8.0  4852.799028  13378.896109  0.354641  0.430600  0.545640  216.635182  37955.437500
validation_wmape       8.0     0.119274      0.008751  0.108916  0.111683  0.117584    0.127764      0.130299
validation_l1          8.0     0.452131      0.032328  0.415299  0.436033  0.443621    0.463682      0.514942
validation_mdae        8.0     0.318440      0.017449  0.288106  0.311116  0.322926    0.330057      0.339814
validation_rmse        8.0     0.639168      0.068747  0.559729  0.609307  0.620219    0.648583      0.792154
[02/22/2024 10:40:29 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count         mean          std       min       25%       50%          75%           max
test_mse_loss    8.0     0.096117     0.013415  0.082148  0.087325  0.096219     0.098967      0.124506
test_r2          8.0     0.907555     0.015811  0.876043  0.899289  0.910924     0.920941      0.922190
test_mape        8.0  4107.554610  7206.676462  0.297727  0.398541  0.606253  5138.838989  20146.849609
test_wmape       8.0     0.123171     0.018052  0.098788  0.110336  0.124274     0.130009      0.158073
test_l1          8.0     0.463413     0.033499  0.411660  0.447721  0.459838     0.481340      0.520428
test_mdae        8.0     0.335356     0.021130  0.303247  0.323648  0.333049     0.346739      0.370731
test_rmse        8.0     0.643311     0.048021  0.592616  0.603093  0.648945     0.654420      0.742377
[02/22/2024 10:40:29 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.891>0.05.
[02/22/2024 10:40:29 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/22/2024 10:40:29 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:00:51.805718 -->

### FreeSolv
The Free Energy of Solvation (FreeSolv) database is a more curated alternative to ESOL developed and maintained by Mobley et al. in 2014 [@freesolv].
It contains approximately 0.6k molecules and their experimental _and_ calculated hydration free energy.
This benchmark is the smallest 'recognized' benchmark often reported in the molecular property prediction literature.

Again the CMPNN study is used as a reference and the same procedure described in [ESOL](#esol) is followed.
`fastprop` achieves an RMSE of 1.33 $\pm$ 0.21, once again trailing trailing the CMPNN at 0.82 $\pm$ 0.15 but matching Chemprop at 1.26 $\pm$ 0.11.
<!-- [02/22/2024 11:08:31 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count          mean           std       min       25%       50%           75%           max
validation_mse_loss    8.0      0.105076      0.021142  0.071070  0.094144  0.108995      0.122287      0.128179
validation_r2          8.0      0.887782      0.031946  0.829467  0.872631  0.887708      0.916983      0.921358
validation_mape        8.0  16893.051336  32489.402467  0.267785  0.288608  0.486189  12784.332201  84006.593750
validation_wmape       8.0      0.168538      0.028897  0.140509  0.149993  0.162564      0.172264      0.229460
validation_l1          8.0      0.793325      0.104319  0.671217  0.725321  0.760419      0.853767      0.973832
validation_mdae        8.0      0.450124      0.073153  0.378300  0.410038  0.427426      0.464366      0.597248
validation_rmse        8.0      1.248167      0.154767  0.977231  1.163309  1.258965      1.383154      1.423775
[02/22/2024 11:08:31 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count          mean           std       min       25%       50%          75%            max
test_mse_loss    8.0      0.122054      0.038057  0.068932  0.086034  0.135498     0.146974       0.172368
test_r2          8.0      0.871179      0.041758  0.820861  0.834757  0.872354     0.904700       0.929744
test_mape        8.0  17740.199359  43945.869444  0.326405  0.664482  0.713992  7779.158203  126142.187500
test_wmape       8.0      0.161002      0.012844  0.144911  0.152154  0.159423     0.166615       0.184403
test_l1          8.0      0.793003      0.073708  0.689533  0.728003  0.811125     0.836852       0.881987
test_mdae        8.0      0.438812      0.037666  0.378906  0.416421  0.439448     0.456648       0.505403
test_rmse        8.0      1.332239      0.212225  1.024511  1.144719  1.421061     1.430947       1.649917
[02/22/2024 11:08:31 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.381>0.05.
[02/22/2024 11:08:31 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/22/2024 11:08:31 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:00:38.625761 -->

### Flash
First assembled and fitted to by Saldana and coauthors [@flash] the dataset (Flash) includes around 0.6k entries, primarily alkanes and some oxygen-containing compounds, and their literature-reported flash point.
The reference study reports the performance on only one repetition, but manually confirms that the distribution of points in the three splits follows the parent dataset.
The split itself was a 70/20/10 random split, which is repeated four times for this study.

Using a complex multi-model ensembling method, the reference study achieved an RMSE of 13.2, an MAE of 8.4, and an MAPE of 2.5%.
`fastprop` matches this performance, achieving 13.3 $\pm$ 2.1 RMSE, 9.4 $\pm$ 0.8 MAE, and 2.6% $\pm$ 0.1% MAPE.
Chemprop, however, struggles to match the accuracy of either method.
It manages an RMSE of 21.2 $\pm$ 2.2 and an MAE of 13.8 $\pm$ 2.1 and does not report MAPE.

Critically, `fastprop` dramatically outperforms both methods in terms of training time.
The reference model required significant manual intervention to create a model ensemble, so no single training time can be fairly identified.
`fastprop` arrived at the indicated performance without any manual intervention in only 30 seconds, 13 of which were spent calculating descriptors.
Chemprop, in addition to not reaching the same level of accuracy, took 5 minutes and 44 seconds to do so - more than ten times the execution time of `fastprop`.
<!-- fastprop:
[02/23/2024 09:18:43 AM fastprop.utils.calculate_descriptors] INFO: Descriptor calculation complete, elapsed time: 0:00:13.064044
[02/23/2024 09:19:00 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count       mean       std        min        25%        50%        75%        max
validation_mse_loss    4.0   0.090098  0.026493   0.060769   0.075516   0.088016   0.102597   0.123590
validation_r2          4.0   0.917885  0.021549   0.897762   0.900850   0.915766   0.932801   0.942246
validation_mape        4.0   0.027317  0.001630   0.025457   0.026290   0.027335   0.028362   0.029143
validation_wmape       4.0   0.028937  0.001957   0.026410   0.027961   0.029222   0.030198   0.030894
validation_l1          4.0   9.414462  0.798012   8.348911   9.052492   9.572168   9.934138  10.164599
validation_mdae        4.0   5.695499  0.645852   4.788015   5.455719   5.916547   6.156327   6.160889
validation_rmse        4.0  16.504705  2.382902  13.781378  15.145635  16.432622  17.791692  19.372200
[02/23/2024 09:19:00 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count       mean       std        min        25%        50%        75%        max
test_mse_loss    4.0   0.058217  0.016954   0.039205   0.048582   0.057268   0.066903   0.079126
test_r2          4.0   0.942541  0.019644   0.914494   0.936482   0.948702   0.954760   0.958264
test_mape        4.0   0.025690  0.001227   0.024379   0.025206   0.025520   0.026004   0.027344
test_wmape       4.0   0.027063  0.001818   0.025431   0.026131   0.026586   0.027518   0.029650
test_l1          4.0   8.765777  0.658962   8.069209   8.341405   8.708392   9.132764   9.577113
test_mdae        4.0   5.984409  0.172704   5.746712   5.941282   6.015377   6.058504   6.160168
test_rmse        4.0  13.285306  2.057740  10.910849  12.108939  13.252343  14.428709  15.725688
[02/23/2024 09:19:00 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.087>0.05.
[02/23/2024 09:19:01 AM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/23/2024 09:19:01 AM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:00:30.493815


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
Assembled by Das and coauthors [@ysi] from a collection of other smaller datasets, this dataset maps ~0.4k molecular structures to a unified-scale Yield Sooting Index (YSI), a molecular property of interest to the combustion community.
The reference study performs leave-one-out cross validation to fit a per-fragment contribution model, effectively a training size of >99%, without a holdout set.
Though this is not standard practice and can lead to overly optimistic reported performance, the results will be carried forward regardless.
The original study did not report overall performance metrics, so they have been re-calculated for this study using the predictions made by the reference model as provided on GitHub [^3].
For comparison `fastprop` and Chemprop use a more typical 60/20/20 random split and 8 repetitions.
Results are summarized in Table \ref{ysi_results_table}.

Table: YSI results. \label{ysi_results_table}

+------------+----------------+----------------+----------------+
| Model      | MAE            | RMSE           | WMAPE          |
+============+================+================+================+
| Reference  | 22.3           | 50             | 0.3            |
+------------+----------------+----------------+----------------+
| `fastprop` | 13.6 $\pm$ 2.1 | 54 $\pm$ 13    | 14.5 $\pm$ 2.2 |
+------------+----------------+----------------+----------------+
| Chemprop   | 28.9 $\pm$ 6.5 | 63 $\pm$ 14    | ~              |
+------------+----------------+----------------+----------------+

`fastprop` significantly outperforms both other models when considering MAE, especially impressive in the case of the reference model which was trained on far more data.
When considering RMSE, which penalizes large errors more than MAE, all models are similarly performant.
Finally the WMAPE shows that the reference model makes much smaller errors on the highest YSI molecules compared to `fastprop`.
Taken in combination with the MAE and RMSE values, which are respectively worse and competitive with `fastprop`, the model is likely highly overfit to the training data due to the cross-validation strategy.

Also notable is the difference in training times.
Chemprop takes 7 minutes and 2 seconds while `fastprop` completes in only 38 seconds, again a factor of ten faster.
<!-- 
[02/23/2024 10:35:08 AM fastprop.utils.calculate_descriptors] INFO: Descriptor calculation complete, elapsed time: 0:00:13.469800
[02/23/2024 10:35:32 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count       mean        std        min        25%        50%        75%        max
validation_mse_loss    8.0   0.036643   0.019237   0.012729   0.021392   0.037871   0.045183   0.063212
validation_r2          8.0   0.966243   0.014797   0.945855   0.951829   0.972803   0.977200   0.981882
validation_mape        8.0   0.372979   0.081809   0.248658   0.321690   0.373215   0.441305   0.474303
validation_wmape       8.0   0.136332   0.018603   0.104015   0.129864   0.132160   0.145879   0.166829
validation_l1          8.0  26.069149   5.297644  19.002356  21.609965  27.703727  29.095233  33.730484
validation_mdae        8.0  12.093825   3.300378   9.047251  10.052302  10.310852  13.844950  18.330185
validation_rmse        8.0  50.956718  13.520214  32.470043  41.267647  51.488636  60.079863  70.025093
[02/23/2024 10:35:32 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count       mean        std        min        25%        50%        75%        max
test_mse_loss    8.0   0.042176   0.022037   0.020046   0.022840   0.041158   0.054149   0.080890
test_r2          8.0   0.959092   0.021130   0.912329   0.955399   0.964561   0.974354   0.975917
test_mape        8.0   0.316280   0.084775   0.221780   0.267679   0.305300   0.321705   0.503944
test_wmape       8.0   0.144770   0.022861   0.118888   0.129254   0.140225   0.155006   0.184557
test_l1          8.0  26.282091   4.350020  21.104954  21.874824  26.894576  29.279588  31.678127
test_mdae        8.0  11.012740   2.510058   8.953960   9.321275  10.470964  11.237217  16.699636
test_rmse        8.0  54.804913  13.585199  39.491203  43.347728  52.980692  65.513039  73.887115
[02/23/2024 10:35:32 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.579>0.05.
[02/23/2024 10:35:32 AM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/23/2024 10:35:32 AM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:00:37.730220
-->

[^3]: Predictions are available at this [permalink](https://github.com/pstjohn/ysi-fragment-prediction/blob/bdf8b16a792a69c3e3e63e64fba6f1d190746abe/data/ysi_predictions.csv) to the CSV file on GitHub.

### HOPV15 Subset
The HOPV15 Subset is a collection of ~0.3k organic photovoltaic compounds curated by Eibeck and coworkers from the larger Harvard Organic Photovoltaic (HOPV15 [@hopv15_original]) based on criteria described in their paper [@hopv15_subset].
This dataset is unique in that the target property Power Conversion Efficiency is both experimentally measurable or can be derived from quantum mechanics simulations, but regardless is not a 'classical' target of QSPR.
After applying a variety of established modeling techniques Eibeck et al. achieved a best-case MAE of 1.32 $\pm$ 0.10 averaged across 5 randomly selected 60/20/20 data splits by using a simple molecular fingerprint representation and support vector regression.

In the course of this study it was found that changing the random seed when using only 5 repetitions would lead to dramatically different model performance.
Thus, for this benchmark the number of repetitions was set _higher_ than the reference study at 15.
`fastprop` reports an average MAE of 1.55 $\pm$ 0.20 and an RMSE of 1.93 $\pm$ 0.20, in line with the performance of Chemprop at an MAE of 1.60 $\pm$ 0.15 and an RMSE of 1.97 $\pm$ 0.16.
Execution time is again dramatically different with Chemprop taking 11 minutes and 35 seconds whereas `fastprop` took only 2 minutes and 3 seconds of which 1 minute and 47 seconds was spent calculating descriptors for the abnormally large molecules in HOPV15.
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
First described by Esaki and coauthors, the Fraction of Unbound Drug in the Brain (Fubrain) dataset is a collection of about 0.3k small molecule drugs and their corresponding experimental experimentally measured unbound fraction in the brain, a critical metric for drug development [@fubrain].
This specific target in combination with this dataset size makes this benchmark highly relevant for typical QSPR studies.

The study that first generated this dataset used `mordred` descriptors but as is convention they strictly applied linear modeling methods.
Using both cross validation and and external test sets, they had an effective training/validation/testing split of 0.64/0.07/0.28 which will be repeated 4 times here for comparison.
All told, their model achieved an RMSE of 0.53 averaged across all testing data.

In only 44 seconds, of which 36 are spent calculating descriptors, `fastprop` far exceeds the reference model with an RMSE of 0.19 $\pm$ 0.03 
Under the same conditions Chemprop approaches `fastprop`'s performance with an RMSE of 0.22 $\pm$ 0.04 but requires 5 minutes and 11 seconds to do so, in this case a 7 times performance improvement for `fastprop` over Chemprop.

#### Delta-Fubrain
Also noteworthy for the Fubrain dataset is that it has been subject to the delta-learning approach to small dataset limitations.
DeepDelta [@deepdelta] performed a 90/0/10 cross-validation study of the Fubrain dataset in which the training and testing molecules were used to generate all possible pairs and then the differences in the property [^4] were predicted rather than absolute values.
They reported an RMSE of 0.830 $\pm$ 0.023, whereas a Chemprop model trained to directly predict property values was only able to reach an accuracy of 0.965 $\pm$ 0.019 when evaluated on its capacity to predict property differences.

`fastprop` is able to overcome these limitations.
Using the same model from above (re-trained to predict log-transformed values), which has _less_ training data than DeepDelta even _before_ the augmentation, `fastprop` achieves 0.740 $\pm$ 0.087 RMSE after pairing all withheld test molecules.
Increasing the amount of training data while retaining some samples for early stopping yields only small improvements, showing that `fastprop` may be approaching the irreducible error of Fubrain.
With an 89/1/10 split the RMSE of `fastprop` decreases to 0.7118 $\pm$ 0.1381, though with significantly increased variance due to small size of the testing data.
Regardless, the execution time and scaling issues of DeepDelta and the inaccuracy of Chemprop are effectively circumvented by `fastprop`.

[^4]: Although the original Fubrain study reported untransformed fractions, the DeepDelta authors confirmed [via GitHub](https://github.com/RekerLab/DeepDelta/issues/2#issuecomment-1917936697) that DeepDelta was trained on log base-10 transformed fraction values, which is replicated here.
<!-- [02/26/2024 11:07:47 PM fastprop.utils.calculate_descriptors] INFO: Descriptor calculation complete, elapsed time: 0:00:36.317700
[02/26/2024 11:07:55 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_mse_loss    4.0  0.528255  0.112356  0.461545  0.462805  0.478035  0.543484  0.695403
validation_r2          4.0  0.520937  0.127055  0.332428  0.502369  0.573573  0.592141  0.604174
validation_mape        4.0  4.051855  0.726136  3.493797  3.521063  3.833108  4.363900  5.047408
validation_wmape       4.0  0.652842  0.125646  0.531584  0.573018  0.629971  0.709794  0.819841
validation_l1          4.0  0.126917  0.021673  0.100416  0.116020  0.127700  0.138596  0.151851
validation_mdae        4.0  0.073359  0.020653  0.047181  0.063825  0.074977  0.084511  0.096301
validation_rmse        4.0  0.188302  0.024534  0.170276  0.174478  0.179318  0.193142  0.224298
[02/26/2024 11:07:55 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%        max
test_mse_loss    4.0  0.581702  0.210334  0.386589  0.406386  0.588264  0.763580   0.763693
test_r2          4.0  0.433427  0.051082  0.389265  0.390404  0.427765  0.470788   0.488912
test_mape        4.0  7.624890  2.673862  4.914618  6.157446  7.179506  8.646950  11.225932
test_wmape       4.0  0.721044  0.043162  0.682854  0.689614  0.712062  0.743492   0.777199
test_l1          4.0  0.130729  0.013527  0.116302  0.120649  0.131922  0.142002   0.142768
test_mdae        4.0  0.080191  0.010658  0.071635  0.072877  0.077059  0.084373   0.095009
test_rmse        4.0  0.194436  0.028416  0.167237  0.171230  0.195749  0.218955   0.219010
[02/26/2024 11:07:55 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.755>0.05.
[02/26/2024 11:07:55 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/26/2024 11:07:55 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:00:44.299743


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


Timing for fastprop: descriptor generation was 30 seconds, total was 54s. -->

### PAHs
Originally compiled by Arockiaraj et al. [@pah] the Polycyclic Aromatic Hydrocarbons (PAH) dataset contains water/octanol partition coefficients (logP) for exactly 55 polycyclic aromatic hydrocarbons ranging in size from napthalene to circumcoronene.
This size of this benchmark is an ideal case study for the application of `fastprop`.
Using expert insight the reference study designed a novel set of molecular descriptors that show a strong correlation to logP, with correlation coefficients ranging from 0.96 to 0.99 among the various new descriptors.

For comparison, `fastprop` and Chemprop are trained using 8 repetitions of a typical 80/10/10 random split - only _44_ molecules in the training data.
`fastprop`matches the performance of the bespoke descriptors with a correlation coefficient of 0.976 $\pm$ 0.027.
This corresponds to an MAE of 0.160 $\pm$ 0.035 and an MAPE of 2.229 $\pm$ 0.061%.
Chemprop effectively fails on this dataset, achieving a correlation coefficient of only 0.59 $\pm$ 0.24, an MAE of 1.04 $\pm$ 0.33 (one anti-correlated outlier replicate removed).
Despite the large parameter size of the `fastprop` model, it readily outperforms Chemprop in the small-data limit.

For this unique dataset, execution time trends are inverted.
`fastprop` takes 2 minutes and 44 seconds, of which 1 minute and 44 seconds were spent calculating descriptors for these unusually large molecules.
Chemprop completes in 1 minute and 17 seconds, on par with the training time of `fastprop` without considering descriptor calculation.
<!--
On only the 21 points that have a retention index, mape: 0.024147   0.008081 
Similar good story on the boiling point, acentric factor only has 12 points left after removing unvalued entries - difficult to call it anything.

[02/27/2024 01:04:31 PM fastprop.utils.calculate_descriptors] INFO: Descriptor calculation complete, elapsed time: 0:01:43.992341
[02/27/2024 01:05:30 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_mse_loss    8.0  0.011793  0.007565  0.002760  0.007052  0.010122  0.016038  0.023678
validation_r2          8.0  0.967643  0.041480  0.887329  0.962079  0.990051  0.991525  0.995362
validation_mape        8.0  0.023872  0.007336  0.012750  0.019830  0.021311  0.030370  0.034264
validation_wmape       8.0  0.023179  0.006904  0.011766  0.019547  0.022500  0.027003  0.033757
validation_l1          8.0  0.180293  0.063113  0.097302  0.142936  0.159853  0.216696  0.297467
validation_mdae        8.0  0.163112  0.076687  0.048465  0.120278  0.155380  0.203896  0.286865
validation_rmse        8.0  0.216918  0.074785  0.119338  0.176945  0.189857  0.275287  0.328269
[02/27/2024 01:05:30 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_mse_loss    8.0  0.009312  0.004077  0.004010  0.005968  0.010208  0.011694  0.015521
test_r2          8.0  0.976755  0.027748  0.910766  0.978237  0.982443  0.991034  0.997106
test_mape        8.0  0.022889  0.006179  0.012255  0.018382  0.024994  0.027098  0.030146
test_wmape       8.0  0.021151  0.005120  0.013177  0.017212  0.023100  0.025260  0.026115
test_l1          8.0  0.160849  0.034691  0.115016  0.132890  0.157501  0.188638  0.205169
test_mdae        8.0  0.141768  0.048483  0.088757  0.108975  0.125542  0.163960  0.224543
test_rmse        8.0  0.197821  0.042304  0.139991  0.166466  0.202003  0.229707  0.248529
[02/27/2024 01:05:30 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.540>0.05.
[02/27/2024 01:05:30 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/27/2024 01:05:30 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:02:43.453523

chemprop

real    1m15.752s
user    1m6.715s
sys     0m30.473s
-->

## Classification Datasets
See Table \ref{classification_results_table} for a summary of all the classification dataset results.
Especially noteworthy is the performance on QuantumScents dataset, which outperforms the best literature result.
Citations for the datasets themselves are included in the sub-sections of this section.

Table: Summary of classification benchmark results. \label{classification_results_table}

+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+
|Benchmark              |Samples (k)         |Metric      |Literature Best             |`fastprop`  |Chemprop                    |
+=======================+====================+============+============================+============+============================+
|HIV                    | ~41                |AUROC       |0.81$^a$                    |0.81        |0.77$^a$                    |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+
|QuantumScents          |~3.5                |AUROC       |0.88$^b$                    |0.91        |0.85$^b$                    |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+
|SIDER                  |~1.4                |AUROC       |0.67$^c$                    |0.64        |0.65$^c$                    |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+
|Pgp                    |~1.3                |AUROC       |0.94$^e$                    |0.92        |0.89$^e$                    |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+
|ARA                    |~0.8                |Accuracy    |91$^d$                      |89          |82*                         |
+-----------------------+--------------------+------------+----------------------------+------------+----------------------------+

a [@unimol] b [@quantumscents] c [@cmpnn] d [@ara] e [@pgp_best] * These results were generated for this study.

### HIV Inhibition
Originally compiled by Riesen and Bunke [@hiv], this dataset includes the reported HIV activity for approximately 41k small molecules.
This is an established benchmark in the molecular property prediction community and the exact version used is that which was standardized in MoleculeNet [@moleculenet].
This dataset is unique in that the labels in the original study include three possible classes (a _multiclass_) regression problem whereas the most common reported metric is instead lumping positive and semi-positive labels into a single class to reduce the task to _binary_ classification; both are reported here.
UniMol is again used as a point of comparison, and thus an 80/10/10 scaffold-based split with three repetitions is used.

For binary classification `fastprop`'s AUROC of 0.81 $\pm$ 0.04 matches the literature best UniMol with and 0.808 $\pm$ 0.003 [@unimol].
This corresponds to an accuracy of 96.8+/-1.0% for `fastprop`, which taken in combination with AUROC hints that the model is prone to false positives.
Chemprop performs worse than both of these models with a reported AUROC of 0.771 $\pm$ 0.005.
<!-- [02/27/2024 02:31:15 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    3.0  0.138464  0.012790  0.123954  0.133643  0.143332  0.145718  0.148104
validation_accuracy    3.0  0.963440  0.003239  0.961333  0.961576  0.961819  0.964494  0.967169
validation_f1          3.0  0.296696  0.024142  0.270270  0.286246  0.302222  0.309909  0.317597
validation_auroc       3.0  0.778699  0.016220  0.764092  0.769970  0.775849  0.786002  0.796155
[02/27/2024 02:31:15 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    3.0  0.122810  0.021838  0.097602  0.116223  0.134844  0.135414  0.135984
test_accuracy    3.0  0.967899  0.009553  0.959630  0.962670  0.965710  0.972033  0.978356
test_f1          3.0  0.298743  0.108394  0.194175  0.242816  0.291457  0.351027  0.410596
test_auroc       3.0  0.805757  0.040305  0.777523  0.782678  0.787833  0.819874  0.851915
[02/27/2024 02:31:15 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing accuracy yielded p value of p=0.487>0.05.
[02/27/2024 02:31:16 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/27/2024 02:31:16 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:08:36.655815 -->

When attempting multiclass classification, `fastprop` maintains a similar AUROC of 0.818+/-0.019 AUROC
Accuracy suffers a prodigious drop to 42.8 $\pm$ 7.6%, now suggesting that the model is prone to false negatives.
Other leading performers do not report performance metrics on this variation of the dataset.
<!-- [02/27/2024 03:39:40 PM fastprop.fastprop_core] INFO: Displaying validation results:
                       count      mean       std       min       25%       50%       75%       max
validation_kldiv_loss    3.0  0.158738  0.014986  0.141436  0.154306  0.167177  0.167390  0.167602
validation_auroc         3.0  0.804446  0.010528  0.792380  0.800787  0.809195  0.810479  0.811763
validation_accuracy      3.0  0.394096  0.033280  0.373434  0.374900  0.376367  0.404427  0.432487
[02/27/2024 03:39:40 PM fastprop.fastprop_core] INFO: Displaying testing results:
                 count      mean       std       min       25%       50%       75%       max
test_kldiv_loss    2.0  0.136514  0.041441  0.107211  0.121862  0.136514  0.151166  0.165817
test_auroc         3.0  0.817601  0.021805  0.792922  0.809270  0.825619  0.829941  0.834263
test_accuracy      3.0  0.428450  0.076047  0.343844  0.397120  0.450396  0.470754  0.491112
[02/27/2024 03:39:40 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.400>0.05.
[02/27/2024 03:39:40 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/27/2024 03:39:40 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:08:51.352367 -->

### QuantumScents
Compiled by Burns and Rogers [@quantumscents], this dataset contains approximately 3.5k SMILES and 3D structures for a collection of molecules labeled with their scents.
Each molecule can have any number of reported scents from a possible 113 different labels, making this benchmark a a Quantitative Structure-Odor Relationship.
Due to the highly sparse nature of the scent labels a unique sampling algorithm (Szymanski sampling [@szymanski]) was used in the reference study and the exact splits are replicated here for a fair comparison.

In the reference study, Chemprop achieved an AUROC of 0.85 with modest hyperparameter optimization and an improved AUROC of 0.88 by incorporating the atomic descriptors calculated as part of QuantumScents.
`fastprop`, using neither these descriptors nor the 3D structures, outperforms both models with an AUROC of 0.910+/-0.001 with only descriptors calculated from the molecules' SMILES.
The GitHub repository contains an example of generating custom descriptors incorporating the 3D information from QuantumScents and passing these to `fastprop`; impact on the performance was negligible.

### SIDER
First described by Kuhn et al. in 2015 [@sider], the Side Effect Resource (SIDER) database has become a standard property prediction benchmark.
This challenging dataset maps around 1.4k compounds, including small molecules, metals, and salts, to any combination of 27 side effects - leading performers are only slightly better than random guessing (AUROC 0.5).

Among the best performers in literature is the previously discussed CMPNN [@cmpnn] with a reported AUROC of 0.666+/-0.007, which narrowly outperforms Chemprop at 0.646+/-0.016.
Using the same approach, `fastprop` achieves a decent AUROC of 0.636+/-0.019.
Despite many of the entries in this dataset being atypical for `mordred` near-leading performance is still possible, supporting the robustness and generalizability of this framework.

### Pgp
First reported in 2011 by Broccatelli and coworkers [@pgp], this dataset has since become a standard benchmark and is included in the Therapeutic Data Commons (TDC) [@tdc] model benchmarking suite.
the dataset maps approximately 1.2k small molecule drugs to a binary label indicating if they inhibit P-glycoprotein (Pgp).
TDC serves this data through a Python package, but due to installation issues the data was retrieved from the original study instead.
The recommended splitting approach is a 70/10/20 scaffold-based split which is done here with 4 replicates.

The model in the original study uses a molecular interaction field but has since been surpassed by other models.
According to TDC the current leader [@pgp_best] on this benchmark has achieved an AUROC of 0.938 $\pm$ 0.002 [^5].
AOn the same leaderboard Chemprop [@chemprop_theory] achieves 0.886 $\pm$ 0.016 with the inclusion of additional molecular features.
`fastprop` yet again approaches the performance of the leading methods and outperforms Chemprop, here with an AUROC of 0.919 $\pm$ 0.013 and an accuracy of 84.5 $\pm$ 0.2%.
<!-- [02/27/2024 08:38:09 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    4.0  0.397071  0.059480  0.345734  0.360633  0.380944  0.417381  0.480661
validation_accuracy    4.0  0.862705  0.057328  0.795082  0.825820  0.868852  0.905738  0.918033
validation_f1          4.0  0.850000  0.035355  0.800000  0.837500  0.862500  0.875000  0.875000
validation_auroc       4.0  0.912848  0.010768  0.901493  0.906710  0.911516  0.917655  0.926868
[02/27/2024 08:38:09 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    4.0  0.408681  0.049120  0.359007  0.388608  0.399594  0.419667  0.476528
test_accuracy    4.0  0.845918  0.025680  0.808163  0.841837  0.855102  0.859184  0.865306
test_f1          4.0  0.860778  0.023598  0.829091  0.851851  0.864501  0.873428  0.885017
test_auroc       4.0  0.919041  0.012997  0.901614  0.912929  0.922293  0.928406  0.929965
[02/27/2024 08:38:09 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing accuracy yielded p value of p=0.612>0.05.
[02/27/2024 08:38:09 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/27/2024 08:38:09 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:00:14.312253 -->

[^5]: See [the TDC Pgp leaderboard](https://tdcommons.ai/benchmark/admet_group/03pgp/).

### ARA
The final benchmark is one which closely mimics typical QSPR studies.
Compiled by Schaduangrat et al. in 2023 [@ara], this dataset maps ~0.8k small molecules to a binary label indicating if the molecule is an Androgen Receptor Antagonist (ARA).
The reference study introduced DeepAR, a highly complex modeling approach, which achieved an accuracy of 0.911 and an AUROC of 0.945.

For this study an 80/10/10 random splitting is repeated four times on the dataset since no analogous split to the reference study can be determined.
Chemprop takes 16 minutes and 55 seconds to run on this dataset and achieves only 0.824+/-0.020 accuracy and 0.898+/-0.022 AUROC.
`fastprop` takes only 2 minutes and 4 seconds (1 minute and 47 seconds for descriptor calculation) and is competitive with the reference study in performance, achieving a 89.1+/-4.0% accuracy and 0.951+/-0.018 AUROC.
<!--
[02/27/2024 09:12:38 PM fastprop.utils.calculate_descriptors] INFO: Descriptor calculation complete, elapsed time: 0:01:47.042475
[02/27/2024 09:12:54 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    4.0  0.342375  0.134405  0.230809  0.246378  0.307217  0.403215  0.524258
validation_accuracy    4.0  0.880952  0.035047  0.833333  0.869048  0.886905  0.898810  0.916667
validation_f1          4.0  0.883563  0.034730  0.837209  0.870017  0.888752  0.902299  0.919540
validation_auroc       4.0  0.936141  0.031894  0.899432  0.914484  0.940432  0.962089  0.964265
[02/27/2024 09:12:54 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    4.0  0.309558  0.074081  0.201878  0.292477  0.334441  0.351521  0.367472
test_accuracy    4.0  0.885294  0.033792  0.858824  0.858824  0.876471  0.902941  0.929412
test_f1          4.0  0.876612  0.048899  0.823529  0.843382  0.875549  0.908779  0.931818
test_auroc       4.0  0.945325  0.023256  0.918286  0.936511  0.943980  0.952794  0.975055
[02/27/2024 09:12:54 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing accuracy yielded p value of p=0.864>0.05.
[02/27/2024 09:12:54 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[02/27/2024 09:12:54 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:02:03.685409


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
## Execution Time
Although `fastprop` is consistently around an order of magnitude faster to train than learned representations when using a GPU, execution time is a minor concern when considering the enormous labor invested in dataset generation.
For day-to-day work it is convenient but the correctness of `fastprop`, especially on small datasets, is more important.
Note that due to the large size of the FNN in `fastprop` it will typically be slower than Chemprop when training on a CPU since Chemprop uses a much smaller FNN and associated components.

Regardless, there is an clear performance improvement to be had by reducing the number of descriptors to a subset of only the most important.
Future work will address this possibility to decrease time requirements for both training by reducing network size and inference by decreasing the number of descriptors to be calculated for new molecules.
This has _not_ been done in this study for two reasons: (1) to emphasize the capacity of the DL framework to effectively perform feature selection on its own via the training process, de-emphasizing unimportant descriptors; (2) as discussed above, training time is small compared ot dataset generation time.

## Coverage of Descriptors
`fastprop` is fundamentally limited by the types of chemicals which can be uniquely described by the `mordred` package.
Domain-specific additions which are not just derived from the descriptors already implemented will be required to expand its application to new domains.

For example, in its current state `mordred` does not include any connectivity based-descriptors that reflect the presence or absence of stereocenters.
While some of the 3D descriptors it implements could implicitly reflect sterochemistry, more explicit descriptors like the Stereo Signature Molecular Descriptor [@stereo_signature] may prove helpful in the future if re-implemented in `mordred`.

## Interpretability
Though not discussed here for the sake of length, `fastprop` already contains the functionality to perform feature importance studies on trained models.
By using SHAP values [@shap] to assign a scalar 'importance' to each of the input features, users can determine which of the `mordred` descriptors has the largest impact on model predictions.
Future studies will demonstrate this in greater detail.

# Availability
 - Project name: fastprop
 - Project home page: github.com/jacksonburns/fastprop
 - Operating system(s): Platform independent
 - Programming language: Python
 - Other requirements: pyyaml, lightning, mordredcommunity, astartes
 - License: MIT

# Declarations

## Availability of data and materials
`fastprop` is Free and Open Source Software; anyone may view, modify, and execute it according to the terms of the MIT license.
See github.com/jacksonburns/fastprop for more information.

All data used in the Benchmarks shown above is publicly available under a permissive license. See the benchmarks directory at the `fastprop` GitHub page for instructions on retrieving each dataset and preparing it for use with `fastprop`, where applicable.

## Competing interests
None.

## Funding
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-SC0023112.

## Authors' contributions
Initial ideation of `fastprop` was a joint effort of Burns and Green.
Implementation, benchmarking, and writing were done by Burns.

## Acknowledgements
The authors acknowledge Haoyang Wu, Hao-Wei Pang, and Xiaorui Dong for their insightful conversations when initially forming the central ideas of `fastprop`.

## Disclaimer
This report was prepared as an account of work sponsored by an agency of the United States Government.
Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its
endorsement, recommendation, or favoring by the United States Government or any agency
thereof.
The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

# Cited Works