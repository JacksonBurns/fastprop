---
title: "Generalizable, Fast, and Accurate Deep-QSPR with `fastprop`. Part 1: Framework and Benchmarks"
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
---

<!-- Graphical Abstract Goes Here -->

# TODO
 - re-run QM9 benchmark with 3 repetitions
 - re-run QM8 with 3 repetitions and scaffold, update table too with these results
 - re-run _all_ benchmarks to ensure the results included here are with the latest version of `fastprop` (and indicate this version of `fastprop` in the benchmark methods section).
 - include the time needed to generate descriptors in this as a separate value from the training time
 - change ESOL to reference against the UniMol paper, or else re-run the CMPNN with repetitions (and Chemprop)
 - FreeSolv: keep the comparison to DeepDelta, but use the correct performance number (the one with property differences instead of absolutes). Use another point of comparison for absolute prediction, probably UniMol.
 - consider adding the delta_fubrain study
 - run chemprop on the HIV dataset, but as multiclass instead of binary
 - re-run the CMPNN and chemprop on SIDER with a holdout set to get a more honest result
 - decide where to report times

# Abstract
Quantitative Structure-Property/Activity Relationship studies, often referred to interchangeably as QS(P/A)R, seek to establish a mapping between molecular structure and an arbitrary Quantity of Interest (QOI).
Since its inception this was done on a QOI-by-QOI basis with new descriptors being devised by researchers to _specifically_ map to their QOI.
This continued for years and culminated in packages like DRAGON (later E-dragon), PaDEL-descriptor (and padelpy),  Mordred, and many others.
The sheer number of different packages resulted in the creation of 'metapackages' which served only to aggregate these other calculators, including tools like molfeat, ChemDes,  Parameter Client, and AIMSim.
Despite the creation of these aggregation tools, QSPR studies have continued to focus on QOI-by-QOI studies rather than attempting to create a generalizable approach which is capable of modeling across chemical domains.

One contributing factor to this is that QSPR studies have relied almost exclusively on linear methods for regression.
Efforts to incorporate Deep Learning (DL) as a regression technique (Deep-QSPR), which would be capable of capturing the non-linear behavior of arbitrary QOIs, have instead focused on using molecular fingerprints as inputs.
The combination of bulk molecular-level descriptors with DL has remained largely unexplored, in significant part due to the orthogonal domain expertise required to combine the two.
Generalizable QSPR has turned to learned representations primarily via message passing graph neural networks.
This approach has proved remarkably effective but is not without drawbacks.
Learning a representation requires large datasets to avoid overfitting or even learn at all, loses interpretability since an embedding's meaning can only be induced retroactively, and needs significant execution time given the complexity of the underlying message passing algorithm.
This paper introduces `fasprop`, a software package and general Deep-QSPR framework that combines a cogent set of molecular descriptors with DL to achieve state-of-the-art performance on datasets ranging from tens to tens of thousands of molecules.
`fastprop` is designed with Research Software Engineering best practices and is free and open source, hosted at github.com/jacksonburns/fastprop.

## Scientific Contribution
<!-- use a maximum of 3 sentences to specifically highlight the scientific contributions that advance the field and what differentiates your contribution from prior work on this topic. -->
`fastprop` is a QSPR framework that achieves state-of-the-art accuracy on datasets of all sizes without sacrificing speed or interpretability.
As a software package `fastprop` emphasizes Research Software Engineering best practices, reproduciblity, and ease of use for experts across domains.

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
A thru-theme in these frameworks is the increasing complexity of DL techniques and consequent uninterpretability.
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
`fastprop` automatically calculates the corresponding molecular descriptors with `mordred`, re-scales both the descriptors and the targets appropriately, and then trains an FNN to predict the indicated target properties.
By default this FNN is two hidden layers with 1800 neurons each, though the configuration can be readily changed via the command line interface or configuration file.
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
rescaling: True
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

<!-- The authors of Chemprop have even suggested on GitHub issues in the past that molecular-level features be added to Chemprop learned representations (see https://github.com/chemprop/chemprop/issues/146#issuecomment-784310581) to avoid overfitting. -->

All of these `fastprop` benchmarks are reproducible, and complete instructions for installation, data retrieval and preparation, and training are publicly available on the `fastprop` GitHub repository at [github.com/jacksonburns/fastprop](https://github.com/jacksonburns/fastprop).

## Benchmark Methods
All benchmarks use 80% of the dataset for training (selected randomly, unless otherwise stated), 10% for validation, and holdout the remaining 10% for testing (unless otherwise stated).
Sampling is performed using the `astartes` package [@astartes], which implements a variety of sampling algorithms and is highly reproducible.

Results for `fastprop` are reported as the average value of a metric and its standard deviation across a number of repetitions (repeated re-sampling of the dataset).
The number of repetitions is chosen to either match referenced literature studies or else increased from two until the standard deviation changes by less than ten percent.
Note that this is _not_ the same as cross-validation.

The evaluation metric used in each of these benchmarks are chosen to match literature precedent, particularly as established by MoleculeNet [@moleculenet].
However, scale-dependent metrics require readers to understand the relative magnitude of the target variables (for regression) or understand the nuances of classification quantification (ARUOC).
The authors prefer more readily interpretable metrics such as accuracy for classification and (Weighted) Mean Absolute Percentage Error (W/MAPE) for regression, both of which are included where relevant.

## Regression Datasets
See Table \ref{regression_results_table} for a summary of all the regression dataset results.
Especially noteworthy is the performance on the ESOL and PAH datasets, which dramatically surpass literature best and have only 55 datapoints, respectively.
Citations for the datasets themselves are included in the sub-sections of this section.

Table: Summary of regression benchmark results. \label{regression_results_table}

+---------------+--------------------+-------------+-----------------------------------+------------+-------------------------+-------------+
|   Benchmark   | Samples (k)        |   Metric    |          Literature Best          | `fastprop` |        Chemprop         |   Speedup   |
+===============+====================+=============+===================================+============+=========================+=============+
|QM9            |~134                |L1           |0.0047$^a$                         |0.0063      |0.0081$^a$               |      ~      |
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
Originally described in Scientific Data [@qm9] and perhaps the most established property prediction benchmark, Quantum Machine 9 (QM9) provides quantum mechanics derived descriptors for all small molecules containing one to nine heavy atoms, totaling ~134k.
The data was retrieved from MoleculeNet [@moleculenet] in a readily usable format.
As a point of comparison, performance metrics are retrieved from the paper presenting the UniMol architecture [@unimol] previously mentioned.
In that study they trained on only three especially difficult QOIs (homo, lumo, and gap) using scaffold-based splitting, reporting mean and standard deviation across 3 repetitions.

`fastprop` achieves ... $\pm$ ... MAE, whereas Chemprop achieves ... $\pm$ ... and the UniMol framework manages ... $\pm$ ....
fastprop beats Chemprop. UniMol is using 3D information here and only achieves a slight performance increase.
<!-- Results go here. -->

### OCELOTv1
The Organic Crystals in Electronic and Light-Oriented Technologies (OCELOTv1) dataset, originally described by Bhat et al. [@ocelot], maps 15 quantum mechanics descriptors and optoelectronic properties to ~25k chromophoric small molecules.
The literature best model is the Molecular Hypergraph Neural Network (MHNN) [@mhnn] which specializes in learned representations for optoelectronic properties and also includes Chemprop as a baseline for comparison.
They used a 70/10/20 random split with three repetitions and final performance reported is the average across those three repetitions.

As done in the referene study, the MAE for each task is shown in Table \ref{ocelot_results_table}.
Meanings for each abbreviation are the same as in the original database publication [@ocelot].
The geometric mean across all tasks, which accounts for the different scales of the target values better than the arithmetic mean, is also included as a summary statistic.
Note also that the relative percentage performance difference between fastprop and chemprop (`fast/chem`) and fastprop and MHNN (`fast/MHNN`) are also included.

Table: Per-task OCELOT dataset results. MHNN and Chemprop results are retrieved from the literature [@mhnn]. \label{ocelot_results_table}

+--------+----------+----------+-----------+-------+-----------+
| Target | fastprop | Chemprop | fast/chem | MHNN  | fast/mhnn |
+========+==========+==========+===========+=======+===========+
| HOMO   | 0.326    | 0.330    | -1.3      | 0.306 | 6.4       |
+--------+----------+----------+-----------+-------+-----------+
| LUMO   | 0.283    | 0.289    | -2.2      | 0.258 | 9.6       |
+--------+----------+----------+-----------+-------+-----------+
| H-L    | 0.554    | 0.548    | 1.1       | 0.519 | 6.7       |
+--------+----------+----------+-----------+-------+-----------+
| VIE    | 0.205    | 0.191    | 7.5       | 0.178 | 15.3      |
+--------+----------+----------+-----------+-------+-----------+
| AIE    | 0.196    | 0.173    | 13.5      | 0.162 | 21.2      |
+--------+----------+----------+-----------+-------+-----------+
| CR1    | 0.056    | 0.055    | 1.1       | 0.053 | 4.9       |
+--------+----------+----------+-----------+-------+-----------+
| CR2    | 0.055    | 0.053    | 4.7       | 0.052 | 6.7       |
+--------+----------+----------+-----------+-------+-----------+
| HR     | 0.106    | 0.133    | -20.4     | 0.099 | 7.0       |
+--------+----------+----------+-----------+-------+-----------+
| VEA    | 0.190    | 0.157    | 21.1      | 0.138 | 37.7      |
+--------+----------+----------+-----------+-------+-----------+
| AEA    | 0.183    | 0.154    | 18.9      | 0.124 | 47.7      |
+--------+----------+----------+-----------+-------+-----------+
| AR1    | 0.054    | 0.051    | 6.6       | 0.050 | 8.8       |
+--------+----------+----------+-----------+-------+-----------+
| AR2    | 0.049    | 0.052    | -6.2      | 0.046 | 6.1       |
+--------+----------+----------+-----------+-------+-----------+
| ER     | 0.099    | 0.098    | 0.6       | 0.092 | 7.2       |
+--------+----------+----------+-----------+-------+-----------+
| S0S1   | 0.281    | 0.249    | 13.0      | 0.241 | 16.7      |
+--------+----------+----------+-----------+-------+-----------+
| S0T1   | 0.217    | 0.150    | 44.9      | 0.145 | 49.9      |
+--------+----------+----------+-----------+-------+-----------+
| G-Mean | 0.148    | 0.140    | 5.9       | 0.128 | 15.9      |
+--------+----------+----------+-----------+-------+-----------+

`fastprop` 'trades places with Chemprop, outperforming on four of the metrics and underperforming on others.
Overall the geometric mean of MAE across all the tasks is ~6% higher, though this result may not be statistically significant.
Both `fastprop` and Chemprop are outperformed by the bespoke MHNN model, which is not itself evaluated on any other common property prediction benchmarks.

Although `fastprop` is not able to reach state-of-the-art accuracy on this dataset this result is still promising.
None of the descriptors implemented in `mordred` were designed to specifically correlate to these QM-derived targets, yet the FNN is able to learn a representation which is nearly as informative as Chemprop.
The fact that a bespoke modeling approach is the most performant is not surprising and instead demonstrates the continued importance of expert input on certain domains.

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
Quantum Machine 8 (QM8) is the predecessor to QM9 first described in 2015 [@qm8].
It follows the same generation procedure as QM9 but includes only up to eight heavy atoms for a total of approximately 22k molecules.
Again, this study used the dataset as prepared by MoleculeNet [@moleculenet] and compares to the UniMol [@unimol] set of benchmarks as a reference point, wherein they used the same data splitting procedure described previsouly but regressed all 12 targets in QM8.

UniMol achieved an average MAE across all tasks of 0.00156 $\pm$ 0.0001, `fastprop` approaches that performance with 0.0166, and Chemprop trails both frameworks with 0.0190 $\pm$ 0.0001.
Much like with QM9 `fastprop` outperforms LR frameworks until 3D information is encoded with UniMol.
As with OCELOTv1, the stated performance is achieved despite the targets being predicted still not directly intended for correlation with the `mordred` descriptors.

Of note is that even though `fastprop` is approaching the leading performance on this benchmark, other performance metrics cast doubt on the model performance.
The weighted mean absolute percentage error (wMAPE) on a per-task basis is shown in Table \ref{qm8_results_table}.

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

At each level of theory (CC2, PBE0, and CAM) `fastprop` is reaching the limit of chemical accuracy on excitation energies (E1 and E2) but is significantly less accurate on oscillator strengths (f1 and f2).
This can at least partially be attributed to dataset itself.
Manual analysis reveals that nearly 90% of the molecules in the dataset fall within only 10% of the total range of f1 values, which is highly imbalanced.
Additionally that 90% of molecules actual f1 values are all near-zero ro zero, which are intentionally less represented in the wMAPE metric.
Future literature studies should take this observation into account and perhaps move away from this splitting approach toward one which accounts for this imbalance.

<!-- Results go here -->

### ESOL
First described in 2004 [@esol] and has since become a critically important benchmark for QSPR/molecular proeprty prediction studies.
The dataset includes molecular structure for approximately 1.1k simple organic molecules are their corresponding experimentally measured free energy of solvation.
This property is a classic target of QSPR studies and is especially well suited for `fastprop`.

 - Reference against the CMPNN paper [@cmpnn], but speficially the amended results shown on their GitHub page: https://github.com/SY575/CMPNN/blob/b647df22ec8fde81785c5a86138ac1efd9ccf9c1/README.md
 - `fastprop` achieves RMSE of 0.566 kcal/mol on an 80/10/10 random split, on a different split (5-fold CV w/o holdout) the CMPNN and Chemprop get 0.547 and 0.665, respectively.
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
First assembled and fitted to by Saldana and coauthors [@flash], the Flash dataset includes around 0.6k compounds, primarily alkanes and some oxygen-containing compounds, and their literature-reported flash point.
The reference study reports the performance on only one repetition, but manuualy confirms that the distribution of points in the three splits follows the parent dataset.
The split itself was a 70/20/10 random split, which is repeated four times for this study.

Using a complex multi-model ensembling method, the reference study achieved an RMSE of 13.2, an MAE of 8.4, and an MAPE of 2.5%.
`fastprop` matches this performance, achieving 13.5 $\pm$ 2.2 RMSE, 9.0 $\pm$ 1.3 MAE, and 2.7% $\pm$ 0.5% MAPE.
Chemprop, however, struggles to match the accuracy of either method.
It manages an RMSE of 21.2 $\pm$ 2.2 and an MAE of 13.8 $\pm$ 2.1 and does not report MAPE.

Critically, `fastprop` dramatically outperforms both methods in terms of training time.
The reference model required significant manual intervention to create an ensemble, whereas `fastprop` arrived at the indicated performance without any manual intervention in only 1 minute and 20 seconds (9 seconds to calculate descriptors).
Chemprop, in addition to not reaching the same level of accuracy, took 5 minutes and 44 seconds to do so - more than 4 times the execution time of `fastprop`.

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
 - `fastprop`, using a 4-repetition 80/10/10 random split, is able to achieve a correlation coefficient of 0.960+/-0.027 in 2m12s (17s descriptor calculation). This is an L1 error of 0.230 and an MAPE of 3.1%.
 - Chemprop using the same scheme achieves only 0.746+/-0.085 in 36s. This is an L1 error 1.07+/-0.20.
 - `fastprop` is able to correlate structure directly to property without expert knowledge in only minutes while having only 44 training points.
<!--
On only the 21 points that have a retention index, mape: 0.024147   0.008081 
Similar good story on the boiling point, acentric factor only has 12 points left after removing unvalued entries - difficult to call it anything.
-->

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
Citations for the datasets themselves are included in the sub-sections of this section.

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
 - ~41k points
 - Originally compiled by Riesen and Bunke [@hiv], this dataset includes the reported HIV activity for many small molecules; this is an established benchmark in the world of molecular property prediction, and the exact version used is that which was prepared in MoleculeNet [@moleculenet].
 - This dataset is unique in that the labels in the original study include three possible classes (a _multiclass_) regression problem whereas the most common reported metric is instead lumping positive and semi-positive labels into a single class to reduce the task to _binary_ classification; both are reported here.
 - To compare to UniMol, use an 80/10/10 scaffold-based split with three repetitions.

#### Binary
 - `fastprop` matches the literature best UniMol (AUROC: 80.8+/-0.3 [@unimol]), AUROC: 0.805+/-0.04. This corresponds to an accuracy of 96.8+/-1.0.
 - Chemprop performs worse than both of these models.
<!-- [01/29/2024 10:53:49 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    3.0  0.137778  0.012579  0.123418  0.133239  0.143060  0.144957  0.146854
validation_accuracy    3.0  0.963035  0.004458  0.959144  0.960603  0.962062  0.964981  0.967899
validation_f1          3.0  0.289890  0.007632  0.282051  0.286187  0.290323  0.293810  0.297297
validation_auroc       3.0  0.781569  0.016585  0.765655  0.772978  0.780301  0.789526  0.798751
[01/29/2024 10:53:49 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    3.0  0.123481  0.022154  0.097957  0.116358  0.134758  0.136243  0.137729
test_accuracy    3.0  0.968142  0.009633  0.959630  0.962913  0.966196  0.972398  0.978599
test_f1          3.0  0.307895  0.113440  0.194175  0.251316  0.308458  0.364755  0.421053
test_auroc       3.0  0.805190  0.041026  0.776199  0.781719  0.787238  0.819685  0.852133
[01/29/2024 10:53:49 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing accuracy yielded p value of p=0.452>0.05.-->


#### Multiclass
 - Not yet found any reported results in the literature.
 - `fastprop` achieves 0.818+/-0.019 AUROC, actually _increasing_ performance from the 'easier' binary classification task; note though that the accuracy has dropped to 0.424+/-0.071.
<!-- [01/29/2024 11:10:11 AM fastprop.fastprop_core] INFO: Displaying validation results:
                       count      mean       std       min       25%       50%       75%       max
validation_kldiv_loss    3.0  0.158862  0.015531  0.140936  0.154146  0.167356  0.167825  0.168295
validation_auroc         3.0  0.801941  0.007333  0.793869  0.798816  0.803762  0.805977  0.808192
validation_accuracy      3.0  0.390970  0.030146  0.370830  0.373641  0.376451  0.401039  0.425628
[01/29/2024 11:10:11 AM fastprop.fastprop_core] INFO: Displaying testing results:
                 count      mean       std       min       25%       50%       75%       max
test_kldiv_loss    3.0  0.142161  0.030653  0.107482  0.130424  0.153366  0.159501  0.165635
test_auroc         3.0  0.818408  0.019140  0.798132  0.809531  0.820930  0.828546  0.836163
test_accuracy      3.0  0.424130  0.071443  0.343844  0.395847  0.447851  0.464273  0.480695
[01/29/2024 11:10:11 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.236>0.05. -->

### QuantumScents
 - ~3.5k points
 - Compiled by Burns and Rogers [@quantumscents], this dataset contains SMILES and 3D structures for a collection of molecules labeled with their scents; each molecule can have any number of reported scents from a possible 113 different labels.
 - This benchmark is specifically a Quantitive Structure-Odor Relationship.
 - In the reference study, Chemprop achieved an AUROC of 0.85 with its default settings and an improved AUROC of 0.88 by incorporating the atomic descriptors calculated as part of QuantumScents.
 - `fastprop`, without using these descriptors and only the SMILES (not 3D) information, achieves an AUROC of 0.915+/-0.004.
 - The GitHub repository contains an example of generating custom descriptors incorporating the 3D information from QuantumScents and passing these to `fastprop`; impact on the performance was negligible.
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
 - ~1.4k compounds, including small molecules, metals, and salts.
 - First described by Kuhn et al. in 2015 [@sider], the Side Effect Resource (SIDER) database has become a standard property prediction benchmark; the challenging dataset requires mapping structure to any combination of 27 side effects.
 - Among the best performers in literature is the CMPNN, with a reported AUROC of 0.666+/-0.0007 (5 repeats of 5-fold CV w/o holdout set); with the same approach, Chemprop got 0.646+/-0.016.
 - On an 80/10/10 random split with 5 repetitions, `fastprop` achieves AUROC of 0.655+/-0.016 in 5m9s whereas Chemprop achieves 0.637+/-0.011 in just under an **hour**. (*during one repetetion side effect 3 was ill-defined in the testing data and was excluded from the average performance for that repetition, repetition 3 was thrown out entirely because a similar issue occoured with the validation set during training, and AUROC for that trial was 0.473 which was an outlier compared to the others.)
<!--
fastprop:
4m21s to calculate features

[01/29/2024 12:06:57 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    5.0  0.513962  0.007272  0.507627  0.509618  0.511895  0.514545  0.526124
validation_auroc       5.0  0.634701  0.024984  0.610214  0.621136  0.628353  0.638365  0.675439
[01/29/2024 12:06:57 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    5.0  0.490345  0.011056  0.476638  0.483128  0.490632  0.496424  0.504902
test_auroc       5.0  0.655232  0.016167  0.628616  0.652302  0.661167  0.664349  0.669729
[01/29/2024 12:06:57 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing auroc yielded p value of p=0.161>0.05.

real    0m47.894s
user    0m27.250s
sys     0m19.838s

Chemprop:
real    56m46.703s
user    50m36.152s
sys     6m25.502s

mean auroc was 0.637+/-0.011
-->

### Pgp
 - ~1.2k small molecule drugs
 - First reported in 2011 by Broccatelli and coworkers [@pgp], this dataset has since become a standard benchmark and is included in the Theraputic Data Commons (TDC) [@tdc] model benchmarkign tool; the dataset maps drugs to a binary label indicating if they inhibit P-glycoprotein.
 - TDC serves this data through a Python package, but due to installation issues the data was retrieved from the original study instead. The reccomended splitting approach is a 70/10/20 scaffold-based split, which is done here.
 - The reference literature model uses a molecular interaction field, bu has since been surpassed by other models; according to TDC the current leader [@pgp_best] on this benchmark has achieved an AUROC of 0.938+/-0.002 (see [the TDC Pgp leaderboard](https://tdcommons.ai/benchmark/admet_group/03pgp/)).
 - According to the same leaderboard, Chemprop [@chemprop_theory] achieves 0.886+/-0.016 with the includsion of additional molecular features.
 - `fastprop` achieves 0.926+/-0.010 AUROC with an accuracy of 86.0+/-0.2%, approaching the performance of the current leader and significantly outperforming Chemprop.
<!--
fastprop does this in 44s w/ cached descriptors
[01/29/2024 11:48:42 AM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_bce_loss    4.0  0.387118  0.069688  0.310836  0.352300  0.379792  0.414611  0.478053
validation_accuracy    4.0  0.864754  0.056194  0.795082  0.831967  0.872951  0.905738  0.918033
validation_f1          4.0  0.851747  0.037130  0.800000  0.837500  0.862500  0.876747  0.881988
validation_auroc       4.0  0.917608  0.018889  0.904749  0.907524  0.910028  0.920112  0.945628
[01/29/2024 11:48:42 AM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_bce_loss    4.0  0.387347  0.032653  0.358048  0.359642  0.386927  0.414632  0.417484
test_accuracy    4.0  0.860204  0.002041  0.857143  0.860204  0.861224  0.861224  0.861224
test_f1          4.0  0.874467  0.008665  0.862903  0.870271  0.876923  0.881119  0.881119
test_auroc       4.0  0.926062  0.009703  0.914280  0.921085  0.926352  0.931329  0.937263
[01/29/2024 11:48:42 AM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing accuracy yielded p value of p=0.877>0.05.-->

### ARA
 - ~0.8k small molecules
 - Compiled by Schaduangrat et al. in 2023 [@ara], this dataset maps molecular structure to a binary label indicating if the molecule is an Androgen Receptor Antagonist (ARA); the reference study introduced DeepAR, a highly complex modeling approach, which achieved an accuracy of 0.911 and an AUROC of 0.945.
 - For this study, an 80/10/10 random splitting is repeated four times on the dataset since no analogous split to the reference study can be determined.
 - Chemprop takes 16m55s to run on this dataset and achieves only 0.824+/-0.020 accuracy and 0.898+/-0.022 AUROC.
 - `fastprop` takes 2m7s and is competitive with the reference study, achieving a 0.882+/-0.02 accuracy and 0.951+/-0.018 AUROC.
<!--
fastprop:
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
Count	4 -->

# Limitations and Future Work
## Execution Time
Execution is ultimately not a significant concern - dataset generation takes a huge amount of time relative to all training methods.
Day to day speedup is nice and significant here, but again just a nice to have.

Execution time is as reported by the unix `time` command using Chemprop version 1.6.1 on Python 3.8 and `fastprop` 1.0.0rc2 on Python 3.11 and include the complete invocation of their respective program: `time chemprop_train ...` and `time fastprop train ...`.
The insignificant time spent manually collating Chemprop results (Chemprop does not natively support repetitions) is excluded.
This coarse comparison is intended to emphasize the scaling of LRs and Deep-QSPR and that `fastprop` is, generally speaking, much faster.

There is an obvious performance improvement to be had (both in training and inference) by reducing the number of descriptors used to a subset that are highly-weighted in the network.
Future work will address the possibility of reducing the required number of descriptors to achieve accurate predictions, which would shrink the required size of the network and remove the training limitation and remediate the inference limitation.
This has _not_ been done in this initial study for two reasons:
 1. To emphasize the capacity of the DL framework to effecitvely perform feature selection on its own via the training process, de-emphasizing unimportant descriptors.
 2. Because the framework as it stands is already _dramatically_ faster than alternatives, and all existing modeling solutions are fast relative to the timescale of drug development where they are deployed (i.e. accuracy is more important than runtime).

### Training
The molecule representation is a series of scalars rather than a graph object, the memory consumption is lower, enabling larger batch sizes and thus faster training.

_Table of training times_

Because `fastprop` does not require message passing like many existing learned representations, all of the operations rqeuired to train the network are much faster.
The limitation is that due to the large size of the molecular descriptor set, the neural network must also be large (primarily in height of hidden layers, rather than depth).
This means that when training with a GPU `fastprop` will be much faster than learned representations, but when using a CPU it will generally be slower.

### Inference
Slow on inference, especially on virtual libaries which may number in the millions of compounds.
Thankfully descriptor calclulation is embarassingly parallel, and in practice the number of descriptors needed to be calculated can be reduced once those which are relevant to the neural network are selected based on their weights.

## Combination with Learned Representations
This seems like an intuitive next step, but we shouldn't do it.
_Not_ just slamming all the mordred features into Chemprop, which would definitely give good results but would take out the `fast` part (FNN would also have to be increased in size).
`fastprop` is good enough on its own.

## Interpretability
Cite Wengao's paper about backing out meaning from intermediate embeddings, but emphasize we are in an even better spot because of the physical meaning of the inputs.

## Additional Descriptors
The `mordredcommunity` featurization package could of course stand to have more features added that could potentially expand its applicability beyond the datasets listed here and improve the performance on those already attempted.
Domain-specific additions which are not just derived from the descriptors already implemented will be required.

### Stereochemical
In its current state, the underlying `mordredcommunity` featurization engine does not include any connectivity based-descriptors that reflect the presence or absence of stereocenters.
While some of the 3D descriptors it implements will inherently reflects these differences somewhat, more explicit descriptors like the Stero Signature Molecular Descriptor (see https://doi.org/10.1021/ci300584r) may prove helpful in the future.

This is really just one named domain - any domain where there are bespoke descriptors could be added to `mordredcommunity`.

### Charged Species

### 3D Descriptors
 - already supported by `mordredcommunity` just need to find a good way to ingest 3D data or embed 3D conformers (cite some lit. like quantumscents for the latter point)

# Availability
 - Project name: fastprop
 - Project home page: github.com/jacksonburns/fastprop
 - Operating system(s): Platform independent
 - Programming language: Python
 - Other requirements: pyyaml, lightning, mordredcommunity, astartes
 - License: MIT

# Declarations

## Availability of data and materials
`fastprop` is Free and Open Source Software; anyone may view, modify, and execute it according to the terms of the MIT license. See github.com/jacksonburns/fastprop for more information.

All data used in the Benchmarks shown above is publicly avialable under a permissive license. See the benchmarks directory at the `fastprop` GitHub page for instructions on retrieving each dataset and preparing it for use with `fastprop`, where applicable.

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