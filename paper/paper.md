---
title: "Generalizable, Fast, and Accurate DeepQSPR with `fastprop`"
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
date: 2 April, 2024
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
note: |
 To prepare the TeX file for submission to arXiv, I used this command instead:
  pandoc --citeproc -s paper.md -o paper.pdf --template default.latex \
    --pdf-engine=pdflatex --pdf-engine-opt=-output-directory=foo
 Which leaves the intermediate TeX file in the `foo` directory. I then manually
 fix an image filepath which pandoc incorrectly leaves.
---

<!-- Graphical Abstract Goes Here -->

# Abstract
Quantitative Structure-Property Relationship studies (QSPR), often referred to interchangeably as QSAR, seek to establish a mapping between molecular structure and an arbitrary Quantity of Interest (QOI).
Historically this was done on a QOI-by-QOI basis with new descriptors being devised by researchers to _specifically_ map to their QOI.
A large number of descriptors have been invented, and can be computed using packages like DRAGON (later E-dragon), PaDEL-descriptor (and padelpy), Mordred, CODESSA, and many others.
The sheer number of different descriptor packages resulted in the creation of 'meta-packages' which served only to aggregate these other calculators, including tools like molfeat, ChemDes, Parameter Client, and AIMSim.

Generalizable descriptor-based modeling was a natural evolution of these meta-packages' development.
Historically QSPR researchers focused almost exclusively on linear methods.
Another community of researchers focused on finding nonlinear correlations between molecular structures and a QOI, often using Deep learning (DL).
The DL community typically used molecular fingerprints instead of the complex descriptors popular in QSPR community.
Recently the DL community has turned to learned representations primarily via message passing graph neural networks.
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
As explained by Muratov et. al [@muratov_qsar] QSPR uses linear methods (some of which are now called machine learning) almost exclusively.
The over-reliance on this category of approaches may be due to priorities; domain experts seek interpretability in their work, especially given that the inputs are physically meaningful descriptors, and linear methods lend themselves well to this approach.
Practice may also have been a limitation, since historically training and deploying neural networks required more computer science expertise than linear methods.

All of this is not to say that Deep Learning (DL) has _never_ been applied to QSPR.
Applications of DL to QSPR, i.e. DeepQSPR, were attempted throughout this time period but focused on the use of molecular fingerprints rather than descriptors.
This may be at least partially attributed to knowledge overlap between deep learning experts and this sub-class of descriptors.
Molecular fingerprints are bit vectors which encode the presence or absence of sub-structures in an analogous manner to the "bag of words" featurization strategy common to natural language processing.
Experts have bridged this gap to open this subdomain and proved its effectiveness.
In Ma and coauthors' review of DL for QSPR [@ma_deep_qsar], for example, it is claimed that DL with fingerprint descriptors is more effective than with molecular-level descriptors.
They also demonstrate that DL outperforms or at least matches classical machine learning methods across a number of ADME-related datasets.
The results of this study demonstrate that molecular-level descriptors actually _are_ effective and reaffirm that DL matches or outperforms baselines, in this case linear.

Despite their differences, both classical- and Deep-QSPR shared a lack of generalizability.
Beyond the domains of chemistry where many of the descriptors had been originally devised, models were either unsuccessful or more likely simply never evaluated.
As interest began to shift toward the prediction of molecular properties which were themselves descriptors (i.e. derived from quantum mechanics simulations) - to which none of the devised molecular descriptors were designed to be correlated - learned representations (LRs) emerged.

## Shift to Learned Representations
The exact timing of the transition from fixed descriptors (molecular-level or fingerprints) to LRs is difficult to ascertain [@Coley2017].
Among the most cited at least is the work of Yang and coworkers in 2019 [@chemprop_theory] which laid the groundwork for applying LRs to "Property Prediction" - QSPR by another name.
In short, the basic idea is to initialize a molecular graph with only information about its bonds and atoms such as order, aromaticity, atomic number, etc.
Then via a Message Passing Neural Network (MPNN) architecture, which is able to aggregate these atom- and bond-level features into a vector in a manner which can be updated, the 'best' representation of the molecule is found during training.
This method proved highly accurate _and_ achieved the generalizability apparently lacking in descriptor-based modeling.
The modern version of the corresponding software package Chemprop (described in [@chemprop_software]) has become a _de facto_ standard for property prediction, partially because of the significant development and maintenance effort supporting that open source software project.

Following the initial success of Chemprop numerous representation learning frameworks have been devised, all of which slightly improve performance.
The Communicative-MPNN (CMPNN) framework is a modified version of Chemprop with a different message passing scheme to increase the interactions between node and edge features [@cmpnn].
Uni-Mol incorporates 3D information and relies extensively on transformers [@unimol].
In a "full circle moment" architectures like the Molecular Hypergraph Neural Network (MHNN) have been devised to learn representations for specific subsets of chemistry, in that case optoelectronic properties [@mhnn].
Myriad others exist including GSL-MPP (accounts for intra-dataset molecular similarity) [@gslmpp], SGGRL (trains three representations simultaneously using different input formats) [@sggrl], and MOCO (multiple representations and contrastive pretraining) [@moco].

### Limitations
Despite the continuous incremental performance improvements, this area of research has had serious drawbacks.
A thru-theme in these frameworks is the increasing complexity of DL techniques and consequent un-interpretability.
This also means that actually _using_ these methods to do research on real-world dataset requires varying amounts of DL expertise, creating a rift between domain experts and these methods.
Perhaps the most significant failure is the inability to achieve good predictions on small [^1] datasets.
This is a long-standing limitation, with the original Chemprop paper stating that linear models are about on par with Chemprop for datasets with fewer than 1000 entries [@chemprop_theory].

This limitation is especially challenging because it is a _fundamental_ drawback of the LR approach.
Without the use of advanced DL techniques like pre-training or transfer learning, the model is essentially starting from near-zero information every time a model is created.
This inherently requires larger datasets to allow the model to effectively 're-learn' the chemical intuition which was built in to descriptor- and fixed fingerprint-based representations.

Efforts are of course underway to address this limitation, though none are broadly successful.
One simple but incredibly computationally expensive approach is to use delta learning, which artificially increases dataset size by generating all possible _pairs_ of molecules from the available data (thus squaring the size of the dataset).
This was attempted by Nalini et al. [@deepdelta], who used an unmodified version of Chemprop referred to as 'DeepDelta' to predict _differences_ in molecular properties for _pairs_ of molecules.
They achieve increased performance over standard LR approaches but _lost_ the ability to train on large datasets due to simple runtime limitations.
Other increasingly complex approaches are discussed in the outstanding review by van Tilborg et al. [@low_data_review].

While iterations on LRs and novel approaches to low-data regimes have been in development, the classical QSPR community has continued their work.
A turning point in this domain was the release of `mordred`, a fast and well-developed package capable of calculating more than 1600 molecular descriptors [@mordred].
Critically this package was fully open source and written in Python, allowing it to readily interoperate with the world-class Python DL software ecosystem that greatly benefitted the LR community.
Despite previous claims that molecular descriptors _cannot_ achieve generalizable QSPR in combination with DL, the opposite is shown here with `fastprop`.

[^1]: What constitutes a 'small' dataset is decidedly _not_ agreed upon by researchers.
For the purposes of this study, it will be used to refer to datasets with ~1000 molecules or fewer, which the authors believe better reflects the size of real-world datasets.

# Implementation
At its core the `fastprop` 'architecture' is simply the `mordred` molecular descriptor calculator [^2] [@mordred] connected to a Feedforward Neural Network (FNN) implemented in PyTorch Lightning [@lightning] (Figure \ref{logo}) - an existing approach formalized into an easy-to-use, reliable, and correct implementation.
`fastprop` is highly modular for seamless integration into existing workflows and includes end-to-end interfaces for general use.
In the latter mode the user simply specifies a set of SMILES [@smiles], a linear textual encoding of molecules, and their corresponding properties.
`fastprop` automatically calculates and caches the corresponding molecular descriptors with `mordred`, re-scales both the descriptors and the targets appropriately, and then trains an FNN to predict the indicated target properties.
By default this FNN is two hidden layers with 1800 neurons each connected by ReLU activation functions, though the configuration can be readily changed via the command line interface or configuration file.
`fastprop` principally owes its success to the cogent set of descriptors assembled by the developers of `mordred`.
Multiple descriptor calculators from the very thorough review by McGibbon et al. [@representation_review] could be used instead, though none are as readily interoperable as `mordred`.
Additionally, the ease of training FNNs with modern software like PyTorch Lightning and the careful application of Research Software Engineering best practices make `fastprop` as user friendly as the best-maintained alternatives.

![`fastprop` logo.\label{logo}](../fastprop_logo.png){ width=2in }

This trivially simple idea has been alluded to in previous published work but neither described in detail nor lauded for its generalizability or accuracy.
Comesana and coauthors, based on a review of the biofuels property prediction landscape, claimed that methods (DL or otherwise) using large numbers of molecular descriptors were unsuccessful, instead proposing a feature selection method [@fuels_qsar_method].
As a baseline in a study of photovoltaic property prediction, Wu et al. reported using the `mordred` descriptors in combination with both a Random Forest and an Artificial Neural Network, though in their hands the performance is worse than their bespoke model and no code is available for inspection [@wu_photovoltaic].

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
descriptor_set: all
# preprocessing
clamp_input: True
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

Training, prediction, and feature importance are then readily accessible via the commands `fastprop train`, `fastprop predict`, or `fastprop shap`, respectively.
The `fastprop` GitHub repository contains a Jupyter notebook runnable from the browser via Google colab which allows users to actually execute the above example, which is also discussed at length in the [PAHs section](#pah), as well as further details about each configurable option.

# Results & Discussion
There are a number of established molecular property prediction benchmarks commonly used in LR studies, especially those standardized by MoleculeNet [@moleculenet].
Principal among them are QM8 [@qm8] and QM9 [@qm9], often regarded as _the_ standard benchmark for property prediction.
These are important benchmarks and QM9 is included for completeness, though the enormous size and rich coverage of chemical space in the QM9 dataset means that nearly all model architectures are highly accurate, including `fastprop`.

Real world experimental datasets, particularly those common in QSPR studies, often number in the hundreds.
To demonstrate the applicability of `fastprop` to these regimes, many smaller datasets are selected including some from the QSPR literature that are not established benchmarks.
These studies relied on more complex and slow modeling techniques ([ARA](#ara)) or the design of a bespoke descriptor ([PAHs](#pah)) and have not yet come to rely on learned representations as a go-to tool.
In these data-limited regimes where LRs sometimes struggle, `fastprop` and its intuition-loaded initialization are highly powerful.
To emphasize this point further, the benchmarks are presented in order of dataset size, descending.

Two additional benchmarks showing the limitations of `fastprop` are included after the main group of benchmarks: Fubrain and QuantumScents.
The former demonstrates how `fastprop` can outperform LRs but still trail approaches like delta learning.
The later is a negative result showing how `fastprop` can fail on especially difficult, atypical targets.

All of these `fastprop` benchmarks are reproducible, and complete instructions for installation, data retrieval and preparation, and training are publicly available on the `fastprop` GitHub repository at [github.com/jacksonburns/fastprop](https://github.com/jacksonburns/fastprop).

## Benchmark Methods
The method for splitting data into training, validation, and testing sets varies on a per-study basis and is described in each sub-section.
Sampling is performed using the `astartes` package [@astartes] which implements a variety of sampling algorithms and is highly reproducible.
For datasets containing missing target values or invalid SMILES strings, those entries were dropped.

Results for `fastprop` are reported as the average value of a metric and its standard deviation across a number of repetitions (repeated re-sampling of the dataset).
The number of repetitions is chosen to either match referenced literature studies or else increased from two until the performance no longer meaningfully changes.
Note that this is _not_ the same as cross-validation.
Each section also includes the performance of a zero-layer (i.e. linear regression) network as a baseline to demonstrate the importance of non-linearity in a deep NN.

For performance metrics retrieved from literature it is assumed that the authors optimized their respective models to achieve the best possible results; therefore, `fastprop` metrics are reported after model optimization using the `fastprop train ... --optimize` option.
When results are generated for this study using Chemprop, the default settings are used except that the number of epochs is increased to allow the model to converge and batch size is increased to match dataset size and speed up training.
Chemprop was chosen as a point of comparison throughout this study since it is among the most accessible and well-maintained software packages for molecular machine learning.

When reported, execution time is as given by the unix `time` command using Chemprop version 1.6.1 on Python 3.8 and includes the complete invocation of Chemprop, i.e. `time chemprop_train ...`.
The insignificant time spent manually collating Chemprop results (Chemprop does not natively support repetitions) is excluded.
`fastprop` is run on version 1.0.6 using Python 3.11 and timing values are reported according to its internal time measurement  which was verified to be nearly identical to the Unix `time` command.
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

## Benchmark Results
See Table \ref{results_table} for a summary of all the results.
Subsequent sections explore each in greater detail.

Table: Summary of benchmark results, best state-of-the-art method vs. `fastprop` and Chemprop. \label{results_table}

+---------------+--------------------+-------------+--------------+------------+-------------------------+------+
|   Benchmark   | Samples (k)        |   Metric    |     SOTA     | `fastprop` |        Chemprop         |  p   |
+===============+====================+=============+==============+============+=========================+======+
|QM9            |~134                |MAE          |0.0047$^a$    |0.0060      |0.0081$^a$               |  ~   |
+---------------+--------------------+-------------+--------------+------------+-------------------------+------+
|Pgp            |~1.3                |AUROC        |0.94$^b$      |0.90        |0.89$^b$                 |  ~   |
+---------------+--------------------+-------------+--------------+------------+-------------------------+------+
|ARA            |~0.8                |Accuracy     |91$^c$        |88          |82*                      |0.083 |
+---------------+--------------------+-------------+--------------+------------+-------------------------+------+
|Flash          |~0.6                |RMSE         |13.2$^d$      |13.0        |21.2*                    |0.021 |
+---------------+--------------------+-------------+--------------+------------+-------------------------+------+
|YSI            |~0.4                |MAE          |22.3$^e$      |25.0        |28.9*                    |0.29  |
+---------------+--------------------+-------------+--------------+------------+-------------------------+------+
|PAH            |~0.06               |R2           |0.96$^f$      |0.97        |0.59*                    |0.0012|
+---------------+--------------------+-------------+--------------+------------+-------------------------+------+

a [@unimol] b [@pgp_best] c [@ara] d [@flash] e [@ysi] f [@pah] * These reference results were generated for this study.

Statistical comparisons of `fastprop` to Chemprop (shown in the `p` column) are performed using the non-parametric Wilcoxon-Mann-Whitney Test as implemented in GNumeric.
Values are only shown for results generated in this study which are known to be performed using the same methods.
Only the results for Flash and PAH are statistically significant at 95% confidence (p<0.05).

### QM9
Originally described in Scientific Data [@qm9] and perhaps the most established property prediction benchmark, Quantum Machine 9 (QM9) provides quantum mechanics derived descriptors for many small molecules containing one to nine heavy atoms, totaling ~134k.
The data was retrieved from MoleculeNet [@moleculenet] in a readily usable format.
As a point of comparison, performance metrics are retrieved from the paper presenting the UniMol architecture [@unimol] previously mentioned.
In that study they trained on only three especially difficult QOIs (homo, lumo, and gap) using scaffold-based splitting (a more challenging alternative to random splitting), reporting mean and standard deviation across 3 repetitions.

`fastprop` achieves 0.0060 $\pm$ 0.0002 mean absolute error, whereas Chemprop achieves 0.00814 $\pm$ 0.00001 and the UniMol framework manages 0.00467 $\pm$ 0.00004.
This places the `fastprop` framework ahead of previous learned representation approaches but still trailing UniMol.
This is not completely unexpected since UniMol encodes 3D information from the dataset whereas Chemprop and `fastprop` use only 2D.
Future work could evaluate the use of 3D-based descriptors to improve `fastprop` performance in the same manner that UniMol has with LRs.
All methods are better than a purely linear model trained on the molecular descriptors, which manages only a 0.0095 $\pm$ 0.0006 MAE.
<!-- 
[07/26/2024 12:29:08 PM fastprop.descriptors] INFO: Descriptor calculation complete, elapsed time: 0:47:15.814133
[07/26/2024 12:42:22 PM fastprop.cli.train] INFO: Displaying validation results:
                                                               count          mean           std           min           25%           50%           75%           max
validation_mse_scaled_loss                                       3.0  5.696597e-02  1.464139e-03  5.533025e-02  5.637197e-02  5.741369e-02  5.778384e-02  5.815398e-02
validation_r2_score                                              3.0  8.941775e-01  8.618863e-03  8.848469e-01  8.903457e-01  8.958445e-01  8.988428e-01  9.018410e-01
validation_homo_r2_score                                         3.0  8.429342e-01  8.330976e-03  8.333260e-01  8.403277e-01  8.473294e-01  8.477383e-01  8.481472e-01
validation_lumo_r2_score                                         3.0  9.560309e-01  6.386675e-03  9.492921e-01  9.530489e-01  9.568058e-01  9.594003e-01  9.619948e-01
validation_gap_r2_score                                          3.0  8.835675e-01  2.337481e-02  8.571015e-01  8.746572e-01  8.922130e-01  8.968005e-01  9.013880e-01
validation_mean_absolute_percentage_error_score                  3.0  4.377785e+09  4.662795e+09  7.852841e+08  1.743133e+09  2.700982e+09  6.174036e+09  9.647091e+09
validation_homo_mean_absolute_percentage_error_score             3.0  2.152999e-02  6.235604e-04  2.091644e-02  2.121344e-02  2.151044e-02  2.183677e-02  2.216310e-02
validation_lumo_mean_absolute_percentage_error_score             3.0  1.313336e+10  1.398838e+10  2.355853e+09  5.229398e+09  8.102944e+09  1.852211e+10  2.894127e+10
validation_gap_mean_absolute_percentage_error_score              3.0  3.167484e-02  4.700025e-04  3.113239e-02  3.153195e-02  3.193152e-02  3.194607e-02  3.196062e-02
validation_weighted_mean_absolute_percentage_error_score         3.0  6.259312e-02  2.938613e-02  4.419936e-02  4.564765e-02  4.709594e-02  7.179000e-02  9.648407e-02
validation_homo_weighted_mean_absolute_percentage_error_score    3.0  2.118536e-02  6.353157e-04  2.054997e-02  2.086774e-02  2.118552e-02  2.150306e-02  2.182060e-02
validation_lumo_weighted_mean_absolute_percentage_error_score    3.0  1.367028e-01  8.790174e-02  8.221404e-02  8.599984e-02  8.978565e-02  1.639472e-01  2.381088e-01
validation_gap_weighted_mean_absolute_percentage_error_score     3.0  2.989116e-02  4.000132e-04  2.952277e-02  2.967842e-02  2.983406e-02  3.007536e-02  3.031666e-02
validation_mean_absolute_error_score                             3.0  5.765846e-03  4.084659e-05  5.724337e-03  5.745771e-03  5.767204e-03  5.786600e-03  5.805996e-03
validation_homo_mean_absolute_error_score                        3.0  4.867187e-03  1.042128e-04  4.750099e-03  4.825895e-03  4.901691e-03  4.925732e-03  4.949773e-03
validation_lumo_mean_absolute_error_score                        3.0  5.256618e-03  1.629768e-04  5.160624e-03  5.162530e-03  5.164436e-03  5.304615e-03  5.444795e-03
validation_gap_mean_absolute_error_score                         3.0  7.173731e-03  1.558325e-04  7.062611e-03  7.084666e-03  7.106722e-03  7.229290e-03  7.351859e-03
validation_root_mean_squared_error_loss                          3.0  7.923986e-03  1.360257e-04  7.804391e-03  7.849997e-03  7.895603e-03  7.983783e-03  8.071964e-03
validation_homo_root_mean_squared_error_loss                     3.0  6.499117e-03  1.275366e-04  6.371946e-03  6.435168e-03  6.498391e-03  6.562703e-03  6.627016e-03
validation_lumo_root_mean_squared_error_loss                     3.0  7.079933e-03  2.307991e-04  6.849300e-03  6.964450e-03  7.079600e-03  7.195249e-03  7.310898e-03
validation_gap_root_mean_squared_error_loss                      3.0  9.656825e-03  3.210587e-04  9.447182e-03  9.472017e-03  9.496853e-03  9.761647e-03  1.002644e-02
[07/26/2024 12:42:22 PM fastprop.cli.train] INFO: Displaying testing results:
                                                         count          mean           std           min           25%           50%           75%           max
test_mse_scaled_loss                                       3.0  5.785314e-02  5.604013e-04  5.739484e-02  5.754075e-02  5.768666e-02  5.808229e-02  5.847792e-02
test_r2_score                                              3.0  8.865663e-01  8.278914e-03  8.796054e-01  8.819889e-01  8.843723e-01  8.900468e-01  8.957213e-01
test_homo_r2_score                                         3.0  8.251316e-01  1.408929e-02  8.093137e-01  8.195298e-01  8.297459e-01  8.330405e-01  8.363351e-01
test_lumo_r2_score                                         3.0  9.503889e-01  1.410442e-02  9.365089e-01  9.432296e-01  9.499502e-01  9.573289e-01  9.647075e-01
test_gap_r2_score                                          3.0  8.841783e-01  2.535874e-02  8.659718e-01  8.696962e-01  8.734207e-01  8.932816e-01  9.131426e-01
test_mean_absolute_percentage_error_score                  3.0  7.237129e+09  9.312357e+08  6.397108e+09  6.736447e+09  7.075786e+09  7.657140e+09  8.238494e+09
test_homo_mean_absolute_percentage_error_score             3.0  2.137491e-02  6.575443e-04  2.093408e-02  2.099701e-02  2.105995e-02  2.159532e-02  2.213069e-02
test_lumo_mean_absolute_percentage_error_score             3.0  2.171139e+10  2.793706e+09  1.919132e+10  2.020934e+10  2.122736e+10  2.297142e+10  2.471548e+10
test_gap_mean_absolute_percentage_error_score              3.0  3.284721e-02  2.545276e-03  2.991102e-02  3.205718e-02  3.420334e-02  3.431530e-02  3.442726e-02
test_weighted_mean_absolute_percentage_error_score         3.0  4.488278e-02  1.316691e-02  2.995482e-02  3.990235e-02  4.984989e-02  5.234677e-02  5.484365e-02
test_homo_weighted_mean_absolute_percentage_error_score    3.0  2.104160e-02  6.489350e-04  2.058042e-02  2.067057e-02  2.076073e-02  2.127219e-02  2.178366e-02
test_lumo_weighted_mean_absolute_percentage_error_score    3.0  8.275962e-02  4.122241e-02  3.565016e-02  6.803221e-02  1.004143e-01  1.063143e-01  1.122144e-01
test_gap_weighted_mean_absolute_percentage_error_score     3.0  3.084714e-02  2.169194e-03  2.837468e-02  3.005538e-02  3.173609e-02  3.208337e-02  3.243065e-02
test_mean_absolute_error_score                             3.0  5.973943e-03  2.720526e-04  5.694849e-03  5.841735e-03  5.988621e-03  6.113491e-03  6.238360e-03
test_homo_mean_absolute_error_score                        3.0  5.009825e-03  1.437896e-04  4.901114e-03  4.928306e-03  4.955497e-03  5.064180e-03  5.172863e-03
test_lumo_mean_absolute_error_score                        3.0  5.407914e-03  3.146776e-04  5.044627e-03  5.313968e-03  5.583308e-03  5.589558e-03  5.595807e-03
test_gap_mean_absolute_error_score                         3.0  7.504091e-03  4.314392e-04  7.084422e-03  7.282932e-03  7.481441e-03  7.713925e-03  7.946408e-03
test_root_mean_squared_error_loss                          3.0  8.073015e-03  3.662954e-04  7.675911e-03  7.910684e-03  8.145457e-03  8.271567e-03  8.397677e-03
test_homo_root_mean_squared_error_loss                     3.0  6.545464e-03  1.376767e-04  6.421414e-03  6.471401e-03  6.521387e-03  6.607489e-03  6.693591e-03
test_lumo_root_mean_squared_error_loss                     3.0  7.188009e-03  4.103632e-04  6.715903e-03  7.052412e-03  7.388921e-03  7.424062e-03  7.459203e-03
test_gap_root_mean_squared_error_loss                      3.0  9.923354e-03  5.759793e-04  9.315705e-03  9.654369e-03  9.993033e-03  1.022718e-02  1.046132e-02
[07/26/2024 12:42:22 PM fastprop.cli.train] INFO: 2-sided T-test between validation and testing mse yielded p value of p=0.383>0.05.
[07/26/2024 12:42:23 PM fastprop.cli.base] INFO: If you use fastprop in published work, please cite https://arxiv.org/abs/2404.02058
[07/26/2024 12:42:23 PM fastprop.cli.base] INFO: Total elapsed time: 1:00:36.573428

linear model:

[08/06/2024 02:10:06 PM fastprop.cli.train] INFO: Displaying testing results:
                                                         count          mean           std           min           25%           50%           75%           max
test_mse_scaled_loss                                       3.0  1.262438e-01  1.304001e-02  1.158136e-01  1.189339e-01  1.220542e-01  1.314590e-01  1.408637e-01
test_r2_score                                              3.0  7.482870e-01  4.923380e-02  7.051906e-01  7.214583e-01  7.377261e-01  7.698352e-01  8.019443e-01
test_homo_r2_score                                         3.0  7.028322e-01  1.334355e-02  6.883467e-01  6.969371e-01  7.055274e-01  7.100749e-01  7.146224e-01
test_lumo_r2_score                                         3.0  8.507954e-01  4.754509e-02  8.059677e-01  8.258645e-01  8.457612e-01  8.732092e-01  9.006573e-01
test_gap_r2_score                                          3.0  6.912336e-01  8.837971e-02  6.212581e-01  6.415739e-01  6.618897e-01  7.262213e-01  7.905529e-01
test_mean_absolute_percentage_error_score                  3.0  1.304759e+10  4.542068e+09  8.576868e+09  1.074249e+10  1.290810e+10  1.528295e+10  1.765779e+10
test_homo_mean_absolute_percentage_error_score             3.0  2.968750e-02  1.181363e-03  2.832339e-02  2.934478e-02  3.036617e-02  3.036956e-02  3.037295e-02
test_lumo_mean_absolute_percentage_error_score             3.0  3.914276e+10  1.362620e+10  2.573061e+10  3.222745e+10  3.872430e+10  4.584884e+10  5.297337e+10
test_gap_mean_absolute_percentage_error_score              3.0  5.384798e-02  4.987951e-03  4.884226e-02  5.136298e-02  5.388371e-02  5.635084e-02  5.881796e-02
test_weighted_mean_absolute_percentage_error_score         3.0  7.036779e-02  1.444914e-02  5.368625e-02  6.606294e-02  7.843963e-02  7.870857e-02  7.897750e-02
test_homo_weighted_mean_absolute_percentage_error_score    3.0  2.920091e-02  1.110940e-03  2.791989e-02  2.885137e-02  2.978285e-02  2.984142e-02  2.989998e-02
test_lumo_weighted_mean_absolute_percentage_error_score    3.0  1.317752e-01  4.443777e-02  8.073083e-02  1.167465e-01  1.527622e-01  1.572974e-01  1.618326e-01
test_gap_weighted_mean_absolute_percentage_error_score     3.0  5.012726e-02  4.418265e-03  4.556634e-02  4.799713e-02  5.042793e-02  5.240772e-02  5.438751e-02
test_mean_absolute_error_score                             3.0  9.520948e-03  5.850805e-04  8.872832e-03  9.276338e-03  9.679845e-03  9.845006e-03  1.001017e-02
test_homo_mean_absolute_error_score                        3.0  6.945320e-03  2.567250e-04  6.648930e-03  6.868839e-03  7.088748e-03  7.093514e-03  7.098280e-03
test_lumo_mean_absolute_error_score                        3.0  9.484590e-03  7.894157e-04  8.628236e-03  9.135252e-03  9.642268e-03  9.912767e-03  1.018327e-02
test_gap_mean_absolute_error_score                         3.0  1.213293e-02  7.230207e-04  1.134133e-02  1.182016e-02  1.229899e-02  1.252873e-02  1.275848e-02
test_root_mean_squared_error_loss                          3.0  1.262690e-02  7.497521e-04  1.183138e-02  1.228012e-02  1.272887e-02  1.302466e-02  1.332045e-02
test_homo_root_mean_squared_error_loss                     3.0  8.922886e-03  3.264740e-04  8.546313e-03  8.821158e-03  9.096003e-03  9.111172e-03  9.126341e-03
test_lumo_root_mean_squared_error_loss                     3.0  1.222515e-02  9.112443e-04  1.131449e-02  1.176923e-02  1.222397e-02  1.268047e-02  1.313698e-02
test_gap_root_mean_squared_error_loss                      3.0  1.556471e-02  8.868964e-04  1.462099e-02  1.515656e-02  1.569214e-02  1.603657e-02  1.638099e-02

 -->

### Pgp
First reported in 2011 by Broccatelli and coworkers [@pgp], this dataset has since become a standard benchmark and is included in the Therapeutic Data Commons (TDC) [@tdc] model benchmarking suite.
The dataset maps approximately 1.2k small molecule drugs to a binary label indicating if they inhibit P-glycoprotein (Pgp).
TDC serves this data through a Python package, but due to installation issues the data was retrieved from the original study instead.
The recommended splitting approach is a 70/10/20 scaffold-based split which is done here with 4 replicates.

The model in the original study uses a molecular interaction field but has since been surpassed by other models.
According to TDC the current leader [@pgp_best] on this benchmark has achieved an AUROC of 0.938 $\pm$ 0.002 [^3].
On the same leaderboard Chemprop [@chemprop_theory] achieves 0.886 $\pm$ 0.016 with the inclusion of additional molecular features.
`fastprop` yet again approaches the performance of the leading methods and outperforms Chemprop, here with an AUROC of 0.903 $\pm$ 0.033 and an accuracy of 83.6 $\pm$ 4.6%.
Remarkably, the linear QSPR model outperforms both Chemprop and `fastprop`, approaching the performance of the current leader with an AUROC of 0.917 $\pm$ 0.016 and an accuracy of 83.8 $\pm$ 3.9%.
<!-- 
[07/26/2024 02:57:14 PM fastprop.descriptors] INFO: Descriptor calculation complete, elapsed time: 0:02:10.865505
[07/26/2024 02:57:31 PM fastprop.cli.train] INFO: Displaying validation results:
                                     count      mean       std       min       25%       50%       75%       max
validation_bce_scaled_loss             4.0  0.354919  0.041701  0.297886  0.344525  0.361862  0.372257  0.398064
validation_binary_accuracy_score       4.0  0.866803  0.030942  0.827869  0.852459  0.868852  0.883197  0.901639
validation_binary_f1_score             4.0  0.729799  0.131026  0.618194  0.619358  0.712975  0.823416  0.875052
validation_binary_auroc                4.0  0.846946  0.089123  0.763769  0.773623  0.844451  0.917774  0.935115
validation_binary_average_precision    4.0  0.819140  0.119679  0.704843  0.723363  0.813648  0.909425  0.944420
[07/26/2024 02:57:31 PM fastprop.cli.train] INFO: Displaying testing results:
                               count      mean       std       min       25%       50%       75%       max
test_bce_scaled_loss             4.0  0.417004  0.089345  0.363406  0.369125  0.377071  0.424949  0.550467
test_binary_accuracy_score       4.0  0.836735  0.045816  0.775510  0.815306  0.846939  0.868367  0.877551
test_binary_f1_score             4.0  0.834970  0.052226  0.766493  0.813489  0.841280  0.862762  0.890827
test_binary_auroc                4.0  0.902870  0.032586  0.858263  0.888738  0.913027  0.927159  0.927163
test_binary_average_precision    4.0  0.921767  0.006116  0.912890  0.920745  0.923646  0.924668  0.926886
[07/26/2024 02:57:31 PM fastprop.cli.train] INFO: 2-sided T-test between validation and testing bce yielded p value of p=0.255>0.05.
[07/26/2024 02:57:31 PM fastprop.cli.base] INFO: If you use fastprop in published work, please cite https://arxiv.org/abs/2404.02058
[07/26/2024 02:57:31 PM fastprop.cli.base] INFO: Total elapsed time: 0:02:27.833184

linear model:

[08/06/2024 01:55:24 PM fastprop.cli.train] INFO: Displaying testing results:
                               count      mean       std       min       25%       50%       75%       max
test_bce_scaled_loss             4.0  0.389468  0.052250  0.346798  0.353294  0.374777  0.410952  0.461519
test_binary_accuracy_score       4.0  0.838776  0.038506  0.783673  0.826531  0.853061  0.865306  0.865306
test_binary_f1_score             4.0  0.841359  0.040467  0.784639  0.830083  0.850547  0.861824  0.879703
test_binary_auroc                4.0  0.917419  0.018365  0.889960  0.915951  0.925768  0.927236  0.928180
test_binary_average_precision    4.0  0.925318  0.015881  0.911060  0.913512  0.922362  0.934168  0.945487

 -->

[^3]: See [the TDC Pgp leaderboard](https://tdcommons.ai/benchmark/admet_group/03pgp/).

### ARA
Compiled by Schaduangrat et al. in 2023 [@ara], this dataset maps ~0.8k small molecules to a binary label indicating if the molecule is an Androgen Receptor Antagonist (ARA).
The reference study introduced DeepAR, a highly complex modeling approach, which achieved an accuracy of 0.911 and an AUROC of 0.945.

For this study an 80/10/10 random splitting is repeated four times on the dataset since no analogous split to the reference study can be determined.
Chemprop takes 16 minutes and 55 seconds to run on this dataset and achieves only 0.824 $\pm$ 0.020 accuracy and 0.898 $\pm$ 0.022 AUROC.
`fastprop` takes only 1 minute and 54 seconds (1 minute and 39 seconds for descriptor calculation) and is competitive with the reference study in performance, achieving a 88.2 $\pm$ 3.7% accuracy and 0.935 $\pm$ 0.034 AUROC.
The purely linear QSPR model falls far behind these methods with a 71.8 $\pm$ 6.6% accuracy and 0.824 $\pm$ 0.052 AUROC.
<!--
[07/26/2024 04:08:23 PM fastprop.descriptors] INFO: Descriptor calculation complete, elapsed time: 0:01:39.167050
[07/26/2024 04:08:38 PM fastprop.cli.train] INFO: Displaying validation results:
                                     count      mean       std       min       25%       50%       75%       max
validation_bce_scaled_loss             4.0  0.350839  0.125927  0.231572  0.269866  0.326317  0.407290  0.519149
validation_binary_accuracy_score       4.0  0.872024  0.026397  0.833333  0.869048  0.880952  0.883929  0.892857
validation_binary_f1_score             4.0  0.875015  0.028110  0.833333  0.871124  0.886305  0.890196  0.894118
validation_binary_auroc                4.0  0.932591  0.031731  0.895455  0.911789  0.936174  0.956976  0.962564
validation_binary_average_precision    4.0  0.928978  0.031425  0.888724  0.911996  0.934706  0.951688  0.957778
[07/26/2024 04:08:38 PM fastprop.cli.train] INFO: Displaying testing results:
                               count      mean       std       min       25%       50%       75%       max
test_bce_scaled_loss             4.0  0.334037  0.089974  0.210850  0.305414  0.350277  0.378899  0.424744
test_binary_accuracy_score       4.0  0.882353  0.037203  0.847059  0.855882  0.876471  0.902941  0.929412
test_binary_f1_score             4.0  0.875653  0.049124  0.821918  0.842979  0.874438  0.907112  0.931818
test_binary_auroc                4.0  0.935258  0.034489  0.889714  0.924770  0.938963  0.949452  0.973392
test_binary_average_precision    4.0  0.908971  0.065320  0.821435  0.881043  0.920907  0.948834  0.972634
[07/26/2024 04:08:38 PM fastprop.cli.train] INFO: 2-sided T-test between validation and testing bce yielded p value of p=0.835>0.05.
[07/26/2024 04:08:38 PM fastprop.cli.base] INFO: If you use fastprop in published work, please cite https://arxiv.org/abs/2404.02058
[07/26/2024 04:08:38 PM fastprop.cli.base] INFO: Total elapsed time: 0:01:54.651789

linear model:

[08/06/2024 01:48:06 PM fastprop.cli.train] INFO: Displaying testing results:
                               count      mean       std       min       25%       50%       75%       max
test_bce_scaled_loss             4.0  0.541295  0.042253  0.478426  0.537446  0.558557  0.562406  0.569639
test_binary_accuracy_score       4.0  0.717647  0.065854  0.623529  0.702941  0.735294  0.750000  0.776471
test_binary_f1_score             4.0  0.704753  0.064048  0.627907  0.664019  0.715806  0.756540  0.759494
test_binary_auroc                4.0  0.823776  0.051937  0.770903  0.795297  0.815315  0.843794  0.893570
test_binary_average_precision    4.0  0.816956  0.074735  0.729266  0.784316  0.813911  0.846552  0.910737

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

### Flash
First assembled and fitted to by Saldana and coauthors [@flash] the dataset (Flash) includes around 0.6k entries, primarily alkanes and some oxygen-containing compounds, and their literature-reported flash point.
The reference study reports the performance on only one repetition, but manually confirms that the distribution of points in the three splits follows the parent dataset.
The split itself was a 70/20/10 random split, which is repeated four times for this study.

Using a complex multi-model ensembling method, the reference study achieved an RMSE of 13.2, an MAE of 8.4, and an MAPE of 2.5%.
`fastprop` matches this performance, achieving 13.0 $\pm$ 2.0 RMSE, 9.0 $\pm$ 0.5 MAE, and 2.7% $\pm$ 0.1% MAPE.
Chemprop, however, struggles to match the accuracy of either method - it manages an RMSE of 21.2 $\pm$ 2.2, an MAE of 13.8 $\pm$ 2.1, and a MAPE of 3.99 $\pm$ 0.36%.
This is worse than the performance of the linear QSPR model, with an RMSE of 16.1 $\pm$ 4.0, an MAE of 11.3 $\pm$ 2.9, and an MAPE of 3.36 $\pm$ 0.77%.

`fastprop` dramatically outperforms both methods in terms of training time.
The reference model required significant manual intervention to create a model ensemble, so no single training time can be fairly identified.
`fastprop` arrived at the indicated performance without any manual intervention in only 30 seconds, 13 of which were spent calculating descriptors.
Chemprop, in addition to not reaching the same level of accuracy, took 5 minutes and 44 seconds to do so - more than ten times the execution time of `fastprop`.
<!-- fastprop:
[07/26/2024 04:24:06 PM fastprop.descriptors] INFO: Descriptor calculation complete, elapsed time: 0:00:12.031769
[07/26/2024 04:25:36 PM fastprop.cli.train] INFO: Displaying validation results:
                                                          count       mean       std        min        25%        50%        75%        max
validation_mse_scaled_loss                                  4.0   0.079637  0.025973   0.056849   0.066836   0.072392   0.085192   0.116915
validation_r2_score                                         4.0   0.922342  0.022873   0.895596   0.909473   0.922302   0.935171   0.949168
validation_mean_absolute_percentage_error_score             4.0   0.027319  0.002744   0.024038   0.025685   0.027497   0.029131   0.030246
validation_weighted_mean_absolute_percentage_error_score    4.0   0.028452  0.002822   0.024768   0.026982   0.029144   0.030615   0.030754
validation_mean_absolute_error_score                        4.0   9.285408  1.079271   7.856185   8.748839   9.563105  10.099674  10.159239
validation_root_mean_squared_error_loss                     4.0  14.775583  2.380761  12.242039  13.859992  14.433841  15.349432  17.992609
[07/26/2024 04:25:36 PM fastprop.cli.train] INFO: Displaying testing results:
                                                    count       mean       std        min        25%        50%        75%        max
test_mse_scaled_loss                                  4.0   0.057471  0.016236   0.039985   0.051368   0.055309   0.061413   0.079281
test_r2_score                                         4.0   0.941629  0.022670   0.907924   0.939550   0.950777   0.952857   0.957040
test_mean_absolute_percentage_error_score             4.0   0.026929  0.001795   0.025454   0.025642   0.026450   0.027737   0.029362
test_weighted_mean_absolute_percentage_error_score    4.0   0.027897  0.001759   0.026271   0.026434   0.027877   0.029340   0.029564
test_mean_absolute_error_score                        4.0   9.040832  0.533608   8.397331   8.708179   9.114861   9.447514   9.536273
test_root_mean_squared_error_loss                     4.0  13.035019  1.983146  11.030409  12.091901  12.675693  13.618810  15.758280
[07/26/2024 04:24:26 PM fastprop.cli.base] INFO: If you use fastprop in published work, please cite https://arxiv.org/abs/2404.02058
[07/26/2024 04:24:26 PM fastprop.cli.base] INFO: Total elapsed time: 0:00:32.380891

linear model:
[08/06/2024 01:44:19 PM fastprop.cli.train] INFO: Displaying testing results:
                                                    count       mean       std        min        25%        50%        75%        max
test_mse_scaled_loss                                  4.0   0.090131  0.038312   0.051589   0.060133   0.091287   0.121285   0.126363
test_r2_score                                         4.0   0.911936  0.038512   0.871687   0.882673   0.914730   0.943993   0.946599
test_mean_absolute_percentage_error_score             4.0   0.033647  0.007672   0.027429   0.028669   0.031345   0.036322   0.044468
test_weighted_mean_absolute_percentage_error_score    4.0   0.034863  0.008066   0.027648   0.029142   0.033169   0.038891   0.045467
test_mean_absolute_error_score                        4.0  11.332869  2.913781   8.746270   9.327742  10.673150  12.678277  15.238905
test_root_mean_squared_error_loss                     4.0  16.066076  3.990219  11.862206  13.073341  16.270329  19.263064  19.861439

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
The original study did not report overall performance metrics, so they have been re-calculated for this study using the predictions made by the reference model as provided on GitHub [^4].
For comparison `fastprop` and Chemprop use a more typical 60/20/20 random split and 8 repetitions.
Results are summarized in Table \ref{ysi_results_table}.

Table: YSI results. \label{ysi_results_table}

+------------+----------------+----------------+-----------------+
| Model      | MAE            | RMSE           | WMAPE           |
+============+================+================+=================+
| Reference  | 22.3           | 50             | 14              |
+------------+----------------+----------------+-----------------+
| `fastprop` | 25.0 $\pm$ 5.2 | 52 $\pm$ 20    | 13.6 $\pm$ 1.3  |
+------------+----------------+----------------+-----------------+
| Chemprop   | 28.9 $\pm$ 6.5 | 63 $\pm$ 14    | ~               |
+------------+----------------+----------------+-----------------+
| Linear     | 82 $\pm$ 39    | 180 $\pm$ 120  | 0.47 $\pm$ 0.23 |
+------------+----------------+----------------+-----------------+

`fastprop` again outperforms Chemprop, in this case approaching the overly-optimistic performance of the reference model.
Taking into account that reference model has been trained on a significantly larger amount of data, this performance is admirable.
Also notable is the difference in training times.
Chemprop takes 7 minutes and 2 seconds while `fastprop` completes in only 42 seconds, again a factor of ten faster.
The linear QSPR model fails entirely, performing dramatically worse than all other models.
<!-- 
[07/26/2024 04:28:28 PM fastprop.descriptors] INFO: Descriptor calculation complete, elapsed time: 0:00:08.613929
[07/26/2024 04:29:01 PM fastprop.cli.train] INFO: Displaying validation results:
                                                          count       mean       std        min        25%        50%        75%        max
validation_mse_scaled_loss                                  8.0   0.023259  0.013221   0.011679   0.016048   0.020006   0.023711   0.051787
validation_r2_score                                         8.0   0.977086  0.006396   0.963429   0.975749   0.979199   0.981055   0.982587
validation_mean_absolute_percentage_error_score             8.0   0.300894  0.087392   0.173184   0.256760   0.301174   0.351737   0.416199
validation_weighted_mean_absolute_percentage_error_score    8.0   0.114728  0.010937   0.093921   0.108248   0.118015   0.123835   0.124697
validation_mean_absolute_error_score                        8.0  21.565024  3.674680  15.578521  19.916584  22.328402  23.966396  25.871998
validation_root_mean_squared_error_loss                     8.0  37.455264  5.757264  27.686756  34.651184  38.412334  41.762099  43.979324
[07/26/2024 04:29:01 PM fastprop.cli.train] INFO: Displaying testing results:
                                                    count       mean        std        min        25%        50%        75%        max
test_mse_scaled_loss                                  8.0   0.062661   0.084937   0.015039   0.023240   0.035300   0.047596   0.270054
test_r2_score                                         8.0   0.952776   0.029499   0.884336   0.950312   0.962499   0.969565   0.974228
test_mean_absolute_percentage_error_score             8.0   0.273237   0.062966   0.179620   0.225779   0.286302   0.315717   0.360405
test_weighted_mean_absolute_percentage_error_score    8.0   0.135877   0.013146   0.118940   0.127922   0.136474   0.143343   0.158092
test_mean_absolute_error_score                        8.0  25.012698   5.164974  18.945625  22.651376  23.246510  26.090505  36.137867
test_root_mean_squared_error_loss                     8.0  51.897439  20.464189  31.176691  41.727784  46.864605  53.754726  98.884499
[07/26/2024 04:29:01 PM fastprop.cli.train] INFO: 2-sided T-test between validation and testing mse yielded p value of p=0.216>0.05.
[07/26/2024 04:29:01 PM fastprop.cli.base] INFO: If you use fastprop in published work, please cite https://arxiv.org/abs/2404.02058
[07/26/2024 04:29:01 PM fastprop.cli.base] INFO: Total elapsed time: 0:00:41.651720

linear model:
[08/06/2024 01:30:55 PM fastprop.cli.train] INFO: Displaying validation results:
                                                          count        mean         std        min        25%        50%         75%         max
validation_mse_scaled_loss                                  8.0    1.782322    2.459989   0.025382   0.046804   0.100248    3.904669    6.047585
validation_r2_score                                         8.0   -2.554322    7.034404 -19.411644  -2.486760   0.906504    0.953913    0.964348
validation_mean_absolute_percentage_error_score             8.0    1.607565    1.332180   0.481278   0.833980   1.062793    2.017346    4.424526
validation_weighted_mean_absolute_percentage_error_score    8.0    0.544973    0.544294   0.187772   0.219358   0.262427    0.697047    1.762439
validation_mean_absolute_error_score                        8.0   89.298786   71.005340  31.348509  41.932364  47.666235  137.419773  226.654160
validation_root_mean_squared_error_loss                     8.0  202.332909  199.009173  43.081791  57.383880  79.881218  371.361580  543.385315
[08/06/2024 01:30:55 PM fastprop.cli.train] INFO: Displaying testing results:
                                                    count        mean         std        min        25%         50%         75%         max
test_mse_scaled_loss                                  8.0    1.110561    1.699195   0.040328   0.101340    0.348682    1.204710    4.954531
test_r2_score                                         8.0   -0.073488    1.694859  -4.114374   0.001652    0.566954    0.828363    0.923891
test_mean_absolute_percentage_error_score             8.0    1.392105    0.877828   0.563580   0.792891    0.938259    1.953995    2.865982
test_weighted_mean_absolute_percentage_error_score    8.0    0.469963    0.229644   0.245020   0.317786    0.466987    0.508767    0.966923
test_mean_absolute_error_score                        8.0   81.887387   38.735396  39.349533  59.377875   78.500698   89.702042  161.834244
test_root_mean_squared_error_loss                     8.0  175.053466  119.504669  56.758713  85.489279  150.429375  223.476109  409.616058
-->

[^4]: Predictions are available at this [permalink](https://github.com/pstjohn/ysi-fragment-prediction/blob/bdf8b16a792a69c3e3e63e64fba6f1d190746abe/data/ysi_predictions.csv) to the CSV file on GitHub.

### PAH
Originally compiled by Arockiaraj et al. [@pah] the Polycyclic Aromatic Hydrocarbons (PAH) dataset contains water/octanol partition coefficients (logP) for 55 polycyclic aromatic hydrocarbons ranging in size from napthalene to circumcoronene.
This size of this benchmark is an ideal case study for the application of `fastprop`.
Using expert insight the reference study designed a novel set of molecular descriptors that show a strong correlation to logP, with correlation coefficients ranging from 0.96 to 0.99 among the various new descriptors.

For comparison, `fastprop` and Chemprop are trained using 8 repetitions of a typical 80/10/10 random split - only **44** molecules in the training data.
`fastprop` matches the performance of the bespoke descriptors with a correlation coefficient of 0.972 $\pm$ 0.025.
This corresponds to an MAE of 0.19 $\pm$ 0.10 and an MAPE of 2.5 $\pm$ 1.5%.
Chemprop effectively fails on this dataset, achieving a correlation coefficient of only 0.59 $\pm$ 0.24, an MAE of 1.04 $\pm$ 0.33 (one anti-correlated outlier replicate removed).
This is worse even than the purely linear QSPR model, which manages a correlation coefficient of 0.78 $\pm$ 0.22, an MAE of 0.59 $\pm$ 0.22, and an RMSE of 0.75 $\pm$ 0.32.
Despite the large parameter size of the `fastprop` model relative to the training data, it readily outperforms Chemprop in the small-data limit.

For this unique dataset, execution time trends are inverted.
`fastprop` takes 1 minute and 43 seconds, of which 1 minute and 31 seconds were spent calculating descriptors for these unusually large molecules.
Chemprop completes in 1 minute and 16 seconds, faster overall but much slower compared with the training time of `fastprop` without descriptor calculation.
<!--
[07/26/2024 04:37:00 PM fastprop.descriptors] INFO: Descriptor calculation complete, elapsed time: 0:01:31.736380
[07/26/2024 04:37:11 PM fastprop.cli.train] INFO: Displaying validation results:
                                                          count      mean       std       min       25%       50%       75%       max
validation_mse_scaled_loss                                  8.0  0.015993  0.014160  0.003091  0.005246  0.010797  0.023142  0.043774
validation_r2_score                                         8.0  0.931177  0.132627  0.604734  0.958733  0.983718  0.985874  0.990619
validation_mean_absolute_percentage_error_score             8.0  0.029330  0.013786  0.010083  0.019395  0.029097  0.042694  0.045405
validation_weighted_mean_absolute_percentage_error_score    8.0  0.027937  0.012704  0.010349  0.018477  0.028200  0.039156  0.043552
validation_mean_absolute_error_score                        8.0  0.215183  0.092659  0.093390  0.142778  0.202674  0.301386  0.336168
validation_root_mean_squared_error_loss                     8.0  0.248758  0.106266  0.107539  0.166125  0.234507  0.334529  0.394562
[07/26/2024 04:37:11 PM fastprop.cli.train] INFO: Displaying testing results:
                                                    count      mean       std       min       25%       50%       75%       max
test_mse_scaled_loss                                  8.0  0.013877  0.009920  0.000536  0.007033  0.014443  0.017946  0.032355
test_r2_score                                         8.0  0.971621  0.025000  0.939595  0.944988  0.979568  0.990349  0.999278
test_mean_absolute_percentage_error_score             8.0  0.025644  0.014867  0.006885  0.013840  0.025514  0.034095  0.051297
test_weighted_mean_absolute_percentage_error_score    8.0  0.024297  0.014311  0.005466  0.013144  0.024334  0.031072  0.048998
test_mean_absolute_error_score                        8.0  0.190078  0.101137  0.041588  0.112903  0.198249  0.234031  0.354014
test_root_mean_squared_error_loss                     8.0  0.236781  0.110437  0.051430  0.178263  0.252217  0.297688  0.401531
[07/26/2024 04:37:11 PM fastprop.cli.train] INFO: 2-sided T-test between validation and testing mse yielded p value of p=0.734>0.05.
[07/26/2024 04:37:11 PM fastprop.cli.base] INFO: If you use fastprop in published work, please cite https://arxiv.org/abs/2404.02058
[07/26/2024 04:37:11 PM fastprop.cli.base] INFO: Total elapsed time: 0:01:43.174466

linear model:

[08/06/2024 01:40:53 PM fastprop.cli.train] INFO: Displaying testing results:
                                                    count      mean       std       min       25%       50%       75%       max
test_mse_scaled_loss                                  8.0  0.155622  0.157510  0.032892  0.057053  0.073111  0.213564  0.469946
test_r2_score                                         8.0  0.781259  0.216684  0.293357  0.742562  0.878880  0.906789  0.942883
test_mean_absolute_percentage_error_score             8.0  0.081609  0.048380  0.051792  0.055160  0.057795  0.079304  0.182489
test_weighted_mean_absolute_percentage_error_score    8.0  0.073567  0.028178  0.048521  0.056440  0.061872  0.082388  0.123405
test_mean_absolute_error_score                        8.0  0.589127  0.216229  0.369163  0.451514  0.486528  0.753469  0.909290
test_root_mean_squared_error_loss                     8.0  0.747754  0.324937  0.421131  0.531061  0.594271  1.011191  1.292799

chemprop

real    1m15.752s
user    1m6.715s
sys     0m30.473s
-->

# Limitations and Future Work
## Negative Results
### Delta Learning with Fubrain
First described by Esaki and coauthors, the Fraction of Unbound Drug in the Brain (Fubrain) dataset is a collection of about 0.3k small molecule drugs and their corresponding experimentally measured unbound fraction in the brain, a critical metric for drug development [@fubrain].
This specific target in combination with the small dataset size makes this benchmark highly relevant for typical QSPR studies, particular via delta learning.
DeepDelta [@deepdelta] performed a 90/0/10 cross-validation study of the Fubrain dataset in which the training and testing molecules were intra-combined to generate all possible pairs and then the differences in the property [^5] were predicted, rather than the absolute values, increasing the amount of training data by a factor of 300.

DeepDelta reported an RMSE of 0.830 $\pm$ 0.023 at predicting differences, whereas a typical Chemprop model trained to directly predict property values was only able to reach an accuracy of 0.965 $\pm$ 0.019 when evaluated on its capacity to predict property differences.
`fastprop` is able to outperform Chemprop, though not DeepDelta, achieving an RMSE of 0.930 $\pm$ 0.029 when using the same splitting procedure above.
It is evident that delta learning is still a powerful technique for regressing small datasets.

For completeness, the performance of Chemprop and `fastprop` on Fubrain are also compared to the original study.
The study that first generated this dataset used `mordred` descriptors but as is convention they strictly applied linear modeling methods.
Using both cross validation and and external test sets, they had an effective training/validation/testing split of 0.64/0.07/0.28 which will be repeated 4 times here for comparison.
All told, their model achieved an RMSE of 0.53 averaged across all testing data.
In only 39 seconds, of which 31 are spent calculating descriptors, `fastprop` far exceeds the reference model with an RMSE of 0.207 $\pm$ 0.024.
This also surpasses Chemprop, itself outperforming the reference model with an RMSE of 0.223 $\pm$ 0.036.

<!--
[08/06/2024 01:24:26 PM fastprop.cli.train] INFO: Displaying validation results:
                                                          count      mean       std       min       25%       50%       75%       max
validation_mse_scaled_loss                                  4.0  0.445959  0.166927  0.199223  0.416297  0.515219  0.544881  0.554174
validation_r2_score                                         4.0  0.595322  0.154344  0.476649  0.511786  0.541765  0.625301  0.821108
validation_mean_absolute_percentage_error_score             4.0  5.567747  2.504782  3.087414  4.139117  5.113917  6.542548  8.955740
validation_weighted_mean_absolute_percentage_error_score    4.0  0.635260  0.175969  0.440934  0.524161  0.628305  0.739404  0.843495
validation_mean_absolute_error_score                        4.0  0.122766  0.027147  0.083292  0.116007  0.132480  0.139240  0.142813
validation_root_mean_squared_error_loss                     4.0  0.171753  0.041262  0.112220  0.159812  0.187137  0.199077  0.200518
[08/06/2024 01:24:26 PM fastprop.cli.train] INFO: Displaying testing results:
                                                    count      mean       std       min       25%       50%       75%        max
test_mse_scaled_loss                                  4.0  0.646342  0.196778  0.439533  0.498014  0.658189  0.806517   0.829457
test_r2_score                                         4.0  0.357382  0.049868  0.311554  0.328430  0.345598  0.374550   0.426779
test_mean_absolute_percentage_error_score             4.0  8.201145  2.286469  5.670279  7.327289  7.954490  8.828346  11.225323
test_weighted_mean_absolute_percentage_error_score    4.0  0.817863  0.129580  0.707232  0.710772  0.798212  0.905304   0.967796
test_mean_absolute_error_score                        4.0  0.147366  0.018656  0.127301  0.140443  0.144859  0.151782   0.172445
test_root_mean_squared_error_loss                     4.0  0.206569  0.024271  0.178577  0.190217  0.209359  0.225711   0.228979

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

[^5]: Although the original Fubrain study reported untransformed fractions, the DeepDelta authors confirmed [via GitHub](https://github.com/RekerLab/DeepDelta/issues/2#issuecomment-1917936697) that DeepDelta was trained on log base-10 transformed fraction values, which is replicated here.

### `fastprop` Fails on QuantumScents
Compiled by Burns and Rogers [@quantumscents], this dataset contains approximately 3.5k SMILES and 3D structures for a collection of molecules labeled with their scents.
Each molecule can have any number of reported scents from a possible 113 different labels, making this benchmark a a Quantitative Structure-Odor Relationship.
Due to the highly sparse nature of the scent labels a unique sampling algorithm (Szymanski sampling [@szymanski]) was used in the reference study and the exact splits are replicated here for a fair comparison.

In the reference study, Chemprop achieved an AUROC of 0.85 with modest hyperparameter optimization and an improved AUROC of 0.88 by incorporating the atomic descriptors calculated as part of QuantumScents.
`fastprop` is incapable of incorporating atomic features, so they are not included.
Using only the 2D structural information, `fastprop` falls far behind the reference study with an AUROC of only 0.651 $\pm$ 0.005.
Even when using the high-quality 3D structures and calculating additional descriptors (demonstrated in the GitHub repository), the performance does not improve.

The exact reason for this failure is unknown.
Possible reasons include that the descriptors in `mordred` are simply not correlated with this target, and thus the model struggles to make predictions.
This is a fundamental drawback of this fixed representation method - whereas a LR could adapt to this unique target, `fastprop` fails.

<!-- 
Displaying testing results:
                                    count      mean       std       min       25%       50%       75%       max
test_bce_scaled_loss                 3.0  0.077605  0.000635  0.077183  0.077240  0.077297  0.077816  0.078335
test_multilabel_auroc                3.0  0.650884  0.005092  0.647141  0.647985  0.648828  0.652755  0.656682
test_multilabel_average_precision    3.0  0.529041  0.006757  0.523960  0.525206  0.526453  0.531581  0.536709
test_multilabel_f1_score             3.0  0.439406  0.022444  0.414642  0.429907  0.445171  0.451788  0.458405
with the addition of 3d descriptors
Displaying testing results:
                                    count      mean       std       min       25%       50%       75%       max
test_bce_scaled_loss                 3.0  0.077164  0.000517  0.076753  0.076873  0.076994  0.077369  0.077745
test_multilabel_auroc                3.0  0.653188  0.005025  0.649045  0.650393  0.651742  0.655260  0.658778
test_multilabel_average_precision    3.0  0.533800  0.004271  0.529235  0.531851  0.534468  0.536083  0.537698
test_multilabel_f1_score             3.0  0.445995  0.017368  0.425965  0.440550  0.455136  0.456010  0.456884
 -->

## Execution Time
`fastprop` is consistently faster to train than Chemprop when using a GPU, helping exploit the 'time value' of data.
Note that due to the large size of the FNN in `fastprop` it can be slower than small Chemprop models when training on a CPU since Chemprop uses a much smaller FNN and associated components.

There is a clear performance improvement to be had by reducing the number of descriptors to a subset of only the most important.
Future work can address this possibility to decrease time requirements for both training by reducing network size and inference by decreasing the number of descriptors to be calculated for new molecules.
This has _not_ been done in this study for two reasons: (1) to emphasize the capacity of the DL framework to effectively perform feature selection on its own via the training process, de-emphasizing unimportant descriptors; (2) as discussed above, training time is small compared to dataset generation time, or even compared to to the time it takes to compute the descriptors using `mordred`.

## Coverage of Descriptors
`fastprop` is fundamentally limited by the types of chemicals which can be uniquely described by the `mordred` package.
Domain-specific additions which are not just derived from the descriptors already implemented will be required to expand its application to new domains.

For example, in its current state `mordred` does not include any connectivity based-descriptors that reflect the presence or absence of stereocenters.
While some of the 3D descriptors it implements could implicitly reflect sterochemistry, more explicit descriptors like the Stereo Signature Molecular Descriptor [@stereo_signature] may prove helpful in the future if re-implemented in `mordred`.

## Interpretability
Though not discussed here for the sake of length, `fastprop` already contains the functionality to perform feature importance studies on trained models.
By using SHAP values [@shap] to assign a scalar 'importance' to each of the input features, users can determine which of the `mordred` descriptors has the largest impact on model predictions.
The utility of these values can be explored in greater detail on a case-by-case basis.

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