<h1 align="center">fastprop</h1> 
<h3 align="center">Fast Molecular Property Prediction with mordredcommunity</h3>

<p align="center">  
  <img alt="fastproplogo" height="400" src="https://github.com/JacksonBurns/fastprop/blob/main/fastprop_logo.png">
</p> 
<p align="center">
  <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/JacksonBurns/fastprop?style=social">
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/fastprop">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/fastprop">
  <img alt="PyPI - License" src="https://img.shields.io/github/license/JacksonBurns/fastprop">
</p>

 - featurization: which to include, etc.
(1) just generate all (2) just generate some (3) generate all but only for subset (configurable size), do pre-processing, then generate rest from subset
 - pre-processing pipeline: no optional to drop missing, optionally include scaling, dropping of zero variance, droppign of colinear, keep the names true or false (?)
 - training: number of interaction layers, size of representations, learning rate, batch size, FNN configs 
 - prediction
