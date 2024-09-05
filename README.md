# Laplacian Segmentation Networks Improve Epistemic Uncertainty Quantification

Kilian Zepf*, Selma Wanna*, Marco Miani, Juston Moore, Jes Frellsen, Søren Hauberg, Frederik Warburg, Aasa Feragen (MICCAI 2024)


$^*$ denotes equal contribution

[[Paper on Arxiv]](https://arxiv.org/abs/2303.13123#)


This repository contains an implementation of the proposed model class as well as the benchmarks presented in the paper. The code is based on PyTorch. 

$\qquad$

<p align="center">
<img src="img/model_overview_grey.png"  width="600"  >
</p>

Figure: Model overview Laplacian Segmentation Network - uncertainty measures are calculated by approximating expectations by Monte Carlo-sampling mean networks from the Laplace approximation $q(θ_*)$ and predicting the respective logit distributions $p(η|x,θ)$ for $x$.
