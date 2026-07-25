# Usage

This repository provides implementations of several spectral risk minimization (SRM) solvers, including *Prospect*, *LSVRG*, *SGD*, and our proposed method, *PSG-SRM*. For the *Sorel* solver, please refer to the **sorel** directory.

The `run.py` script performs the **fairness evaluation** and **out-of-distribution classification** experiments described in the manuscript.

The scripts in the **script** directory summarize the corresponding experimental results reported in Tables 3 and 4 of the main text. To reproduce the reported results, simply execute these scripts directly.

Please note that the repository is currently intended for review purposes only and has not yet been fully refined for user-friendly use.

# Acknowledgment

This repository is built upon the [Prospect](https://github.com/ronakdm/prospect) implementation provided by Mehta et al.

Mehta, R., Roulet, V., Pillutla, K., & Harchaoui, Z. (2024). Distributionally robust optimization with bias and variance reduction. In *Proceedings of the International Conference on Learning Representations (ICLR 2024)*.
