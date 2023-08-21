# Uncertainty in Deep Learning on HPC Systems

This repository houses our research code, experimental setups, and findings concerning the quantification of uncertainty in deep learning, particularly when leveraging High-Performance Computing (HPC) systems.

## Overview
We explore the quantification of uncertainty using both vision transformers (ViTs) and convolutional neural networks (CNNs) through two primary techniques: deep ensembles and evidential learning. Our experiments span multiple HPC systems, including the Polaris supercomputer, Cerebras CS-2, and SambaNova DataScale at Argonne National Laboratory (ANL).

## Table of Contents
- [Methods](#methods)
- [Systems](#systems)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
- [Recent Updates & Ongoing Work](#recent-updates--ongoing-work)
- [Contributors](#contributors)
- [Publications](#publications)
- [License](#license)

## Methods
- **Deep Ensembles**: Harnessing the power of multiple model configurations to attain better predictive performance.
- **Evidential Learning**: Focusing on understanding the data's inherent noise and uncertainty.
  
[Detailed breakdown of methods and related experiments](link_to_detailed_method_folder_or_file)

## Systems
- **Polaris Supercomputer**: Known for its computational prowess, Polaris was instrumental in our deep ensemble and evidential learning experiments using CNNs.
- **Cerebras CS-2**: Leveraged mainly for CNNs, its unique architecture provides distinct opportunities and challenges.
- **SambaNova DataScale**: A versatile system where we delved into both deep ensemble techniques and distributed single model training for both ViTs and CNNs.

[Further details on system-specific experiments](link_to_systems_folder_or_file)

## Datasets
- **ImageNet-1K**: A diverse dataset instrumental for our experiments involving Vision Transformers.
- **MNIST**: Classic dataset used predominantly with our CNN-based experiments.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/epautsch/UncertaintyANL.git
   ```

## Recent Updates & Ongoing Work
- **08-12-2023**: Submitted *Evaluation of Novel AI Architectures for Uncertainty Estimation* to CARLA 2023: Workshop on Advanced Computing Trends
- **08-12-2023**: Submitted *Optimized Uncertainty Estimation for Vision Transformers: Enhancing Adversarial Robustness and Performance Using Selective Classification* to SC23 Workshop: EPSOUQ-HPC
- **08-11-2023**: Submitted *Optimizing Uncertainty Quantification of Vision Transformers in Deep Learning on Novel AI Architectures* to SC23 Research Posters
- **08-11-2023**: Submitted *Using MPI for Distributed Hyper-Parameter Optimization and Uncertainty Evaluation* to SC23 Workshop: EduHPC Peachy Assignment

## Contributors
- [John Li](https://www.google.com)
- [Maria Pantoja](https://www.google.com)
- [Erik Pautsch](https://www.google.com)
- [Silvio Rizzi](https://www.google.com)
- [George Thiruvathukal](https://www.google.com)

## Publications
*Coming soon...*

## License
*Coming soon...*

