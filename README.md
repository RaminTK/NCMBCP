# NCMBCP Codes and Dataset

In the GitHub repository associated with the paper titled "A non-clustered approach to platelet collection routing problem" by Ramin Talebi Khameneh, Milad Elyasi, O. Örsan Özener, and Ali Ekici, we provide a comprehensive suite of resources aimed at addressing the complex issue of optimizing platelet collection from blood donation centers. The repository includes Python code scripts and datasets that are instrumental in implementing the innovative solutions proposed in the study. The datasets consist of the locations of blood donation centers and the frequency of donations at each center, which are crucial for understanding and solving the blood collection problem (BCP) with a focus on platelet collection.

The uniqueness of our approach lies in the handling of platelet perishability and the strategic routing without clustering blood donation sites, thereby enhancing the efficiency and reach of blood collection operations. To facilitate the analysis, the geographical location data of the donation centers have been meticulously converted into a distance matrix, representing the distances between each pair of centers. This transformation simplifies the complex problem of routing and scheduling for platelet collection, making the data more accessible for computational algorithms.

Our Python code scripts embody the implementation of two advanced heuristic methods: a hybrid genetic algorithm (HGA) and an invasive weed optimization (IWO) algorithm. These scripts are designed to tackle the non-clustered maximum blood collection problem (NCMBCP), a variant of the BCP that takes into account the processing time limit (PTL) of blood and the varied donation patterns of donors, without assigning each blood collection vehicle (BCV) to a predetermined set of blood donation sites (BDSs). The proposed algorithms have demonstrated significant improvements over existing solutions, offering more efficient and feasible routing decisions for the collection of platelets, which are critical for various medical applications. The repository serves as a valuable resource for researchers and practitioners in the field, providing the necessary tools to replicate our findings and further explore the potential of non-clustered approaches in enhancing platelet collection operations.




# Dataset and Code Overview

This repository contains datasets and Python scripts related to the research paper titled "A non-clustered approach to platelet collection routing problem" by Ramin Talebi Khameneh, Milad Elyasi, O. Örsan Özener, and Ali Ekici. The resources provided here are aimed at optimizing platelet collection from blood donation centers by employing a non-clustered routing approach.

## Key Features of the Dataset

- **Location Data of Blood Donation Centers**: Includes the geographical coordinates of blood donation centers, crucial for routing optimization.

- **Donation Frequencies**: Provides data on the frequency of donations at each center, indicating the potential volume of blood available for platelet extraction.

- **Distance Matrix**: The dataset is transformed into a distance matrix format, where each cell represents the distance between two centers, simplifying routing analysis.

- **Multiple Instances**: Contains seven distinct datasets for different scenarios, allowing for a comprehensive evaluation of the proposed solutions.

- **Platelet Time Limits**: Analyses are conducted under three different platelet time limits (300, 500, and 700 minutes) to accommodate the perishable nature of platelets.

## Python Scripts

The repository includes Python scripts that implement the heuristic algorithms proposed in the study:

- **Hybrid Genetic Algorithm (HGA)**
- **Invasive Weed Optimization (IWO) Algorithm**
- **Flower Pollination Algorithm (FPA)**

These scripts are pivotal in solving the non-clustered maximum blood collection problem (NCMBCP), enhancing the efficiency and effectiveness of platelet collection routes.

## Non-Clustered Routing Approach

This innovative approach avoids clustering blood donation centers for routing, thereby increasing the flexibility and potential reach of blood collection operations.

## Algorithm Performance

The dataset and scripts facilitate the assessment of the proposed algorithms against existing solutions, demonstrating their superiority in improving platelet collection logistics.

---

For more details on the algorithms, methodology, and results, please refer to the full paper linked [here](https://doi.org/10.1016/j.cor.2023.106366).
