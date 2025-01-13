# Network Flow Analysis and Threat Diffusion Simulation

This project demonstrates an end-to-end pipeline for analyzing network flow data, building a graph-based model, and simulating threat diffusion using machine learning techniques. It combines data preprocessing, feature engineering, graph construction, and diffusion modeling to predict and visualize the impact of network threats.

![Graph Illustration](images/image.png)

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Graph Construction](#graph-construction)
5. [Threat Diffusion Simulation](#threat-diffusion-simulation)
6. [Training a Diffusion Model](#training-a-diffusion-model)
7. [Testing IP Impact](#testing-ip-impact)
8. [Visualizing Results](#visualizing-results)
9. [How to Use](#how-to-use)
10. [Requirements](#requirements)

---

## Introduction

This project processes network flow data to simulate and predict how threats propagate through a network. Using a RandomForestClassifier, the model predicts the likelihood of infection for each node based on graph-based features.

Key functionalities:
- Preprocessing raw network flow data.
- Constructing a graph to represent communication between IPs.
- Simulating threat diffusion using probabilistic models.
- Training a machine learning model to predict infection spread.
- Visualizing the impact of specific IPs on the network.

---

## Dataset

The dataset contains network flow information, including source and destination IPs, ports, protocols, and packet counts. It is read from a CSV file and limited to the first 5000 rows for demonstration purposes.

Example dataset structure:
- `src_ip`, `dst_ip`: Source and destination IP addresses.
- `pktTotalCount`: Total packet count between source and destination.
- Additional columns representing flow metadata.

---

## Preprocessing

1. **Imputation**:
   - Numerical columns: Imputed using the median.
   - Categorical columns: Imputed using the most frequent value.

2. **Scaling**:
   - Numerical features are scaled using Min-Max normalization.

3. **Outlier Removal**:
   - Outliers are removed using the Interquartile Range (IQR) method.

4. **Encoding**:
   - Categorical features are label-encoded for machine learning compatibility.

---

## Graph Construction

A graph is built using `networkx`, where:
- Nodes represent IP addresses.
- Edges represent communication between IPs.
- Edge weights correspond to the total packet count (`pktTotalCount`).

Visualization of the graph includes a subset of nodes for clarity.

---

## Threat Diffusion Simulation

Threat diffusion is simulated using a probabilistic model:
- Nodes are either infected (1) or uninfected (0).
- Infection spreads to neighbors based on a given probability.
- Simulation runs for a specified number of steps.

---

## Training a Diffusion Model

1. **Feature Engineering**:
   - Node degree, infection state, and neighbor infection states are used as features.

2. **Training**:
   - Data is split into training and testing sets.
   - A RandomForestClassifier is trained to predict node infection states.

3. **Evaluation**:
   - Model performance is assessed using accuracy and a classification report.

---

## Testing IP Impact

The effect of a specific IP on the network is tested:
- Simulate infection starting from the test IP.
- Predict the final infection states using the trained model.
- Results are visualized to show affected and unaffected nodes.

---

## Visualizing Results

1. **Graph Visualization**:
   - Nodes are colored based on their infection state:
     - Red: Test IP.
     - Orange: Infected nodes.
     - Blue: Uninfected nodes.

2. **Impact Summary**:
   - Counts of affected and unaffected nodes are displayed.

---

## How to Use

1. **Prepare the Dataset**:
   - Place your dataset file in the specified path (`data_path`).
   - Ensure it follows the expected structure.

2. **Run the Notebook**:
   - Execute the cells in order to preprocess data, construct the graph, simulate diffusion, and train the model.

3. **Test Specific IPs**:
   - Use the `test_ip_impact` function to analyze the effect of specific IPs on the network.

4. **Visualize Results**:
   - Run the `visualize_impact` function to generate visualizations.

---

## Requirements

Install the required Python packages using:

```bash
pip install pandas numpy scikit-learn networkx matplotlib
