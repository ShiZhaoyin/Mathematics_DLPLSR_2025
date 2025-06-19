# DLPLSR: Dual-Label Propagation Least Squares Regression for Semi-Supervised Classification

This repository provides the official MATLAB implementation of **DLPLSR**, a dual-label propagation–based least squares regression framework for semi-supervised classification.  
DLPLSR jointly exploits local manifold information and global clustering structure without generating explicit pseudo-labels, and employs an orthogonal-sparse projection to select compact and discriminative features.

## Contents

| File / Folder | Description |
|---------------|-------------|
| `DLPLSR.m` | Core routine of the DLPLSR algorithm |
| `Demo.m` | Example script showing end-to-end training and testing |
| `Testing.m` | Utility for evaluating classification metrics |
| `Multi_Class_Metrics.m` | Functions for ACC, F1-Score, etc. |
| `.mat` files | Example datasets (e.g., `Jaffe.mat`, etc.) |

## Quick Start

1. Launch MATLAB and add the project folder to the path.
2. Execute the demo:

```matlab
>> Demo
