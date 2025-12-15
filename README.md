# MoCo-PGCo-Traffic-Analysis

Automated Traffic Enforcement: Predicting Injury Crash Reductions (DC + PG County)

This repository contains an end-to-end analysis pipeline that extends an earlier Medium post on automated traffic enforcement in Washington, DC and Prince George’s County (PG County), MD. The goal is to move beyond the question “Do cameras work?” and instead answer:

Which camera sites are most likely to experience a significant reduction in injury crashes after activation, and what site characteristics predict success?

The pipeline includes:

Site-level pre/post injury crash aggregation
Supervised learning + model evaluation (logistic regression, random forest; ROC-AUC, confusion matrix)
Clustering of enforcement sites into similar “context” groups
Network/spillover exploration using a camera proximity graph
Stakeholder & Decision Use
Stakeholder: Local transportation/public safety agencies (e.g., DDOT, PG County DPW&T)

Decision this supports:

Prioritizing where to install new automated enforcement cameras.
Identifying which sites are likely to produce meaningful safety benefits (injury crash reductions)
Supporting corridor-based planning by exploring network adjacency / spillover patterns.
