# Personalized-Offer-Recommendation-Amex-Campus-Challenge-
Personalized Offer Recommendation Engine
American Express Campus Challenge 2025

 Project Overview
This repository contains the winning approach developed for the American Express Campus Challenge 2025. The goal was to build a robust recommendation system that predicts and ranks the most relevant credit card offers for users, optimizing for engagement and conversion.
Our solution focuses on three core pillars: Advanced NLP-based feature engineering, Adversarial Validation to mitigate severe data drift, and a Stacked Ensemble of gradient-boosted rankers.

 Key Highlights
Ensemble Strategy: A stacked architecture using XGBoost, LightGBM, and CatBoost meta-tuned with Logistic Regression.
Drift Mitigation: Utilized Adversarial Validation (AUC > 0.7) to identify and remove 45+ high-drift features, ensuring leaderboard stability.

NLP Pipeline: Transformed offer descriptions using SentenceTransformer (384D) reduced via PCA (32D) to capture semantic user-offer relevance.

Ranking Optimization: Optimized directly for MAP@7 (Mean Average Precision) using LambdaRank and NDCG-based objectives.
 Data Pipeline & Feature Engineering
 
1. Data Cleaning & Preprocessing
Handling Sparsity: Removed 26 columns with >90% missing values.
Zero Variance: Eliminated 59 columns with no predictive signal (Standard Deviation = 0).
Normalization: Applied log1p transformation to skewed numeric features to ensure distribution symmetry.
Imputation: Mode imputation for categorical features; Mean/Median for numerical features based on skewness.

2. Feature Engineering
Temporal Features: Created time_since_last_event, time_since_last_system_event, and time_band to capture user behavioral patterns.
Event Velocity: Calculated rolling windows (1h, 3h, 6h, 9h, 24h, 3d) for offer impressions and clicks.
NLP & Embeddings:
Offer titles processed through SentenceTransformers.
Dimensionality reduction via PCA (32 components) to capture latent themes.
Session Similarity: Calculated cosine similarity between current offer and user’s live session activity.
Categorical Encoding: Target encoding on offer_id to map historical conversion performance.
3. Drift Detection (Adversarial Validation)
A critical challenge was the distributional shift between training and test sets. We implemented:
Univariate Detection: KS-statistic (> 0.1) and Chi-squared tests.
Multivariate Detection: Trained a binary classifier to distinguish between Train/Test. High AUC (1.0) confirmed severe drift; by removing high-drift features, we stabilized the AUC to 0.82.

 Modeling Approach
We employed a Stacked Ensembling approach to leverage the strengths of different ranking algorithms:
Base Models (Rankers):
XGBoost: Optimized using rank:ndcg.
LightGBM: Optimized using lambdarank.
CatBoost: Optimized using YetiRank.
Meta-Model:
Logistic Regression: Used to blend raw rank scores from base models into a final engagement probability.
Ablation Study:
An ablation study revealed that a LightGBM + XGBoost stack dominated the ensemble, allowing us to remove CatBoost to create a leaner, more efficient model.

 Model Performance
Internal Validation (Time-split): Achieved an initial MAP@7 of ~0.71.
Leaderboard Stability: Successfully stabilized leaderboard performance from a collapsed 0.56 to a robust ~0.53 by eliminating unstable features.

Improvement: The final ensemble provided a 5% boost over the single best standalone model (LightGBM).

📈 Future Scope for Improvement
Graph Embeddings: Implementing Node2Vec to capture latent user-offer relationships via interaction graphs.
FFMs: Exploring Field-aware Factorization Machines to capture granular pairwise feature interactions.
Cold Start Modeling: Developing a separate metadata-only model specifically for new offers with zero historical interaction.
