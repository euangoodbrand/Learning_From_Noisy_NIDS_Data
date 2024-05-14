# Notes on Learning from Label Noise in Network Intrusion Detection Data

## Sample Selection
- Promising technique but results are inconsistent and vary significantly.

## Noise Adaptation
- Performs well with no noise and imbalance.
- Performs poorly with added noise or data imbalance, except at 0.05 imbalance which acts as a regularizer and enhances performance for all techniques.

## LIO (Label-Noise Robust Model)
- Delivers the best and most consistent results.

## Baseline Comparison
- Techniques hover around the baseline with some performing worse and some slightly better.

## Regularization Effect
- Low levels of synthetic noise and imbalance improve performance due to a regularization effect.

## Impact of Noise vs. Data Imbalance
- Noise has a larger impact on results than data imbalance.
  - Example: 60% noise reduces performance more than extreme class imbalance (minority class 100 times smaller than majority), with a 20-40% difference in F1 score.

## Data Augmentation vs. Sample Reweighting
- Data augmentation generally performs better but is more computationally expensive.
  - **SMOTE:** Best overall data augmentation technique, except at 0.6 high noise level where random upsampling is superior.
  - **Focal Loss:** Best sample reweighting technique but only slightly better than others. Naive technique is best for high noise (0.6).
  - If noise level is unknown, naive technique is preferred for consistent performance across noise levels.

## Synthetic Noise
- Feature-dependent, class-dependent, and MIMICRY noise outperform uniform noise.
- Non-uniform noise techniques trained models perform better on clean data across all metrics.
  - However, these techniques might bias results since they modify label noise based on classes.
  - Further research is needed to measure the similarity of synthetic noise to real noise for clearer insights.

## Conclusion
- Noise significantly impacts performance more than data imbalance.
- Regularization and appropriate noise handling techniques are crucial for improving performance.
- Continued research is required to accurately measure and handle noise in network intrusion detection data.
