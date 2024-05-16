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



### Problem
The primary issue observed is the significant drop in performance after the initial epochs during training, likely due to overfitting to noisy labels. This overfitting is indicated by the models performing well initially but then learning the noise, leading to degraded performance.

### Mitigation Strategies
To address the overfitting and improve the robustness of models to label noise, consider implementing the following strategies:

1. **Early Stopping:**
   - Monitor validation performance and stop training when performance starts to degrade.
   - This prevents the model from overfitting to noisy labels.

2. **Regularization Techniques:**
   - Use L2 regularization (weight decay) to penalize large weights.
   - Incorporate dropout layers to prevent overfitting by randomly dropping units during training.

3. **Label Smoothing:**
   - Apply label smoothing to soften the labels, reducing sensitivity to noise.


### Best Perfomig Methods
Morse performs the best followed by LIO close second, and then GCE. Co-teaching is very strong also but with unpredictable training and very large swings, a concern that this would not generalise well at all for different data distributions/ shifts.


### Comparison to MORSE(Grim Reality) paper
Morse performs the best as expected from the paper, the rest of the techniques dont perform that much better or worse than baseline except for morse and LIO, LIO is different from the paper.
In the morse paper LIO was the worst performer but for us it is the best.
In the paper except for MORSE, noise adaptation performed the best but our reserch showed the oposite also with GCE performing worse across the metrics. It should be noted that the difference between these techniques is not large so increased numbers of runs should be performed to confirm these numbers.

