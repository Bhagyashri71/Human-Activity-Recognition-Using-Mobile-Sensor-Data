# Human-Activity-Recognition-Using-Mobile-Sensor-Data - Walking vs Standing

<img width="1536" height="1024" alt="ChatGPT Image Sep 15, 2026, 02_48_58 PM" src="https://github.com/user-attachments/assets/ef499825-a977-475c-a879-110075971fbc" />

This project focuses on detecting **Walking and Standing activities** using Inertial Measurement Unit (IMU) sensor data and traditional Machine Learning techniques.

The dataset contains accelerometer and gyroscope measurements collected at approximately **10 Hz**. The main objective is to extract meaningful time-domain features from sensor signals and build a machine learning model that can distinguish between walking and standing.

## Dataset Features

The dataset contains:

- `accX`, `accY`, `accZ` — Accelerometer measurements
- `gyroX`, `gyroY`, `gyroZ` — Gyroscope measurements
- `timestamp` — Sensor timestamp
- `Activity` — Activity label
     - `Activity 0 → Standing`
     - `Activity 1 → Walking`

Additional features such as acceleration magnitude, gyroscope magnitude, and frame-to-frame magnitude changes are derived during preprocessing.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Google Colab

## Machine Learning Approach
## Methodology

The project follows the pipeline:

1. Data loading and exploration
2. Timestamp conversion and correction
3. Duplicate removal
4. Timestamp discontinuity detection
5. Activity-based segmentation
6. 2-second sliding window creation
7. Feature engineering
8. Feature selection
9. Segment-level train-test split
10. Class imbalance handling
11. Machine Learning model training
12. Model evaluation
13. Unseen activity-episode validation

### Windowing

- Sampling rate: ~10 Hz
- Window duration: 2 seconds
- Samples per window: 20
- Overlap: 50%
- Step size: 10 samples

Windows are created within valid segments so that they do not cross activity transitions or timestamp discontinuities.

## Machine Learning Models

The following traditional ML models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Class imbalance was addressed using:

- Class weighting
- SMOTE

## Best Model

The best-performing model on the segment-level holdout was:

**XGBoost with Class Weighting**

Performance:

| Metric | Score |
|---|---:|
| Accuracy | 99.56% |
| Macro F1 | 86.25% |
| Balanced Accuracy | 83.26% |
| ROC-AUC | 99.14% |
| Standing F1 | 72.73% |
| Standing Recall | 66.67% |

The dataset is highly imbalanced, with approximately:
- 98.2% Walking
- 1.8% Standing

Therefore, evaluation focuses not only on accuracy but also on Precision, Recall, F1-score, Balanced Accuracy, and ROC-AUC.

## Generalization Analysis

A separate unseen-standing-episode evaluation was performed to test whether the model could recognize standing episodes that were not represented in training.

The model detected approximately **23.8% of unseen standing windows**.

This showed that the main challenge was not only class imbalance, but also the limited diversity of independent standing episodes.

Different standing episodes showed variation in sensor orientation, acceleration patterns, and movement characteristics. SMOTE improved the training class balance but did not create genuinely new activity patterns, so it did not solve the unseen-episode generalization problem.

## Limitations

- Highly imbalanced activity classes
- Limited number of independent standing episodes
- Overlapping windows are not completely independent observations
- Timestamp discontinuities/resets are present in the raw data
- Sensor orientation and placement may affect the recorded patterns
- Holdout performance may be higher than performance on completely unseen activity episodes

## Recommendation 

Future work can focus on:

- Collecting more independent standing episodes
- Collecting data from multiple subjects
- Using different sensor orientations and device placements
- Increasing the diversity of walking and standing patterns
- Performing subject-level or session-level validation
- Collecting more balanced activity data
- Improving timestamp reliability

##  Conclusion

This project demonstrates that traditional machine learning can effectively identify walking and standing patterns from IMU sensor data.

XGBoost with class weighting achieved strong performance on a leakage-safe segment-level holdout. However, evaluation on unseen standing episodes revealed a significant generalization gap.

Therefore, the final model should be considered a **strong baseline rather than a fully generalizable real-world activity recognition system**. The most important improvement is increasing the diversity and number of independent activity episodes and subjects.
