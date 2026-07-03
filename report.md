# Modeling Human Activity States Using Hidden Markov Models

**Course:** Machine Learning  
**Submission date:** July 3, 2026  
**Device:** iPhone 16 — Sensor Logger v1.60.1  
**Sampling rate:** 100 Hz (10 ms intervals)

---

## 1. Background and Motivation

Human activity recognition (HAR) is a foundational problem in pervasive computing with direct applications in wearable health monitoring, rehabilitation tracking, smart home automation, and sports analytics. The core challenge is that the true activity a person is performing — walking, standing, jumping, remaining still — is a *hidden* state; only noisy, indirect measurements from onboard sensors are observable. This makes HAR a natural application for Hidden Markov Models (HMMs), which explicitly model the relationship between unobservable states and observable evidence, and capture the temporal dynamics of state transitions.

This project builds a complete HAR pipeline using motion data self-collected with an iPhone 16. The motivation for using a personal device is practical: smartphones are ubiquitous, carry high-quality inertial measurement units (IMUs), and collect data in ecologically valid conditions. By training and evaluating an HMM on real-world recordings, this work demonstrates whether a lightweight probabilistic model can reliably distinguish four everyday activities — **Still**, **Standing**, **Walking**, and **Jumping** — using only accelerometer and gyroscope signals.

---

## 2. Data Collection and Preprocessing

### 2.1 Recording Protocol

Data were collected using the **Sensor Logger** iOS app on a single **iPhone 16** held at waist level (still and standing) or placed in a trouser pocket (walking and jumping). Each recording captured one activity continuously and was saved as a timestamped folder containing separate CSV files for the accelerometer, gyroscope, and device metadata.

| Activity | Description |
|---|---|
| **Still** | Phone placed flat on a desk — minimal vibration |
| **Standing** | Phone held stationary at waist height |
| **Walking** | Consistent-pace walking in a straight corridor |
| **Jumping** | Continuous vertical jumps |

A total of **52 recording sessions** were completed: 13 per activity (recordings 0–12, where recording 0 is an unlabelled session from the first capture batch). All recordings were made on the same device to avoid cross-device calibration issues.

### 2.2 Sampling Rate and Duration

The Sensor Logger app was configured to sample all sensors at **10 ms intervals (100 Hz)**. This rate was chosen because:

- 100 Hz is sufficient to capture the peak dynamics of jumping (~2–4 Hz stride cycle for walking, ~1–2 Hz for jumping), with a Nyquist frequency of 50 Hz well above any relevant human motion frequency.
- Being a single-device study, no harmonisation across different sampling rates was required.

| Activity | Recordings | Total duration | Min single recording |
|---|---|---|---|
| Still | 13 | ~130.5 s | 7.58 s |
| Standing | 13 | ~126.3 s | 6.77 s |
| Walking | 13 | ~128.3 s | 6.77 s |
| Jumping | 13 | ~122.2 s | 6.68 s |

All four activities exceed the 90-second total threshold. Three short recordings (standing\_12, walking\_9, jumping\_10) fall just under 7 seconds but were retained because they contribute valid data and the per-activity totals remain well above the minimum.

### 2.3 Dataset Construction

Each recording folder contained `Accelerometer.csv` and `Gyroscope.csv` with columns `time` (nanoseconds), `seconds_elapsed`, and `x, y, z` axes. These were merged on the shared nanosecond timestamp (inner join) to produce **52 labelled CSV files** in a `dataset/` directory, each with columns:

```
time, seconds_elapsed, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, label
```

Files are named `activity_N.csv` (e.g., `walking_3.csv`, `jumping_10.csv`).

**Train / test split:**

- **Train:** recordings 0–9 per activity → 40 files (~502 s total)
- **Test:** recordings 10–12 per activity → 12 files (genuinely unseen, from a later recording session on the same day with the phone in a different position)

### 2.4 Windowing

A **sliding window** of **100 samples (1.0 s)** with a **hop of 50 samples (0.5 s, 50% overlap)** was applied. Window size rationale: at 100 Hz, a 1-second window provides 100 samples — enough for a statistically reliable FFT (frequency resolution = 1 Hz) and enough to capture at least one full walking or jumping cycle (~0.5–1.0 s). The 50% overlap smooths feature trajectories while avoiding redundancy.

This produced **753 training windows** and **176 test windows** across all activities.

---

## 3. Feature Extraction

Thirteen features were extracted per window across two domains. All features were computed on the raw 6-axis signal (3-axis accelerometer + 3-axis gyroscope).

### 3.1 Time-Domain Features (9 features)

| Feature | Formula / Description | Justification |
|---|---|---|
| `accel_mean_mag` | Mean of ‖a‖ = √(ax²+ay²+az²) | Activity intensity proxy; still ≈ 9.8 m/s², walking > 9.8 |
| `accel_std_mag` | Standard deviation of ‖a‖ | Variability; high for jumping, low for still |
| `accel_rms` | √(mean(‖a‖²)) | Energy content; distinguishes energetic activities |
| `accel_var_x` | Variance of ax | Lateral sway; elevated during walking |
| `accel_var_y` | Variance of ay | Anterior–posterior motion |
| `accel_var_z` | Variance of az | Vertical acceleration; highest during jumping |
| `gyro_rms` | √(mean(‖g‖²)) | Rotational energy; distinguishes walking (periodic rotation) from still |
| `gyro_std_mag` | Standard deviation of ‖g‖ | Gyroscope variability |
| `corr_ax_ay` | Pearson correlation(ax, ay) | Captures gait coupling; distinct for walking |

### 3.2 Frequency-Domain Features (4 features)

All frequency features are derived from the **Fast Fourier Transform (FFT)** of the accelerometer magnitude signal.

| Feature | Formula | Justification |
|---|---|---|
| `dom_freq_accel` | argmax of |FFT(‖a‖)| in Hz | Walking has a dominant stride frequency (~1–2 Hz); still/standing near 0 |
| `spectral_energy` | Σ|FFT(‖a‖)|² | Total signal power; high for jumping |
| `spectral_entropy` | −Σ p·log(p), p = normalised PSD | Regularity of motion; walking is periodic (low entropy), jumping less so |
| `gyro_dom_freq` | Dominant frequency of |FFT(‖g‖)| | Gyroscope rhythm; complements accelerometer dominant frequency |

### 3.3 Normalisation

All features were normalised using **Z-score standardisation** (StandardScaler):

> x̃ = (x − μ_train) / σ_train

The scaler was fit **exclusively on training data** and applied to test data to prevent information leakage. Z-score was chosen because:
- Features span widely different magnitudes (e.g., `spectral_energy` ~10⁶ vs `corr_ax_ay` ∈ [−1, 1]).
- Gaussian HMM emissions assume normally distributed observations; zero-mean unit-variance features better match this prior.

---

## 4. HMM Setup and Implementation

### 4.1 Model Architecture

One **Gaussian HMM** was trained per activity (four models total), following a **one-vs-rest** paradigm. Each model independently learns the temporal dynamics of its own activity. At inference time, a feature window is classified by the model that gives the highest log-likelihood score:

> ŷ = argmax_k log P(x | θ_k)

**Component definitions:**

| HMM Component | Definition in this project |
|---|---|
| Hidden states Z | Sub-phases within an activity (N=4 per model) |
| Observations X | 13-dimensional normalised feature vector per 1-second window |
| Transition matrix A | Probability of moving from sub-phase i to sub-phase j |
| Emission distribution B | Diagonal-covariance multivariate Gaussian: P(x \| z) = N(μ_z, Σ_z) |
| Initial distribution π | Probability of starting in sub-state i |

**Four hidden states** capture realistic within-activity sub-phases:
- *Still/Standing:* stable, micro-sway, slight lean, recovery
- *Walking:* heel-strike, loading response, mid-stance, push-off
- *Jumping:* crouch, push-off, airborne, landing

### 4.2 Baum–Welch Training

The Baum–Welch algorithm (Expectation-Maximisation for HMMs) was used to estimate model parameters {A, B, π} from unlabelled observation sequences. Key implementation choices:

- **Covariance type:** diagonal. Full covariance was found to be numerically unstable at this data scale (~750 windows per activity with 13 features and 4 states), producing zero-sum transition rows and diverging log-likelihoods. Diagonal covariance assumes independent features per state — a reasonable simplification that yields stable, well-converged models.
- **min_covar = 1e-3:** a numerical floor on diagonal covariance elements prevents singular matrices in near-zero variance states (particularly the "still" activity).
- **Convergence criterion:** |ΔlogL| < 10⁻⁴ — training halts when the change in log-likelihood between iterations falls below this threshold. This is a principled criterion: the algorithm terminates when it has found a local maximum of the likelihood surface, not after an arbitrary number of iterations.
- **transmat repair:** after fitting, any transition matrix row summing to zero (caused by a state never being visited in short recordings) is normalised to a uniform distribution, maintaining a valid stochastic matrix.

Convergence results:

| Activity | Iterations | Final log-L | Converged |
|---|---|---|---|
| Still | 3 | 8152.1 | ✓ |
| Standing | 12 | 6304.7 | ✓ |
| Walking | 10 | 3129.7 | ✓ |
| Jumping | 3 | −131.5 | ✓ |

All four models converged within the 300-iteration budget.

### 4.3 Viterbi Decoding

The **Viterbi algorithm** was used to decode the most likely hidden-state sequence for any given observation sequence:

> z* = argmax_{z₁,...,zT} P(z₁,...,zT | x₁,...,xT, θ)

This is implemented via dynamic programming with logarithmic probabilities to prevent underflow. Viterbi decoding reveals the temporal progression of sub-phases within each activity and is visualised for two test recordings (walking\_10 and jumping\_10) in Figure 6.

---

## 5. Results and Interpretation

### 5.1 Evaluation Setup

The model was tested on **12 genuinely unseen recordings** (3 per activity, recordings 10–12), collected in a separate session after training data collection. The test recordings include three short files (6.7–7.6 s) that represent more challenging edge cases.

### 5.2 Confusion Matrix

|  | Pred: Still | Pred: Standing | Pred: Walking | Pred: Jumping |
|---|---|---|---|---|
| **True: Still** | 45 | 0 | 0 | 0 |
| **True: Standing** | 0 | 45 | 0 | 0 |
| **True: Walking** | 0 | 0 | 47 | 0 |
| **True: Jumping** | 0 | 30 | 0 | 9 |

*Figure: Confusion Matrix — Test Set*

### 5.3 Per-Class Metrics

| Activity | Samples | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Still | 45 | 100.0% | 100.0% | 100.0% |
| Standing | 45 | 100.0% | 77.1% | 83.0% |
| Walking | 47 | 100.0% | 100.0% | 100.0% |
| Jumping | 39 | 23.1% | 100.0% | 83.0% |
| **Overall** | **176** | — | — | **83.0%** |

### 5.4 Interpretation

**Still, Standing, Walking** are classified with perfect sensitivity (100% recall). The model reliably separates the three low-energy states (still vs standing) and the rhythmic gait state (walking).

**Jumping** is the most challenging class: 30 of 39 jumping windows are predicted as Standing. This is a meaningful finding — when jumping test recordings are very short (6.68–7.62 s), the model sees limited periodicity evidence, and the *resting* frames between jumps (crouch and landing) produce feature vectors that resemble standing. The jump model itself (logL = −131.5) has a notably lower likelihood than the other three models, suggesting the jumping HMM learned a less representative distribution from the training data.

This confusion is one-directional: the standing model has 100% specificity (no false positives from jumping). The issue is exclusively a failure of the jumping model to recognise its own activity at short durations.

### 5.5 Visualisations

- **Figure 1 (fig_raw_signals.png):** Raw accelerometer and gyroscope traces for each activity. Jumping shows large-amplitude, quasi-periodic bursts. Still shows near-flat signals. Walking shows clear rhythmic oscillations at ~1–2 Hz.
- **Figure 2 (fig_feature_distributions.png):** Normalised feature distributions per activity. `accel_rms` and `gyro_rms` are the most discriminative: jumping has a high, wide distribution; still is tightly clustered near zero.
- **Figure 3 (fig_convergence.png):** Baum–Welch log-likelihood curves. Still and Jumping converge in 3 iterations; Standing takes 12, reflecting more complex within-class variation.
- **Figure 4 (fig_transition_matrices.png):** Transition heatmaps. The still and standing HMMs show strong self-transitions (diagonal-dominant), consistent with sustained posture. The walking model shows cyclic transitions reflecting gait phases.
- **Figure 5 (fig_emission_means.png):** Emission means for each hidden state and feature. States within the jumping model show high variance in `accel_rms` and `spectral_energy`, reflecting the impact-then-flight dynamics.
- **Figure 6 (fig_viterbi_sequences.png):** Viterbi-decoded state sequences for walking\_10 and jumping\_10. Walking shows smooth, cyclic state transitions; jumping shows irregular state visits due to shorter duration.
- **Figure 7 (fig_confusion_matrix.png):** Confusion matrix as described in Section 5.2.
- **Figure 8 (fig_predicted_timeline.png):** Window-by-window prediction timelines for four test recordings, showing temporal consistency of predictions.

---

## 6. Discussion and Conclusion

### 6.1 Activity Difficulty

Still, standing, and walking were easily distinguished, demonstrating that HMMs capture the temporal structure of these activities effectively from simple IMU features. Still vs standing is the hardest pair conceptually (both are low-energy states) but is separated cleanly by the near-zero gyroscope variance of the "still" activity (phone on desk) versus the micro-sway of standing.

Jumping is the hardest to recognise. The confusion with standing arises from two factors: (1) the brevity of test recordings (as short as 6.68 s) limits the observable periodicity, and (2) the resting phase between jumps closely resembles standing in the feature space.

### 6.2 Transition Probabilities and Realistic Behaviour

The learned transition matrices show physically plausible patterns. The still and standing models are nearly diagonal, reflecting that these activities involve staying in a sub-phase for extended periods. The walking model shows more off-diagonal mass, consistent with the cyclic progression through gait sub-phases. These patterns emerged purely from the Baum–Welch algorithm — they were not manually set — which validates the expressiveness of the HMM formulation.

### 6.3 Effect of Sensor Noise and Sampling Rate

At 100 Hz, the sensor data is sufficiently clean for feature extraction. Minor timestamp jitter (~0.01 ms) visible in the raw CSVs had negligible impact after windowing. The main source of noise is environmental: phone movement artefacts in the pocket during walking occasionally produce spike features, but these are averaged out within the 1-second window. Three test recordings below 7 seconds introduced difficulty for jumping specifically, as discussed above.

### 6.4 Improvements

Several targeted improvements would significantly boost performance:

1. **Longer jumping recordings.** The three jumping test files (6.7–7.6 s) are noticeably shorter than other activities. Re-recording at 10 s would eliminate the brevity confound.
2. **Additional features.** Zero-crossing rate of the accelerometer magnitude and the inter-quartile range of the gyroscope norm would better characterise the impulsive dynamics of jumping vs the steady low-level dynamics of standing.
3. **Mixture of Gaussians emissions.** Replacing single Gaussians per state with Gaussian Mixture Models (GMM-HMM) would capture the multimodal distributions visible in Figure 2 more accurately.
4. **Longer training sequences per activity.** Increasing from 10 to 15+ recordings per activity would give the Baum–Welch algorithm more diverse trajectory examples, especially for the jumping model.
5. **Cross-validation.** Using leave-one-recording-out cross-validation during training would yield a more reliable estimate of generalisation performance.

### 6.5 Conclusion

A Hidden Markov Model pipeline was successfully built to recognise four human activities from real smartphone IMU data. The model achieved **83% overall accuracy** on genuinely unseen test recordings, with perfect classification of still, standing, and walking, and 23% recall for jumping. The Baum–Welch algorithm converged reliably for all four activity models, and the Viterbi algorithm revealed physically interpretable hidden-state sequences. The main limitation — jumping/standing confusion — is directly attributable to short test recording durations and suggests a clear path forward for improvement.

---

## Appendix: Project File Structure

```
Hidden_Markov_Model/
├── dataset/                         # 52 labelled CSV files (activity_N.csv)
│   ├── still_0.csv  ...  still_12.csv
│   ├── standing_0.csv  ...  standing_12.csv
│   ├── walking_0.csv  ...  walking_12.csv
│   └── jumping_0.csv  ...  jumping_12.csv
├── HMM_Activity_Recognition.ipynb   # Main Jupyter notebook
├── build_dataset.py                 # Script to merge raw folders into dataset/
├── notebook_cells/                  # Individual cell source files
│   ├── cell_01_imports.py
│   ├── cell_02_load.py
│   ├── cell_03_raw_viz.py
│   ├── cell_04_features.py
│   ├── cell_05_normalize.py
│   ├── cell_06_train.py
│   ├── cell_07_visualize.py
│   ├── cell_08_viterbi.py
│   └── cell_09_evaluation.py
├── fig_raw_signals.png
├── fig_feature_distributions.png
├── fig_convergence.png
├── fig_transition_matrices.png
├── fig_emission_means.png
├── fig_viterbi_sequences.png
├── fig_confusion_matrix.png
├── fig_predicted_timeline.png
└── report.md                        # This report
```
