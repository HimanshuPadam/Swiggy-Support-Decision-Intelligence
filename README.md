Domain: Support

## How to run

### Prerequisites
- Python 3.10+ is required.
- Standard libraries (`math`, `csv`, `os`, `sys`) and `numpy` are used.
- NumPy is the only external dependency for the graded numerical work.
- All algorithms (Ordinary Least Squares, gradient descent, logistic regression, K-Nearest Neighbours, K-Means, hierarchical linkage, Principal Component Analysis, decision tree split evaluation, and AdaBoost) are implemented manually from first principles rather than using `scikit-learn` or other black-box machine learning libraries.

### Environment Setup (Windows PowerShell)
From the project root:

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install required dependencies
pip install -r requirements.txt
```

### Dataset Generation & Execution
The dataset can be generated or regenerated using the provided script:

```powershell
# Generate data/records.csv (800 records)
python generate_data.py
```

Each assignment part can be executed independently from the repository root:

```powershell
# Part 1: Statistical inference, regression, and gradient descent
python part1.py

# Part 2: Classification, threshold analysis, KNN CV, and cost optimization
python part2.py

# Part 3: K-Means, segment profiling, hierarchical clustering, and PCA
python part3.py

# Part 4: Decision tree splits, voting ensemble, AdaBoost, and neural architectures
python part4.py
```

All numerical results and CSV deliverables are automatically saved in the `outputs/` directory.

---

## Part 1 findings

Part 1 establishes parametric baselines, statistical inferences, and regression relationships for ticket closure times ($y = \text{resolution\_minutes}$).

### 1. Point Estimation & Confidence Interval
- **Sample size ($n$)**: 800 support tickets.
- **Sample mean ($\bar{y}$)**: 8.9382 minutes.
- **Sample standard deviation ($s$)**: 9.1723 minutes (computed with Bessel's correction $n-1 = 799$).
- **Standard error ($SE$)**: $s / \sqrt{n} = 0.3243$ minutes.
- **95% Confidence Interval**: $\bar{y} \pm 1.96 \times SE = [8.3026, 9.5738]$ minutes.
- *Practical implication*: We are 95% confident that the mean resolution time of the generated synthetic dataset lies between 8.30 and 9.57 minutes, rather than reflecting the true mean of real Swiggy support interactions.

### 2. Hypothesis Test against 8.0-Minute SLA Benchmark
- $H_0: \mu = 8.0$ minutes versus $H_1: \mu \ne 8.0$ minutes.
- **$z$-statistic**: $z = (8.9382 - 8.0) / 0.3243 = 2.8930$.
- **Decision**: Because $|z| = 2.8930 > 1.96$, we reject $H_0$ at the 5% significance level (`reject_h0 = yes`).
- *Practical implication*: The generated sample mean resolution time is statistically significantly above the 8.0-minute reference benchmark at the 5% significance level.

### 3. Sample Size Determination
- Margin of error $E = 0.5$ minutes at 95% confidence:
  $$n_{\text{required}} = \left\lceil \left(\frac{1.96 \times 9.1723}{0.5}\right)^2 \right\rceil = 1293$$
- *Practical implication*: The available 800 records are insufficient to guarantee a tight $\pm 0.5$-minute precision; an audit sample of at least 1,293 tickets is required for future operational benchmarking.

### 4. Simple Linear Regression & Gradient Descent
- Regressing $y$ on account value $X_4$ (`account_value_inr_thousands`):
  - Closed-form OLS: slope $\hat{\beta}_1 = 0.6701$, intercept $\hat{\beta}_0 = -3.6598$.
  - Fit metrics: $R^2 = 0.5196$, $RMSE = 6.3537$, $MAE = 3.7636$.
  - Variance decomposition: $TSS = 67221.4285 = ESS (34925.6088) + RSS (32295.8197)$ (`tss_equals_ess_plus_rss = yes`).
- Batch gradient descent ($\alpha = 0.001$, 500 iterations):
  - Loss converges from $J_0 = 81.9589 \to J_{500} = 20.1849$.
  - Final slope $= 0.6701$, intercept $= -3.6598$.
  - `gd_slope_gap = 0.0000`: Exact numerical convergence confirming first-principles correctness.

### 5. Heteroscedasticity Diagnosis & Log-Log Transformation
- Residual standard deviation for high-value accounts ($X_4 > \text{median}$) was $8.6891$ versus $2.3155$ for low-value accounts, yielding a raw ratio of **3.7525** (severe variance instability).
- Fitting a log-log model $\ln(y) = \alpha + \beta \ln(X_4)$:
  - Slope $= 1.5030$, Intercept $= -2.4370$, $R^2 = 0.6565$, $RMSE = 0.5268$.
  - Residual standard deviation ratio compressed to **1.0396** ($0.5369 / 0.5165$).
  - *Practical implication*: Log transformation stabilizes multiplicative variance across account tiers, bringing residual spread to nearly 1:1.

### 6. Multiple Regression Analysis
- Regressing $\ln(y)$ on $\ln(X_4)$ and operational covariates via normal equations:
  - Intercept $\beta_0 = -1.8613$.
  - Model fit: Multiple regression achieved **Adjusted $R^2 = 0.7403$**, substantially higher than the single-predictor log-log model (**Adjusted $R^2 = 0.6561$**).
- **Coefficients exceeding 0.05 in absolute value** (excluding intercept):
  - **$\beta_1 = 1.1576$** ($\ln(X_4)$, Account Value): Higher account value is positively associated with log-resolution time, conditional on the other predictors.
  - **$\beta_3 = 0.0940$** ($X_2$, Earlier Ticket Count): A higher earlier-ticket count is positively associated with log-resolution time.
  - **$\beta_4 = 0.1067$** ($X_3$, SLA Consumed Ratio): Higher SLA consumption is positively associated with log-resolution time.
  - **$\beta_6 = 0.1518$** ($X_6$, Reopened Flag): The reopened flag is positively associated with log-resolution time, conditional on the other predictors.
- *Excluded coefficients ($|\beta| \le 0.05$)*: Customer tenure $X_1$ ($\beta_2 = -0.0074$), satisfaction score $X_5$ ($\beta_5 = 0.0028$), and self-service flag $X_7$ ($\beta_7 = 0.0172$) have negligible linear effects once account scale and operational distress are accounted for.

---

## Part 2 findings

Part 2 addresses the binary prediction of customer escalations (`escalated = 1`).

### 1. Stratified Data Split & Model Training
- **Dataset Partition**: Stratified 70/30 split into 560 training rows (119 escalations, 21.25%) and 240 test rows (51 escalations, 21.25%).
- **Training Trajectory**: Cross-entropy loss decreased monotonically from $0.6931$ (epoch 0) $\to 0.5685$ (epoch 10) $\to 0.3952$ (epoch 100) $\to 0.3704$ (epoch 1000).
- **Out-of-sample Test Loss**: Test binary cross-entropy $= 0.3818$.
- **Learning Rate Sensitivity (200 Epochs)**:
  - $\eta = 0.01 \implies 0.5091$ (slow progress).
  - $\eta = 0.1 \implies 0.3776$ (stable, well-paced convergence).
  - $\eta = 1.0 \implies 0.3703$ (produced the lowest loss among the three tested learning rates after 200 epochs).

### 2. Feature Importance via Odds Ratios
The largest absolute standardized logistic coefficients identify key escalation drivers:
- **$X_3$ (`sla_consumed_ratio`)**: Weight $= +0.9013$, **Odds Ratio $= 2.4629$**. A 1 SD increase in SLA consumption increases escalation odds by $146.3\%$.
- **$X_1$ (`customer_tenure_months`)**: Weight $= -0.5554$, **Odds Ratio $= 0.5739$**. A one-standard-deviation increase in customer tenure is associated with approximately 42.6% lower escalation odds.
- **$X_2$ (`earlier_ticket_count_90d`)**: Weight $= +0.5505$, **Odds Ratio $= 1.7341$**. Frequent past contacts increase escalation odds by $73.4\%$.

### 3. Evaluation Metrics & Accuracy Pitfalls
- **Blind Baseline**: Always predicting negative yields an accuracy of **$78.75\%$** ($189 / 240$), but recall is **$0.00\%$** (fails to catch all 51 escalations). Accuracy alone is deeply misleading for class-imbalanced datasets ($21.25\%$ positive rate).
- **Threshold 0.5**: Accuracy $= 80.42\%$, Precision $= 54.55\%$, Recall $= 47.06\%$, F1 $= 0.5053$.
- **Threshold 0.3**: Accuracy $= 76.25\%$, Precision $= 46.34\%$, Recall $= 74.51\%$, F1 $= 0.5714$. Lowering the threshold shifts the decision boundary, capturing 38 of 51 escalations ($74.51\%$ recall) at the expense of more false alarms ($44$ FPs vs $20$ FPs).

### 4. K-Nearest Neighbours Cross-Validation
- Evaluated odd $K \in \{1..21\}$ across 5 folds using mean validation F1 score.
- **Best Model**: Selected $K = 7$ with mean CV F1 of **$0.4309$** (from `outputs/part2_knn_cv.csv`).
- **Test Performance ($K=7$)**: Accuracy $= 76.67\%$, Precision $= 43.24\%$, Recall $= 31.37\%$, F1 $= 0.3636$ (substantially weaker than logistic regression).

### 5. Cost-Sensitive Threshold Selection
Using the operational loss matrix $C = ₹50 \times FP + ₹400 \times FN$:
- Evaluated candidate thresholds $\tau \in [0.05, 0.95]$ strictly on the 560 training records.
- Selected threshold: **$\tau = 0.1500$** through training-data cost minimization (training cost: ₹12,150), rather than as a proven optimal production policy.
- Out-of-sample evaluation on the 240 test records:
  - Flag Nobody ($FN=51$): **₹20,400**.
  - Standard Threshold $\tau = 0.5$ ($FP=20, FN=27$): **₹11,800**.
  - **Threshold $\tau = 0.15$ ($TP=45, FP=73, TN=116, FN=6$)**: Observed test cost of **₹6,050** on this particular split (FP cost $= 73 \times ₹50 = ₹3,650$; FN cost $= 6 \times ₹400 = ₹2,400$; Total $= ₹6,050$).
- *Operational value*: Selecting the threshold through training-data cost minimization yields an observed **$48.73\%$ financial savings** over default classification and **$70.34\%$ savings** over the unmonitored baseline on this test split.

---

## Part 3 findings

Part 3 analyzes customer ticket patterns using unsupervised clustering and dimensionality reduction on features $X_1$–$X_5$.

### 1. K-Means Clustering & Elbow Method
Evaluated K-Means from scratch for $K \in \{2..6\}$ using standardized features:
- $K = 2$: WCSS $= 2262.6753$
- $K = 3$: WCSS $= 1602.8471$
- $K = 4$: WCSS $= 1494.7732$
- $K = 5$: WCSS $= 1323.7157$
- $K = 6$: WCSS $= 1278.1691$
- **Elbow K = 3**: Drop $\Delta WCSS(3) = 108.0739 < 0.5 \times \Delta WCSS(2) = 329.9141$.
- *Optimization note*: WCSS reductions do not decrease monotonically between successive $K$ increments ($\Delta WCSS$ from $K=3 \to 4$ is $108.07$, whereas $K=4 \to 5$ is $171.06$). This demonstrates that the solution represents a deterministic result under the prescribed initialization; it is not guaranteed to be the global minimum.

### 2. Segment Profiles & Operational Strategy
Using $K=3$ ordered by ascending unstandardized mean of $X_3$ (`sla_consumed_ratio`):

| Metric | Segment A | Segment B | Segment C |
| :--- | :---: | :---: | :---: |
| **Segment Name** | *Stable, low-friction tickets* | *High-value repeat-contact tickets* | *SLA-stressed, low-satisfaction tickets* |
| **Cluster Size** | 322 (40.25%) | 255 (31.88%) | 223 (27.88%) |
| **Mean Tenure ($X_1$)** | 47.5028 months | 28.6071 months | 14.3892 months |
| **Mean Earlier Tickets ($X_2$)** | 1.9503 | 6.0706 | 4.2287 |
| **Mean SLA Ratio ($X_3$)** | 0.2250 | 0.4534 | 0.7118 |
| **Mean Account Value ($X_4$)** | ₹12.5301k | ₹28.7192k | ₹16.5099k |
| **Mean CSAT Score ($X_5$)** | 62.1584 | 55.0082 | 38.5215 |
| **Escalation Outcome Rate** | **0.0186** (1.86%) | **0.2353** (23.53%) | **0.4664** (46.64%) |
| **Mean Resolution Time** | **3.5976 minutes** | **17.1248 minutes** | **7.2883 minutes** |

- **Operational Strategies**:
  - **Segment A**: Standard resolution and automated self-service with routine background monitoring.
  - **Segment B**: Priority specialist assignment to prevent repeated customer friction, as these high-value accounts exhibit the highest resolution time (17.12 minutes).
  - **Segment C**: Proactive escalation and senior agent routing before SLA breach occurs, as this segment exhibits the highest escalation rate (46.64%).

### 3. Hierarchical Dissimilarity Clustering
Agglomerative clustering on dissimilarity elements E1–E6:
- **Single Linkage**: Merges at minimum pairwise distance, creating elongated chaining effects across intermediate points $\implies$ **4 clusters at cut** (cut between merges 2 and 3).
- **Complete Linkage**: Merges based on maximum pairwise distance, enforcing compact diameter clusters $\implies$ **2 clusters at cut** (cut between merges 4 and 5).
- **Average Linkage**: Uses average pairwise distance, balancing sensitivity $\implies$ **2 clusters at cut** (cut between merges 4 and 5).

### 4. Principal Component Analysis (PCA)
- Total variance: $\sum \lambda_j = 5.0000$.
- **Eigenvalues & Variance Ratios**:
  - PC1: $\lambda_1 = 2.6136$ ($52.27\%$ variance explained).
  - PC2: $\lambda_2 = 1.1828$ ($23.66\%$ variance explained).
  - PC3: $\lambda_3 = 0.5114$ ($10.23\%$ variance explained).
  - Components to explain $\ge 80\%$ variance: **3 components** (cumulative variance $= 86.16\%$).
- **PC1 Loadings**:
  - $X_1$ (`tenure`): $+0.5317$
  - $X_2$ (`earlier_tickets`): $-0.4100$
  - $X_3$ (`sla_ratio`): $-0.5187$
  - $X_4$ (`account_value`): $-0.2916$
  - $X_5$ (`satisfaction`): $+0.4418$
  - *Interpretation*: PC1 represents an observational contrast between stable, satisfied, long-tenured customer interactions and high-friction tickets marked by repeated contacts and high SLA consumption (reflecting association, not causal impact).
- **Mean PC1 by Segment**:
  - Segment A: **$+1.7773$** (highly aligned with stability and satisfaction).
  - Segment B: **$-0.7831$** (shifted toward high ticket volume and account scale).
  - Segment C: **$-1.6709$** (heavily distressed along the friction/SLA dimension).

---

## Part 4 findings

Part 4 evaluates non-linear decision tree splits, ensemble voting, boosting theory, bootstrap sampling, and deep neural architectures.

### 1. Decision-Tree Split Quality
Evaluated on 560 training records ($N=560, P=119$):
- Root node impurity: `root_entropy = 0.7462`, `root_gini = 0.3347`.
- **Candidate Split Comparison**:
  - S1 ($X_6 = 1$): Info Gain $= 0.0110$, Gini Gain $= 0.0052$.
  - S2 ($X_7 = 1$): Info Gain $= 0.0033$, Gini Gain $= 0.0015$.
  - S3 ($X_3 > 0.5$): Info Gain $= 0.1046$, Gini Gain $= 0.0494$.
  - **S4 ($X_1 \le 24$)**: Info Gain **$= 0.1252$**, Gini Gain **$= 0.0606$**.
- S4 is the strongest candidate because partitioning on customer tenure $\le 24$ months maximizes class separation (positive rate of $45.74\%$ on the Yes branch vs. $8.87\%$ on the No branch).
- *Overfitting & Regularization*: A fully grown tree recursively partitions data until leaves are pure, capturing idiosyncrasies and noise in the training set. Tree growth must be regularized using pre-pruning stopping rules such as maximum tree depth (`max_depth`) or minimum samples per leaf (`min_samples_leaf`).

### 2. Decision Stump and Voting Comparison
Evaluated on 240 test records (from `outputs/part4_voting.csv`):

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| **Logistic** | **0.4634** | 0.7451 | **0.5714** |
| **KNN** | 0.4324 | 0.3137 | 0.3636 |
| **Stump** | 0.4333 | **0.7647** | 0.5532 |
| **Hard vote** | 0.4430 | 0.6863 | 0.5385 |
| **Soft vote** | 0.4235 | 0.7059 | 0.5294 |

- *Finding*: Voting ensembles did not improve F1 score over standalone logistic regression on this test partition. Logistic regression remains the strongest choice for overall F1 balance ($0.5714$), while the single-split decision stump achieves slightly higher raw recall ($0.7647$).

### 3. AdaBoost from Scratch
Evaluated over 2 rounds on the 8-point benchmark:
- **Round 1**: Base learner $h_1$ misclassifies points 3 and 7.
  - Weighted error: $\epsilon_1 = 0.2500$.
  - Learner weight: $\alpha_1 = 0.5493$.
  - Post-round normalized weights: Points 3 and 7 each receive weight $0.2500$ (combined normalized weight $= 0.5000$), while correctly classified points receive weight $\approx 0.0833$.
- **Round 2**: Base learner $h_2$ achieves lower weighted error $\epsilon_2 = 0.1667$ on updated weights, receiving a higher weight $\alpha_2 = 0.8047$.
- **Ensemble Accuracy**: Final ensemble $H = \text{sign}(\alpha_1 h_1 + \alpha_2 h_2)$ achieves **$75.00\%$ accuracy**.

### 4. Out-of-Bag (OOB) Fraction
$$\text{oob\_fraction} = \left(1 - \frac{1}{560}\right)^{560} = \mathbf{0.3676}$$
In bootstrap resampling with replacement, the probability that any given record is omitted from a bootstrap sample approaches $e^{-1} \approx 0.3679$. For $n=560$, exactly $36.76\%$ of samples are left out, serving as an internal validation partition.

### 5. Neural Network Parameter Counts
*Note: Neural networks were not trained; calculations represent theoretical layer and parameter sizing.*

#### Model A — MLP ($784 \to 16 \to 16 \to 10$)
- Hidden Layer 1: $784 \times 16 + 16 = 12560$ parameters.
- Hidden Layer 2: $16 \times 16 + 16 = 272$ parameters.
- Output Layer: $16 \times 10 + 10 = 170$ parameters.
- **Total MLP Parameters**: **$13,002$**.

#### Model A — CNN ($28 \times 28 \times 1$)

| Layer | Output shape | Parameters |
|---|---|---:|
| Input | 28×28×1 | 0 |
| Conv1 | 28×28×6 | 156 |
| MaxPool1 | 14×14×6 | 0 |
| Conv2 | 10×10×16 | 2,416 |
| MaxPool2 | 5×5×16 | 0 |
| Flatten | 400 | 0 |
| Dense1 | 32 | 12,832 |
| Dense2 | 10 | 330 |

- **Total CNN Parameters**: **$15,734$**.

#### Convolution Scaling Comparison ($128 \times 128 \times 3$ Input)
- **Fully Connected Layer** (to 1024 neurons): $(128 \times 128 \times 3) \times 1024 + 1024 = \mathbf{50,332,672}$ parameters.
- **Convolutional Layer** (64 filters of $3\times3$): $(3 \times 3 \times 3 + 1) \times 64 = \mathbf{1,792}$ parameters.
- *Insight*: Convolutions reduce parameter burden by over $28,000\times$ by enforcing local receptive fields and translation-invariant weight sharing.

#### Model B — Recurrent Neural Network (RNN)
Input dimension $= 64$, Hidden dimension $= 128$, Output classes $= 16$:
- $W_{xh}$: Shape $128 \times 64 \implies 8,192$ parameters.
- $W_{hh}$: Shape $128 \times 128 \implies 16,384$ parameters.
- $W_{hy}$: Shape $16 \times 128 \implies 2,048$ parameters.
- $b_h$: Shape $128 \times 1 \implies 128$ parameters.
- $b_y$: Shape $16 \times 1 \implies 16$ parameters.
- **Total RNN Parameters**: **$26,768$**.
- *Independence of sequence length*: RNN parameter counts do not grow with sequence length $T$ because the transition matrices ($W_{xh}, W_{hh}$) and biases are identically reused at each recurrent time step.

#### Architectural Activation & Training Principles
- Standard choices: **ReLU** is standard for hidden layers in MLPs and CNNs to mitigate saturation; **$\tanh$** maintains bounded recurrent hidden state dynamics ($[-1, 1]$) in RNNs; **Softmax** combined with categorical cross-entropy provides normalized probability distributions for 10-class classification.
- *Vanishing Gradients*: In deep plain feedforward architectures, backpropagated gradients repeatedly scaled by weight matrices and derivative factors decay exponentially toward zero, making early-layer learning difficult. Residual/skip connections help mitigate vanishing gradients and improve gradient flow by providing an identity shortcut $x + \mathcal{F}(x)$, allowing gradient signals to flow directly through deep layers without attenuation.

---

## Model-choice memo

**To:** Head of Customer-Support Operations  
**From:** Lead Decision Intelligence Team  
**Subject:** Production Support Escalation Model Recommendation

We recommend deploying **Logistic Regression** for Swiggy's production support decision intelligence.

On the 240-record test partition evaluated at the Part 4.2 benchmark threshold of 0.3, logistic regression achieved the highest overall F1 score of 0.5714 with 0.7451 recall (capturing 38 of 51 escalations). The decision stump yielded slightly higher recall (0.7647) but lower F1 (0.5532), while voting ensembles failed to outperform standalone logistic regression (hard vote F1 = 0.5385; soft vote F1 = 0.5294).

While Part 4.2 compared models at threshold 0.3, our Part 2.8 cost analysis selected a cost-sensitive threshold of 0.15 through training-data cost minimization rather than as a proven production policy. Under Swiggy's asymmetric operational losses (₹50 per false alarm versus ₹400 per missed escalation), this threshold achieved an observed test cost of ₹6,050 on this particular split. This represents a 48.7% cost reduction compared to default threshold 0.5 (₹11,800) and a 70.3% saving compared to flagging nobody (₹20,400).

An operational limitation is that the dataset is synthetic and evaluated on a single stratified split; performance should be monitored across rolling operational windows prior to full automation.

Regarding deep neural architectures: we recommend building neither Model A (image-based MLP/CNN) nor Model B (sequence-based RNN) before deploying structured-data production. The current support tickets are tabular operational records, making the tested logistic model a more direct, transparent, and computationally lightweight foundation. Neural models may be revisited if unstructured ticket screenshots or sequential conversational turn transcripts become operationally integrated.

---

## Limitations

1. **Synthetic Dataset Scope**: The underlying dataset was generated via synthetic operational equations and may not fully reflect real-world Swiggy support interactions, unmeasured customer behaviors, or human agent variability.
2. **Single Stratified Split Evaluation**: Model selection and threshold tuning rely on a single 70/30 stratified train/test split. Although stratified, performance estimates may vary across alternative resamplings or live time partitions.
3. **Observational Associations versus Causation**: Regression coefficients, odds ratios, and PCA loadings indicate statistical associations within observational data and must not be interpreted as direct causal mechanisms.
4. **Absence of Production & Temporal Validation**: The models have not been evaluated against live streaming support traffic, seasonal holiday spikes, regional monsoon disruptions, or temporal concept drift.
5. **Simplified Cost Matrix Assumptions**: The cost optimization framework assumes constant penalties of ₹50 per false positive and ₹400 per false negative, whereas real customer lifetime values and compensation costs vary dynamically across account tiers.
6. **Sensitivity of Clustering to Scaling and Initialization**: K-Means and hierarchical clustering outcomes depend directly on the chosen sample standard deviation scaling and deterministic initial centroid selections.
