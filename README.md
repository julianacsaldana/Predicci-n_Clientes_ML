## Customer Churn Prediction – Model Fitness

##  Project Description
This project predicts customer churn for a gym chain using supervised and unsupervised Machine Learning techniques. The goal is to identify which customers are most likely to cancel their service, enabling the implementation of retention strategies before losing them.
Machine Learning project combining supervised and unsupervised learning to predict gym customer churn and identify behavioral segments using Logistic Regression, Random Forest, and K-Means clustering.

---
## 🎯 Objectives
- Predict customer churn using classification models
- Segment customers by behavioral profile using clustering
- Identify the key variables driving churn
- Generate actionable recommendations to reduce churn rate

---
##  Problem
A gym chain needed to understand and predict which customers are likely to cancel their membership (churn). Without a predictive model, the business had no way to proactively intervene before losing customers.

The goal was to:

- Predict churn probability for each customer using supervised ML models.
- Segment customers into behavioral profiles using unsupervised clustering.
- Identify which features most strongly influence a customer's decision to leave.

## 🛠️ Tools and Libraries

| Library | Usage |
|---|---|
| `Pandas` | Data manipulation and analysis |
| `NumPy` | Numerical operations |
| `Scikit-learn` | Machine Learning models and preprocessing |
| `Matplotlib / Seaborn` | Data visualization |
| `SciPy` | Hierarchical clustering (whiten, pdist, linkage) |

---
## 📐 Methodology

### 1. 📊 Exploratory Data Analysis (EDA)
- Identification of null values, duplicates, and data types
- Analysis of mean values grouped by churn status
- Correlation matrix to detect multicollinearity
- Visualization of active vs churned customer distribution by contract month

### 2. 🔧 Data Preprocessing
- Checked for missing values and duplicates — none found.
- Identified binary (gender, Near_Location, Partner, etc.) and numeric columns (Age, Lifetime, Contract_period, etc.).
- Applied StandardScaler to numeric features to avoid scale-driven bias and prevent overfitting.
- Applied whitening (standard deviation normalization) before running clustering algorithms, ensuring each feature contributed equally.
<img width="876" height="750" alt="image" src="https://github.com/user-attachments/assets/f73059cb-34b0-4e46-8f0d-880d87dcff25" />

### 3. Supervised Learning – Classification Models

#### 📈 Model 1 Logistic Regression
Statistical model that estimates the probability of a binary outcome (churn or no churn) based on input variables. It finds the optimal linear boundary to separate both classes.

```python
model = LogisticRegression(random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
| Metric | Value |
|--------|-------|
| Accuracy | 93% |
| Precision (active clients) | 94% |
| Recall (active clients) | 97% |
| MAE | 7% |
| MSE | 0.07 |
| R² Score | 62% |

### Confusion Matrix
<img width="374" height="314" alt="image" src="https://github.com/user-attachments/assets/2395c0d1-c896-4938-97eb-5176ddee9b0b" />


| | Predicted Active | Predicted Churned |
|------------------|------------------|-------------------|
| **Actually Active** | ✅ 580 (True Positives) | 18 |
| **Actually Churned** | ❌ 162 (False Negatives) | — |

Interpretation

- The model correctly identifies **580 active customers** and catches most retention cases with **97% recall**.
- **162 false negatives** — churned customers that the model missed — represent the main area for improvement.
- **MAE of 7%** indicates that predicted probabilities are close to actual outcomes on average.
- **R² of 62%** means the model explains 62% of the variance in customer behavior; the remaining 38% is driven by factors not captured in the dataset.
  
---

#### 🌲 Random Forest Model 2 — Random Forest

```python
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=5,
    criterion='gini', max_features='sqrt', random_state=42
)
```
Ensemble method that builds multiple decision trees on random subsets of data and combines their predictions. Reduces overfitting and improves model generalization.

**Results:**
| Metric | Value |
|--------|-------|
| Accuracy | 90% |
| Precision (active clients) | 93% |
| Recall / F1 | 94% |
| MAE | 10% |
| MSE | 10% |
| R² Score | 47% |
### Confusion Matrix
<img width="374" height="314" alt="image" src="https://github.com/user-attachments/assets/4ddb73c8-015a-4033-9440-881d354aa620" />


| | Predicted Active | Predicted Churned |
|------------------|------------------|-------------------|
| **Actually Active** | ✅ 574 (True Positives) | 24 |
| **Actually Churned** | ❌ 145 (False Negatives) | — |

### Interpretation

- Achieved **90% accuracy** with **574 correctly identified active customers**.
- **145 false negatives** — slightly fewer than Logistic Regression in absolute terms.
- **MAE of 10%** — predictions deviate more from reality compared to Logistic Regression.
- **R² of 47%** — the model explains less than half the variance in the target, meaning more than 50% of churn behavior is driven by untracked factors.

## Model Comparison

| Metric | Logistic Regression | Random Forest | Winner |
|--------|---------------------|---------------|--------|
| Accuracy | 93% | 90% | ✅ LR |
| Precision | 94% | 93% | ✅ LR |
| Recall | 97% | 94% | ✅ LR |
| MAE | 7% | 10% | ✅ LR |
| R² Score | 62% | 47% | ✅ LR |
| False Negatives | 162 | 145 | ✅ RF |


### 4. 🔍 Model Comparison Unsupervised Learning – Customer Segmentation 

#### 📉 Hierarchical Clustering (Dendrogram)
Ward and Average methods with Euclidean distance were used to explore the natural clustering structure before applying K-Means.
<img width="834" height="529" alt="image" src="https://github.com/user-attachments/assets/2f6e43dc-ac7f-4d37-b3c5-7ed1a42b9bc3" />


#### 🎯 K-Means
Customers were grouped into **5 behavioral segments**:

| Cluster | Churn Rate | Contract Length | Attendance | Social Ties | Profile |
|---------|------------|-----------------|------------|-------------|---------|
| Cluster 0 | 2% | Long (1.72) | Good | High Partner & Promo | 🟢 Stable client |
| Cluster 1 | 84% | Very short (0.33) | Low | Weak | 🔴 High-risk client |
| Cluster 2 | 19% | Short–medium | Acceptable | Average | 🟡 Moderate risk |
| Cluster 3 | 1% | Long | Very high (2.55–2.69) | Very strong | 💎 VIP / Loyal client |
| Cluster 4 | 4% | Average | Very high monthly | Good | 🟢 Long-standing client |

### Key Cluster Insights

- **Cluster 3 (1% churn)** is the gym's most valuable segment — high-frequency visitors with strong social connections and long contracts. These customers are almost impossible to lose.
- **Cluster 1 (84% churn)** is the critical at-risk group — short contracts, low attendance, and weak engagement. These customers represent the primary churn driver.
- **Cluster 2 (19% churn)** is a convertible segment — with targeted actions (promotions, group classes, contract upgrades), these customers could be moved toward Cluster 0 or 3.
- **Clusters 0 and 4 (2–4% churn)** are stable and loyal, driven by consistent attendance and longer membership tenure.
---

## 📊 Key Findings
- **Logistic Regression is the best-performing model**  With 93% accuracy, 94% precision, and only 7% MAE, it provides reliable predictions for identifying which customers are likely to stay or leave. It outperforms Random Forest across all key metrics except false negatives.

- **Random Forest catches more churners**  Despite lower overall accuracy (90%), it produces 145 false negatives vs. 162 for Logistic Regression — meaning it misses fewer customers who are about to leave. This trade-off is relevant if the business priority is proactive churn intervention.

- **Contract length is the strongest retention driver**  Customers on long contracts (6–12 months) show churn rates as low as 1–2%, while those on monthly contracts reach up to 84% churn. Promoting longer contract commitments is the single highest-impact retention strategy.

- **Attendance frequency is a leading churn indicator**  Cluster 3 customers visit an average of 2.55–2.69 times per week and almost never churn. Cluster 1 customers have low and declining attendance before canceling — making frequency a key early-warning signal.

- **Social integration reduces churn significantly**  Customers with Partner affiliation and Promo_friends participation cluster into the lowest-churn groups. Group visit programs and referral promotions are effective retention tools.

- **Seasonality drives churn patterns** → A notable spike in churn occurs at the end of June (mid-year), aligning with the transition between gym activity semesters. Targeted re-engagement campaigns before this window could reduce seasonal drop-off.

- **62% of churn behavior is explained by current features**  The R² of the best model leaves 38% of variance unexplained. Incorporating additional data — such as customer satisfaction scores, payment history, or app engagement — could significantly improve predictive power.




---

## ✅ Recommendations
- Offer incentives to convert monthly contracts into longer-term plans
- Design reactivation campaigns targeting Cluster 1 customers
- Monitor class frequency as an early warning sign of churn risk
- Prioritize retention efforts in the first 3 months of the customer lifecycle
