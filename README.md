<img width="1200" height="420" alt="predictive-insight-engine-header-animated" src="https://github.com/user-attachments/assets/d7ac7691-d839-41a1-bde5-2f7d4f7d96d1" />

---

## 🎯 Objective

The purpose of this project is to demonstrate **complete, practical knowledge of Supervised Learning regression techniques** — walking the full path from raw housing data to a compared, diagnosed and business-ready predictive model.

Every stage is deliberate: **understand → split → model (simple, multiple, polynomial) → evaluate → optimize with gradient descent → diagnose bias/variance → report.** Nothing is skipped, and every technique is followed by a 🎯 *Insight* note explaining what it revealed and whether it was the right call.

---

## 📄 Problem Statement

You are hired as a **Junior Data Scientist** working on a real-estate analytics team. The company holds a **House Price dataset** of 4,200 properties and wants a model that can predict a house's market price from its physical and locational attributes.

Your manager asks you to build and compare multiple regression approaches, optimize them with gradient descent from scratch, diagnose overfitting/underfitting, and deliver a final report recommending which model the business should use.

The dataset contains:

- **Physical attributes** — area, bedrooms, bathrooms, lot size, age.
- **Locational attributes** — location score, distance to city.
- **Amenities** — garage, pool, renovation history.
- **Target variable** — House Price (₹).

---

# 📂 Project Files

| 📄 File / Folder | 📌 Description |
|------------------|----------------|
| 📓 `Supervised_Learning_1_fixed.ipynb` | Main notebook — the complete, annotated regression & gradient-descent pipeline |
| 📊 `RealEstate_HousePrice_Dataset_4200.csv` | Raw housing dataset (4,200 records) |
| 📑 `House_Price_Prediction_Final_Analysis_Report.docx` | Final analysis report — best model, gradient descent impact, bias–variance diagnostics, business interpretation |
| 📘 `README.md` | Project documentation and workflow guide |

---

## 🛠️ Tools Used

<div>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Gradient--Descent-Batch%20%7C%20SGD%20%7C%20Mini--Batch-EC4899?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Cross--Validation-KFold-059669?style=for-the-badge"/>

</div>

---

## 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/Watch%20Demo-Add%20Your%20Link-blue?style=for-the-badge&logo=googledrive&logoColor=white)](#)

📹 Add a link to your project walkthrough video here.

---

### 🧬 Dataset Structure — House Price Dataset

| Field Name | Data Type | Description | Notes |
|------------|-----------|-------------|-------|
| `house_id` | Integer | Unique identifier for each property | Not used as a model feature |
| `area_sqft` | Integer | House area in square feet | Model feature — used in all three regressions |
| `bedrooms` | Integer | Number of bedrooms | Model feature — used in MLR |
| `bathrooms` | Integer | Number of bathrooms | Model feature — used in MLR |
| `location_score` | Float | Locational desirability score | Model feature — strongest MLR coefficient |
| `age_years` | Integer | Age of the property in years | Model feature — negative effect on price |
| `distance_city_km` | Float | Distance from city center (km) | Present in raw data, not used as a model feature |
| `lot_size_sqft` | Integer | Total lot size in square feet | Present in raw data, not used as a model feature |
| `has_garage` | Binary Int | Whether the property has a garage | Present in raw data, not used as a model feature |
| `has_pool` | Binary Int | Whether the property has a pool | Present in raw data, not used as a model feature |
| `renovation_years_ago` | Integer | Years since last renovation | Present in raw data, not used as a model feature |
| `house_price_inr` | Integer | House price (₹) | 🎯 **Target variable** |

---

## 🧠 Part B : Dataset Understanding & Preparation

### 7️⃣ Identify independent and dependent variables

```python
independent_variables = [
    'area_sqft', 'bedrooms', 'bathrooms', 'location_score', 'age_years'
]
dependent_variable = 'house_price_inr'
```

💡 **Insight:** Area, bedrooms, bathrooms, location score and age are the independent variables; house price (INR) is the dependent variable. All 5 features are numeric, so no encoding is needed. 🎯

---

### 8️⃣ Visualize relationships between features and target variable

```python
for feature in features:
    sns.scatterplot(data=data, x=feature, y=target, color="salmon")
    plt.title(f"{feature} vs House Price")
    plt.show()
```

💡 **Insight:** Scatter plots show area and location score rise clearly with price, while age declines with price — confirming all 5 features carry real signal. 📊

---

### 9️⃣ Split the dataset into training and testing sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

💡 **Insight:** The 80/20 split gives 3,360 training and 840 testing rows — enough data on both sides for reliable evaluation. 🔀

---

## 📊 Part C : Simple Linear Regression

### 🔟 Implement Simple Linear Regression using House Area

```python
slr_model = LinearRegression()
slr_model.fit(X_train_slr, y_train_slr)
```

**Output:** Coefficient (Slope): 14,788.31 · Intercept: −1,163,519.18

💡 **Insight:** Slope ≈ ₹14,788 per sq.ft — every extra square foot adds about ₹14.8K to the predicted price. 📐

---

### 1️⃣1️⃣ Plot the regression line and interpret slope & intercept

```python
plt.scatter(X_test_slr['area_sqft'], y_test_slr, label='Actual Data', color="pink")
plt.plot(X_sorted['area_sqft'], y_sorted, label='Regression Line')
```

**Regression Equation:** House Price = −1,163,519.18 + (14,788.31 × Area)

💡 **Insight:** The intercept (−₹11.6 lakh) has no practical meaning — a 0 sq.ft house doesn't exist; it only anchors the line. The slope is the meaningful number. 📏

---

### 1️⃣2️⃣ Validate linear regression assumptions using plots

```python
residuals = y_test_slr - y_pred_slr
sns.scatterplot(x=y_pred_slr, y=residuals, color="plum")
plt.axhline(y=0, linestyle='--', color="red")
```

💡 **Insight:** Residual plots show scattered, roughly normal errors — linearity and equal-variance assumptions hold reasonably, so linear regression is valid here. ✅

---

## 📈 Part D : Model Evaluation Metrics

### 1️⃣3️⃣ Evaluate Simple Linear Regression

```python
mse  = mean_squared_error(y_test_slr, y_pred_slr)
mae  = mean_absolute_error(y_test_slr, y_pred_slr)
rmse = np.sqrt(mse)
r2   = r2_score(y_test_slr, y_pred_slr)
```

**Output:** MAE ₹62,94,594 · RMSE ₹81,84,697 · R² 0.5625 · Adjusted R² 0.5620

💡 **Insight:** R² = 0.56, MAE ≈ ₹63 lakh — area alone leaves ~44% of price variation unexplained. 📉

---

### 1️⃣4️⃣ Interpret each metric

- **MSE** — average of squared errors; punishes big mistakes more.
- **MAE** — average error in rupees; how wrong the average prediction is.
- **RMSE** — square root of MSE, back in rupee units, more sensitive to large errors than MAE.
- **R² Score** — how much price variation is explained by the feature(s) used.

---

## 📉 Part E : Multiple Linear Regression

### 1️⃣5️⃣ Implement Multiple Linear Regression using all relevant features

```python
X_mlr = data[['area_sqft', 'bedrooms', 'bathrooms', 'location_score', 'age_years']]
mlr_model = LinearRegression()
mlr_model.fit(X_train_mlr, y_train_mlr)
```

**Coefficients:** area ₹13,991 · bedrooms ₹2,05,468 · bathrooms ₹1,92,233 · location score ₹33,85,623 · age −₹67,608

💡 **Insight:** Location score has the biggest coefficient (+₹33.9 lakh per point) and age is negative (−₹67.6K per year) — location matters more than size. 📍

---

### 1️⃣6️⃣ Compare its performance with Simple Linear Regression

| Metric | Simple Linear Regression | Multiple Linear Regression |
|--------|---------------------------|------------------------------|
| MSE | 6.70 × 10¹³ | 1.28 × 10¹³ |
| MAE | ₹62,94,594 | ₹26,41,320 |
| RMSE | ₹81,84,697 | ₹35,84,509 |
| R² | 0.5625 | 0.9161 |
| Adjusted R² | 0.5620 | 0.9156 |

💡 **Insight:** MLR clearly beats SLR — R² jumps 0.56 → 0.916 and RMSE falls from ₹82 lakh to ₹36 lakh. 🚀

---

### 1️⃣7️⃣ Explain why performance improves

💡 **Insight:** Performance improves because house price does not depend only on area — location score, bathrooms, bedrooms and age also affect price. Giving the model all these features reduces error and raises R², and the increase in Adjusted R² confirms the added features carry genuine signal, not noise.

---

## 🔁 Part F : Polynomial Regression

### 1️⃣8️⃣ Implement Polynomial Regression (Degree 2)

```python
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_poly)
```

**Coefficients:** [0, 15,819.79, −0.30] · Intercept: −19,32,924.64

💡 **Insight:** The squared term's coefficient is tiny (−0.30), meaning the curve barely bends. 〰️

---

### 1️⃣9️⃣ Compare linear vs polynomial regression visually and numerically

| Metric | Simple Linear Regression | Polynomial Regression |
|--------|---------------------------|------------------------|
| MSE | 6.699 × 10¹³ | 6.696 × 10¹³ |
| MAE | ₹62,94,594 | ₹62,92,395 |
| RMSE | ₹81,84,697 | ₹81,83,089 |
| R² | 0.5625 | 0.5627 |

💡 **Insight:** Linear and polynomial results are nearly identical (R² 0.5625 vs 0.5627) — the area–price relationship is genuinely linear; extra complexity buys nothing. ➖

---

### 2️⃣0️⃣ Identify signs of overfitting or underfitting

**Output:** Training R² 0.5725 · Testing R² 0.5627 · Difference 0.0098

💡 **Insight:** Train R² (0.572) ≈ Test R² (0.563), gap < 0.01 — no overfitting, but both being low signals slight underfitting (high bias). ⚖️

---

## ⚙️ Part G : Gradient Descent Optimization

### 2️⃣1️⃣ Explain Gradient Descent conceptually

Gradient Descent is an optimization algorithm used to find the parameter values that minimize the loss function. It starts with initial parameters and repeatedly updates them in the direction that reduces error, with the learning rate controlling step size: **initialize → predict → compute loss → compute gradient → update parameters → repeat.**

---

### 2️⃣2️⃣ – 2️⃣4️⃣ Implement Batch, Stochastic and Mini-Batch Gradient Descent from scratch

```python
X_scaled = (X_gd - X_gd.mean()) / X_gd.std()
y_scaled = (y_gd - y_gd.mean()) / y_gd.std()

for epoch in range(epochs):
    error = (beta0 + beta1 * X_scaled) - y_scaled
    beta0 -= learning_rate * np.mean(error)
    beta1 -= learning_rate * np.mean(error * X_scaled)
```

| Method | Final MSE | Insight |
|--------|-----------|---------|
| 🟦 **Batch GD** | 0.4293 | Smoothest convergence, lowest final MSE, but slow — one update per full dataset pass |
| 🟨 **Stochastic GD** | 0.4411 | Reaches low error in very few epochs, but the loss curve is noisy and total time is highest |
| 🟩 **Mini-Batch GD** | 0.4293 | Matches Batch GD's final MSE with far less time than SGD — best balance of speed and stability |

---

### 2️⃣5️⃣ Compare convergence behavior and training time

```python
epochs_compare = 100  # fair, equal-time comparison
```

| Method | Final MSE | Training Time (100 epochs) |
|--------|-----------|------------------------------|
| Batch GD | 0.5073 | 0.036 s |
| Stochastic GD | 0.4314 | 1.992 s |
| Mini-Batch GD | 0.4293 | 0.753 s |

💡 **Insight:** Batch GD = fastest per epoch but needs more epochs; SGD = fast convergence, slowest overall; Mini-Batch = the practical winner. 🏆

---

## 🔍 Part H : Bias–Variance & Model Diagnostics

### 2️⃣6️⃣ Analyze bias and variance across all three models

```python
cv_scores = cross_val_score(model, X, y, cv=KFold(n_splits=5), scoring='r2')
```

| Model | Training R² | Mean CV R² | CV R² Std |
|-------|--------------|------------|-----------|
| Simple Linear Regression | 0.5707 | 0.5693 | 0.0101 |
| Multiple Linear Regression | 0.9211 | 0.9206 | 0.0033 |
| Polynomial Regression | 0.5708 | 0.5692 | 0.0099 |

💡 **Insight:** SLR and Polynomial show high bias (CV R² ≈ 0.57) but low variance (CV std ≈ 0.01). MLR shows low bias **and** low variance (CV R² ≈ 0.921, std ≈ 0.003) — it is the best-balanced model for this data. ⚖️

---

### 2️⃣7️⃣ How model complexity affects prediction error

A model that is **too simple** (one feature) has high bias — both training and testing error stay high (underfitting). As complexity increases (more features, polynomial terms), bias falls and training error keeps dropping. If the model becomes **too complex**, it starts learning training noise — variance rises, training error keeps falling but testing error rises (overfitting). Total error is minimized at the balance point between the two.

---

### 2️⃣8️⃣ Identify the model with the best bias–variance balance

| Model | Training R² | Mean CV R² | Train–CV Gap |
|-------|--------------|------------|----------------|
| Simple Linear Regression | 0.5707 | 0.5693 | 0.00134 |
| Multiple Linear Regression | 0.9211 | 0.9206 | **0.00047** |
| Polynomial Regression | 0.5708 | 0.5692 | 0.00160 |

💡 **Insight:** Multiple Linear Regression has both the best validation performance and the smallest train–CV gap — the most consistent, best-generalizing model in this project. 🥇

---

## 📊 Part I : Final Analysis & Reporting

### 2️⃣9️⃣ Final Report Summary

| ❓ Question | ✅ Answer |
|-------------|----------|
| Best-performing model and why? | **Multiple Linear Regression** — R² 0.916 vs ~0.563 for SLR/Polynomial, with much lower RMSE and MAE, because it uses area, bedrooms, bathrooms, location score and age instead of area alone. |
| Impact of Gradient Descent optimization? | Batch GD converged smoothly to MSE 0.4293; Mini-Batch GD matched it with a much better speed/stability balance; SGD converged fast per-epoch but was noisiest and slowest overall. |
| Evidence of overfitting / underfitting? | SLR and Polynomial Regression underfit (R² ≈ 0.57); Polynomial showed no significant overfitting (tiny train/test gap); MLR generalized best (Train–CV gap 0.00047). |
| Practical business interpretation? | House price depends on more than area — location, bedrooms, bathrooms and age all add predictive information, so a real-estate business can use MLR as a supporting tool for price estimation and property comparison. |
| Final conclusion? | Multiple Linear Regression is the best model overall; Mini-Batch Gradient Descent is the best optimizer for accuracy, speed and stability. |

---

## 📂 Project Workflow

1. **Dataset Understanding** → Identify independent/dependent variables, visualize relationships
2. **Train/Test Split** → 80/20 split (3,360 / 840 rows)
3. **Simple Linear Regression** → Fit on area alone, plot regression line, validate assumptions
4. **Model Evaluation** → MSE, MAE, RMSE, R², Adjusted R²
5. **Multiple Linear Regression** → Fit on all 5 relevant features, compare against SLR
6. **Polynomial Regression** → Degree-2 fit on area, compare vs linear, check overfitting
7. **Gradient Descent Optimization** → Batch, Stochastic and Mini-Batch GD from scratch, compare convergence & time
8. **Bias–Variance Diagnostics** → Cross-validation R², Train–CV gap, best-balanced model
9. **Final Reporting** → Best model, GD impact, over/underfitting evidence, business interpretation, conclusion

---

## 📈 Results & Insights

- ✅ **Three regression models** built and compared — Simple Linear, Multiple Linear, Polynomial (Degree 2)
- ✅ **Multiple Linear Regression** selected as best model — **R² 0.916**, RMSE ₹35.8 lakh, MAE ₹26.4 lakh
- ✅ **Three gradient descent methods** implemented from scratch — Batch, Stochastic, Mini-Batch
- ✅ **Mini-Batch Gradient Descent** identified as the practical winner — matches Batch GD's MSE (0.4293) at a fraction of SGD's training time
- ✅ **Bias–variance diagnostics** confirm MLR has the smallest Train–CV gap (0.00047) of all three models
- ✅ Final report delivered with model comparison, optimization impact, diagnostics and business interpretation

---

## 📌 Expected Outcomes

- Understand how to **plan and execute a complete supervised-learning regression workflow**
- Build and compare **Simple, Multiple and Polynomial Regression** models with intent, not habit
- Implement **Gradient Descent optimization from scratch** and reason about the speed/stability trade-off
- Diagnose **bias, variance, overfitting and underfitting** using train/test and cross-validation metrics
- Translate model results into a **practical business recommendation**

---

## ⚙️ Installation & Setup

```bash
# clone the repository
git clone https://github.com/yourusername/house-price-supervised-learning.git
cd house-price-supervised-learning

# create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter

# launch the notebook
jupyter notebook Supervised_Learning_1_fixed.ipynb
```

---

## 🚀 Future Scope

- [ ] Add regularized regression (Ridge, Lasso, ElasticNet) as further baselines
- [ ] Engineer features from `distance_city_km`, `lot_size_sqft`, `has_garage`, `has_pool` and `renovation_years_ago`
- [ ] Hyperparameter-tune Mini-Batch Gradient Descent (learning rate, batch size)
- [ ] Deploy the trained MLR model behind a simple prediction API

---

## 🙏 Thank You

Thank you for taking the time to explore this project!
Your feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to **star the repository** and share it with others.
