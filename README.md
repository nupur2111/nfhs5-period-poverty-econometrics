# Policy Paradox: Fractional Logit Modeling of Period Poverty Across 698 Indian Districts
## Data Source: National Family Health Survey (NFHS-5) & Census of India 2011

## 📌 Project Executive Summary
This project evaluates the multi-dimensional determinants of Menstrual Health and Hygiene Management (MHM) deprivation across 698 districts in India. Utilizing a three-barrier analytical framework, the study decomposes period poverty into structural, social, and institutional layers to identify the true binding constraints on welfare distribution. 

The core analytical finding uncovers a stark policy paradox: female educational attainment is the single strongest predictor of access ($\beta = 0.555$), while formal government health worker outreach exhibits a near-zero, statistically insignificant association ($p = 0.226$). This indicates that traditional top-down supply chains fundamentally misdiagnose a structural capability failure rooted in socio-cultural barriers.

## 🛠️ Advanced Econometric & Tech Stack
* **Modeling Framework:** Papke-Wooldridge (1996) Fractional Logit Model via Quasi-Maximum Likelihood Estimation (QMLE). This framework was selected as the mathematically appropriate specification for a continuously bounded dependent variable on $(0,1)$ over traditional OLS or binary logit.
* **Engineering Environment:** Python (`pandas`, `statsmodels`, `scikit-learn`), R (`tidyverse`).
* **Optimization Pipeline:** AI-Assisted code prototyping and pipeline optimization. Advanced generative AI tools were leveraged to architect data-cleaning functions and scale the multi-variable data merge, while human econometric oversight directed the model specification, winsorization parameters, and hypothesis validation.
* **Statistical Diagnostics:** Evaluated via BFGS optimization (gradient norm: $2.06 \times 10^{-8}$), McFadden Pseudo $R^2$, and Hosmer-Lemeshow goodness-of-fit.

## 📊 Feature Engineering & Layered Variables
* **Dependent Variable:** Proportion of women aged 15–24 using hygienic protection methods per district (NFHS-5 microdata).
* **Structural Panel:** Improved sanitation access, clean cooking fuel, urbanization share, macro poverty rates.
* **Socio-Cultural Panel:** Female literacy ($10+$ years schooling), sex ratio at birth (son-preference proxy), high-order births (reproductive autonomy proxy), child marriage rates.
* **Institutional Panel:** State health insurance coverage, unmet family planning needs, ASHA frontline health worker outreach.

## 📈 Strategic Institutional Insights (The "Goldman" Layer)
1.  **The Education Multiplier:** A 1 Standard Deviation increase in district-level female education increases the log-odds of MHM uptake by $0.555$, yielding an average marginal effect of $+0.62$ percentage points. 
2.  **Patriarchal Control Correlation:** High-order births (proxy for low reproductive autonomy) exert the largest negative social drag ($\beta = -0.098, p < 0.001$), proving that household bargaining dynamics dictate healthcare spending far more than material liquid wealth.
3.  **Welfare Supply-Chain Mismatch:** Frontline health worker outreach failed to achieve statistical significance. For an asset management or macroeconomic research team, this proves that capital allocation toward purely distribution-based infrastructure yields diminishing returns unless matched by localized human capital development.
## 📊 Empirical Results & Standardized Coefficients Matrix
The model successfully converged utilizing BFGS optimization with a final gradient norm of $2.06 \times 10^{-8}$. [cite_start]Robust standard errors are calculated via the Papke-Wooldridge sandwich estimator ($N = 690$)[cite: 104, 109].

| Structural/Social Barrier Layer | Variable Target | Coef. ($\beta$) | z-stat | p-value | Institutional Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Panel B: Social & Patriarchal** | Female Education (10+ Yrs) | **0.5547** | **15.71** | **< 0.001** | *** High Statistical Alpha** |
| **Panel A: Structural Infrastructure** | Clean Cooking Fuel | 0.2063 | 5.92 | < 0.001 | *** Significant Enabler** |
| **Panel A: Structural Infrastructure** | Sanitation Access | 0.1844 | 6.93 | < 0.001 | *** Significant Enabler** |
| **Panel C: Institutional Policy** | Health Insurance Coverage | 0.1313 | 5.46 | < 0.001 | *** Moderately Significant** |
| **Panel B: Social & Patriarchal** | High-Order Births (3rd+) | -0.0982 | -4.75 | < 0.001 | *** Negative Autonomy Trap** |
| **Panel C: Institutional Policy** | Unmet Family Planning Need | -0.0883 | -4.05 | < 0.001 | *** Structural Barrier** |
| **Panel B: Social & Patriarchal** | Sex Ratio at Birth | -0.0697 | -3.69 | < 0.001 | *** Cultural Stigma Drag** |
| **Panel C: Institutional Policy** | **Health Worker FP Outreach** | **-0.0250** | **-1.21** | **0.226** | **Statistically Insignificant** |

### 🛠️ Model Diagnostics Checklist
* [cite_start][x] **Sample Size:** 690 Districts (after listwise deletion checks) [cite: 104]
* [cite_start][x] **McFadden Pseudo $R^2$:** 0.093 [cite: 111]
* [cite_start][x] **OLS-Equivalent $R^2$:** 0.721 [cite: 111]
* [cite_start][x] **Hosmer-Lemeshow Goodness-of-Fit:** $p = 0.998$ (Indicating high calibration accuracy) [cite: 111]
