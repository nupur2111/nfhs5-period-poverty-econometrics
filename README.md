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
