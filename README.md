# Period Poverty in India: A Fractional Logit Analysis Across 690 Districts

**Data Sources:** National Family Health Survey (NFHS-5, 2019–21) · Census of India 2011

## Overview

This project models the determinants of menstrual health and hygiene (MHM) management deprivation across Indian districts using a Papke-Wooldridge (1996) fractional logit specification — the appropriate model for a continuously bounded dependent variable on (0,1).

A three-barrier framework decomposes MHM outcomes into structural (sanitation, fuel access, poverty), social (female education, child marriage, son preference, caste/religion composition), and institutional (health insurance, family planning outreach) determinants.

**Key finding:** female educational attainment is the strongest predictor of MHM access (β = 0.555, p < 0.001), while government health worker outreach shows no statistically significant relationship with outcomes (p = 0.226) — suggesting current welfare delivery may be misaligned with the actual binding constraints.

## Data & Sample

- NFHS-5 district factsheet data (698 districts) merged with Census 2011 demographic variables
- District matching used a three-tier approach: exact name match → fuzzy match within state → state-average imputation for districts created after 2011
- Final analytical sample after listwise deletion: **N = 690 districts**

## Methodology

- **Model:** Fractional logit via quasi-maximum likelihood estimation (QMLE), estimated using manual BFGS optimization (`scipy.optimize`)
- **Standard errors:** Robust sandwich (Papke-Wooldridge) estimator
- **Stack:** Python — `pandas`, `numpy`, `scipy`
- Convergence: gradient norm 2.06 × 10⁻⁸

## Results

| Variable | Layer | β | z-stat | p-value |
|---|---|---|---|---|
| Female education (10+ yrs) | Social | 0.5547 | 15.71 | <0.001 |
| Clean cooking fuel | Structural | 0.2063 | 5.92 | <0.001 |
| Sanitation access | Structural | 0.1844 | 6.93 | <0.001 |
| Health insurance coverage | Institutional | 0.1313 | 5.46 | <0.001 |
| High-order births (3rd+) | Social | -0.0982 | -4.75 | <0.001 |
| Unmet family planning need | Institutional | -0.0883 | -4.05 | <0.001 |
| Sex ratio at birth | Social | -0.0697 | -3.69 | <0.001 |
| Health worker FP outreach | Institutional | -0.0250 | -1.21 | 0.226 (n.s.) |

**Diagnostics:** OLS-equivalent R² = 0.721 · McFadden Pseudo R² = 0.093 · Hosmer-Lemeshow p = 0.998

Full 15-variable model and standardised/unstandardised coefficients available in `analysis.py`.

## Repository Structure

```
├── analysis.py                          # Full cleaning, merging, and modeling pipeline
├── merged_cleaned_analysis_dataset.csv  # Final analytical dataset (690 districts)
├── PeriodPoverty_TermPaper.md           # Full term paper with literature review and discussion
└── README.md
```

## Reproducing the Analysis

```bash
pip install pandas numpy scipy
python analysis.py
```

Requires the raw NFHS-5 district factsheet and Census 2011 district-level extract (file paths configurable at the top of `analysis.py`).

## Notes on AI Assistance

Data cleaning code and pipeline structure were developed with AI assistance (Claude); model specification, variable selection, and interpretation of results were directed and validated by the author.

## Limitations

This is a cross-sectional analysis and identifies associations, not causal relationships. See the term paper's Future Research section for proposed panel-data extensions (NFHS-4 vs NFHS-5) to address directionality.
