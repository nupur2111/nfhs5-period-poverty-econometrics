"""
=============================================================================
PERIOD POVERTY & FEMINISATION OF POVERTY IN INDIA
Full Econometric Analysis Pipeline — Fractional Logit Model
=============================================================================
Data sources:
  1. NFHS5_DistricFactsheet_Data.csv   (National Family Health Survey-5, 2019-21)
  2. india-districts-census-2011.csv   (Census of India, 2011 — district level)

Pipeline:
  1. Load & clean both datasets
  2. Harmonise district/state names (standardisation + state mapping)
  3. Fuzzy-match districts; impute post-2011 new districts with state averages
  4. Engineer Census-derived variables (SC/ST/Muslim share, urban share, etc.)
  5. Estimate Fractional Logit model (Papke-Wooldridge 1996) via QMLE/BFGS
  6. Compute robust (sandwich) standard errors
  7. Compute full goodness-of-fit metrics, AME, VIF diagnostics
  8. Print final regression equation (standardised + unstandardised)
=============================================================================
"""

import pandas as pd
import numpy as np
from difflib import get_close_matches
from scipy import stats
from scipy.special import expit, logit
from scipy.optimize import minimize

# -----------------------------------------------------------------------
# FILE PATHS — update these if your filenames differ
# -----------------------------------------------------------------------
NFHS_FILE   = "NFHS5_DistricFactsheet_Data.csv"
CENSUS_FILE = "india-districts-census-2011.csv"

# =========================================================================
# STEP 1 — LOAD DATA
# =========================================================================
nfhs   = pd.read_csv(NFHS_FILE)
census = pd.read_csv(CENSUS_FILE)

print(f"NFHS-5 shape: {nfhs.shape}")
print(f"Census 2011 shape: {census.shape}")

# =========================================================================
# STEP 2 — HELPER FUNCTIONS
# =========================================================================
def std(s):
    """Standardise district/state name strings for matching."""
    return (str(s).strip().lower()
            .replace('&', 'and').replace('-', ' ')
            .replace('(', '').replace(')', '')
            .replace('.', '').replace("'", ''))

def clean_nfhs(series):
    """Clean NFHS columns: strip brackets/asterisks, convert to numeric."""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[\(\)\*\s]', '', regex=True),
        errors='coerce'
    )

# =========================================================================
# STEP 3 — STANDARDISE NAMES FOR MERGING
# =========================================================================
nfhs['dk']   = nfhs['District Names'].apply(std)
nfhs['sk']   = nfhs['State/UT'].apply(std)
census['dk'] = census['District name'].apply(std)
census['sk'] = census['State name'].apply(std)

# State name harmonisation map: NFHS-5 (2021) state names -> Census 2011 state names
# Handles: J&K/Ladakh bifurcation, Telangana/AP bifurcation, NCT naming, spelling diffs
state_map = {
    'andaman and nicobar islands': 'andaman and nicobar islands',
    'andhra pradesh': 'andhra pradesh',
    'arunachal pradesh': 'arunachal pradesh',
    'assam': 'assam',
    'bihar': 'bihar',
    'chandigarh': 'chandigarh',
    'chhattisgarh': 'chhattisgarh',
    'dadra and nagar haveli and daman and diu': 'dadra and nagar haveli',
    'goa': 'goa',
    'gujarat': 'gujarat',
    'haryana': 'haryana',
    'himachal pradesh': 'himachal pradesh',
    'jammu and kashmir': 'jammu and kashmir',
    'ladakh': 'jammu and kashmir',          # Ladakh was part of undivided J&K in 2011
    'jharkhand': 'jharkhand',
    'karnataka': 'karnataka',
    'kerala': 'kerala',
    'lakshadweep': 'lakshadweep',
    'madhya pradesh': 'madhya pradesh',
    'maharastra': 'maharashtra',             # NFHS spelling variant
    'manipur': 'manipur',
    'meghalaya': 'meghalaya',
    'mizoram': 'mizoram',
    'nagaland': 'nagaland',
    'nct of delhi': 'delhi',
    'odisha': 'orissa',                      # Census 2011 uses old spelling
    'puducherry': 'puducherry',
    'punjab': 'punjab',
    'rajasthan': 'rajasthan',
    'sikkim': 'sikkim',
    'tamil nadu': 'tamil nadu',
    'telangana': 'andhra pradesh',           # Telangana was part of undivided AP in 2011
    'tripura': 'tripura',
    'uttar pradesh': 'uttar pradesh',
    'uttarakhand': 'uttarakhand',
    'west bengal': 'west bengal',
}
nfhs['sk2'] = nfhs['sk'].map(state_map).fillna(nfhs['sk'])

# =========================================================================
# STEP 4 — ENGINEER CENSUS 2011 VARIABLES
# =========================================================================
census = census.copy()
census['sc_pct']       = census['SC'] / census['Population'] * 100
census['st_pct']       = census['ST'] / census['Population'] * 100
census['muslim_pct']   = census['Muslims'] / census['Population'] * 100
census['urban_pct']    = census['Urban_Households'] / census['Households'] * 100
census['fem_work_pct'] = census['Female_Workers'] / census['Female'] * 100
census['poverty_pct']  = (census['Power_Parity_Less_than_Rs_45000']
                           / census['Total_Power_Parity'] * 100)

cvars = ['sc_pct', 'st_pct', 'muslim_pct', 'urban_pct', 'fem_work_pct', 'poverty_pct']

# =========================================================================
# STEP 5 — BUILD LOOKUP TABLES FOR MATCHING (exact -> fuzzy -> state-average)
# =========================================================================
census_lookup = {(r['sk'], r['dk']): {v: r[v] for v in cvars}
                  for _, r in census.iterrows()}

state_avgs = {sk: {v: grp[v].mean() for v in cvars}
              for sk, grp in census.groupby('sk')}

def find_match(dk, sk2):
    """3-tier matching: exact name match -> fuzzy match within state -> state average (for new post-2011 districts)."""
    if (sk2, dk) in census_lookup:
        return census_lookup[(sk2, dk)], 'exact'
    state_dists = [k[1] for k in census_lookup if k[0] == sk2]
    fuzzy = get_close_matches(dk, state_dists, n=1, cutoff=0.6)
    if fuzzy:
        return census_lookup[(sk2, fuzzy[0])], 'fuzzy'
    return state_avgs.get(sk2, {v: np.nan for v in cvars}), 'imputed'

# =========================================================================
# STEP 6 — MERGE NFHS-5 + CENSUS 2011 INTO ANALYSIS DATAFRAME
# =========================================================================
rows = []
for _, nrow in nfhs.iterrows():
    cdata, method = find_match(nrow['dk'], nrow['sk2'])
    r = {
        'District':        nrow['District Names'],
        'State':           nrow['State/UT'],
        'method':          method,
        # ---- Dependent variable ----
        'MH':              clean_nfhs(pd.Series([nrow[nfhs.columns[23]]])).iloc[0],   # menstrual hygiene %
        # ---- NFHS-5 independent variables ----
        'sanitation':      clean_nfhs(pd.Series([nrow[nfhs.columns[13]]])).iloc[0],   # improved sanitation
        'clean_fuel':      clean_nfhs(pd.Series([nrow[nfhs.columns[14]]])).iloc[0],   # clean cooking fuel
        'female_edu':      clean_nfhs(pd.Series([nrow[nfhs.columns[19]]])).iloc[0],   # women 10+ yrs schooling
        'sex_ratio_birth': clean_nfhs(pd.Series([nrow[nfhs.columns[8]]])).iloc[0],    # sex ratio at birth
        'high_order_b':    clean_nfhs(pd.Series([nrow[nfhs.columns[21]]])).iloc[0],   # 3rd+ order births
        'unmet_fp':        clean_nfhs(pd.Series([nrow[nfhs.columns[32]]])).iloc[0],   # unmet FP need
        'hw_outreach':     clean_nfhs(pd.Series([nrow[nfhs.columns[34]]])).iloc[0],   # health worker FP outreach
        'health_ins':      clean_nfhs(pd.Series([nrow[nfhs.columns[16]]])).iloc[0],   # health insurance coverage
        'child_marriage':  clean_nfhs(pd.Series([nrow[nfhs.columns[20]]])).iloc[0],   # child marriage rate
    }
    r.update(cdata)   # adds sc_pct, st_pct, muslim_pct, urban_pct, fem_work_pct, poverty_pct
    rows.append(r)

df = pd.DataFrame(rows).dropna()

print(f"\nFinal merged sample: {len(df)} districts")
print(f"Match methods: {df['method'].value_counts().to_dict()}")

# -------------------------------------------------------------------
# SAVE THE MERGED & CLEANED DATASET
# -------------------------------------------------------------------
output_col_order = [
    'District', 'State', 'method',
    'MH',
    'sanitation', 'clean_fuel', 'urban_pct', 'poverty_pct',
    'female_edu', 'sex_ratio_birth', 'high_order_b', 'child_marriage',
    'sc_pct', 'st_pct', 'muslim_pct', 'fem_work_pct',
    'health_ins', 'unmet_fp', 'hw_outreach',
]
df_export = df[output_col_order].rename(columns={
    'MH': 'menstrual_hygiene_pct',
    'method': 'census_match_method',
})

EXPORT_PATH = "merged_cleaned_analysis_dataset.csv"
df_export.to_csv(EXPORT_PATH, index=False)
print(f"Merged & cleaned dataset saved to: {EXPORT_PATH}")
print(f"  Rows: {len(df_export)}  |  Columns: {len(df_export.columns)}")

# =========================================================================
# STEP 7 — PREPARE MODEL VARIABLES
# =========================================================================
# Dependent variable: proportion in (0,1), winsorised to avoid log(0)/log(1) issues
Y = np.clip(df['MH'].values / 100, 1e-6, 1 - 1e-6)

feature_cols = [
    'sanitation', 'clean_fuel', 'female_edu', 'sex_ratio_birth', 'high_order_b',
    'child_marriage', 'unmet_fp', 'hw_outreach', 'health_ins',
    'sc_pct', 'st_pct', 'muslim_pct', 'urban_pct', 'fem_work_pct', 'poverty_pct'
]

X_df = df[feature_cols]

# Standardise predictors (mean=0, sd=1) for comparable coefficient magnitudes
Xm = X_df.mean()
Xs = X_df.std()
Xs[Xs == 0] = 1   # safety guard against zero-variance columns
Xn = (X_df - Xm) / Xs
X  = np.column_stack([np.ones(len(Xn)), Xn.values])   # add intercept column

# =========================================================================
# STEP 8 — FRACTIONAL LOGIT MODEL (Papke-Wooldridge, 1996)
# =========================================================================
def neg_ll(b):
    """Negative Bernoulli quasi-log-likelihood for fractional response."""
    mu = np.clip(expit(X @ b), 1e-10, 1 - 1e-10)
    return -(Y * np.log(mu) + (1 - Y) * np.log(1 - mu)).sum()

def grad(b):
    """Analytical gradient of the negative log-likelihood."""
    mu = expit(X @ b)
    return -X.T @ (Y - mu)

res = minimize(
    neg_ll, np.zeros(X.shape[1]), jac=grad,
    method='BFGS', options={'maxiter': 3000, 'gtol': 1e-7}
)
beta   = res.x
mu_hat = expit(X @ beta)

print(f"\nModel converged: {res.success}")
print(f"Final gradient norm: {np.linalg.norm(res.jac):.2e}")

# =========================================================================
# STEP 9 — ROBUST (SANDWICH) STANDARD ERRORS
# =========================================================================
resid = Y - mu_hat
H    = sum(mu_hat[i] * (1 - mu_hat[i]) * np.outer(X[i], X[i]) for i in range(len(Y)))
meat = sum((resid[i] ** 2) * np.outer(X[i], X[i]) for i in range(len(Y)))
Hinv = np.linalg.inv(H)
V    = Hinv @ meat @ Hinv
se   = np.sqrt(np.diag(V))
z    = beta / se
pv   = 2 * (1 - stats.norm.cdf(np.abs(z)))

# =========================================================================
# STEP 10 — PRINT REGRESSION TABLE
# =========================================================================
layer_map = {
    'Intercept': '-',
    'sanitation': 'STRUCTURAL', 'clean_fuel': 'STRUCTURAL',
    'urban_pct': 'STRUCTURAL', 'poverty_pct': 'STRUCTURAL',
    'female_edu': 'SOCIAL', 'sex_ratio_birth': 'SOCIAL',
    'high_order_b': 'SOCIAL', 'child_marriage': 'SOCIAL',
    'sc_pct': 'SOCIAL', 'st_pct': 'SOCIAL',
    'muslim_pct': 'SOCIAL', 'fem_work_pct': 'SOCIAL',
    'unmet_fp': 'INSTITUTIONAL', 'hw_outreach': 'INSTITUTIONAL',
    'health_ins': 'INSTITUTIONAL',
}
all_cols = ['Intercept'] + feature_cols

print(f"\n{'='*82}")
print(f"  FRACTIONAL LOGIT  |  Papke-Wooldridge (1996)  |  Robust SEs")
print(f"  DV: Menstrual Hygiene (proportion)  |  N={len(Y)}")
print(f"{'='*82}")
print(f"{'Variable':<22} {'Coef':>7} {'Rob.SE':>8} {'z':>7} {'p':>8}  {'':>4}  Layer")
print(f"{'-'*82}")
for i, col in enumerate(all_cols):
    sig = "***" if pv[i] < 0.01 else "**" if pv[i] < 0.05 else "*" if pv[i] < 0.1 else ""
    print(f"{col:<22} {beta[i]:>7.4f} {se[i]:>8.4f} {z[i]:>7.2f} {pv[i]:>8.4f}  {sig:>3}  {layer_map.get(col, '')}")

# =========================================================================
# STEP 11 — GOODNESS OF FIT METRICS
# =========================================================================
ll_full = -neg_ll(beta)
b0 = np.zeros(len(beta)); b0[0] = logit(Y.mean())
ll_null = -neg_ll(b0)

mcfadden   = 1 - ll_full / ll_null
adj_mcf    = 1 - (ll_full - len(beta)) / ll_null
nagelkerke = (1 - np.exp(-2 * (ll_full - ll_null) / len(Y))) / (1 - np.exp(2 * ll_null / len(Y)))
aic        = -2 * ll_full + 2 * len(beta)
bic        = -2 * ll_full + len(beta) * np.log(len(Y))
ols_r2     = 1 - np.sum((Y - mu_hat) ** 2) / np.sum((Y - Y.mean()) ** 2)
corr_yhat  = np.corrcoef(Y, mu_hat)[0, 1]
rmse       = np.sqrt(np.mean((Y - mu_hat) ** 2))
mae        = np.mean(np.abs(Y - mu_hat))
lr_stat    = 2 * (ll_full - ll_null)
lr_p       = 1 - stats.chi2.cdf(lr_stat, df=len(beta) - 1)

# Hosmer-Lemeshow goodness-of-fit test (10 deciles)
n_grp = 10
sidx  = np.argsort(mu_hat)
gsz   = len(Y) // n_grp
hl    = 0
for g in range(n_grp):
    idx = sidx[g * gsz:(g + 1) * gsz]
    o = Y[idx].sum(); e = mu_hat[idx].sum(); ng = len(idx)
    if e > 0 and (ng - e) > 0:
        hl += (o - e) ** 2 / e + ((ng - o) - (ng - e)) ** 2 / (ng - e)
hl_p = 1 - stats.chi2.cdf(hl, df=n_grp - 2)

print(f"\n-- Goodness of Fit -----------------------------------------")
print(f"  McFadden Pseudo R^2:        {mcfadden:.6f}")
print(f"  Adjusted McFadden R^2:      {adj_mcf:.6f}")
print(f"  Nagelkerke R^2:             {nagelkerke:.6f}")
print(f"  OLS-equivalent R^2 (Y,Yhat):  {ols_r2:.6f}")
print(f"  Corr(Y, Yhat):                {corr_yhat:.6f}")
print(f"  AIC:                       {aic:.4f}")
print(f"  BIC:                       {bic:.4f}")
print(f"  RMSE (% scale):            {rmse*100:.4f}%")
print(f"  MAE  (% scale):            {mae*100:.4f}%")
print(f"  LR chi^2({len(beta)-1}):          {lr_stat:.4f}  (p = {lr_p:.2e})")
print(f"  Hosmer-Lemeshow chi^2(8):   {hl:.4f}  (p = {hl_p:.4f})")

# =========================================================================
# STEP 12 — AVERAGE MARGINAL EFFECTS (unstandardised, % scale)
# =========================================================================
dmu = mu_hat * (1 - mu_hat)
ame = {col: np.mean(dmu * beta[j + 1] / Xs[col]) for j, col in enumerate(feature_cols)}

print(f"\n-- Average Marginal Effects (pp per 1pp increase in X) -")
for col, val in sorted(ame.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {col:<22} {val*100:>+8.4f} pp")

# =========================================================================
# STEP 13 — MULTICOLLINEARITY CHECK (VIF)
# =========================================================================
Xraw = X_df.values
vif = {}
for j, col in enumerate(feature_cols):
    yj = Xraw[:, j]
    Xj = np.delete(Xraw, j, axis=1)
    Xjc = np.column_stack([np.ones(len(Xj)), Xj])
    bj  = np.linalg.lstsq(Xjc, yj, rcond=None)[0]
    r2j = 1 - np.sum((yj - Xjc @ bj) ** 2) / np.sum((yj - yj.mean()) ** 2)
    vif[col] = 1 / (1 - r2j) if r2j < 1 else float('inf')

print(f"\n-- Multicollinearity -- VIF ---------------------------")
for col, v in sorted(vif.items(), key=lambda x: x[1], reverse=True):
    status = "HIGH" if v > 10 else "Moderate" if v > 5 else "OK"
    print(f"  {col:<22} {v:>7.3f}  {status}")

# =========================================================================
# STEP 14 — REGRESSION EQUATION (standardised + unstandardised forms)
# =========================================================================
print(f"\n-- Standardised coefficients (beta per 1 SD change) ---")
for i, col in enumerate(all_cols):
    print(f"  {col}: {beta[i]:.6f}")

print(f"\n-- Unstandardised coefficients (beta per 1 unit change) ---")
intercept_unstd = beta[0] - sum(beta[j + 1] * Xm.iloc[j] / Xs.iloc[j] for j in range(len(feature_cols)))
print(f"  Intercept (adjusted): {intercept_unstd:.6f}")
for j, col in enumerate(feature_cols):
    b_raw = beta[j + 1] / Xs.iloc[j]
    print(f"  {col}: {b_raw:.6f}  (mean={Xm[col]:.3f}, sd={Xs[col]:.3f})")

print("\n=== ANALYSIS COMPLETE ===")
