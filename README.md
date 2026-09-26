# US Treasury Yield Curve Forecasting & Return Engine

A quantitative pipeline that forecasts the US Treasury yield curve (3M-30Y) using 30+ years of macroeconomic and market data, benchmarks five forecasting approaches against a random-walk baseline under a rigorous walk-forward protocol, and converts those forecasts into bond prices and holding-period returns via a full-repricing engine, with an interactive dashboard on top.

## Motivation

Forecasting interest rates one month ahead is famously hard: the random walk is the benchmark nearly every model fails to beat in the literature. This project treats that as the actual research question rather than assuming otherwise. Every model is checked for whether its apparent RMSE edge is statistically real (Diebold-Mariano, multiple-testing corrected) before being called a "win."

## Pipeline

| Notebook | What it does |
|---|---|
| `01_preprocessing.ipynb` | Loads 18 FRED series (6 yields plus 12 macro indicators), aligns mixed frequencies (daily/weekly/monthly/quarterly to monthly), handles missing values, saves a clean panel. |
| `02_eda_feature_engineering.ipynb` | Exploratory analysis, seasonality testing (Kruskal-Wallis plus seasonal ACF, confirming no seasonal cycle in yields and ruling out SARIMA), ACF/PACF diagnostics, PCA on yield changes (level/slope/curvature), VIF-based multicollinearity screening, feature engineering with publication-lag shifts to prevent look-ahead leakage. |
| `03_stationarity.ipynb` | ADF and KPSS tests confirm all six yields are I(1); differenced series feed directly into ARIMA's `d` parameter and the VAR-in-differences specification. |
| `04_forecasting.ipynb` | Walk-forward (expanding window, one-step-ahead) benchmark: Random Walk, ARIMA, ARIMAX (with macro exogenous regressors), and VAR-in-differences. Orders identified once on the training window only. |
| `05_dns.ipynb` | Dynamic Nelson-Siegel (Diebold-Li 2006): collapses the 6-yield curve into level, slope, and curvature factors, forecasts them via AR(1) and VAR(1), rebuilds the curve. Also tests whether DNS's advantage over the random walk grows at longer horizons (1, 3, 6, 12 months). |
| `06_return_engine.ipynb` | Converts forecasted yields into bond prices and returns via full repricing. Zero-coupon core (price = 100/(1+y)^T) plus a coupon-bond layer (real carry via accrued coupon income), decomposed into price, roll, and carry. Includes directional-accuracy evaluation: did the forecast get the sign of the return right, not just a lower RMSE. |
| `dashboard/app.py` | Streamlit dashboard: yield-curve forecasts (single maturity plus full curve snapshot) and the return/pricing engine, both driven by the pre-computed parquet outputs above. |

## Models benchmarked

Random Walk, ARIMA, ARIMAX, VAR (differenced), Dynamic Nelson-Siegel (AR and VAR variants).

## Evaluation protocol

- Walk-forward, expanding window, one-step-ahead. Every model is refit at each step on data up to time t and forecasts t+1.
- Model orders (ARIMA p,q; VAR lag) identified once, on the initial training window only, never re-selected inside the walk-forward loop, to avoid look-ahead.
- RMSE and MAE per maturity, plus Diebold-Mariano significance tests (Harvey-Leybourne-Newbold small-sample correction) with FDR correction across the six maturities.
- Residual diagnostics (Ljung-Box, Engle's ARCH) run both in-sample and on out-of-sample forecast errors.

## Results

**By RMSE (h=1, test window October 2020 to September 2025):**

| Maturity | Best model | RMSE | Random Walk | Improvement |
|---|---|---|---|---|
| 3M | DNS-VAR | 0.157 | 0.229 | 31.5% |
| 1Y | ARIMA | 0.228 | 0.257 | 11.2% |
| 2Y | VAR-diff | 0.303 | 0.307 | 1.4% |
| 5Y | DNS-AR | 0.304 | 0.308 | 1.2% |
| 10Y | DNS-AR | 0.265 | 0.281 | 5.6% |
| 30Y | Random Walk | 0.249 | 0.249 | 0.0% |

**Statistical significance: the honest finding.** After Diebold-Mariano testing with FDR correction across maturities, none of the apparent RMSE gains above are statistically significant at one-month-ahead. This is consistent with the term-structure forecasting literature: beating the random walk one-step-ahead is genuinely difficult, and a correctly-hedged negative result here is more informative than an overstated win.

**Where the structure actually pays off: longer horizons.** Extending DNS-VAR's forecast horizon from 1 to 12 months shows its RMSE advantage over the random walk growing monotonically, turning negative (DNS wins) at nearly every maturity by h=12, reproducing the classic Diebold-Li result on this dataset.

## Return / pricing engine

Two return components are computed via full repricing (not a linear approximation):
- **Price return**: value change from the curve moving.
- **Roll-down**: value earned purely from the bond aging down a sloped curve, even with no yield change (present in both the zero and coupon layers; often the dominant term at the short-to-belly end).
- **Carry**: coupon income. Zero for the zero-coupon core; a real, computed component in the coupon-bond layer (fixed rate, at-par, or a custom rule).

Duration and convexity are reported as a diagnostic check on the price-return leg, not used to compute returns. Full repricing is the ground truth throughout.

## Limitations & next steps

- Point forecasts only; no calibrated interval/fan-chart forecasts (motivated by the out-of-sample ARCH diagnostics, which show volatility clustering mostly present in-sample but not out-of-sample).
- Coupon accrual is modeled linearly over the holding period (no ACT/360 vs 30/360 day-count distinction), a reasonable simplification for a monthly holding period, not appropriate for daily P&L attribution.
- DGS30 is excluded from the return engine and CV-facing model comparison (scope decision), though it remains in the stationarity/forecasting notebooks for completeness.
- Natural extensions: GARCH errors on the mean-model residuals; a simple curve-positioning backtest (steepener/flattener) evaluated on realized P&L rather than RMSE alone.

## Running it

```bash
pip install -r requirements.txt
# Run notebooks 01 through 06 in order (each saves parquet outputs the next one reads)
streamlit run dashboard/app.py
```

## Data

FRED series: `DGS3MO, DGS1, DGS2, DGS5, DGS10, DGS30, CPIAUCSL, CPILFESL, FEDFUNDS, UNRATE, GDPC1, INDPRO, RSAFS, HOUST, ICSA, UMCSENT, VIXCLS, DCOILWTICO`. Monthly, 1992 to 2025.

## Reference

Ayliffe, K. and Rubin, T. "A Quantitative Comparison of Yield Curve Models in the MINT Economies." EPFL Infoscience. http://infoscience.epfl.ch/record/279314
