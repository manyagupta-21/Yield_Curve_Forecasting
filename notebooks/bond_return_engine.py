"""
Bond Return Engine — Yields to Prices to Returns
==================================================
Converts forecasted (and actual) constant-maturity Treasury yields into
zero-coupon bond prices, then decomposes one-month holding-period returns
into PRICE, CARRY, and ROLL-DOWN components via full repricing.

Design notes (matching the write-up):
  - Compounding convention: DISCRETE, annual. DGS yields are quoted on a
    bond-equivalent basis, so P = 100 / (1+y)^T is the defensible default.
    (100 * exp(-yT) is a DIFFERENT convention — continuous yield y_c =
    ln(1+y), NOT y itself. Don't conflate the two.)
  - Duration used in the Δy approximation is MODIFIED duration
    D_mod = T / (1+y), not Macaulay duration (= T). They're only equal in
    the continuous-compounding limit.
  - Zero-coupon core first (this file). Coupon-bond layer is a separate,
    later extension — see the stub at the bottom.
  - Excludes DGS30 (per your professor's suggestion / current CV scope).

Expected inputs (from your existing pipeline):
  actual_curves   : pd.DataFrame, index = monthly dates, columns = yield_col
                     (e.g. Y from 04_forecasting / 05_dns, in PERCENT, e.g. 4.25)
  forecast_curves : pd.DataFrame, same shape, your model's one-step-ahead
                     forecasts (e.g. preds["DNS_VAR"] or preds["ARIMA"])
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# ----------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------
MATURITIES = np.array([0.25, 1, 2, 5, 10])          # years
YIELD_COLS = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10"]


# ----------------------------------------------------------------------
# I. zero-coupon bond, discrete compounding
# ----------------------------------------------------------------------
def zero_price(y_pct: float, T: float, face: float = 100.0) -> float:
    """Price of a zero-coupon bond. y_pct in PERCENT (e.g. 4.25), T in years."""
    y = y_pct / 100.0
    return face / (1 + y) ** T


def zero_modified_duration(y_pct: float, T: float) -> float:
    """Modified duration of a zero: T / (1+y). (Macaulay duration = T.)"""
    y = y_pct / 100.0
    return T / (1 + y)


def zero_convexity(y_pct: float, T: float) -> float:
    """Convexity of a zero: T(T+1) / (1+y)^2."""
    y = y_pct / 100.0
    return T * (T + 1) / (1 + y) ** 2


def duration_convexity_return(y0_pct: float, y1_pct: float, T: float) -> float:
    """Second-order Taylor approximation of the return from a yield move,
    for validation against full repricing (should track closely for small Δy)."""
    D = zero_modified_duration(y0_pct, T)
    C = zero_convexity(y0_pct, T)
    dy = (y1_pct - y0_pct) / 100.0
    return -D * dy + 0.5 * C * dy ** 2


# ----------------------------------------------------------------------
# Curve interpolation: needed because "age the bond one month" moves
#    its maturity off the fixed CMT grid (e.g. 10Y -> 9Y 11M)
# ----------------------------------------------------------------------
def interp_yield(curve_row: pd.Series, target_T: float,
                  maturities: np.ndarray = MATURITIES) -> float:
    """Cubic-spline interpolate the yield curve at an arbitrary maturity."""
    order = np.argsort(maturities)
    cs = CubicSpline(maturities[order], curve_row.values[order], extrapolate=True)
    return float(cs(target_T))


# ----------------------------------------------------------------------
# Full-repricing return decomposition: PRICE + CARRY + ROLL
# ----------------------------------------------------------------------
def decompose_return(curve_t0: pd.Series, curve_t1: pd.Series,
                      T: float, dt: float = 1 / 12) -> dict:
    """
    One-period return of a zero-coupon bond bought at maturity T on curve_t0,
    held to t1 (curve_t1), decomposed into:

      total_return  = P1_actual / P0 - 1                (full repricing, ground truth)
      price_return  = value change from the curve moving,   bond NOT aged
      roll_return   = value change from the bond aging,      curve held fixed at t0
      carry_return  = 0 for a zero-coupon bond (no coupon income)
      residual      = total - (price + roll + carry)         [cross term; small]

    This additive split is a decomposition of full-repricing P&L, not a
    separate model — duration/convexity below is a *diagnostic* check on
    the price_return leg, not used to compute it.
    """
    y0 = interp_yield(curve_t0, T)
    P0 = zero_price(y0, T)

    T1 = T - dt  # bond has aged by one holding period

    # (a) full repricing — the actual realized/forecasted outcome
    y1_actual = interp_yield(curve_t1, T1)
    P1_actual = zero_price(y1_actual, T1)
    total_return = P1_actual / P0 - 1

    # (b) price-only: curve moves to t1, but pretend the bond didn't age
    y1_pricechg = interp_yield(curve_t1, T)
    P1_pricechg = zero_price(y1_pricechg, T)
    price_return = P1_pricechg / P0 - 1

    # (c) roll-only: bond ages, curve stays at t0 (nothing moves)
    y1_roll = interp_yield(curve_t0, T1)
    P1_roll = zero_price(y1_roll, T1)
    roll_return = P1_roll / P0 - 1

    carry_return = 0.0  # zero-coupon: no accrued coupon

    residual_interaction = total_return - (price_return + roll_return + carry_return)

    # not part of the decomposition itself
    dur_conv_check = duration_convexity_return(y0, y1_actual, T)

    return {
        "T": T, "y0": y0, "y1": y1_actual,
        "P0": P0, "P1": P1_actual,
        "total_return": total_return,
        "price_return": price_return,
        "roll_return": roll_return,
        "carry_return": carry_return,
        "residual_interaction": residual_interaction,
        "duration_convexity_check": dur_conv_check,
        "modified_duration": zero_modified_duration(y0, T),
        "convexity": zero_convexity(y0, T),
    }


# ----------------------------------------------------------------------
# Sanity check: a near-zero-maturity bond should behave like cash
#    (price_return -> 0, total_return dominated by roll as T shrinks)
# ----------------------------------------------------------------------
def sanity_check(curve_t0: pd.Series, curve_t1: pd.Series):
    """Run before trusting the engine on real data."""
    out = decompose_return(curve_t0, curve_t1, T=0.30, dt=1 / 12)
    print("Sanity check (T≈0.30y, ~zero-coupon 3M-ish bond):")
    for k, v in out.items():
        print(f"  {k:28s}: {v:.6f}" if isinstance(v, float) else f"  {k:28s}: {v}")
    print("  Expect: small total_return, price_return ≈ total_return, "
          "roll_return small (little curve to roll down at the short end).")


# ----------------------------------------------------------------------
# realized vs. forecast-implied returns
# ----------------------------------------------------------------------
def run_return_engine(actual_curves: pd.DataFrame,
                       forecast_curves: pd.DataFrame,
                       target_maturities=MATURITIES,
                       dt: float = 1 / 12) -> pd.DataFrame:
    """
    For every date t1 in the test period, and every target maturity T:
      - REALIZED return: curve_t0 (actual) -> curve_t1 (actual)
      - EXPECTED return : curve_t0 (actual) -> curve_t1 (FORECAST, e.g. DNS_VAR)

    Returns a long-format DataFrame you can pivot / evaluate directional
    accuracy on (did the forecast get the SIGN of the return right?).
    """
    dates = actual_curves.index
    records = []

    for i in range(len(dates) - 1):
        t0, t1 = dates[i], dates[i + 1]
        curve_t0 = actual_curves.loc[t0]
        curve_t1_actual = actual_curves.loc[t1]

        for T in target_maturities:
            rec = decompose_return(curve_t0, curve_t1_actual, T, dt)
            rec.update(date=t1, maturity=T, kind="realized")
            records.append(rec)

            if t1 in forecast_curves.index and not forecast_curves.loc[t1].isna().any():
                curve_t1_fcst = forecast_curves.loc[t1]
                rec_f = decompose_return(curve_t0, curve_t1_fcst, T, dt)
                rec_f.update(date=t1, maturity=T, kind="expected")
                records.append(rec_f)

    return pd.DataFrame.from_records(records)


# ----------------------------------------------------------------------
# Directional accuracy: did the forecast get the sign right?
# ----------------------------------------------------------------------
def directional_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """
    Merge realized vs. expected total_return per (date, maturity) and check
    sign agreement. A naive random-walk-implied forecast (zero yield change)
    predicts pure roll-down every period — compare against that baseline too.
    """
    real = results[results.kind == "realized"].set_index(["date", "maturity"])["total_return"]
    exp = results[results.kind == "expected"].set_index(["date", "maturity"])["total_return"]
    df = pd.concat([real.rename("realized"), exp.rename("expected")], axis=1).dropna()

    df["sign_match"] = np.sign(df["realized"]) == np.sign(df["expected"])

    out = (df.groupby(level="maturity")["sign_match"]
             .agg(["mean", "count"])
             .rename(columns={"mean": "directional_accuracy", "count": "n_obs"}))
    return out


# ----------------------------------------------------------------------
# II. Coupon-bond layer
#    Adds a real carry_return (coupon income) instead of the zero-coupon
#    engine's carry_return = 0. Same discrete-compounding convention as
#    zero_price(): P = CF / (1+y)^t, y in PERCENT, t in years.
# ----------------------------------------------------------------------

def coupon_bond_price(y_pct: float, coupon_pct: float, T: float,
                       freq: int = 2, face: float = 100.0) -> float:
    """
    Clean price of a fixed-coupon bond, flat-yield discounting.

    Bug check: coupon_pct=0 must reproduce zero_price(y_pct, T) exactly,
    since a zero-coupon bond is just this bond with every coupon cash
    flow set to zero. Verified in validate_coupon_engine() below.
    """
    y = y_pct / 100.0
    coupon_per_period = (coupon_pct / 100.0) * face / freq
    n_periods = round(T * freq)

    if n_periods <= 0:
        return face + coupon_per_period

    periods = np.arange(1, n_periods + 1)
    times = periods / freq
    discount_factors = (1 + y) ** (-times)

    pv_coupons = coupon_per_period * discount_factors.sum()
    pv_face = face * (1 + y) ** (-T)
    return pv_coupons + pv_face


def par_coupon_rate(y_pct: float, T: float, freq: int = 2, face: float = 100.0) -> float:
    """
    The coupon rate (in PERCENT) that makes coupon_bond_price(y_pct, ., T)
    exactly equal to `face` — i.e. a bond issued 'at par' on this curve.

    coupon_bond_price is LINEAR in coupon_pct (coupon terms just scale with
    it), so this is solved in closed form rather than by root-finding:

        face = coupon_pct/100 * face/freq * S + face * (1+y)^-T
        =>  coupon_pct = 100 * freq * (1 - (1+y)^-T) / S

    where S = sum of per-period discount factors.

    Note: because zero_price/coupon_bond_price use ANNUAL discrete
    compounding with a FRACTIONAL exponent per coupon period, i.e.
    (1+y)^(k/freq), not the more common (1+y/freq)^k, this is the exact
    par rate *for this specific convention*, not the textbook par-yield
    formula you'd get under standard semiannual-compounding bond math.
    Consistent with the convention already documented at the top of this
    module for zero_price().
    """
    y = y_pct / 100.0
    n_periods = round(T * freq)
    if n_periods <= 0:
        return 0.0

    periods = np.arange(1, n_periods + 1)
    times = periods / freq
    S = ((1 + y) ** (-times)).sum()

    return 100.0 * freq * (1 - (1 + y) ** (-T)) / S


def _resolve_coupon_pct(coupon_pct, curve_t0: pd.Series, T: float,
                         freq: int = 2, face: float = 100.0) -> float:
    """
    Turns the user-facing `coupon_pct` argument into an actual number for
    this (curve_t0, T). Accepts three forms:

      - a fixed float/int      -> used as-is for every date/maturity
      - the string "par"       -> exact par coupon rate at this t0, via
                                    par_coupon_rate() above
      - a callable(curve, T)   -> full custom control, e.g. your own
                                    par-minus-spread rule, a fixed spread
                                    over the on-the-run yield, etc.
    """
    if isinstance(coupon_pct, str):
        if coupon_pct != "par":
            raise ValueError(f"Unknown coupon_pct string option: {coupon_pct!r} "
                              f"(only 'par' is supported)")
        y0 = interp_yield(curve_t0, T)
        return par_coupon_rate(y0, T, freq, face)
    if callable(coupon_pct):
        return coupon_pct(curve_t0, T)
    return float(coupon_pct)


def coupon_bond_macaulay_duration(y_pct: float, coupon_pct: float, T: float,
                                   freq: int = 2, face: float = 100.0) -> float:
    """Macaulay duration: PV-weighted average time to each cash flow."""
    y = y_pct / 100.0
    coupon_per_period = (coupon_pct / 100.0) * face / freq
    n_periods = round(T * freq)
    if n_periods <= 0:
        return 0.0

    periods = np.arange(1, n_periods + 1)
    times = periods / freq
    cashflows = np.full(n_periods, coupon_per_period, dtype=float)
    cashflows[-1] += face

    discount_factors = (1 + y) ** (-times)
    pv_cashflows = cashflows * discount_factors
    price = pv_cashflows.sum()
    return float((times * pv_cashflows).sum() / price)


def coupon_bond_modified_duration(y_pct: float, coupon_pct: float, T: float,
                                   freq: int = 2, face: float = 100.0) -> float:
    """Modified duration = Macaulay / (1+y). Reduces to zero_modified_duration
    when coupon_pct=0 (Macaulay = T for a zero)."""
    y = y_pct / 100.0
    D_mac = coupon_bond_macaulay_duration(y_pct, coupon_pct, T, freq, face)
    return D_mac / (1 + y)


def coupon_bond_convexity(y_pct: float, coupon_pct: float, T: float,
                           freq: int = 2, face: float = 100.0) -> float:
    """Convexity: PV-weighted average of t(t+1), matching zero_convexity's
    T(T+1)/(1+y)^2 form when coupon_pct=0."""
    y = y_pct / 100.0
    coupon_per_period = (coupon_pct / 100.0) * face / freq
    n_periods = round(T * freq)
    if n_periods <= 0:
        return 0.0

    periods = np.arange(1, n_periods + 1)
    times = periods / freq
    cashflows = np.full(n_periods, coupon_per_period, dtype=float)
    cashflows[-1] += face

    discount_factors = (1 + y) ** (-times)
    pv_cashflows = cashflows * discount_factors
    price = pv_cashflows.sum()
    weighted = (times * (times + 1) * pv_cashflows).sum()
    return float(weighted / (price * (1 + y) ** 2))


def decompose_return_coupon(curve_t0: pd.Series, curve_t1: pd.Series,
                             T: float, coupon_pct, freq: int = 2,
                             dt: float = 1 / 12) -> dict:
    """
    Coupon-bond analogue of decompose_return(). Same PRICE / ROLL split via
    full repricing, but CARRY is now real coupon accrual instead of 0.

    `coupon_pct` accepts a fixed float, "par", or a callable — see
    _resolve_coupon_pct() above. It's resolved ONCE at t0 (the coupon rate
    is fixed at issuance/purchase; it doesn't change as the curve moves).

      total_return  = (P1_actual + coupon_income) / P0 - 1   (cash-flow-inclusive HPR)
      price_return  = curve moves, bond NOT aged, no coupon accrued yet
      roll_return   = bond ages, curve held at t0, no coupon accrued
      carry_return  = coupon income accrued over dt, as a fraction of P0
      residual      = total - (price + roll + carry)          [small cross term]

    Coupon accrual is modeled linearly over the holding period (accrued =
    coupon_pct/100 * face * dt), i.e. the fraction of the ANNUAL coupon
    earned during this holding period — independent of `freq`, since accrual
    is continuous-in-time regardless of how often it's actually paid out.
    This ignores day-count-convention edge cases (ACT/360 vs 30/360 etc.),
    which is a reasonable simplification for a monthly holding period.
    """
    face = 100.0
    coupon_pct_resolved = _resolve_coupon_pct(coupon_pct, curve_t0, T, freq, face)

    y0 = interp_yield(curve_t0, T)
    P0 = coupon_bond_price(y0, coupon_pct_resolved, T, freq, face)

    T1 = T - dt
    coupon_income = (coupon_pct_resolved / 100.0) * face * dt

    y1_actual = interp_yield(curve_t1, T1)
    P1_actual = coupon_bond_price(y1_actual, coupon_pct_resolved, T1, freq, face)
    total_return = (P1_actual + coupon_income) / P0 - 1

    y1_pricechg = interp_yield(curve_t1, T)
    price_return = coupon_bond_price(y1_pricechg, coupon_pct_resolved, T, freq, face) / P0 - 1

    y1_roll = interp_yield(curve_t0, T1)
    roll_return = coupon_bond_price(y1_roll, coupon_pct_resolved, T1, freq, face) / P0 - 1

    carry_return = coupon_income / P0
    residual_interaction = total_return - (price_return + roll_return + carry_return)

    return {
        "T": T, "coupon_pct": coupon_pct_resolved,
        "y0": y0, "y1": y1_actual, "P0": P0, "P1": P1_actual,
        "total_return": total_return,
        "price_return": price_return,
        "roll_return": roll_return,
        "carry_return": carry_return,
        "residual_interaction": residual_interaction,
        "modified_duration": coupon_bond_modified_duration(y0, coupon_pct_resolved, T, freq, face),
        "convexity": coupon_bond_convexity(y0, coupon_pct_resolved, T, freq, face),
    }


def run_return_engine_coupon(actual_curves: pd.DataFrame,
                              forecast_curves: pd.DataFrame,
                              coupon_pct,
                              target_maturities=MATURITIES,
                              freq: int = 2,
                              dt: float = 1 / 12) -> pd.DataFrame:
    """
    Coupon-bond analogue of run_return_engine() has the same structure, so the
    existing directional_accuracy() works unchanged on its output.

    For every date t1 in the test period and every target maturity T:
      - REALIZED return: curve_t0 (actual) -> curve_t1 (actual)
      - EXPECTED return : curve_t0 (actual) -> curve_t1 (forecast)
    priced as a coupon_pct-coupon bond (fixed rate, "par", or a callable —
    see _resolve_coupon_pct) instead of a zero.
    """
    dates = actual_curves.index
    records = []

    for i in range(len(dates) - 1):
        t0, t1 = dates[i], dates[i + 1]
        c0, c1_actual = actual_curves.loc[t0], actual_curves.loc[t1]

        for T in target_maturities:
            rec = decompose_return_coupon(c0, c1_actual, T, coupon_pct, freq, dt)
            rec.update(date=t1, maturity=T, kind="realized")
            records.append(rec)

            if t1 in forecast_curves.index and not forecast_curves.loc[t1].isna().any():
                rec_f = decompose_return_coupon(c0, forecast_curves.loc[t1], T, coupon_pct, freq, dt)
                rec_f.update(date=t1, maturity=T, kind="expected")
                records.append(rec_f)

    return pd.DataFrame.from_records(records)


# ----------------------------------------------------------------------
# Validation: coupon_pct=0 must reduce EXACTLY to the zero-coupon engine.
# Run this once before trusting the coupon layer on real data.
# ----------------------------------------------------------------------
def validate_coupon_engine(curve_t0: pd.Series, curve_t1: pd.Series,
                            T: float = 5.0, tol: float = 1e-8):
    """Bug check called for in the original stub: a zero-coupon bond priced
    through coupon_bond_price() should match zero_price() almost exactly,
    and decompose_return_coupon(coupon_pct=0) should match decompose_return()
    on every field except the (still-zero) carry_return."""
    y0 = interp_yield(curve_t0, T)

    p_zero = zero_price(y0, T)
    p_coupon_zero = coupon_bond_price(y0, coupon_pct=0.0, T=T)
    price_ok = abs(p_zero - p_coupon_zero) < tol

    out_zero = decompose_return(curve_t0, curve_t1, T)
    out_coupon = decompose_return_coupon(curve_t0, curve_t1, T, coupon_pct=0.0)

    mismatches = {
        k: (out_zero[k], out_coupon[k])
        for k in ("total_return", "price_return", "roll_return")
        if abs(out_zero[k] - out_coupon[k]) > tol
    }

    print(f"zero_price vs coupon_bond_price(coupon=0): "
          f"{'MATCH' if price_ok else 'MISMATCH'}  ({p_zero:.6f} vs {p_coupon_zero:.6f})")
    print(f"decompose_return vs decompose_return_coupon(coupon=0): "
          f"{'MATCH' if not mismatches else 'MISMATCH -> ' + str(mismatches)}")

    # Bonus check: par_coupon_rate should make P0 == face exactly.
    par_rate = par_coupon_rate(y0, T)
    p_par = coupon_bond_price(y0, par_rate, T)
    par_ok = abs(p_par - 100.0) < tol
    print(f"par_coupon_rate check: coupon={par_rate:.4f}%  ->  P0={p_par:.6f}  "
          f"{'MATCH (par)' if par_ok else 'MISMATCH'}")

    return price_ok and not mismatches and par_ok



if __name__ == "__main__":
    dates = pd.date_range("2021-01-01", periods=6, freq="MS")
    rng = np.random.default_rng(0)
    base = np.array([4.5, 4.3, 4.0, 3.8, 3.9])  # 3M,1Y,2Y,5Y,10Y
    actual = pd.DataFrame(
        base + rng.normal(0, 0.05, size=(6, 5)).cumsum(axis=0),
        index=dates, columns=YIELD_COLS
    )
    forecast = actual.shift(1).bfill()  # placeholder: can be replaced with e.g. preds["DNS_VAR"]

    sanity_check(actual.iloc[0], actual.iloc[1])
    print()

    res = run_return_engine(actual, forecast)
    print(res.groupby(["kind", "maturity"])[["total_return", "price_return", "roll_return"]].mean())
    print()

    print(directional_accuracy(res))
