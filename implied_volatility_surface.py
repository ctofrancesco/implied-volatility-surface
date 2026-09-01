"""Build an implied-volatility surface from listed option chains.

The program:
1. downloads the spot price and option chains across several target maturities;
2. filters observations by liquidity, relative spread, and intrinsic value;
3. keeps out-of-the-money options only: calls for K >= F and puts for K < F;
4. solves for implied volatility by inverting Black-Scholes;
5. interpolates each smile on a common log-moneyness grid;
6. displays a 3D surface, maturity skews, and an ATM term structure in Spyder.

The risk-free rate is a user-supplied constant and dividends are not modeled
explicitly. Results are therefore intended for exploratory research rather
than production valuation.
"""

from __future__ import annotations

import argparse
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm


# ===========================================================================
# DATA DOWNLOAD
# ===========================================================================

def get_yfinance_ticker(ticker: str):
    """
    Return a yfinance Ticker object.

    Raise an explanatory message when yfinance is not installed.
    """
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "yfinance is missing. Install it with:\n"
            "python -m pip install -r requirements.txt"
        ) from exc

    return yf.Ticker(ticker)


def is_valid_price(value: object) -> bool:
    """Return whether a value represents a positive finite price."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return False

    return bool(np.isfinite(price) and price > 0)


def get_spot_price(ticker: str) -> float:
    """
    Download a valid spot price.

    Attempt order:
    1. latest valid close over the previous 30 days;
    2. last_price from fast_info;
    3. previous_close from fast_info.

    This prevents a final NaN row from propagating through later calculations.
    """
    yf_ticker = get_yfinance_ticker(ticker)

    # -----------------------------------------------------------------------
    # Attempt 1: latest valid daily close
    # -----------------------------------------------------------------------
    try:
        history = yf_ticker.history(
            period="1mo",
            interval="1d",
            auto_adjust=False,
        )

        if not history.empty and "Close" in history.columns:
            valid_closes = pd.to_numeric(
                history["Close"],
                errors="coerce",
            )

            valid_closes = (
                valid_closes
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if not valid_closes.empty:
                spot = float(valid_closes.iloc[-1])

                if is_valid_price(spot):
                    return spot

    except Exception as exc:
        print(
            "    Warning: historical-series download failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # -----------------------------------------------------------------------
    # Attempts 2 and 3: fast_info
    # -----------------------------------------------------------------------
    try:
        fast_info = yf_ticker.fast_info

        for field_name in ("last_price", "previous_close"):
            try:
                candidate = fast_info[field_name]
            except (KeyError, TypeError, AttributeError):
                candidate = getattr(fast_info, field_name, np.nan)

            if is_valid_price(candidate):
                return float(candidate)

    except Exception as exc:
        print(
            "    Warning: fast_info is unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    raise ValueError(
        f"Unable to obtain a valid spot price for {ticker}. Historical data "
        "and fast_info returned missing or non-numeric values."
    )


def select_target_expiries(
    available_expiries: list[str],
    today: dt.date,
    target_days: list[int],
) -> list[str]:
    """
    Select the available expiry closest to each target horizon.

    For example, target_days = [7, 14, 30, 60, 90] selects expiries distributed
    across a relatively broad term structure.
    """
    if not available_expiries:
        raise ValueError(
            "yfinance returned no available expiry."
        )

    parsed_expiries = []

    for expiry in available_expiries:
        try:
            expiry_date = dt.datetime.strptime(
                expiry,
                "%Y-%m-%d",
            ).date()
        except ValueError:
            continue

        if expiry_date > today:
            parsed_expiries.append((expiry, expiry_date))

    if not parsed_expiries:
        raise ValueError(
            "No valid future expiry was found."
        )

    selected: list[str] = []

    for target in target_days:
        target_date = today + dt.timedelta(days=target)

        closest_expiry, _ = min(
            parsed_expiries,
            key=lambda pair: abs(
                (pair[1] - target_date).days
            ),
        )

        if closest_expiry not in selected:
            selected.append(closest_expiry)

    selected.sort(
        key=lambda expiry: dt.datetime.strptime(
            expiry,
            "%Y-%m-%d",
        )
    )

    return selected


def download_option_chain(
    ticker: str,
    expiries: list[str],
) -> pd.DataFrame:
    """
    Download calls and puts for each expiry and combine them in one DataFrame.
    """
    yf_ticker = get_yfinance_ticker(ticker)
    frames: list[pd.DataFrame] = []

    for expiry in expiries:
        try:
            chain = yf_ticker.option_chain(expiry)
        except Exception as exc:
            print(
                f"    Warning: unable to download expiry {expiry}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        calls = chain.calls.copy()
        calls["option_type"] = "call"

        puts = chain.puts.copy()
        puts["option_type"] = "put"

        combined = pd.concat(
            [calls, puts],
            ignore_index=True,
        )

        combined["expiry"] = expiry
        frames.append(combined)

    if not frames:
        raise ValueError(
            "No option chain could be downloaded."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ===========================================================================
# DATA CLEANING
# ===========================================================================

def clean_option_data(
    options: pd.DataFrame,
    spot: float,
    today: dt.date,
    max_relative_spread: float,
    min_open_interest: int,
    allow_stale_price: bool,
) -> pd.DataFrame:
    """
    Filter illiquid options and calculate:

    - mid price;
    - spread relativo;
    - time to expiry;
    - intrinsic value;
    - time value.

    Filters:
    - valid spot price;
    - positive bid and ask;
    - ask greater than or equal to bid;
    - relative spread within the threshold;
    - positive volume or sufficient open interest;
    - price at least 0.02 above intrinsic value.
    """
    if not is_valid_price(spot):
        raise ValueError(
            f"Invalid spot price received by the cleaning function: "
            f"{spot}. Intrinsic value cannot be calculated."
        )

    if options.empty:
        raise ValueError(
            "The options DataFrame is empty."
        )

    required_columns = {
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "volume",
        "openInterest",
        "option_type",
        "expiry",
    }

    missing_columns = required_columns.difference(options.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns in the option chain: "
            + ", ".join(sorted(missing_columns))
        )

    result = options.copy()

    numeric_columns = [
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "volume",
        "openInterest",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # A quote book is valid only if:
    # - bid > 0;
    # - ask > 0;
    # - ask >= bid.
    has_valid_book = (
        (result["bid"] > 0)
        & (result["ask"] > 0)
        & (result["ask"] >= result["bid"])
    )

    quoted_mid = (
        result["bid"] + result["ask"]
    ) / 2.0

    if allow_stale_price:
        result["mid_price"] = np.where(
            has_valid_book,
            quoted_mid,
            result["lastPrice"],
        )

        book_ok = (
            has_valid_book
            | (
                pd.notna(result["lastPrice"])
                & (result["lastPrice"] > 0)
            )
        )
    else:
        result["mid_price"] = quoted_mid
        book_ok = has_valid_book

    result["relative_spread"] = np.nan

    result.loc[
        has_valid_book,
        "relative_spread",
    ] = (
        (
            result.loc[has_valid_book, "ask"]
            - result.loc[has_valid_book, "bid"]
        )
        / result.loc[has_valid_book, "mid_price"]
    )

    if allow_stale_price:
        spread_ok = (
            ~has_valid_book
            | (
                result["relative_spread"]
                <= max_relative_spread
            )
        )
    else:
        spread_ok = (
            has_valid_book
            & (
                result["relative_spread"]
                <= max_relative_spread
            )
        )

    has_liquidity = (
        result["volume"].fillna(0) > 0
    ) | (
        result["openInterest"].fillna(0)
        >= min_open_interest
    )

    result["expiry_date"] = pd.to_datetime(
        result["expiry"],
        errors="coerce",
    ).dt.date

    result["time_to_expiry"] = result["expiry_date"].apply(
        lambda expiry_date: (
            max(
                (expiry_date - today).days,
                1,
            )
            / 365.0
            if pd.notna(expiry_date)
            else np.nan
        )
    )

    intrinsic = np.where(
        result["option_type"] == "call",
        np.maximum(
            spot - result["strike"],
            0.0,
        ),
        np.maximum(
            result["strike"] - spot,
            0.0,
        ),
    )

    result["intrinsic_value"] = intrinsic
    result["time_value"] = (
        result["mid_price"]
        - result["intrinsic_value"]
    )

    min_time_value = 0.02

    price_above_intrinsic = (
        result["time_value"]
        > min_time_value
    )

    finite_values = (
        np.isfinite(result["strike"])
        & np.isfinite(result["mid_price"])
        & np.isfinite(result["time_to_expiry"])
    )

    valid = (
        book_ok
        & finite_values
        & (result["strike"] > 0)
        & (result["mid_price"] > 0)
        & (result["time_to_expiry"] > 0)
        & has_liquidity
        & spread_ok
        & price_above_intrinsic
    )

    return result.loc[valid].copy()


# ===========================================================================
# OTM SELECTION AND MONEYNESS
# ===========================================================================

def select_otm_options(
    options: pd.DataFrame,
    spot: float,
    rate: float,
    moneyness_band: float,
) -> pd.DataFrame:
    """
    Calculate the forward price and retain only:

    - OTM calls when K >= F;
    - OTM puts when K < F.

    Also apply a moneyness band around K/F = 1.
    """
    if options.empty:
        return options.copy()

    result = options.copy()

    result["forward"] = spot * np.exp(
        rate * result["time_to_expiry"]
    )

    result["moneyness"] = (
        result["strike"]
        / result["forward"]
    )

    result["log_moneyness"] = np.log(
        result["moneyness"]
    )

    is_otm_call = (
        (result["option_type"] == "call")
        & (
            result["strike"]
            >= result["forward"]
        )
    )

    is_otm_put = (
        (result["option_type"] == "put")
        & (
            result["strike"]
            < result["forward"]
        )
    )

    result = result.loc[
        is_otm_call | is_otm_put
    ].copy()

    lower_bound = 1.0 - moneyness_band
    upper_bound = 1.0 + moneyness_band

    result = result.loc[
        result["moneyness"].between(
            lower_bound,
            upper_bound,
            inclusive="both",
        )
    ].copy()

    return result


# ===========================================================================
# BLACK-SCHOLES AND IMPLIED VOLATILITY
# ===========================================================================

def bs_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str,
) -> float:
    """
    Return the Black-Scholes price of a European call or put.
    """
    if (
        not is_valid_price(spot)
        or strike <= 0
        or volatility <= 0
        or time_to_expiry <= 0
    ):
        return np.nan

    sqrt_time = np.sqrt(time_to_expiry)

    d1 = (
        np.log(spot / strike)
        + (
            rate
            + 0.5 * volatility**2
        )
        * time_to_expiry
    ) / (
        volatility * sqrt_time
    )

    d2 = (
        d1
        - volatility * sqrt_time
    )

    discounted_strike = (
        strike
        * np.exp(
            -rate * time_to_expiry
        )
    )

    if option_type == "call":
        return (
            spot * norm.cdf(d1)
            - discounted_strike * norm.cdf(d2)
        )

    if option_type == "put":
        return (
            discounted_strike * norm.cdf(-d2)
            - spot * norm.cdf(-d1)
        )

    return np.nan


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str,
) -> float:
    """
    Find the implied volatility that reproduces the observed price.
    """
    if (
        not np.isfinite(market_price)
        or market_price <= 0
        or not is_valid_price(spot)
        or strike <= 0
        or time_to_expiry <= 0
    ):
        return np.nan

    def price_difference(
        volatility: float,
    ) -> float:
        model_price = bs_price(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            volatility=volatility,
            option_type=option_type,
        )

        return model_price - market_price

    try:
        lower_value = price_difference(1e-4)
        upper_value = price_difference(5.0)

        if not (
            np.isfinite(lower_value)
            and np.isfinite(upper_value)
        ):
            return np.nan

        if lower_value * upper_value > 0:
            return np.nan

        return brentq(
            price_difference,
            1e-4,
            5.0,
            maxiter=200,
            xtol=1e-10,
        )

    except (
        ValueError,
        RuntimeError,
        OverflowError,
        FloatingPointError,
    ):
        return np.nan


def compute_implied_vols(
    options: pd.DataFrame,
    spot: float,
    rate: float,
) -> pd.DataFrame:
    """
    Calculate implied volatility for each selected OTM option.
    """
    if options.empty:
        return options.copy()

    result = options.copy()

    result["implied_vol"] = result.apply(
        lambda row: implied_volatility(
            market_price=float(row["mid_price"]),
            spot=spot,
            strike=float(row["strike"]),
            time_to_expiry=float(row["time_to_expiry"]),
            rate=rate,
            option_type=str(row["option_type"]),
        ),
        axis=1,
    )

    result = result.dropna(
        subset=["implied_vol"]
    ).copy()

    result = result.loc[
        (
            result["implied_vol"] > 0
        )
        & (
            result["implied_vol"] < 1.0
        )
    ].copy()

    return result


# ===========================================================================
# SURFACE INTERPOLATION
# ===========================================================================

def build_interpolated_surface(
    options: pd.DataFrame,
    min_days_for_surface: int,
    grid_points: int = 21,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Interpolate each smile on a common log-moneyness grid.

    Expiries with fewer days remaining than the configured threshold
    are excluded from the 3D surface.
    """
    long_enough = options.loc[
        options["time_to_expiry"] * 365.0
        >= min_days_for_surface
    ].copy()

    if long_enough.empty:
        raise ValueError(
            "No expiry exceeds the minimum day threshold "
            "for the 3D surface."
        )

    expiries = sorted(
        long_enough["expiry"].unique(),
        key=lambda expiry: dt.datetime.strptime(
            expiry,
            "%Y-%m-%d",
        ),
    )

    valid_smiles: list[
        tuple[str, pd.DataFrame]
    ] = []

    for expiry in expiries:
        smile = long_enough.loc[
            long_enough["expiry"] == expiry,
            [
                "log_moneyness",
                "implied_vol",
                "time_to_expiry",
            ],
        ].copy()

        # If several contracts share the same moneyness,
        # use the median implied volatility.
        smile = (
            smile.groupby(
                "log_moneyness",
                as_index=False,
            )
            .agg(
                implied_vol=("implied_vol", "median"),
                time_to_expiry=("time_to_expiry", "first"),
            )
            .sort_values("log_moneyness")
        )

        if len(smile) >= 4:
            valid_smiles.append(
                (expiry, smile)
            )

    if not valid_smiles:
        raise ValueError(
            "No expiry has at least four distinct "
            "log-moneyness points."
        )

    # Intersect the observed ranges to avoid extrapolation
    # and large NaN regions.
    common_left = max(
        smile["log_moneyness"].min()
        for _, smile in valid_smiles
    )

    common_right = min(
        smile["log_moneyness"].max()
        for _, smile in valid_smiles
    )

    if common_left >= common_right:
        raise ValueError(
            "The smiles do not share a common "
            "log-moneyness range."
        )

    log_moneyness_grid = np.linspace(
        common_left,
        common_right,
        grid_points,
    )

    # Explicitly insert the ATM point log(K/F)=0 when it lies
    # inside the common range.
    if common_left <= 0.0 <= common_right:
        log_moneyness_grid = np.unique(
            np.append(
                log_moneyness_grid,
                0.0,
            )
        )

    times: list[float] = []
    iv_rows: list[np.ndarray] = []

    for _, smile in valid_smiles:
        interpolator = interp1d(
            smile["log_moneyness"].to_numpy(),
            smile["implied_vol"].to_numpy(),
            kind="linear",
            bounds_error=True,
        )

        interpolated_iv = interpolator(
            log_moneyness_grid
        )

        if not np.all(
            np.isfinite(interpolated_iv)
        ):
            continue

        times.append(
            float(
                smile["time_to_expiry"].iloc[0]
            )
        )

        iv_rows.append(
            np.asarray(
                interpolated_iv,
                dtype=float,
            )
        )

    if not iv_rows:
        raise ValueError(
            "No expiry produced a valid interpolated row."
        )

    times_array = np.asarray(
        times,
        dtype=float,
    )

    iv_matrix = np.vstack(
        iv_rows
    )

    sort_order = np.argsort(
        times_array
    )

    return (
        log_moneyness_grid,
        times_array[sort_order],
        iv_matrix[sort_order],
    )


# ===========================================================================
# VISUALIZATION
# ===========================================================================

def plot_smooth_surface(
    log_moneyness_grid: np.ndarray,
    times: np.ndarray,
    iv_matrix: np.ndarray,
    ticker: str,
) -> None:
    """
    Plot the interpolated 3D surface.
    """
    grid_x, grid_y = np.meshgrid(
        log_moneyness_grid,
        times,
    )

    fig = plt.figure(
        figsize=(12, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot_surface(
        grid_x,
        grid_y,
        iv_matrix,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )

    ax.set_title(
        f"Interpolated Implied Volatility Surface - {ticker}"
    )

    ax.set_xlabel(
        "Log-moneyness ln(K/F)"
    )

    ax.set_ylabel(
        "Time to expiry (years)"
    )

    ax.set_zlabel(
        "Implied volatility"
    )

    fig.tight_layout()
    plt.show()


def plot_skew(
    options: pd.DataFrame,
    ticker: str,
) -> None:
    """
    Plot the OTM option skew for each expiry.
    """
    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for expiry, group in options.groupby("expiry"):
        group_sorted = group.sort_values(
            "moneyness"
        )

        ax.plot(
            group_sorted["moneyness"],
            group_sorted["implied_vol"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=expiry,
        )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_title(
        f"Volatility Skew (OTM options) - {ticker}"
    )

    ax.set_xlabel(
        "Moneyness K/F"
    )

    ax.set_ylabel(
        "Implied volatility"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()
    plt.show()


def plot_term_structure(
    options: pd.DataFrame,
    ticker: str,
) -> None:
    """
    Plot the term structure by interpolating ATM IV for each expiry.

    The ATM point corresponds to:

        log(K/F) = 0
    """
    atm_rows: list[dict[str, object]] = []

    for expiry, group in options.groupby("expiry"):
        smile = (
            group[
                [
                    "log_moneyness",
                    "implied_vol",
                ]
            ]
            .groupby(
                "log_moneyness",
                as_index=False,
            )["implied_vol"]
            .median()
            .sort_values(
                "log_moneyness"
            )
        )

        if len(smile) < 4:
            continue

        minimum_log_moneyness = float(
            smile["log_moneyness"].min()
        )

        maximum_log_moneyness = float(
            smile["log_moneyness"].max()
        )

        if not (
            minimum_log_moneyness
            <= 0.0
            <= maximum_log_moneyness
        ):
            continue

        interpolator = interp1d(
            smile["log_moneyness"].to_numpy(),
            smile["implied_vol"].to_numpy(),
            kind="linear",
            bounds_error=True,
        )

        atm_iv = float(
            interpolator(0.0)
        )

        atm_rows.append(
            {
                "expiry": expiry,
                "time_to_expiry": float(
                    group["time_to_expiry"].iloc[0]
                ),
                "implied_vol": atm_iv,
            }
        )

    if not atm_rows:
        print(
            "    Warning: unable to build the ATM term structure. "
            "No smile contains the K/F = 1 point."
        )
        return

    atm_surface = (
        pd.DataFrame(atm_rows)
        .sort_values("time_to_expiry")
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        atm_surface["time_to_expiry"] * 365.0,
        atm_surface["implied_vol"],
        marker="o",
        linewidth=2,
    )

    ax.set_title(
        f"Term Structure ATM - {ticker}"
    )

    ax.set_xlabel(
        "Days to expiry"
    )

    ax.set_ylabel(
        "ATM implied volatility"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()
    plt.show()


# ===========================================================================
# ARGUMENTS AND ORCHESTRATION
# ===========================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Define command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Implied Volatility Surface Builder"
    )

    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Ticker symbol to analyze",
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=0.04,
        help="Annual risk-free rate",
    )

    parser.add_argument(
        "--target-days",
        type=int,
        nargs="+",
        default=[7, 14, 30, 60, 90],
        help=(
            "Target horizons in days used to select "
            "available expiries"
        ),
    )

    parser.add_argument(
        "--moneyness-band",
        type=float,
        default=0.10,
        help=(
            "Band around K/F=1. "
            "0.10 corresponds to the 90%%-110%% range"
        ),
    )

    parser.add_argument(
        "--min-days-for-surface",
        type=int,
        default=3,
        help=(
            "Expiries with fewer remaining days are excluded "
            "from the 3D surface"
        ),
    )

    parser.add_argument(
        "--max-relative-spread",
        type=float,
        default=0.25,
        help=(
            "Maximum allowed relative spread: "
            "(ask-bid)/mid"
        ),
    )

    parser.add_argument(
        "--min-open-interest",
        type=int,
        default=10,
        help="Minimum open interest",
    )

    parser.add_argument(
        "--allow-stale-price",
        action="store_true",
        help=(
            "Use lastPrice when bid and ask are unavailable. "
            "This is less rigorous and may introduce stale prices."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the complete pipeline.
    """
    args = parse_arguments()
    today = dt.date.today()

    # -----------------------------------------------------------------------
    # 1. Spot
    # -----------------------------------------------------------------------
    print(
        f"1/6 - Downloading the spot price for {args.ticker}..."
    )

    spot = get_spot_price(
        args.ticker
    )

    if not is_valid_price(spot):
        raise ValueError(
            f"Invalid spot price: {spot}"
        )

    print(
        f"    Spot price: {spot:.2f}"
    )

    # -----------------------------------------------------------------------
    # 2. Expiries
    # -----------------------------------------------------------------------
    print(
        "2/6 - Selecting target expiries "
        "(broad-horizon term structure)..."
    )

    yf_ticker = get_yfinance_ticker(
        args.ticker
    )

    available_expiries = list(
        yf_ticker.options
    )

    selected_expiries = select_target_expiries(
        available_expiries=available_expiries,
        today=today,
        target_days=args.target_days,
    )

    print(
        f"    Selected expiries: {selected_expiries}"
    )

    # -----------------------------------------------------------------------
    # 3. Option chain
    # -----------------------------------------------------------------------
    print(
        "3/6 - Downloading the option chain..."
    )

    raw_options = download_option_chain(
        ticker=args.ticker,
        expiries=selected_expiries,
    )

    print(
        f"    Downloaded rows (raw): {len(raw_options)}"
    )

    # -----------------------------------------------------------------------
    # 4. Cleaning
    # -----------------------------------------------------------------------
    print(
        "4/6 - Cleaning data "
        "(liquidity, spread, intrinsic value)..."
    )

    clean_options = clean_option_data(
        options=raw_options,
        spot=spot,
        today=today,
        max_relative_spread=args.max_relative_spread,
        min_open_interest=args.min_open_interest,
        allow_stale_price=args.allow_stale_price,
    )

    print(
        f"    Rows after cleaning: {len(clean_options)}"
    )

    if clean_options.empty:
        raise ValueError(
            "No option passed the quality filters. "
            "Check bid/ask, spreads, and liquidity, or rerun "
            "during US market hours. You can also try "
            "--allow-stale-price, noting that lastPrice may be stale."
        )

    # -----------------------------------------------------------------------
    # 5. OTM selection and IV
    # -----------------------------------------------------------------------
    print(
        "5/6 - Selecting OTM options "
        "and calculating implied volatility..."
    )

    otm_options = select_otm_options(
        options=clean_options,
        spot=spot,
        rate=args.rate,
        moneyness_band=args.moneyness_band,
    )

    print(
        f"    OTM rows inside the moneyness band: {len(otm_options)}"
    )

    if otm_options.empty:
        raise ValueError(
            "No option remained after OTM selection "
            "and the moneyness filter."
        )

    iv_options = compute_implied_vols(
        options=otm_options,
        spot=spot,
        rate=args.rate,
    )

    print(
        f"    Rows with valid IV: {len(iv_options)}"
    )

    if iv_options.empty:
        raise ValueError(
            "No valid implied volatility could be calculated."
        )

    print(
        "\nIV summary by expiry:"
    )

    summary = (
        iv_options
        .groupby("expiry")["implied_vol"]
        .describe()
    )

    available_summary_columns = [
        column
        for column in [
            "count",
            "mean",
            "std",
            "min",
            "50%",
            "max",
        ]
        if column in summary.columns
    ]

    print(
        summary[available_summary_columns]
    )

    # -----------------------------------------------------------------------
    # 6. Charts
    # -----------------------------------------------------------------------
    print(
        "\n6/6 - Generating charts..."
    )

    (
        log_moneyness_grid,
        times,
        iv_matrix,
    ) = build_interpolated_surface(
        options=iv_options,
        min_days_for_surface=args.min_days_for_surface,
    )

    plot_smooth_surface(
        log_moneyness_grid=log_moneyness_grid,
        times=times,
        iv_matrix=iv_matrix,
        ticker=args.ticker,
    )

    plot_skew(
        options=iv_options,
        ticker=args.ticker,
    )

    plot_term_structure(
        options=iv_options,
        ticker=args.ticker,
    )

    print(
        "\nDone. The charts are available in Spyder's Plots pane."
    )


if __name__ == "__main__":
    main()
