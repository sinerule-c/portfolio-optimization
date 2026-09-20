import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.optimize import minimize


# ----------------------------
# 1. Define assets and period
# ----------------------------

tickers = ["JPM", "AAPL", "XOM", "JNJ", "GLD"]

start_date = "2020-01-01"
end_date = "2026-01-01"


# ----------------------------
# 2. Download price data
# ----------------------------

data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=False
)

prices = data["Adj Close"]

print("First five rows:")
print(prices.head())

print("\nLast five rows:")
print(prices.tail())


# ----------------------------
# 3. Calculate daily returns
# ----------------------------

daily_returns = prices.pct_change().dropna()

print("\nFirst five rows of daily returns:")
print(daily_returns.head())


# ----------------------------
# 4. Calculate expected annual returns
# ----------------------------

mean_daily_returns = daily_returns.mean()

annual_returns = (
    mean_daily_returns * 252
)

print("\nExpected annual returns (%):")
print(
    (annual_returns * 100).round(2)
)


# ----------------------------
# 5. Calculate annualized volatility
# ----------------------------

daily_volatility = (
    daily_returns.std()
)

annual_volatility = (
    daily_volatility * np.sqrt(252)
)

print("\nAnnualized volatility (%):")
print(
    (annual_volatility * 100).round(2)
)


# ----------------------------
# 6. Calculate covariance matrix
# ----------------------------

daily_cov_matrix = (
    daily_returns.cov()
)

annual_cov_matrix = (
    daily_cov_matrix * 252
)

print("\nAnnualized covariance matrix:")
print(
    annual_cov_matrix.round(4)
)


# ----------------------------
# 7. Create equal-weight portfolio
# ----------------------------

number_of_assets = len(
    prices.columns
)

weights = np.array(
    [1 / number_of_assets]
    * number_of_assets
)

print("\nEqual portfolio weights (%):")

print(
    pd.Series(
        weights * 100,
        index=prices.columns
    ).round(2)
)

portfolio_return = np.dot(
    weights,
    annual_returns
)

portfolio_variance = np.dot(
    weights.T,
    np.dot(
        annual_cov_matrix,
        weights
    )
)

portfolio_volatility = np.sqrt(
    portfolio_variance
)

print("\nEqual-weight portfolio:")

print(
    f"Expected annual return: "
    f"{portfolio_return * 100:.2f}%"
)

print(
    f"Annual volatility: "
    f"{portfolio_volatility * 100:.2f}%"
)


# ----------------------------
# 8. Calculate Sharpe ratio
# ----------------------------

risk_free_rate = 0.04

sharpe_ratio = (
    portfolio_return
    - risk_free_rate
) / portfolio_volatility

print(
    f"Sharpe ratio: "
    f"{sharpe_ratio:.4f}"
)


# ----------------------------
# 9. Simulate random portfolios
# ----------------------------

number_of_portfolios = 10000

rng = np.random.default_rng(42)

portfolio_results = []

for i in range(number_of_portfolios):

    random_weights = rng.random(
        number_of_assets
    )

    random_weights = (
        random_weights
        / random_weights.sum()
    )

    random_return = np.dot(
        random_weights,
        annual_returns
    )

    random_variance = np.dot(
        random_weights.T,
        np.dot(
            annual_cov_matrix,
            random_weights
        )
    )

    random_volatility = np.sqrt(
        random_variance
    )

    random_sharpe = (
        random_return
        - risk_free_rate
    ) / random_volatility

    portfolio_results.append(
        [
            random_return,
            random_volatility,
            random_sharpe,
            *random_weights
        ]
    )


# ----------------------------
# 10. Convert simulation to DataFrame
# ----------------------------

columns = [
    "Return",
    "Volatility",
    "Sharpe"
] + list(prices.columns)

portfolio_results = pd.DataFrame(
    portfolio_results,
    columns=columns
)

print("\nFirst five simulated portfolios:")

print(
    portfolio_results
    .head()
    .round(4)
)


# ----------------------------
# 11. Find simulated maximum Sharpe
# ----------------------------

max_sharpe_index = (
    portfolio_results[
        "Sharpe"
    ].idxmax()
)

max_sharpe_portfolio = (
    portfolio_results.loc[
        max_sharpe_index
    ]
)

print(
    "\nSimulated maximum Sharpe portfolio:"
)

print(
    f"Expected annual return: "
    f"{max_sharpe_portfolio['Return'] * 100:.2f}%"
)

print(
    f"Annual volatility: "
    f"{max_sharpe_portfolio['Volatility'] * 100:.2f}%"
)

print(
    f"Sharpe ratio: "
    f"{max_sharpe_portfolio['Sharpe']:.4f}"
)

print(
    "\nSimulated maximum Sharpe weights (%):"
)

print(
    (
        max_sharpe_portfolio[
            prices.columns
        ] * 100
    ).round(2)
)


# ----------------------------
# 12. Find simulated minimum volatility
# ----------------------------

min_volatility_index = (
    portfolio_results[
        "Volatility"
    ].idxmin()
)

min_volatility_portfolio = (
    portfolio_results.loc[
        min_volatility_index
    ]
)

print(
    "\nSimulated minimum volatility portfolio:"
)

print(
    f"Expected annual return: "
    f"{min_volatility_portfolio['Return'] * 100:.2f}%"
)

print(
    f"Annual volatility: "
    f"{min_volatility_portfolio['Volatility'] * 100:.2f}%"
)

print(
    f"Sharpe ratio: "
    f"{min_volatility_portfolio['Sharpe']:.4f}"
)


# ----------------------------
# 13. Portfolio performance function
# ----------------------------

def portfolio_performance(
    weights,
    annual_returns,
    annual_cov_matrix
):

    portfolio_return = np.dot(
        weights,
        annual_returns
    )

    portfolio_variance = np.dot(
        weights.T,
        np.dot(
            annual_cov_matrix,
            weights
        )
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    return (
        portfolio_return,
        portfolio_volatility
    )


# ----------------------------
# 14. Negative Sharpe function
# ----------------------------

def negative_sharpe_ratio(
    weights,
    annual_returns,
    annual_cov_matrix,
    risk_free_rate
):

    portfolio_return, portfolio_volatility = (
        portfolio_performance(
            weights,
            annual_returns,
            annual_cov_matrix
        )
    )

    sharpe = (
        portfolio_return
        - risk_free_rate
    ) / portfolio_volatility

    return -sharpe


# ----------------------------
# 15. Volatility function
# ----------------------------

def portfolio_volatility_function(
    weights,
    annual_returns,
    annual_cov_matrix
):

    portfolio_return, portfolio_volatility = (
        portfolio_performance(
            weights,
            annual_returns,
            annual_cov_matrix
        )
    )

    return portfolio_volatility


# ----------------------------
# 16. Optimization constraints
# ----------------------------

constraints = (
    {
        "type": "eq",
        "fun": lambda weights:
            np.sum(weights) - 1
    },
)

bounds = tuple(
    (0, 1)
    for asset
    in range(number_of_assets)
)

initial_weights = np.array(
    [1 / number_of_assets]
    * number_of_assets
)


# ----------------------------
# 17. Optimize maximum Sharpe
# ----------------------------

max_sharpe_optimization = minimize(

    negative_sharpe_ratio,

    initial_weights,

    args=(
        annual_returns,
        annual_cov_matrix,
        risk_free_rate
    ),

    method="SLSQP",

    bounds=bounds,

    constraints=constraints
)

optimized_max_sharpe_weights = (
    max_sharpe_optimization.x
)

optimized_max_return, optimized_max_volatility = (
    portfolio_performance(
        optimized_max_sharpe_weights,
        annual_returns,
        annual_cov_matrix
    )
)

optimized_max_sharpe = (
    optimized_max_return
    - risk_free_rate
) / optimized_max_volatility


# ----------------------------
# 18. Optimize minimum volatility
# ----------------------------

min_volatility_optimization = minimize(

    portfolio_volatility_function,

    initial_weights,

    args=(
        annual_returns,
        annual_cov_matrix
    ),

    method="SLSQP",

    bounds=bounds,

    constraints=constraints
)

optimized_min_vol_weights = (
    min_volatility_optimization.x
)

optimized_min_return, optimized_min_volatility = (
    portfolio_performance(
        optimized_min_vol_weights,
        annual_returns,
        annual_cov_matrix
    )
)

optimized_min_sharpe = (
    optimized_min_return
    - risk_free_rate
) / optimized_min_volatility


# ----------------------------
# 19. Print optimized portfolios
# ----------------------------

print(
    "\nOptimized maximum Sharpe portfolio:"
)

print(
    f"Expected annual return: "
    f"{optimized_max_return * 100:.2f}%"
)

print(
    f"Annual volatility: "
    f"{optimized_max_volatility * 100:.2f}%"
)

print(
    f"Sharpe ratio: "
    f"{optimized_max_sharpe:.4f}"
)

print(
    "\nOptimized maximum Sharpe weights (%):"
)

print(
    pd.Series(
        optimized_max_sharpe_weights
        * 100,
        index=prices.columns
    ).round(2)
)


print(
    "\nOptimized minimum volatility portfolio:"
)

print(
    f"Expected annual return: "
    f"{optimized_min_return * 100:.2f}%"
)

print(
    f"Annual volatility: "
    f"{optimized_min_volatility * 100:.2f}%"
)

print(
    f"Sharpe ratio: "
    f"{optimized_min_sharpe:.4f}"
)

print(
    "\nOptimized minimum volatility weights (%):"
)

print(
    pd.Series(
        optimized_min_vol_weights
        * 100,
        index=prices.columns
    ).round(2)
)


# ----------------------------
# 20. Calculate efficient frontier
# ----------------------------

target_returns = np.linspace(
    optimized_min_return,
    annual_returns.max(),
    100
)

frontier_returns = []
frontier_volatilities = []

for target_return in target_returns:

    frontier_constraints = (
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1
        },
        {
            "type": "eq",
            "fun": lambda weights,
            target=target_return:
                np.dot(
                    weights,
                    annual_returns
                ) - target
        }
    )

    frontier_optimization = minimize(

        portfolio_volatility_function,

        initial_weights,

        args=(
            annual_returns,
            annual_cov_matrix
        ),

        method="SLSQP",

        bounds=bounds,

        constraints=frontier_constraints
    )

    if frontier_optimization.success:

        frontier_return, frontier_volatility = (
            portfolio_performance(
                frontier_optimization.x,
                annual_returns,
                annual_cov_matrix
            )
        )

        frontier_returns.append(
            frontier_return
        )

        frontier_volatilities.append(
            frontier_volatility
        )


# ----------------------------
# 21. Plot efficient frontier
# ----------------------------

plt.figure(
    figsize=(10, 6)
)

scatter = plt.scatter(
    portfolio_results[
        "Volatility"
    ] * 100,

    portfolio_results[
        "Return"
    ] * 100,

    c=portfolio_results[
        "Sharpe"
    ],

    cmap="viridis",

    alpha=0.35,

    s=20
)

plt.plot(
    np.array(
        frontier_volatilities
    ) * 100,

    np.array(
        frontier_returns
    ) * 100,

    linewidth=3,

    label="Efficient Frontier"
)

plt.scatter(
    optimized_max_volatility * 100,
    optimized_max_return * 100,

    marker="*",
    s=300,

    label="Optimized Maximum Sharpe"
)

plt.scatter(
    optimized_min_volatility * 100,
    optimized_min_return * 100,

    marker="*",
    s=300,

    label="Optimized Minimum Volatility"
)

plt.scatter(
    portfolio_volatility * 100,
    portfolio_return * 100,

    marker="X",
    s=180,

    label="Equal Weight"
)

plt.xlabel(
    "Annual Volatility (%)"
)

plt.ylabel(
    "Expected Annual Return (%)"
)

plt.title(
    "Efficient Frontier"
)

plt.colorbar(
    scatter,
    label="Sharpe Ratio"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "charts/efficient_frontier.png",
    dpi=300
)

plt.show()


# ----------------------------
# 22. Create portfolio comparison table
# ----------------------------

comparison_table = pd.DataFrame(
    {
        "Portfolio": [
            "Equal Weight",
            "Maximum Sharpe",
            "Minimum Volatility"
        ],

        "Expected Return (%)": [
            portfolio_return * 100,
            optimized_max_return * 100,
            optimized_min_return * 100
        ],

        "Volatility (%)": [
            portfolio_volatility * 100,
            optimized_max_volatility * 100,
            optimized_min_volatility * 100
        ],

        "Sharpe Ratio": [
            sharpe_ratio,
            optimized_max_sharpe,
            optimized_min_sharpe
        ],

        "AAPL Weight (%)": [
            weights[
                prices.columns.get_loc("AAPL")
            ] * 100,

            optimized_max_sharpe_weights[
                prices.columns.get_loc("AAPL")
            ] * 100,

            optimized_min_vol_weights[
                prices.columns.get_loc("AAPL")
            ] * 100
        ],

        "GLD Weight (%)": [
            weights[
                prices.columns.get_loc("GLD")
            ] * 100,

            optimized_max_sharpe_weights[
                prices.columns.get_loc("GLD")
            ] * 100,

            optimized_min_vol_weights[
                prices.columns.get_loc("GLD")
            ] * 100
        ],

        "JNJ Weight (%)": [
            weights[
                prices.columns.get_loc("JNJ")
            ] * 100,

            optimized_max_sharpe_weights[
                prices.columns.get_loc("JNJ")
            ] * 100,

            optimized_min_vol_weights[
                prices.columns.get_loc("JNJ")
            ] * 100
        ],

        "JPM Weight (%)": [
            weights[
                prices.columns.get_loc("JPM")
            ] * 100,

            optimized_max_sharpe_weights[
                prices.columns.get_loc("JPM")
            ] * 100,

            optimized_min_vol_weights[
                prices.columns.get_loc("JPM")
            ] * 100
        ],

        "XOM Weight (%)": [
            weights[
                prices.columns.get_loc("XOM")
            ] * 100,

            optimized_max_sharpe_weights[
                prices.columns.get_loc("XOM")
            ] * 100,

            optimized_min_vol_weights[
                prices.columns.get_loc("XOM")
            ] * 100
        ]
    }
)

comparison_table = (
    comparison_table.round(2)
)

print(
    "\nPortfolio comparison:"
)

print(
    comparison_table.to_string(
        index=False
    )
)

comparison_table.to_csv(
    "data/portfolio_comparison.csv",
    index=False
)


# ----------------------------
# 23. Calculate correlation matrix
# ----------------------------

correlation_matrix = (
    daily_returns.corr()
)

print(
    "\nCorrelation matrix:"
)

print(
    correlation_matrix.round(2)
)


# ----------------------------
# 24. Plot correlation matrix
# ----------------------------

fig, ax = plt.subplots(
    figsize=(8, 6)
)

image = ax.imshow(
    correlation_matrix,
    vmin=-1,
    vmax=1
)

# Asset names on x-axis
ax.set_xticks(
    np.arange(
        len(correlation_matrix.columns)
    )
)

ax.set_xticklabels(
    correlation_matrix.columns
)

# Asset names on y-axis
ax.set_yticks(
    np.arange(
        len(correlation_matrix.index)
    )
)

ax.set_yticklabels(
    correlation_matrix.index
)

# Add correlation values inside each box
for i in range(
    len(correlation_matrix.index)
):

    for j in range(
        len(correlation_matrix.columns)
    ):

        ax.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

ax.set_title(
    "Asset Return Correlation Matrix"
)

fig.colorbar(
    image,
    ax=ax,
    label="Correlation"
)

plt.tight_layout()

plt.savefig(
    "charts/correlation_matrix.png",
    dpi=300
)

plt.show()