> **Project Title:** EGIER Warehouse Capacity & Production Trend Forecasting
> **Project Overview:**
> Analyzed 12 years (144 months, Jan 2013 – Dec 2024) of monthly bag production data for EGIER to model manufacturing growth and project future warehouse capacity constraints.
> 
> 
> **Key Technical Highlights & Methodology:**
> * **Non-Linear Mathematical Modeling:** Built and fitted an exponential growth model ($y = a \cdot b^x$) via logarithmic regression, outperforming linear models with an exceptional coefficient of determination ($R^2 = 0.995$ vs $R^2 = 0.9299$).
> 
> 
> * **Polynomial Transformation & Validation:** Transformed the non-linear trend model into computable higher-degree Taylor/polynomial expansions to ensure computational efficiency and long-term numerical accuracy without relying on early reference bias.
> * **Capacity Constraint & Root-Finding Analysis:** Solved the nonlinear root equation $y(x) \ge 25,000$ to predict the exact month when production exceeds current warehouse capacity.
> * **Actionable Supply Chain Planning:** Factored in a 13-month construction lead time to pinpoint the exact date EGIER must initiate construction on a new warehouse facility.
