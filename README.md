# Civic Engagement Uplift Lab

Civic Engagement Uplift Lab is a analytics project that test civic engagement messaging, estimate persuasion lift, and prioritize outreach segments. It mirrors the kind of work campaign analytics teams do in canvassing, turnout, and persuasion: controlled experiments, uplift modeling, segmentation, and decision-focused visualization.

## What This Project Does

- Generates a realistic synthetic voter engagement dataset with treatment arms, demographics, district context, prior turnout behavior, and downstream actions such as click, pledge, and donation.
- Uses `duckdb` and `pandas` for ETL and experiment summaries.
- Includes explicit experimental design documentation and reusable SQL analysis queries for DuckDB.
- Trains predictive response models with logistic regression and gradient boosting.
- Estimates message-level uplift and heterogeneous treatment effects by district, age, race, and turnout history.
- Ships with a Streamlit app for interactive filtering and message recommendations.

## Project Structure

```text
.
├── app.py
├── docs/experimental_design.md
├── generate_assets.py
├── requirements.txt
├── sql/analysis_queries.sql
├── data/processed/
├── reports/figures/
├── src/civic_uplift/
│   ├── analysis.py
│   ├── data.py
│   ├── modeling.py
│   └── visuals.py
└── tests/test_pipeline.py
```

## How To Run

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Generate the dataset, scored outputs, and figures:

```bash
python3 generate_assets.py
```

3. Launch the app:

```bash
python3 -m streamlit run app.py
```

4. Run the test:

```bash
python3 -m pytest tests/test_pipeline.py
```

