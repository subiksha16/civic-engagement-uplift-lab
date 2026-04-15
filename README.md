# Civic Engagement Uplift Lab

Civic Engagement Uplift Lab is a resume-ready analytics project that simulates how a data science team could test civic engagement messaging, estimate persuasion lift, and prioritize outreach segments. It mirrors the kind of work campaign analytics teams do in canvassing, turnout, and persuasion: controlled experiments, uplift modeling, segmentation, and decision-focused visualization.

## What This Project Does

- Generates a realistic synthetic voter engagement dataset with treatment arms, demographics, district context, prior turnout behavior, and downstream actions such as click, pledge, and donation.
- Uses `duckdb` and `pandas` for ETL and experiment summaries.
- Includes explicit experimental design documentation and reusable SQL analysis queries for DuckDB.
- Trains predictive response models with logistic regression and gradient boosting.
- Estimates message-level uplift and heterogeneous treatment effects by district, age, race, and turnout history.
- Ships with a Streamlit app for interactive filtering and message recommendations.
- Exports polished figures that you can use directly in a portfolio or on a project page.

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

## Key Storyline For Recruiters

This project answers a practical campaign question: which message should we send to which voters if we want to maximize sign-ups, pledges, or donations?

The workflow is designed like a mini consulting engagement:

- Construct an experiment dataset with multiple treatment arms.
- Estimate the average treatment effect of each message against control.
- Train response models and compare discrimination plus calibration.
- Estimate heterogeneous treatment effects to identify who is most persuadable.
- Turn the analysis into an interface an operator can actually use.

## Skills This Demonstrates

- **Experimental design and causal inference:** the project is framed as a multi-arm randomized test with ATE and subgroup uplift analysis. See [docs/experimental_design.md](/Users/subiksha/Documents/DS/bluelabs/docs/experimental_design.md).
- **SQL for data manipulation:** the ETL and summary logic are written with DuckDB SQL, and standalone queries live in [sql/analysis_queries.sql](/Users/subiksha/Documents/DS/bluelabs/sql/analysis_queries.sql).
- **User interface development:** the current app is built in Streamlit for fast delivery and demoability. If you want a Django version later, this same analytics layer can be dropped behind Django views and templates without changing the modeling code.

## Recommended Portfolio Screenshots

Use these four visuals from `reports/figures/`:

1. `01_message_uplift.png`
   Message-level lift vs. control. Good hero chart for showing causal thinking.
2. `02_age_segment_heatmap.png`
   Segment-level uplift heatmap. Good for showing heterogeneous treatment effects.
3. `03_model_diagnostics.png`
   ROC / precision and calibration quality. Good for showing statistical rigor.
4. `04_top_segments.png`
   High-leverage audience clusters. Good for showing business actionability.

You can also take one dashboard screenshot from the Streamlit app showing filters, KPIs, and the uplift charts on one screen.

## Resume Bullet Ideas

- Built an end-to-end voter engagement uplift modeling app that simulated randomized message tests across 12K voter records and identified the highest-performing persuasion messages by district, age, and turnout history.
- Estimated average and heterogeneous treatment effects using logistic regression, gradient boosting, and subgroup analysis; translated model outputs into a deployable recommendation dashboard in Streamlit.
- Developed a reproducible analytics workflow in Python, DuckDB, pandas, and scikit-learn with exportable visual assets for campaign strategy and stakeholder reporting.

## Portfolio Description

**Civic Engagement Uplift Lab** is a campaign analytics simulation focused on a core civic engagement problem: identifying which persuasion message is most likely to move which voter. I designed a synthetic experiment with multiple treatment arms, modeled response probability and treatment lift, and built an interactive dashboard that lets users filter audiences and compare predicted uplift across segments. The result is a decision-support prototype that blends causal inference, machine learning, and stakeholder-ready data storytelling.
