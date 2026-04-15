# Experimental Design Notes

## Business Question

Which outreach message most increases the probability that a voter will take a meaningful action such as clicking, pledging to vote, or donating?

## Experiment Structure

- **Population:** synthetic voter universe across seven battleground-style districts.
- **Unit of randomization:** individual voter.
- **Treatment arms:** `Control`, `Economic Security`, `Community Voice`, `Future Forward`.
- **Primary outcome:** `action_taken`.
- **Secondary outcomes:** `clicked_message`, `pledged_to_vote`, `donated`.

## Causal Inference Framing

This project is set up as a multi-arm randomized experiment.

- Random assignment supports identification of average treatment effects.
- The control arm provides a clean baseline for comparison.
- Subgroup comparisons estimate heterogeneous treatment effects by district, age, race, and turnout history.
- Predictive models are used to rank likely responders, while uplift estimates are used to recommend the best message for each segment.

## Core Design Choices

1. **Randomization**
   Treatment assignment is simulated independently of observed covariates so message effects are interpretable as causal contrasts.

2. **Primary estimand**
   The key estimand is:

   `ATE(message) = P(action | message) - P(action | control)`

3. **Heterogeneous effects**
   The project estimates how treatment effects vary across:

- Age bands
- Race / ethnicity
- Turnout history
- District context

4. **Modeling**
   Two predictive baselines are compared:

- Logistic regression for interpretability
- Gradient boosting for nonlinear response patterns

5. **Decision support**
   A recommendation layer assigns the best message based on estimated uplift rather than response likelihood alone.

## What To Say In An Interview

You can describe the project as a hybrid of experimental design and machine learning:

> I structured the problem as a multi-arm randomized outreach experiment, estimated average treatment effects against a control group, then layered on uplift modeling to understand which message should be shown to which voter segment. That let me separate “who is likely to respond” from “who is actually persuadable,” which is a key distinction in campaign analytics.
