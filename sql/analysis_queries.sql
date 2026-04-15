-- Civic Engagement Uplift Lab SQL walkthrough
-- These queries are written for DuckDB and mirror the calculations used in the Python pipeline.

-- 1. Action rate by message and ATE versus control
with message_rates as (
    select
        treatment,
        message_label,
        count(*) as audience_size,
        avg(action_taken) as action_rate,
        avg(donated) as donation_rate,
        avg(pledged_to_vote) as pledge_rate
    from feature_table
    group by 1, 2
),
control_rate as (
    select avg(action_taken) as control_action_rate
    from feature_table
    where treatment = 'control'
)
select
    m.*,
    m.action_rate - c.control_action_rate as ate_vs_control
from message_rates m
cross join control_rate c
order by ate_vs_control desc, audience_size desc;

-- 2. Segment-level lift by age band
select
    age_band,
    treatment,
    avg(action_taken) as action_rate
from feature_table
group by 1, 2
order by age_band, action_rate desc;

-- 3. High-value operational segments
select
    district,
    age_band,
    turnout_history,
    best_message,
    avg(best_uplift) as avg_uplift,
    avg(score_gradient_boosting) as avg_response_score,
    count(*) as audience
from scored_voters
group by 1, 2, 3, 4
having count(*) >= 80
order by avg_uplift desc, avg_response_score desc
limit 15;
