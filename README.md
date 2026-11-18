# Olist Delivery Risk
[Open the interactive demo](https://olist-delivery-risk-dagchqjkwfywkjgsrwagta.streamlit.app/)

A data science portfolio project that predicts late-delivery risk
and presents historical orders in an interactive review queue.

## Demo features

- Select an approval date.
- Rank orders by predicted delivery risk.
- Adjust the number of orders to review.
- Reveal actual delivery outcomes.
- Download the selected queue as a CSV.

The app replays historical data using previously generated model
scores. It does not connect to a live store or score new orders.

## Business question

If an operations team can review only a limited number of orders
each day, can a model help identify more late deliveries than
random selection?

## Data

Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The project combines orders, customers, items, sellers, and products.

After eligibility checks, the modelling dataset contains 96,127
completed orders. An order is labelled late when its actual delivery
calendar date is after its promised delivery date.

Original CSV files are kept unchanged in data/raw.

## Approach

1. Inspect missing values, identifiers, timestamps, and table relationships.
2. Aggregate item records before joining to preserve one row per order.
3. Build 14 inputs describing order contents, locations, and approval timing.
4. Split orders chronologically into training, validation, and test periods.
5. Compare a constant baseline, logistic regression, and boosted trees.
6. Evaluate prioritisation using a budget of up to 50 reviews per day.

Preprocessing is fitted on training data only.

## Model selection

Logistic regression outperformed the tested boosted-tree configuration
on validation data under the same daily review budget:

| Model | Orders reviewed | Late orders caught | Average precision |
|---|---:|---:|---:|
| Logistic regression | 2,913 | 185 | 0.117 |
| Boosted trees | 2,913 | 141 | 0.083 |

Logistic regression was selected before the final test evaluation.

## Test results

For 12,516 eligible test orders, including 599 late deliveries:

| Measure | Result |
|---|---:|
| Orders reviewed | 2,774 |
| Late orders caught | 143 |
| Precision | 5.16% |
| Recall | 23.87% |
| Expected late orders caught by random daily selection | 114.45 |
| Lift over random daily selection | 1.25× |
| Average precision | 0.0769 |

The model identified approximately 29 more late orders than expected
random selection with the same daily workload.

## Limitations

- Results describe eligible completed deliveries, not every incoming order.
- Training and validation delivery-outcome cutoffs disproportionately
  excluded late orders. An audit documents this selection bias.
- The test period appeared in earlier descriptive exploration.
- Product and seller information are catalog snapshots; availability
  at order approval is assumed.
- Recorded delivery promises may have been revised; revision history
  is unavailable.
- Performance weakened from validation to test.
- Scores should not be interpreted as verified delivery probabilities.
- The project does not establish that reviewing orders prevents delays.

These are preliminary portfolio results, not evidence of readiness
for operational deployment.

## Project structure

- notebooks/01_data_exploration.ipynb — data audit and exploration
- notebooks/02_prepare_modelling_data.ipynb — preparation, modelling,
  evaluation, and artifact export
- app/streamlit_app.py — interactive historical demo
- data/raw/ — original CSV files
- data/processed/demo_orders.csv — exported demo orders and scores
- models/logistic_delivery_risk_v1.joblib — saved model and metadata

Raw data, processed data, and model artifacts are excluded from Git.

## Run locally

From the project root, create and activate a Python environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install the dependencies:

    python -m pip install pandas numpy matplotlib jupyter ipykernel scikit-learn joblib streamlit

Download the dataset from Kaggle and place its nine CSV files
directly in data/raw, preserving the original filenames.

Select the project environment in Jupyter or VS Code.
Run notebook 1 for exploration, then notebook 2 from top to bottom.
Notebook 2 exports the model and demo data required by the app.

Launch the demo:

    python -m streamlit run app/streamlit_app.py

Open the Local URL shown in the terminal.

## Tools

Python, pandas, NumPy, matplotlib, scikit-learn, Jupyter, and Streamlit.