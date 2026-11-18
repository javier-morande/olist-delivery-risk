from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Olist Delivery Risk",
    page_icon="📦",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "demo_orders.csv"


@st.cache_data
def load_orders():
    data = pd.read_csv(
        DATA_PATH,
        dtype={"order_id": "string"},
        parse_dates=["order_approved_at"]
    )
    data["approval_day"] = data["order_approved_at"].dt.date
    return data


st.title("📦 Olist Delivery Risk")
st.write(
    "Explore historical orders ranked by predicted late-delivery risk."
)

if not DATA_PATH.exists():
    st.error("Demo data is missing. Export demo_orders.csv first.")
    st.stop()

orders = load_orders()

with st.sidebar:
    st.header("Review settings")

    selected_day = st.selectbox(
        "Approval date",
        options=sorted(orders["approval_day"].unique())
    )

    review_limit = st.slider(
        "Maximum orders to review",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )

    reveal_outcomes = st.checkbox(
        "Reveal actual delivery outcomes"
    )

    st.caption(
        "Scores come from the saved logistic regression model. "
        "This demo uses historical completed orders."
    )

daily_orders = (
    orders.loc[orders["approval_day"] == selected_day]
    .sort_values(
        ["risk_score", "order_id"],
        ascending=[False, True]
    )
    .copy()
)

queue = daily_orders.head(review_limit).copy()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Orders available",
    f"{len(daily_orders):,}"
)

col2.metric(
    "Orders selected",
    f"{len(queue):,}"
)

col3.metric(
    "Share selected",
    f"{len(queue) / len(daily_orders):.1%}"
)

st.subheader("Priority review queue")
st.caption(
    "Highest scores appear first. Scores are model estimates, "
    "not guaranteed delivery probabilities."
)

display_queue = queue[
    [
        "order_id",
        "customer_state",
        "risk_score",
        "item_count",
        "seller_count",
        "promised_days_from_approval"
    ]
].copy()

display_queue["risk_score"] = display_queue["risk_score"].map(
    lambda value: f"{value:.1%}"
)

display_queue.columns = [
    "Order ID",
    "Customer state",
    "Risk score",
    "Items",
    "Sellers",
    "Promised days after approval"
]

if reveal_outcomes:
    display_queue["Actual outcome"] = queue["is_late"].map(
        {0: "On time or early", 1: "Late"}
    )

st.dataframe(
    display_queue,
    hide_index=True,
    width="stretch"
)

st.download_button(
    "Download this review queue",
    data=display_queue.to_csv(index=False).encode("utf-8"),
    file_name=f"review_queue_{selected_day}.csv",
    mime="text/csv"
)

if reveal_outcomes:
    st.subheader("What actually happened")

    caught = int(queue["is_late"].sum())
    total_late = int(daily_orders["is_late"].sum())
    random_expected = (
        len(queue) * daily_orders["is_late"].mean()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Late orders caught",
        f"{caught} of {total_late}"
    )

    col2.metric(
        "Selected orders actually late",
        f"{caught / len(queue):.1%}"
    )

    col3.metric(
        "Share of late orders caught",
        f"{caught / total_late:.1%}" if total_late else "N/A"
    )

    st.write(
        f"Randomly selecting {len(queue)} orders from this day "
        f"would catch an expected {random_expected:.2f} late orders."
    )

with st.expander("About this demo and its results"):
    st.write(
        "The model uses order, customer, seller and product "
        "information available or assumed available at approval. "
        "Actual delivery information is used only for evaluation."
    )

    st.write(
        "Across 12,516 test orders, reviewing up to 50 orders "
        "per day selected 2,774 orders and caught 143 of 599 "
        "late deliveries. Random selection with the same daily "
        "budgets would catch about 114, giving a lift of 1.25×."
    )

    st.write(
        "These results describe the fixed 50-order daily budget. "
        "Changing the slider changes the selected day's queue."
    )

    st.write(
        "These are preliminary historical results. Only eligible "
        "completed deliveries are included. Catalog information "
        "is a snapshot, and development outcome cutoffs introduced "
        "selection bias. The test period also appeared in earlier "
        "exploration."
    )

    st.write(
        "The app displays previously generated model scores. "
        "It does not connect to a live store. The project has not "
        "demonstrated that reviewing orders prevents delays."
    )

    st.markdown(
        "[Dataset: Brazilian E-Commerce by Olist]"
        "(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)"
    )