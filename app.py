import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import eda
from modules.nl_query import ask

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

MAX_FILE_MB = 50


@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


def show_result(result):
    """Display whatever type of result the generated code produced."""
    if isinstance(result, go.Figure):
        st.plotly_chart(result)
    elif isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif isinstance(result, pd.Series):
        st.dataframe(result.to_frame())
    else:
        st.write(result)


def render_assistant(msg: dict):
    if msg.get("error"):
        st.error(msg["error"])
    else:
        show_result(msg["result"])
    if msg.get("code"):
        with st.expander("View generated code"):
            st.code(msg["code"], language="python")
            if msg.get("attempts", 1) > 1:
                st.caption("The first attempt failed, so the code was auto-corrected.")


st.title("📊 AI Data Analyst")
st.caption("Upload a CSV, explore it, and ask questions in plain English.")

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

if uploaded.size > MAX_FILE_MB * 1024 * 1024:
    st.error(f"File is too large. Please upload a file under {MAX_FILE_MB} MB.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not read this file as a CSV: {e}")
    st.stop()

if df.empty:
    st.warning("This file has no data.")
    st.stop()

# Reset the chat when a different file is uploaded
file_key = f"{uploaded.name}-{uploaded.size}"
if st.session_state.get("file_key") != file_key:
    st.session_state.file_key = file_key
    st.session_state.messages = []

tab_ask, tab_overview, tab_missing, tab_dist, tab_corr = st.tabs(
    ["💬 Ask AI", "Overview", "Missing Values", "Distributions", "Correlations"]
)

with tab_ask:
    st.caption(
        "Try: 'What is the average of each numeric column?', "
        "'Show the top 10 rows by <column>', or 'Plot the distribution of <column>'."
    )

    question = st.chat_input("Ask a question about your data")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                outcome = ask(df, question)
                reply = {
                    "role": "assistant",
                    "result": outcome["result"],
                    "error": None if outcome["ok"] else outcome["error"],
                    "code": outcome["code"],
                    "attempts": outcome["attempts"],
                }
            except Exception as e:
                reply = {"role": "assistant", "result": None, "error": str(e), "code": None}
        st.session_state.messages.append(reply)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                render_assistant(msg)

with tab_overview:
    stats = eda.get_overview(df)
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats.items()):
        col.metric(label, f"{value:,}")

    st.subheader("Preview")
    st.dataframe(df.head(20))

    st.subheader("Column info")
    st.dataframe(eda.get_column_info(df))

    st.subheader("Summary statistics")
    st.dataframe(df.describe(include="all").T.astype(str))

with tab_missing:
    missing_df = eda.get_missing_values(df)
    fig = eda.plot_missing(missing_df)
    if fig is None:
        st.success("No missing values found. 🎉")
    else:
        st.plotly_chart(fig)
    st.dataframe(missing_df)

with tab_dist:
    column = st.selectbox("Choose a column", df.columns)
    st.plotly_chart(eda.plot_distribution(df, column))

with tab_corr:
    fig = eda.plot_correlation(df)
    if fig is None:
        st.info("Need at least two numeric columns to show correlations.")
    else:
        st.plotly_chart(fig)
