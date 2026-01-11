import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")
st.title("T-AutoNLP. Human-Centered AutoML")

pareto_data = load_pareto_results()  # serialize from AutoML

df = pd.DataFrame(pareto_data)

fig = px.scatter_3d(
    df,
    x="accuracy",
    y="efficiency",
    z="interpretability",
    hover_data=["vectorizer", "classifier", "mode"],
)

st.plotly_chart(fig, use_container_width=True)

selected = st.selectbox("Inspect solution", df.index)

st.subheader("Model Explanation")
st.write(df.loc[selected])
