import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Solubility Demo", page_icon="🧪")
st.title("🧪 Drug Solubility Prediction — Demo")
st.caption("Regression workflow using molecular-property-style input columns.")
data = pd.read_csv("data/solubility_demo.csv")
features = ["molecular_weight", "logP", "hydrogen_bond_donors", "rotatable_bonds"]
X, y = data[features], data["solubility_demo"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
st.warning("Demo values and target are illustrative, not experimental measurements. This model cannot guide drug design or clinical decisions.")
with st.form("solubility"):
    mw = st.number_input("Molecular weight", min_value=1.0, value=180.0)
    logp = st.number_input("logP", value=1.5)
    hbd = st.number_input("Hydrogen-bond donors", min_value=0, value=2, step=1)
    rot = st.number_input("Rotatable bonds", min_value=0, value=4, step=1)
    submitted = st.form_submit_button("Estimate demo value")
if submitted:
    row = pd.DataFrame([[mw, logp, hbd, rot]], columns=features)
    st.subheader(f"Illustrative predicted target: {model.predict(row)[0]:.3f}")
    st.caption("The output is in arbitrary demo units and is not a real solubility estimate.")
with st.expander("Data and example evaluation"):
    st.dataframe(data, use_container_width=True)
    if len(data) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        eval_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        st.write("Holdout MAE on demo data:", round(float(mean_absolute_error(y_test, eval_model.predict(X_test))), 3))
