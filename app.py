from pathlib import Path

import streamlit as st

from predict_smiles import (
    load_gine_model,
    predict_pic50,
    pic50_to_ic50_nm,
)


MODEL_PATH = Path("models/gine_egfr_chembl203.pt")


@st.cache_resource
def get_model():
    return load_gine_model(MODEL_PATH)


st.set_page_config(
    page_title="EGFR pIC50 Predictor",
    page_icon="🧪",
)

st.title("EGFR pIC50 Predictor")
st.write(
    "Model GINE predicts biological activity of a molecule against "
    "Epidermal growth factor receptor (EGFR / CHEMBL203)."
)

model, checkpoint, device = get_model()

st.sidebar.header("Model info")
st.sidebar.write("Target:", checkpoint["target_name"])
st.sidebar.write("Target ChEMBL ID:", checkpoint["target_chembl_id"])
st.sidebar.write("Test R²:", round(checkpoint["test_r2"], 3))
st.sidebar.write("Test RMSE:", round(checkpoint["test_rmse"], 3))

smiles = st.text_input(
    "Enter SMILES:",
    value="CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
)

if st.button("Predict pIC50"):
    try:
        pred_pic50 = predict_pic50(smiles, model, device)
        pred_ic50_nm = pic50_to_ic50_nm(pred_pic50)

        st.success(f"Predicted pIC50: {pred_pic50:.3f}")
        st.write(f"Approx. IC50: {pred_ic50_nm:.2f} nM")

    except Exception as e:
        st.error(f"Prediction failed: {e}")