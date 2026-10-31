from pathlib import Path
import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# Load saved models and metadata
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

regression_model = joblib.load(
    MODELS / "regression_random_forest.joblib"
)

classification_model = joblib.load(
    MODELS / "classification_logistic.joblib"
)

with open(MODELS / "metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

options = metadata["category_options"]


# --------------------------------------------------
# Helper function for creating model input
# --------------------------------------------------

def create_input_row(
    business_age,
    months,
    paid_workers,
    unpaid_workers,
    raw_material_cost,
    operating_cost,
    sector,
    acquisition,
    funding,
    location,
    expansion,
    vat,
    income_tax,
    current_revenue=None,
):
    row = {
        "business_age": float(business_age),
        "months_operated_last_12_months": float(months),
        "paid_workers": float(paid_workers),
        "unpaid_workers": float(unpaid_workers),

        "log_monthly_raw_material_cost":
            float(np.log1p(raw_material_cost)),

        "log_monthly_other_operating_cost":
            float(np.log1p(operating_cost)),

        "business_sector": sector,
        "business_acquisition_method": acquisition,
        "startup_funding_source": funding,
        "business_location": location,
        "used_funding_for_expansion": expansion,
        "vat_registration_status": vat,
        "income_tax_registration_status": income_tax,
    }

    if current_revenue is not None:
        row["log_monthly_gross_revenue"] = float(
            np.log1p(current_revenue)
        )

    return pd.DataFrame([row])


# --------------------------------------------------
# Regression prediction
# --------------------------------------------------

def predict_revenue(
    business_age,
    months,
    paid_workers,
    unpaid_workers,
    raw_material_cost,
    operating_cost,
    sector,
    acquisition,
    funding,
    location,
    expansion,
    vat,
    income_tax,
):
    input_df = create_input_row(
        business_age,
        months,
        paid_workers,
        unpaid_workers,
        raw_material_cost,
        operating_cost,
        sector,
        acquisition,
        funding,
        location,
        expansion,
        vat,
        income_tax,
    )

    predicted_log = regression_model.predict(input_df)[0]

    predicted_ugx = max(
        float(np.expm1(predicted_log)),
        0.0
    )

    metrics = metadata["regression"]["notebook_test_metrics"]

    result = f"""
## Predicted Monthly Gross Revenue

### UGX {predicted_ugx:,.0f}

**Model:** Random Forest Regression

**Notebook test performance**

- R²: {metrics["R2"]:.3f}
- MAE: UGX {metrics["MAE_UGX"]:,.0f}
- RMSE: UGX {metrics["RMSE_UGX"]:,.0f}

*This prediction is an estimate for academic demonstration and decision support.*
"""

    return result


# --------------------------------------------------
# Classification prediction
# --------------------------------------------------

def classify_revenue_trend(
    business_age,
    months,
    paid_workers,
    unpaid_workers,
    raw_material_cost,
    operating_cost,
    current_revenue,
    sector,
    acquisition,
    funding,
    location,
    expansion,
    vat,
    income_tax,
):
    input_df = create_input_row(
        business_age,
        months,
        paid_workers,
        unpaid_workers,
        raw_material_cost,
        operating_cost,
        sector,
        acquisition,
        funding,
        location,
        expansion,
        vat,
        income_tax,
        current_revenue=current_revenue,
    )

    predicted_class = classification_model.predict(input_df)[0]

    probabilities = classification_model.predict_proba(
        input_df
    )[0]

    classes = classification_model.named_steps[
        "model"
    ].classes_

    probability_dict = {
        cls: float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    metrics = metadata[
        "classification"
    ]["notebook_test_metrics"]

    result = f"""
## Predicted Revenue Trend

### {predicted_class}

**Model:** Multinomial Logistic Regression

**Notebook test performance**

- Accuracy: {metrics["Accuracy"]:.3f}
- Balanced Accuracy: {metrics["Balanced_Accuracy"]:.3f}
- Macro F1: {metrics["Macro_F1"]:.3f}

*Class probabilities represent model confidence, not certainty.*
"""

    return result, probability_dict


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

with gr.Blocks(
    title="Uganda Household Enterprise ML Predictor"
) as demo:

    gr.Markdown(
        """
# Uganda Household Enterprise ML Predictor

### Group 1 Developing Data Products Capstone Models

Using **Uganda National Panel Survey 2019–2020**
household-enterprise data.

> These predictions are for academic demonstration
> and decision support and should not be treated as
> guaranteed business outcomes.
"""
    )

    # ==================================================
    # REGRESSION TAB
    # ==================================================

    with gr.Tab("💰 Revenue Regression"):

        gr.Markdown(
            """
## Predict Monthly Gross Revenue

Enter the enterprise characteristics below.
The deployed **Random Forest Regression** model
will estimate monthly gross revenue.
"""
        )

        with gr.Row():

            with gr.Column():

                reg_age = gr.Number(
                    label="Business age (years)",
                    value=8
                )

                reg_months = gr.Number(
                    label="Months operated in last 12 months",
                    value=12
                )

                reg_paid = gr.Number(
                    label="Paid workers",
                    value=0
                )

                reg_unpaid = gr.Number(
                    label="Unpaid workers",
                    value=0
                )

                reg_raw = gr.Number(
                    label="Monthly raw-material cost (UGX)",
                    value=100000
                )

                reg_operating = gr.Number(
                    label="Other monthly operating cost (UGX)",
                    value=10000
                )

            with gr.Column():

                reg_sector = gr.Dropdown(
                    options["business_sector"],
                    label="Business sector",
                    value=options["business_sector"][0]
                )

                reg_acquisition = gr.Dropdown(
                    options["business_acquisition_method"],
                    label="How was the business acquired?",
                    value=options[
                        "business_acquisition_method"
                    ][0]
                )

                reg_funding = gr.Dropdown(
                    options["startup_funding_source"],
                    label="Main startup funding source",
                    value=options[
                        "startup_funding_source"
                    ][0]
                )

                reg_location = gr.Dropdown(
                    options["business_location"],
                    label="Business location",
                    value=options["business_location"][0]
                )

                reg_expansion = gr.Dropdown(
                    options["used_funding_for_expansion"],
                    label="Used funding for expansion?",
                    value=options[
                        "used_funding_for_expansion"
                    ][0]
                )

                reg_vat = gr.Dropdown(
                    options["vat_registration_status"],
                    label="VAT registration status",
                    value=options[
                        "vat_registration_status"
                    ][0]
                )

                reg_income = gr.Dropdown(
                    options[
                        "income_tax_registration_status"
                    ],
                    label="Income-tax registration status",
                    value=options[
                        "income_tax_registration_status"
                    ][0]
                )

        reg_button = gr.Button(
            "Predict Monthly Revenue",
            variant="primary"
        )

        reg_result = gr.Markdown()

        reg_button.click(
            fn=predict_revenue,
            inputs=[
                reg_age,
                reg_months,
                reg_paid,
                reg_unpaid,
                reg_raw,
                reg_operating,
                reg_sector,
                reg_acquisition,
                reg_funding,
                reg_location,
                reg_expansion,
                reg_vat,
                reg_income,
            ],
            outputs=reg_result,
        )

    # ==================================================
    # CLASSIFICATION TAB
    # ==================================================

    with gr.Tab("📈 Revenue Trend Classification"):

        gr.Markdown(
            """
## Classify Revenue Performance Trend

The deployed **Multinomial Logistic Regression**
model predicts whether revenue is:

- Increased
- Decreased
- Remained the same
"""
        )

        with gr.Row():

            with gr.Column():

                clf_age = gr.Number(
                    label="Business age (years)",
                    value=8
                )

                clf_months = gr.Number(
                    label="Months operated in last 12 months",
                    value=12
                )

                clf_paid = gr.Number(
                    label="Paid workers",
                    value=0
                )

                clf_unpaid = gr.Number(
                    label="Unpaid workers",
                    value=0
                )

                clf_raw = gr.Number(
                    label="Monthly raw-material cost (UGX)",
                    value=100000
                )

                clf_operating = gr.Number(
                    label="Other monthly operating cost (UGX)",
                    value=10000
                )

                clf_revenue = gr.Number(
                    label="Current monthly gross revenue (UGX)",
                    value=400000
                )

            with gr.Column():

                clf_sector = gr.Dropdown(
                    options["business_sector"],
                    label="Business sector",
                    value=options["business_sector"][0]
                )

                clf_acquisition = gr.Dropdown(
                    options["business_acquisition_method"],
                    label="How was the business acquired?",
                    value=options[
                        "business_acquisition_method"
                    ][0]
                )

                clf_funding = gr.Dropdown(
                    options["startup_funding_source"],
                    label="Main startup funding source",
                    value=options[
                        "startup_funding_source"
                    ][0]
                )

                clf_location = gr.Dropdown(
                    options["business_location"],
                    label="Business location",
                    value=options["business_location"][0]
                )

                clf_expansion = gr.Dropdown(
                    options["used_funding_for_expansion"],
                    label="Used funding for expansion?",
                    value=options[
                        "used_funding_for_expansion"
                    ][0]
                )

                clf_vat = gr.Dropdown(
                    options["vat_registration_status"],
                    label="VAT registration status",
                    value=options[
                        "vat_registration_status"
                    ][0]
                )

                clf_income = gr.Dropdown(
                    options[
                        "income_tax_registration_status"
                    ],
                    label="Income-tax registration status",
                    value=options[
                        "income_tax_registration_status"
                    ][0]
                )

        clf_button = gr.Button(
            "Classify Revenue Trend",
            variant="primary"
        )

        clf_result = gr.Markdown()

        clf_probabilities = gr.Label(
            label="Class probabilities"
        )

        clf_button.click(
            fn=classify_revenue_trend,
            inputs=[
                clf_age,
                clf_months,
                clf_paid,
                clf_unpaid,
                clf_raw,
                clf_operating,
                clf_revenue,
                clf_sector,
                clf_acquisition,
                clf_funding,
                clf_location,
                clf_expansion,
                clf_vat,
                clf_income,
            ],
            outputs=[
                clf_result,
                clf_probabilities,
            ],
        )

    # ==================================================
    # ABOUT TAB
    # ==================================================

    with gr.Tab("ℹ️ About the Models"):

        gr.Markdown(
            f"""
## Deployment Information

**Regression model:**  
{metadata["regression"]["algorithm"]}

**Classification model:**  
{metadata["classification"]["algorithm"]}

**Dataset:**  
Uganda National Panel Survey 2019–2020 household
enterprise data.

### Deployment flow

User inputs  
↓  
Validation and transformation  
↓  
Saved Scikit-learn preprocessing pipeline  
↓  
Fitted machine-learning model  
↓  
Prediction  
↓  
Human-readable result

The `.joblib` files contain the already fitted
machine-learning pipelines.

The application therefore **does not retrain the
models when a user makes a prediction**.
"""
        )


# --------------------------------------------------
# Start Gradio
# --------------------------------------------------

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )