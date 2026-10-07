
import os
import joblib
import pandas as pd
import streamlit as st


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(
    page_title="AutoValue | Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_PATH = "models/car_price_model.joblib"
INFO_PATH = "models/model_info.joblib"


# -----------------------------------
# CUSTOM STYLING
# -----------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #1e3a8a, #2563eb);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    div[data-testid="stForm"] {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 14px;
        padding: 22px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }

    .result-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    info = joblib.load(INFO_PATH)
    return model, info


# -----------------------------------
# HEADER
# -----------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🚘 AutoValue</h1>
        <p>Smart car valuation powered by Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if not os.path.exists(MODEL_PATH) or not os.path.exists(INFO_PATH):
    st.warning(
        "The trained model was not found. Train the model first."
    )
    st.code("python train_model.py")
    st.stop()


try:
    model, info = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.title("AutoValue")
    st.caption("Machine Learning Project")
    st.divider()

    st.markdown("### Model Information")
    st.write(f"**Algorithm:** {info.get('model_name', 'Regression')}")

    st.write(
        "**Input features:**",
        len(info.get("feature_columns", [])),
    )

    st.divider()
    st.markdown("### How it works")
    st.write("1. Enter the car details.")
    st.write("2. Submit the valuation form.")
    st.write("3. Review the estimated price.")

    st.divider()
    st.caption("Built using Python, pandas, scikit-learn and Streamlit.")


# -----------------------------------
# INTRODUCTION
# -----------------------------------
st.subheader("Estimate your car's value")
st.write(
    "Enter the vehicle details below to estimate its selling price "
    "based on patterns learned from the training dataset."
)

st.info(
    "For the most useful estimate, enter accurate vehicle details. "
    "Predictions depend on the quality and coverage of the dataset."
)


# -----------------------------------
# INPUT FORM
# -----------------------------------
inputs = {}

with st.form("car_valuation_form"):
    st.markdown("### Vehicle details")

    features = info["feature_columns"]
    numeric_features = info["numeric_features"]
    categorical_features = info["categorical_features"]

    # Put car name and company first, followed by remaining fields.
    preferred_order = [
        col for col in ["name", "company", "year", "kms_driven", "fuel_type"]
        if col in features
    ]
    ordered_features = preferred_order + [
        col for col in features if col not in preferred_order
    ]

    left, right = st.columns(2)

    for index, feature in enumerate(ordered_features):
        column = left if index % 2 == 0 else right

        label = (
            feature.replace("_", " ")
            .replace("-", " ")
            .title()
        )

        with column:
            if feature in numeric_features:
                default = float(info.get("defaults", {}).get(feature, 0.0))
                limits = info.get("ranges", {}).get(feature, [])

                min_value = None
                max_value = None

                if len(limits) == 2:
                    low, high = float(limits[0]), float(limits[1])
                    if low < high:
                        min_value = low
                        max_value = high

                is_whole_number = (
                    feature.lower() in {
                        "year", "kms_driven", "kilometers_driven",
                        "km_driven", "mileage", "age"
                    }
                    or "year" in feature.lower()
                )

                if is_whole_number:
                    default_value = int(round(default))
                    min_int = int(min_value) if min_value is not None else None
                    max_int = int(max_value) if max_value is not None else None

                    if min_int is not None and max_int is not None:
                        default_value = max(min_int, min(default_value, max_int))

                    inputs[feature] = st.number_input(
                        label,
                        min_value=min_int,
                        max_value=max_int,
                        value=default_value,
                        step=1,
                        key=f"input_{feature}",
                    )
                else:
                    kwargs = {
                        "label": label,
                        "value": default,
                        "step": 0.1,
                        "format": "%.2f",
                        "key": f"input_{feature}",
                    }

                    if min_value is not None and max_value is not None:
                        kwargs["min_value"] = min_value
                        kwargs["max_value"] = max_value
                        kwargs["value"] = max(
                            min_value, min(default, max_value)
                        )

                    inputs[feature] = st.number_input(**kwargs)

            elif feature in categorical_features:
                options = info.get("options", {}).get(feature, [])

                if options:
                    options = [str(option) for option in options]
                    inputs[feature] = st.selectbox(
                        label,
                        options=options,
                        key=f"input_{feature}",
                    )
                else:
                    inputs[feature] = st.text_input(
                        label,
                        key=f"input_{feature}",
                    )

            else:
                inputs[feature] = st.text_input(
                    label,
                    key=f"input_{feature}",
                )

    st.markdown("")
    submitted = st.form_submit_button(
        "🔍  Estimate Car Price",
        type="primary",
        use_container_width=True,
    )


# -----------------------------------
# PREDICTION RESULTS
# -----------------------------------
if submitted:
    try:
        input_df = pd.DataFrame(
            [{feature: inputs[feature] for feature in info["feature_columns"]}],
            columns=info["feature_columns"],
        )

        prediction = float(model.predict(input_df)[0])

        st.divider()
        st.subheader("Valuation result")

        if not pd.notna(prediction):
            st.error("The model returned an invalid prediction.")

        elif prediction < 0:
            st.warning(
                "The model returned a negative price. "
                "This is not a valid selling-price estimate."
            )

        else:
            st.success("Your car valuation is ready!")

            result_col, details_col = st.columns([1.2, 1])

            with result_col:
                st.metric(
                    label="Estimated Selling Price",
                    value=f"₹{prediction:,.0f}",
                )

                st.caption(
                    "Estimated value based on the trained model. "
                    "Actual market prices may vary."
                )

            with details_col:
                st.markdown("#### Submitted vehicle details")

                summary = pd.DataFrame(
                    [
                        {
                            "Feature": feature.replace("_", " ").title(),
                            "Value": str(inputs[feature]),
                        }
                        for feature in info["feature_columns"]
                    ]
                )

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True,
                    height=min(350, 45 * len(summary) + 40),
                )

            if "prediction_history" not in st.session_state:
                st.session_state.prediction_history = []

            st.session_state.prediction_history.append(
                {
                    "Predicted Price (₹)": round(prediction, 2),
                    **{
                        feature.replace("_", " ").title(): inputs[feature]
                        for feature in info["feature_columns"]
                    },
                }
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# -----------------------------------
# PREDICTION HISTORY
# -----------------------------------
if st.session_state.get("prediction_history"):
    st.divider()
    st.subheader("Recent predictions")

    history = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(
        history.iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download prediction history (CSV)",
        data=csv_data,
        file_name="car_price_predictions.csv",
        mime="text/csv",
    )


# -----------------------------------
# FOOTER
# -----------------------------------
st.divider()
st.caption(
    "AutoValue | Educational Machine Learning Project | "
    "Predictions are estimates and are not guaranteed market valuations."
)
