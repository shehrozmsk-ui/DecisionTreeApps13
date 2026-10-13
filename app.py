
with tab2:
    st.header("Decision Tree ML Prediction")
    st.write(
        "Enter nutrition values below and press the button "
        "to get a model prediction."
    )

    tree_image = ROOT / "outputs" / "decision_tree.png"

    if tree_image.exists():
        with st.expander(
            "View the actual trained Decision Tree diagram",
            expanded=True
        ):
            st.caption(
                "Each box is a decision rule. Follow the branches "
                "to understand how the model reaches a class."
            )
            st.image(
                str(tree_image),
                caption="Trained Starbucks Decision Tree",
                use_container_width=True
            )
    else:
        st.info(
            "The tree diagram is not available. Upload "
            "outputs/decision_tree.png to GitHub if you want to display it."
        )

    if model is None or meta is None:
        st.warning(
            "The trained model files are missing. Upload both "
            "decision_tree_model.joblib and model_metadata.json "
            "to the same GitHub repository."
        )

    else:
        st.caption(
            f"Prediction target: {meta.get('target_column', 'Not specified')} "
            f" • Test accuracy: {meta.get('accuracy', 0):.1%}"
        )

        if meta.get("generated_calorie_target"):
            st.info(
                "This model predicts Low / Medium / High calorie "
                "categories learned from your dataset."
            )

        numeric_features = meta.get("numeric_features", [])
        categorical_features = meta.get("categorical_features", [])
        feature_columns = meta.get("feature_columns", [])

        if not feature_columns:
            st.error(
                "No feature columns were found in model_metadata.json."
            )
        else:
            with st.form("prediction_form"):
                st.subheader("Enter nutrition values")
                values = {}

                left, right = st.columns(2)

                for i, feature in enumerate(feature_columns):
                    pane = left if i % 2 == 0 else right

                    with pane:
                        if feature in numeric_features:
                            default = 0.0

                            if df is not None and feature in df.columns:
                                series = pd.to_numeric(
                                    df[feature], errors="coerce"
                                ).dropna()

                                if not series.empty:
                                    default = float(series.median())

                            values[feature] = st.number_input(
                                f"Enter {feature}",
                                value=default,
                                step=1.0,
                                format="%.2f"
                            )

                        elif feature in categorical_features:
                            options = []

                            if df is not None and feature in df.columns:
                                options = (
                                    df[feature]
                                    .dropna()
                                    .astype(str)
                                    .value_counts()
                                    .head(100)
                                    .index.tolist()
                                )

                            if options:
                                values[feature] = st.selectbox(
                                    f"Choose {feature}", options
                                )
                            else:
                                values[feature] = st.text_input(
                                    f"Enter {feature}"
                                )

                        else:
                            values[feature] = st.text_input(
                                f"Enter {feature}"
                            )

                submitted = st.form_submit_button(
                    "Predict calorie category",
                    use_container_width=True
                )

            if submitted:
                input_df = pd.DataFrame(
                    [values], columns=feature_columns
                )

                for feature in numeric_features:
                    input_df[feature] = pd.to_numeric(
                        input_df[feature], errors="coerce"
                    )

                try:
                    prediction = model.predict(input_df)[0]

                    st.success(
                        f"Predicted category: **{prediction}**"
                    )

                    if hasattr(model, "predict_proba"):
                        try:
                            probs = model.predict_proba(input_df)[0]

                            if hasattr(model, "named_steps"):
                                classifier = list(
                                    model.named_steps.values()
                                )[-1]
                            else:
                                classifier = model

                            classes = getattr(
                                classifier,
                                "classes_",
                                range(len(probs))
                            )

                            prob_df = pd.DataFrame({
                                "Category": [
                                    str(item) for item in classes
                                ],
                                "Probability": probs
                            }).sort_values(
                                "Probability",
                                ascending=False
                            )

                            prob_df["Probability"] = (
                                prob_df["Probability"] * 100
                            ).round(2)

                            st.subheader("Prediction probabilities")
                            st.dataframe(
                                prob_df,
                                use_container_width=True,
                                hide_index=True
                            )

                            st.bar_chart(
                                prob_df.set_index("Category")
                            )

                        except Exception as exc:
                            st.warning(
                                "The prediction worked, but probabilities "
                                f"could not be displayed: {exc}"
                            )

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Nutrition results depend on the CSV dataset you provide. "
    "ML predictions are educational estimates."
)
