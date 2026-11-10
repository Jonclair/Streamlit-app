import streamlit as st
import pandas as pd
import plotly.express as px
import io

df = pd.read_csv("hypertension_dataset_5000.csv")

OUTCOME_COL = "Hypertension"
POSITIVE_LABEL = "Yes"
AGE_COL = "Age"
GENDER_COL = "Gender"
BMI_COL = "BMI"

BMI_LABELS = ["Underweight", "Healthy weight", "Overweight", "Obese"]

df["BMI_Category"] = pd.cut(
    df[BMI_COL],
    bins=[0, 18.5, 25, 30, float("inf")],
    labels=BMI_LABELS,
    right=False
)

CATEGORY_ORDERS = {
    "Salt_Intake": ["Low", "Normal", "High", "Very High"],
    "Physical_Activity": ["Sedentary", "Light", "Moderate", "Active"],
    "Alcohol_Consumption": ["Occasional", "Moderate", "Heavy"],
    "Smoking_Status": ["Never", "Former", "Current"],
    "Stress_Level": ["Low", "Moderate", "High", "Very High"],
    "BP_Category": ["Normal", "Elevated", "Stage 1 Hypertension", "Stage 2 Hypertension"],
    "BMI_Category": BMI_LABELS,
}


def order_for(variable):
    if variable in CATEGORY_ORDERS:
        return {variable: CATEGORY_ORDERS[variable]}
    return {}


categorical_variables = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_variables = df.select_dtypes(
    include="number"
).columns.tolist()

group_variables = [
    c for c in categorical_variables if c not in (OUTCOME_COL, "BP_Category")
]

st.title("app.py")

st.info(
    "My app is to help GP and medical staff "
    "and that it is not a diagnostic tool."
)

page = st.sidebar.radio(
    "Select Section",
    ["Data Overview", "Interactive Exploration"]
)

if page == "Data Overview":

    rows_to_show = st.selectbox(
        "Number of rows to display",
        [5, 10, 20, 50, 100]
    )

    st.dataframe(
        df.head(rows_to_show),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    col1.metric("Number of Rows", df.shape[0])
    col2.metric("Number of Columns", df.shape[1])

    st.write(df.columns.tolist())

    st.write("Numerical variables:", numerical_variables)
    st.write("Categorical variables:", categorical_variables)

    with st.expander("View Dataset Information"):

        buffer = io.StringIO()
        df.info(buf=buffer)

        info_text = buffer.getvalue()
        st.text(info_text)

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    col1, col2, col3, col4 = st.columns(4)

    prevalence = (df[OUTCOME_COL].astype(str) == POSITIVE_LABEL).mean() * 100
    gender_share = df[GENDER_COL].value_counts(normalize=True) * 100

    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Hypertension Prevalence", f"{prevalence:.1f}%")
    col3.metric("Age Range", f"{int(df[AGE_COL].min())} - {int(df[AGE_COL].max())}")
    col4.metric(f"Most Common {GENDER_COL}", f"{gender_share.index[0]} ({gender_share.iloc[0]:.0f}%)")

    with st.expander("Data Quality and Preprocessing"):

        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        missing_total = int(missing.sum())
        duplicate_total = int(df.duplicated().sum())

        inconsistent_columns = []
        for column in categorical_variables:
            values = df[column].dropna().astype(str).unique()
            standardised = {v.strip().lower() for v in values}
            if len(standardised) < len(values):
                inconsistent_columns.append(column)

        if missing.empty:
            st.write("Missing values: none remain, so every summary uses complete records.")
        else:
            missing_table = pd.DataFrame({
                "Column": missing.index,
                "Missing Values": missing.values,
                "Percent of Records": (missing.values / len(df) * 100).round(1),
            })
            st.dataframe(missing_table, hide_index=True)
            st.write(
                f"Missing values: {missing_total} values are missing across {len(missing)} columns. "
                "They were not imputed. Each chart and summary excludes the missing values "
                "of the variable being shown, so counts can differ slightly between views."
            )

        if duplicate_total == 0:
            st.write("Duplicate rows: none remain, so no patient is counted twice.")
        else:
            st.write(f"Duplicate rows: {duplicate_total} remain and may slightly inflate counts.")

        if not inconsistent_columns:
            st.write(
                "Category labels: each category uses one consistent spelling, "
                "so groups are not split by wording."
            )
        else:
            st.write(f"Category labels: inconsistent spellings were found in {inconsistent_columns}.")

        st.write("Imputation: all numerical values shown are the original recorded values.")

        st.write(
            "Derived variable: BMI_Category was created from BMI using the standard adult bands "
            "(under 18.5, 18.5 to 24.9, 25 to 29.9, 30 and over)."
        )

        st.write(
            "Why this matters: consistently labelled data means differences between groups "
            "reflect the population rather than data errors. This app supports population-level "
            "exploration only and is not a diagnostic tool."
        )

    st.subheader("Population Context")

    outcome_table = df[OUTCOME_COL].value_counts().reset_index()
    outcome_table.columns = [OUTCOME_COL, "Count"]

    fig = px.bar(
        outcome_table,
        x=OUTCOME_COL,
        y="Count",
        title="Number of Patients With and Without Hypertension"
    )
    st.plotly_chart(fig)

    fig = px.histogram(
        df,
        x=AGE_COL,
        nbins=20,
        title="Age Distribution of the Patient Population"
    )
    st.plotly_chart(fig)

elif page == "Interactive Exploration":
    st.header("Interactive Exploration")

    st.sidebar.subheader("Population Filters")

    age_options = list(range(int(df[AGE_COL].min()), int(df[AGE_COL].max()) + 1))

    minimum_age = st.sidebar.selectbox("Minimum age", age_options, index=0)
    maximum_age = st.sidebar.selectbox("Maximum age", age_options, index=len(age_options) - 1)

    gender_options = sorted(df[GENDER_COL].dropna().unique().tolist())
    selected_genders = st.sidebar.multiselect(
        "Select gender",
        gender_options,
        default=gender_options
    )

    if minimum_age > maximum_age:
        st.warning("Minimum age is higher than maximum age. Please change the age filters.")
        st.stop()

    filtered_df = df[
        (df[AGE_COL] >= minimum_age)
        & (df[AGE_COL] < maximum_age + 1)
        & (df[GENDER_COL].isin(selected_genders))
    ]

    if filtered_df.empty:
        st.warning("No patients match the selected filters.")
        st.stop()

    col1, col2 = st.columns(2)
    col1.metric("Patients in Selection", f"{len(filtered_df):,}")
    col2.metric(
        "Hypertension Prevalence in Selection",
        f"{(filtered_df[OUTCOME_COL] == POSITIVE_LABEL).mean() * 100:.1f}%"
    )

    analysis_type = st.radio(
        "Choose Analysis Type",
        ["Univariate", "Bivariate", "Multivariate"]
    )

    if analysis_type == "Univariate":
        variable_type = st.radio("Variable type", ["Numerical", "Categorical"])

        if variable_type == "Numerical":
            variable = st.selectbox("Select Numerical Variable", numerical_variables)

            fig = px.histogram(
                filtered_df,
                x=variable,
                nbins=20,
                title=f"Distribution of {variable}"
            )
            st.plotly_chart(fig)

            st.write(f"This histogram shows how {variable} is spread across the selected patients.")
        else:
            variable = st.selectbox("Select Categorical Variable", categorical_variables)

            counts = filtered_df[variable].value_counts().reset_index()
            counts.columns = [variable, "Count"]

            fig = px.bar(
                counts,
                x=variable,
                y="Count",
                title=f"Distribution of {variable}",
                category_orders=order_for(variable)
            )
            st.plotly_chart(fig)

            st.write(f"This chart shows how many selected patients fall into each {variable} group.")

    elif analysis_type == "Bivariate":
        comparison = st.radio(
            "Choose Comparison",
            ["Numerical vs Numerical", "Population Group vs Hypertension Outcome"]
        )

        if comparison == "Numerical vs Numerical":
            x_variable = st.selectbox("Select X Variable", numerical_variables, index=0)
            y_variable = st.selectbox("Select Y Variable", numerical_variables, index=2)

            fig = px.scatter(
                filtered_df,
                x=x_variable,
                y=y_variable,
                title=f"{x_variable} vs {y_variable}"
            )
            st.plotly_chart(fig)

            correlation = filtered_df[x_variable].corr(filtered_df[y_variable])
            st.metric("Correlation", f"{correlation:.2f}")

            st.write(
                "Correlation shows how two measurements move together in this population. "
                "It does not show that one causes the other."
            )
        else:
            group = st.selectbox("Select Population Group", group_variables)

            percentage_table = pd.crosstab(
                filtered_df[group],
                filtered_df[OUTCOME_COL],
                normalize="index"
            ) * 100

            plot_table = percentage_table[POSITIVE_LABEL].reset_index()
            plot_table.columns = [group, "Hypertension Percentage"]

            fig = px.bar(
                plot_table,
                x=group,
                y="Hypertension Percentage",
                title=f"Percentage With Hypertension by {group}",
                category_orders=order_for(group)
            )
            st.plotly_chart(fig)

            st.write(
                "Percentages are used because the groups can differ in size. "
                "A difference between groups does not explain why it occurs."
            )

    else:
        x_variable = st.selectbox("Select X Variable", numerical_variables, index=0)
        y_variable = st.selectbox("Select Y Variable", numerical_variables, index=2)

        fig = px.scatter(
            filtered_df,
            x=x_variable,
            y=y_variable,
            color=OUTCOME_COL,
            title=f"{x_variable} vs {y_variable} by Hypertension Status"
        )
        st.plotly_chart(fig)

        st.write(
            "Each point is a patient, coloured by hypertension status. "
            "Use this to compare how groups cluster, not to assess an individual."
        )