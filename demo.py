import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

INPUT_FILE = "patient_data.csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cleaned_patients.csv"
SUMMARY_FILE = OUTPUT_DIR / "patient_summary.csv"
ERROR_FILE = OUTPUT_DIR / "error_records.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# 2. CREATE SAMPLE DATA
# --------------------------------------------------

def create_sample_data():
    data = {
        "patient_id": [
            101, 102, 103, 104, 105, 101, 106, 107, 108, 109
        ],
        "patient_name": [
            "Rahul", "Priya", "Amit", "Sneha", "Rohan",
            "Rahul", "Neha", "Karan", None, "Pooja"
        ],
        "age": [
            35, 28, 45, 32, 29, 35, 150, 40, 25, 31
        ],
        "gender": [
            "Male", "Female", "Male", "Female", "Male",
            "Male", "Female", "Male", "Female", "Female"
        ],
        "city": [
            "Mumbai", "Pune", "Delhi", "Mumbai", "Pune",
            "Mumbai", "Delhi", None, "Pune", "Mumbai"
        ],
        "bill_amount": [
            1500, 2500, 3200, 1800, 2100,
            1500, 5000, 2700, 1600, -500
        ],
        "admission_date": [
            "2026-01-10", "2026-01-11", "2026-01-12",
            "2026-01-13", "2026-01-14", "2026-01-10",
            "2026-01-15", "2026-01-16", "2026-01-17",
            "2026-01-18"
        ],
        "department": [
            "Cardiology", "Neurology", "Orthopedics",
            "Cardiology", "Neurology", "Cardiology",
            "Orthopedics", "Cardiology", "Neurology",
            "Cardiology"
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv(INPUT_FILE, index=False)

    logging.info("Sample patient data created.")


# --------------------------------------------------
# 3. READ DATA
# --------------------------------------------------

def read_data(file_path):
    try:
        df = pd.read_csv(file_path)

        logging.info("Input data loaded successfully.")
        logging.info("Total records: %s", len(df))

        return df

    except FileNotFoundError:
        logging.error("Input file not found: %s", file_path)
        raise

    except pd.errors.ParserError:
        logging.error("Input CSV contains parsing errors.")
        raise


# --------------------------------------------------
# 4. VALIDATE DATA
# --------------------------------------------------

def validate_data(df):
    required_columns = [
        "patient_id",
        "patient_name",
        "age",
        "gender",
        "city",
        "bill_amount",
        "admission_date",
        "department"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    logging.info("Schema validation completed.")

    return df


# --------------------------------------------------
# 5. CLEAN DATA
# --------------------------------------------------

def clean_data(df):
    df = df.copy()

    # Remove leading and trailing spaces from text columns.
    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # Convert columns to their expected data types.
    df["patient_id"] = pd.to_numeric(
        df["patient_id"], errors="coerce"
    )

    df["age"] = pd.to_numeric(
        df["age"], errors="coerce"
    )

    df["bill_amount"] = pd.to_numeric(
        df["bill_amount"], errors="coerce"
    )

    df["admission_date"] = pd.to_datetime(
        df["admission_date"], errors="coerce"
    )

    # Store records with invalid critical fields separately.
    invalid_mask = (
        df["patient_id"].isna()
        | df["age"].isna()
        | df["bill_amount"].isna()
        | df["admission_date"].isna()
        | df["patient_name"].isna()
    )

    invalid_records = df[invalid_mask].copy()

    # Save invalid records for investigation.
    invalid_records.to_csv(ERROR_FILE, index=False)

    # Continue processing valid records.
    df = df[~invalid_mask].copy()

    # Remove duplicate patient records.
    df = df.drop_duplicates(
        subset=["patient_id"],
        keep="first"
    )

    # Validate age and billing values.
    df = df[
        (df["age"] >= 0)
        & (df["age"] <= 120)
        & (df["bill_amount"] >= 0)
    ].copy()

    # Fill missing city values.
    df["city"] = df["city"].fillna("Unknown")

    # Normalize text values.
    df["gender"] = df["gender"].str.title()
    df["city"] = df["city"].str.title()
    df["department"] = df["department"].str.title()

    logging.info("Data cleaning completed.")
    logging.info("Valid records: %s", len(df))
    logging.info("Invalid records: %s", len(invalid_records))

    return df


# --------------------------------------------------
# 6. TRANSFORM DATA
# --------------------------------------------------

def transform_data(df):
    df = df.copy()

    # Create age groups.
    df["age_group"] = pd.cut(
        df["age"],
        bins=[-1, 17, 35, 60, 120],
        labels=[
            "Child",
            "Young Adult",
            "Middle Aged",
            "Senior"
        ]
    )

    # Create billing categories.
    df["bill_category"] = pd.cut(
        df["bill_amount"],
        bins=[-1, 2000, 4000, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    # Extract admission year and month.
    df["admission_year"] = (
        df["admission_date"].dt.year
    )

    df["admission_month"] = (
        df["admission_date"].dt.month
    )

    # Calculate tax at 5% for demonstration.
    df["tax_amount"] = (
        df["bill_amount"] * 0.05
    ).round(2)

    # Calculate final bill.
    df["total_bill"] = (
        df["bill_amount"] + df["tax_amount"]
    ).round(2)

    logging.info("Data transformation completed.")

    return df


# --------------------------------------------------
# 7. GENERATE DEPARTMENT SUMMARY
# --------------------------------------------------

def generate_department_summary(df):
    summary = (
        df.groupby("department", as_index=False)
        .agg(
            total_patients=("patient_id", "count"),
            total_billing=("bill_amount", "sum"),
            average_bill=("bill_amount", "mean"),
            maximum_bill=("bill_amount", "max"),
            minimum_bill=("bill_amount", "min")
        )
    )

    summary["average_bill"] = (
        summary["average_bill"].round(2)
    )

    summary["total_billing"] = (
        summary["total_billing"].round(2)
    )

    summary = summary.sort_values(
        by="total_billing",
        ascending=False
    )

    logging.info("Department summary generated.")

    return summary


# --------------------------------------------------
# 8. GENERATE CITY SUMMARY
# --------------------------------------------------

def generate_city_summary(df):
    city_summary = (
        df.groupby("city", as_index=False)
        .agg(
            patient_count=("patient_id", "count"),
            total_revenue=("total_bill", "sum")
        )
        .sort_values(
            by="total_revenue",
            ascending=False
        )
    )

    logging.info("City summary generated.")

    return city_summary


# --------------------------------------------------
# 9. GENERATE PATIENT REPORT
# --------------------------------------------------

def generate_patient_report(df):
    report_columns = [
        "patient_id",
        "patient_name",
        "age",
        "age_group",
        "gender",
        "city",
        "department",
        "bill_amount",
        "tax_amount",
        "total_bill",
        "admission_date"
    ]

    report = df[report_columns].copy()

    report = report.sort_values(
        by="total_bill",
        ascending=False
    )

    logging.info("Patient report generated.")

    return report


# --------------------------------------------------
# 10. DATA QUALITY CHECKS
# --------------------------------------------------

def run_quality_checks(df):
    checks = {
        "total_records": len(df),
        "duplicate_patient_ids": int(
            df["patient_id"].duplicated().sum()
        ),
        "missing_patient_names": int(
            df["patient_name"].isna().sum()
        ),
        "missing_cities": int(
            df["city"].isna().sum()
        ),
        "invalid_ages": int(
            ((df["age"] < 0) | (df["age"] > 120)).sum()
        ),
        "negative_bills": int(
            (df["bill_amount"] < 0).sum()
        )
    }

    print("\n========== DATA QUALITY REPORT ==========")

    for check_name, result in checks.items():
        print(f"{check_name}: {result}")

    print("==========================================\n")

    logging.info("Data quality checks completed.")

    return checks


# --------------------------------------------------
# 11. SAVE OUTPUT FILES
# --------------------------------------------------

def save_outputs(df, department_summary, city_summary, report):
    df.to_csv(OUTPUT_FILE, index=False)

    department_summary.to_csv(
        OUTPUT_DIR / "department_summary.csv",
        index=False
    )

    city_summary.to_csv(
        OUTPUT_DIR / "city_summary.csv",
        index=False
    )

    report.to_csv(
        SUMMARY_FILE,
        index=False
    )

    logging.info("All output files saved successfully.")


# --------------------------------------------------
# 12. PRINT FINAL STATISTICS
# --------------------------------------------------

def print_statistics(df):
    print("\n========== FINAL STATISTICS ==========")

    print("Total patients:", len(df))

    print(
        "Total billing amount:",
        round(df["bill_amount"].sum(), 2)
    )

    print(
        "Average bill amount:",
        round(df["bill_amount"].mean(), 2)
    )

    print(
        "Total tax amount:",
        round(df["tax_amount"].sum(), 2)
    )

    print(
        "Total revenue including tax:",
        round(df["total_bill"].sum(), 2)
    )

    print("\nDepartment-wise patient count:")

    print(
        df.groupby("department")["patient_id"]
        .count()
    )

    print("\nTop 3 patients by bill amount:")

    print(
        df.nlargest(3, "total_bill")[
            ["patient_name", "department", "total_bill"]
        ]
    )

    print("======================================\n")


# --------------------------------------------------
# 13. MAIN PIPELINE
# --------------------------------------------------

def main():
    start_time = datetime.now()

    logging.info("Healthcare data pipeline started.")

    try:
        # Step 1: Create sample input.
        create_sample_data()

        # Step 2: Read source data.
        raw_df = read_data(INPUT_FILE)

        # Step 3: Validate schema.
        validated_df = validate_data(raw_df)

        # Step 4: Clean records.
        cleaned_df = clean_data(validated_df)

        # Step 5: Transform records.
        transformed_df = transform_data(cleaned_df)

        # Step 6: Run data quality checks.
        run_quality_checks(transformed_df)

        # Step 7: Generate reports.
        department_summary = (
            generate_department_summary(transformed_df)
        )

        city_summary = generate_city_summary(
            transformed_df
        )

        patient_report = generate_patient_report(
            transformed_df
        )

        # Step 8: Save results.
        save_outputs(
            transformed_df,
            department_summary,
            city_summary,
            patient_report
        )

        # Step 9: Print final statistics.
        print_statistics(transformed_df)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logging.info(
            "Pipeline completed successfully in %.2f seconds.",
            duration
        )

    except Exception:
        logging.exception("Pipeline failed.")
        raise


# --------------------------------------------------
# 14. EXECUTION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()