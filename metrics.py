import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def calculate_metrics(clean_file: pd.DataFrame, cleaned_file: pd.DataFrame, error_file: pd.DataFrame) -> pd.DataFrame:
    if not all(clean_file.columns == cleaned_file.columns) or not all(clean_file.columns == error_file.columns):
        raise ValueError("Files have mismatched columns.")

    metrics = []

    total_correct_repairs = 0
    total_repairs = 0
    total_errors = 0

    for column in clean_file.columns:
        # Ground truth, cleaned output, and input with errors for the column
        ground_truth = clean_file[column]
        cleaned_output = cleaned_file[column]
        input_with_errors = error_file[column]

        # Calculate correct repairs
        correct_repairs = (input_with_errors != ground_truth) & (cleaned_output == ground_truth)
        repairs = (input_with_errors != cleaned_output)
        errors = (input_with_errors != ground_truth)

        precision = correct_repairs.sum() / repairs.sum() if repairs.sum() > 0 else 0
        recall = correct_repairs.sum() / errors.sum() if errors.sum() > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        total_correct_repairs += correct_repairs.sum()
        total_repairs += repairs.sum()
        total_errors += errors.sum()

        metrics.append({
            "column": column,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    overall_precision = total_correct_repairs / total_repairs if total_repairs > 0 else 0
    overall_recall = total_correct_repairs / total_errors if total_errors > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0

    return pd.DataFrame(metrics), overall_precision, overall_recall, overall_f1

# File paths
clean_file_path = "clean_testfile.csv"
error_file_path = "testfile.csv"
cleaned_file_path = "cleaned_data.csv"

# Load the files
clean_file = load_csv(clean_file_path)
error_file = load_csv(error_file_path)
cleaned_file = load_csv(cleaned_file_path)

# Calculate column-wise metrics
metrics_df, overall_precision, overall_recall, overall_f1 = calculate_metrics(clean_file, cleaned_file, error_file)

# Display results
print("Column-wise Metrics:")
print(metrics_df)
print("\nOverall Precision Score:", overall_precision)
print("\nOverall Recall Score:", overall_recall)
print("\nOverall F1 Score:", overall_f1)
