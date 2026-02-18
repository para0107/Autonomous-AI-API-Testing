import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import os


def analyze_dataset(file_path='simplified_tests.json', output_dir='analysis_output'):
    """
    Performs detailed statistical analysis on QASE test data and generates
    visualizations and a Markdown report.
    """

    # 1. Setup and Loading
    print(f"--- Starting Analysis of {file_path} ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}/")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} test cases.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load file. {e}")
        return

    # 2. Data Processing & Flattening
    # We convert the nested JSON into a flat DataFrame for analysis
    df_tests = pd.DataFrame(data)

    # Calculate character counts for text fields
    df_tests['name_len'] = df_tests['test_name'].fillna('').astype(str).apply(len)
    df_tests['objective_len'] = df_tests['objective'].fillna('').astype(str).apply(len)
    df_tests['preconditions_len'] = df_tests['preconditions'].fillna('').astype(str).apply(len)
    df_tests['step_count'] = df_tests['steps'].apply(len)

    # Flatten steps to analyze individual actions
    all_steps = []
    for test in data:
        for step in test.get('steps', []):
            all_steps.append(step)
    df_steps = pd.DataFrame(all_steps)

    # Analyze steps if they exist
    if not df_steps.empty:
        df_steps['action_len'] = df_steps['action'].fillna('').astype(str).apply(len)

        # Detect HTTP methods in the 'action' text
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        for method in methods:
            # Case-insensitive search for method names
            df_steps[method] = df_steps['action'].str.contains(method, case=False, regex=False)

    # 3. Generating Visualizations
    sns.set_theme(style="whitegrid")  # Clean plotting style

    # Plot A: Distribution of Text Lengths (Histograms)
    print("Generating Text Length Distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.histplot(df_tests['name_len'], kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Test Name Length Distribution')

    sns.histplot(df_tests['objective_len'], kde=True, ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title('Objective/Description Length Distribution')

    sns.histplot(df_tests['preconditions_len'], kde=True, ax=axes[1, 0], color='salmon')
    axes[1, 0].set_title('Preconditions Length Distribution')

    if not df_steps.empty:
        sns.histplot(df_steps['action_len'], kde=True, ax=axes[1, 1], color='orange')
        axes[1, 1].set_title('Step Action Length Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'text_distributions.png'))
    plt.close()

    # Plot B: Steps per Test (Bar Chart)
    print("Generating Steps per Test Chart...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df_tests['step_count'], palette='viridis')
    plt.title('Distribution of Steps per Test Case')
    plt.xlabel('Number of Steps')
    plt.ylabel('Count of Tests')
    plt.savefig(os.path.join(output_dir, 'steps_count.png'))
    plt.close()

    # Plot C: HTTP Method Distribution
    if not df_steps.empty:
        print("Generating HTTP Method Chart...")
        method_counts = df_steps[methods].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(x=method_counts.index, y=method_counts.values, palette='magma')
        plt.title('Frequency of HTTP Methods in Steps')
        plt.ylabel('Count')
        # Add labels on top of bars
        for i, v in enumerate(method_counts.values):
            barplot.text(i, v, str(v), ha='center', va='bottom')
        plt.savefig(os.path.join(output_dir, 'http_methods.png'))
        plt.close()

    # Plot D: Boxplots for Outliers
    print("Generating Outlier Boxplots...")
    plt.figure(figsize=(12, 6))
    data_to_plot = [df_tests['name_len'], df_tests['objective_len'], df_tests['preconditions_len']]
    labels = ['Name', 'Objective', 'Preconditions']

    sns.boxplot(data=data_to_plot, orient='h', palette='Set2')
    plt.yticks(range(len(labels)), labels)
    plt.title('Text Length Outliers')
    plt.savefig(os.path.join(output_dir, 'outliers.png'))
    plt.close()

    # 4. Calculating Statistics & Generating Report
    print("Calculating Statistics...")

    def get_stats(series):
        return {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'skew': skew(series),
            'kurtosis': kurtosis(series),
            'max': series.max()
        }

    stats_name = get_stats(df_tests['name_len'])
    stats_obj = get_stats(df_tests['objective_len'])
    stats_steps = get_stats(df_tests['step_count'])

    # Determine dominant method
    dominant_method = "N/A"
    if not df_steps.empty:
        dominant_method = method_counts.index[0]
        dominant_pct = (method_counts.iloc[0] / len(df_steps)) * 100

    # Write Markdown Report
    report_content = f"""# 📊 Data Analysis Report
**Dataset:** `{file_path}`
**Total Tests:** {len(df_tests)}
**Total Steps:** {len(df_steps)}

## 1. Executive Summary
The dataset is highly **{"atomic" if stats_steps['mean'] < 2 else "complex"}**, with an average of **{stats_steps['mean']:.2f}** steps per test.
The dominant HTTP method is **{dominant_method}**, appearing in **{dominant_pct:.1f}%** of all steps.

## 2. Structural Analysis (Complexity)
- **Skewness:** {stats_steps['skew']:.2f} (High positive skew means mostly short tests).
- **Kurtosis:** {stats_steps['kurtosis']:.2f} (High kurtosis means outlier "long" tests are rare).

![Steps Distribution](steps_count.png)

## 3. Content Analysis (Verbosity)
| Field | Mean Length | Median | Max | Std Dev |
| :--- | :--- | :--- | :--- | :--- |
| **Name** | {stats_name['mean']:.1f} | {stats_name['median']} | {stats_name['max']} | {stats_name['std']:.1f} |
| **Objective** | {stats_obj['mean']:.1f} | {stats_obj['median']} | {stats_obj['max']} | {stats_obj['std']:.1f} |

![Text Distributions](text_distributions.png)

## 4. API Coverage
The following chart shows the distribution of HTTP methods detected in the 'action' fields.
![HTTP Methods](http_methods.png)

---
*Report generated by Auto-Analysis Script*
"""

    report_path = os.path.join(output_dir, 'analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✅ Analysis Complete!")
    print(f"📂 Results saved to: {os.path.abspath(output_dir)}")
    print(f"📄 Report: {report_path}")


if __name__ == "__main__":
    # Ensure dependencies are installed:
    # pip install pandas numpy matplotlib seaborn scipy
    analyze_dataset()