# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis Tools
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def create_distribution_plots(df, output_dir):
    """Create distribution plots for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return []
    
    print("\nCreating distribution plots...")
    plot_paths = []
    
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_title('Distribution of {}'.format(col))
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot(df[col].dropna())
        axes[1].set_title('Box Plot of {}'.format(col))
        axes[1].set_ylabel(col)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = Path(output_dir) / 'dist_{}.png'.format(col.lower())
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_paths.append(str(plot_path))
        print("  - Created: {}".format(plot_path.name))
    
    return plot_paths

def create_correlation_matrix(df, output_dir):
    """Create correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    print("\nCreating correlation matrix...")
    
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'correlation_matrix.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print("  - Created: {}".format(plot_path.name))
    return str(plot_path)

def create_categorical_plots(df, output_dir, max_categories=10):
    """Create bar plots for categorical columns"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) == 0:
        return []
    
    print("\nCreating categorical plots...")
    plot_paths = []
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        
        if len(value_counts) > max_categories:
            print("  - Skipped '{}': too many categories".format(col))
            continue
        
        plt.figure(figsize=(10, 6))
        value_counts.plot(kind='bar', edgecolor='black', alpha=0.7)
        plt.title('Distribution of {}'.format(col))
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plot_path = Path(output_dir) / 'cat_{}.png'.format(col.lower())
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_paths.append(str(plot_path))
        print("  - Created: {}".format(plot_path.name))
    
    return plot_paths

def calculate_statistics(df):
    """Calculate comprehensive statistics"""
    stats = {
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        stats['numeric_summary'][col] = {
            'count': int(df[col].count()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'q25': float(df[col].quantile(0.25)),
            'median': float(df[col].median()),
            'q75': float(df[col].quantile(0.75)),
            'max': float(df[col].max())
        }
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        stats['categorical_summary'][col] = {
            'count': int(df[col].count()),
            'unique': int(df[col].nunique()),
            'top_value': str(value_counts.index[0]),
            'top_count': int(value_counts.iloc[0])
        }
    
    return stats

def generate_insights(df, stats):
    """Generate key insights"""
    insights = []
    
    insights.append("Dataset Overview:")
    insights.append("- Total records: {:,}".format(len(df)))
    insights.append("- Total features: {}".format(len(df.columns)))
    insights.append("")
    
    if stats['numeric_summary']:
        insights.append("Numeric Features:")
        for col, s in stats['numeric_summary'].items():
            insights.append("  {}: mean={:.2f}, range=[{:.2f}, {:.2f}]".format(
                col, s['mean'], s['min'], s['max']))
        insights.append("")
    
    if stats['categorical_summary']:
        insights.append("Categorical Features:")
        for col, s in stats['categorical_summary'].items():
            insights.append("  {}: {} unique values".format(col, s['unique']))
        insights.append("")
    
    return "\n".join(insights)

def create_eda_report(input_path, output_dir):
    """Main EDA function"""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print("\nLoading: {}".format(input_path))
    df = pd.read_csv(input_path)
    print("Loaded: {} rows, {} columns".format(len(df), len(df.columns)))
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\nCalculating statistics...")
    stats = calculate_statistics(df)
    
    dist_plots = create_distribution_plots(df, output_dir)
    corr_plot = create_correlation_matrix(df, output_dir)
    cat_plots = create_categorical_plots(df, output_dir)
    
    print("\nGenerating insights...")
    insights_text = generate_insights(df, stats)
    
    insights_path = Path(output_dir) / 'insights.md'
    with open(insights_path, 'w') as f:
        f.write("# EDA Insights\n\n")
        f.write(insights_text)
    print("Insights saved: {}".format(insights_path))
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE!")
    print("=" * 60)
    print("\nGenerated:")
    print("  - insights.md")
    print("  - {} plots".format(len(dist_plots) + len(cat_plots) + (1 if corr_plot else 0)))
    
    return {
        'insights': str(insights_path),
        'plots': dist_plots + cat_plots + ([corr_plot] if corr_plot else [])
    }

if __name__ == "__main__":
    print("TESTING EDA TOOLS\n")
    
    report_info = create_eda_report(
        input_path='artifacts/analyst/clean_data.csv',
        output_dir='artifacts/analyst'
    )
    
    print("\nDONE!")
