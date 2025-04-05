"""
Data Analyzer Module

This module handles data analysis and visualization.
It processes data files and generates insights and visualizations.
"""

import io
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union

class DataAnalyzer:
    def __init__(self):
        """Initialize the data analyzer."""
        self.data = None
        self.analysis_results = {}
    
    def analyze_file(self, file) -> Dict[str, Any]:
        """
        Analyze the given data file.
        
        Args:
            file: The file to analyze (CSV or Excel)
            
        Returns:
            dict: Analysis results
        """
        try:
            file_size = file.size
            
            # Log the file size for debugging
            print(f"Analyzing file: {file.name}, Size: {file_size} bytes")
            
            # Determine file type and read accordingly
            if file.name.endswith('.csv'):
                # For large CSV files, use chunking to avoid memory issues
                if file_size > 50 * 1024 * 1024:  # If file is larger than 50MB
                    chunks = pd.read_csv(file, chunksize=10000)
                    self.data = next(chunks)  # Get first chunk for analysis
                    return {
                        "warning": f"Large file detected ({file_size:,} bytes). Analyzing first 10,000 rows. For full analysis, consider using a smaller dataset.",
                        "analysis": self._perform_analysis()
                    }
                else:
                    self.data = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                try:
                    if file_size > 20 * 1024 * 1024:  # If Excel file is larger than 20MB
                        return {
                            "warning": f"Large Excel file detected ({file_size:,} bytes). For better performance, consider converting to CSV or using a smaller dataset.",
                            "recommendation": "The file may take a long time to process. Please wait..."
                        }
                    # Explicitly use openpyxl engine
                    self.data = pd.read_excel(file, engine='openpyxl')
                except Exception as excel_error:
                    return {"error": f"Error analyzing Excel file: {str(excel_error)}. If this is an Excel file, make sure openpyxl is installed."}
            else:
                return {"error": "Unsupported file format. Please upload a CSV or Excel file."}
            
            # Perform basic analysis
            return self._perform_analysis()
        except Exception as e:
            return {"error": f"Error analyzing file: {str(e)}"}
    
    def analyze_csv(self, csv_content: str) -> Dict[str, Any]:
        """
        Analyze CSV content provided as a string.
        
        Args:
            csv_content: CSV content as a string
            
        Returns:
            dict: Analysis results
        """
        try:
            self.data = pd.read_csv(io.StringIO(csv_content))
            return self._perform_analysis()
        except Exception as e:
            return {"error": f"Error analyzing CSV content: {str(e)}"}
    
    def _perform_analysis(self) -> Dict[str, Any]:
        """
        Perform basic analysis on the loaded data.
        
        Returns:
            dict: Analysis results
        """
        if self.data is None:
            return {"error": "No data loaded for analysis."}
        
        try:
            # Basic information
            analysis = {
                "shape": {
                    "rows": self.data.shape[0],
                    "columns": self.data.shape[1]
                },
                "columns": list(self.data.columns),
                "data_types": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
                "missing_values": {col: int(self.data[col].isna().sum()) for col in self.data.columns},
                "summary": {}
            }
            
            # Summary statistics for numeric columns
            numeric_columns = self.data.select_dtypes(include=['number']).columns
            if not numeric_columns.empty:
                analysis["summary"]["numeric"] = self.data[numeric_columns].describe().to_dict()
            
            # Summary for categorical columns
            categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
            if not categorical_columns.empty:
                analysis["summary"]["categorical"] = {
                    col: {
                        "unique_values": int(self.data[col].nunique()),
                        "top_5_values": self.data[col].value_counts().head(5).to_dict()
                    } for col in categorical_columns
                }
            
            # Store results
            self.analysis_results = analysis
            
            # Generate human-readable summary
            return self._generate_summary_text()
        except Exception as e:
            return {"error": f"Error during analysis: {str(e)}"}
    
    def _generate_summary_text(self) -> Dict[str, str]:
        """
        Generate a human-readable summary of the analysis results.
        
        Returns:
            dict: Text summary
        """
        if not self.analysis_results:
            return {"summary": "No analysis results available."}
        
        # Basic dataset information
        summary = f"**Dataset Overview**\n\n"
        summary += f"- Dataset contains {self.analysis_results['shape']['rows']} rows and {self.analysis_results['shape']['columns']} columns.\n"
        
        # Missing values
        missing_values = sum(self.analysis_results['missing_values'].values())
        if missing_values > 0:
            summary += f"- Dataset contains {missing_values} missing values.\n"
            # Highlight columns with most missing values
            cols_with_missing = {k: v for k, v in self.analysis_results['missing_values'].items() if v > 0}
            if cols_with_missing:
                sorted_cols = sorted(cols_with_missing.items(), key=lambda x: x[1], reverse=True)
                summary += f"- Top columns with missing values: " + ", ".join([f"{col} ({val})" for col, val in sorted_cols[:3]]) + "\n"
        else:
            summary += "- No missing values found in the dataset.\n"
        
        # Numeric data
        if 'numeric' in self.analysis_results.get('summary', {}):
            summary += "\n**Numeric Data**\n\n"
            for col in self.analysis_results['summary']['numeric'].keys():
                col_data = self.analysis_results['summary']['numeric'][col]
                summary += f"- {col}: Range [{col_data.get('min', 'N/A'):.2f} - {col_data.get('max', 'N/A'):.2f}], "
                summary += f"Mean: {col_data.get('mean', 'N/A'):.2f}, Median: {col_data.get('50%', 'N/A'):.2f}\n"
        
        # Categorical data
        if 'categorical' in self.analysis_results.get('summary', {}):
            summary += "\n**Categorical Data**\n\n"
            for col, data in self.analysis_results['summary']['categorical'].items():
                summary += f"- {col}: {data['unique_values']} unique values\n"
                if data['unique_values'] <= 10:
                    # If few unique values, show all
                    summary += f"  - Values: {', '.join(list(data['top_5_values'].keys()))}\n"
                else:
                    # Otherwise show most common
                    summary += f"  - Most common: {', '.join(list(data['top_5_values'].keys())[:3])}\n"
        
        # Add potential insights
        summary += "\n**Potential Insights**\n\n"
        # For numeric columns, look at correlations (in a real implementation)
        summary += "- To further analyze this data, you could:\n"
        summary += "  - Create visualizations to see distributions and relationships\n"
        summary += "  - Look for correlations between numeric variables\n"
        summary += "  - Perform statistical tests to validate hypotheses\n"
        summary += "  - Clean and transform data for modeling\n"
        
        return {"summary": summary}
    
    def create_visualization(self, file, chart_type: str):
        """
        Create a visualization based on the uploaded file and chart type.
        
        Args:
            file: The data file
            chart_type: The type of chart to create
            
        Returns:
            A Plotly figure object
        """
        try:
            file_size = file.size
            
            # Log the file size for debugging
            print(f"Creating visualization for file: {file.name}, Size: {file_size} bytes")
            
            # Read the data with size limits
            if file.name.endswith('.csv'):
                # For large CSV files, use sampling to avoid memory issues
                if file_size > 20 * 1024 * 1024:  # If file is larger than 20MB
                    # Use pandas sampling for large files
                    data = pd.read_csv(file, nrows=10000)  # Read only first 10,000 rows
                    print(f"Large CSV file. Sampling first 10,000 rows for visualization.")
                else:
                    data = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                try:
                    if file_size > 10 * 1024 * 1024:  # If Excel file is larger than 10MB
                        print(f"Large Excel file. May take longer to process.")
                    # Explicitly use openpyxl engine
                    data = pd.read_excel(file, engine='openpyxl')
                except Exception as excel_error:
                    raise ValueError(f"Error reading Excel file: {str(excel_error)}. Make sure openpyxl is installed.")
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
            
            # Store the data
            self.data = data
            
            # Determine appropriate columns for the chart type
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not numeric_columns:
                raise ValueError("No numeric columns found in the data. Visualization requires numeric data.")
            
            # Create the appropriate chart
            if chart_type == "Bar Chart":
                if categorical_columns:
                    x = categorical_columns[0]  # Use first categorical column for x-axis
                    y = numeric_columns[0]      # Use first numeric column for y-axis
                    fig = px.bar(data, x=x, y=y, title=f'Bar Chart of {y} by {x}')
                else:
                    # If no categorical columns, use the numeric column itself
                    x = numeric_columns[0]
                    fig = px.bar(data, x=x, title=f'Bar Chart of {x}')
            
            elif chart_type == "Line Chart":
                if len(numeric_columns) >= 2:
                    x = numeric_columns[0]
                    y = numeric_columns[1]
                else:
                    # If only one numeric column, try using index as x-axis
                    x = data.index
                    y = numeric_columns[0]
                fig = px.line(data, x=x, y=y, title=f'Line Chart of {y}')
            
            elif chart_type == "Scatter Plot":
                if len(numeric_columns) >= 2:
                    x = numeric_columns[0]
                    y = numeric_columns[1]
                    fig = px.scatter(data, x=x, y=y, title=f'Scatter Plot of {y} vs {x}')
                else:
                    raise ValueError("Scatter plot requires at least two numeric columns.")
            
            elif chart_type == "Pie Chart":
                if categorical_columns:
                    values = numeric_columns[0]
                    names = categorical_columns[0]
                    fig = px.pie(data, values=values, names=names, title=f'Pie Chart of {values} by {names}')
                else:
                    raise ValueError("Pie chart requires at least one categorical column and one numeric column.")
            
            elif chart_type == "Heatmap":
                if len(numeric_columns) >= 2:
                    # Create correlation matrix
                    corr_matrix = data[numeric_columns].corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='Viridis',
                        colorbar=dict(title='Correlation')
                    ))
                    fig.update_layout(title='Correlation Heatmap')
                else:
                    raise ValueError("Heatmap requires at least two numeric columns for correlation analysis.")
            
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            # Update layout for better appearance
            fig.update_layout(
                template="plotly_dark",
                xaxis_title=x,
                yaxis_title=y if chart_type != "Pie Chart" else None,
                legend_title="Legend"
            )
            
            return fig
        
        except Exception as e:
            # Return a simple error message figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def generate_report(self, file) -> str:
        """
        Generate a comprehensive report based on the uploaded file.
        
        Args:
            file: The data file
            
        Returns:
            str: HTML or markdown report
        """
        try:
            # Analyze the file
            analysis = self.analyze_file(file)
            
            # Create a report
            report = "# Data Analysis Report\n\n"
            
            # Dataset overview
            report += "## Dataset Overview\n\n"
            if "shape" in self.analysis_results:
                report += f"- **Rows**: {self.analysis_results['shape']['rows']}\n"
                report += f"- **Columns**: {self.analysis_results['shape']['columns']}\n\n"
            
            # Column information
            report += "## Column Information\n\n"
            report += "| Column | Type | Missing Values |\n"
            report += "| ------ | ---- | -------------- |\n"
            for col in self.analysis_results.get('columns', []):
                col_type = self.analysis_results['data_types'].get(col, 'Unknown')
                missing = self.analysis_results['missing_values'].get(col, 0)
                report += f"| {col} | {col_type} | {missing} |\n"
            
            # Statistical summary
            if 'numeric' in self.analysis_results.get('summary', {}):
                report += "\n## Statistical Summary\n\n"
                for col in self.analysis_results['summary']['numeric'].keys():
                    report += f"### {col}\n\n"
                    col_data = self.analysis_results['summary']['numeric'][col]
                    report += f"- **Min**: {col_data.get('min', 'N/A'):.2f}\n"
                    report += f"- **25th Percentile**: {col_data.get('25%', 'N/A'):.2f}\n"
                    report += f"- **Median**: {col_data.get('50%', 'N/A'):.2f}\n"
                    report += f"- **Mean**: {col_data.get('mean', 'N/A'):.2f}\n"
                    report += f"- **75th Percentile**: {col_data.get('75%', 'N/A'):.2f}\n"
                    report += f"- **Max**: {col_data.get('max', 'N/A'):.2f}\n"
                    report += f"- **Standard Deviation**: {col_data.get('std', 'N/A'):.2f}\n\n"
            
            # Categorical data
            if 'categorical' in self.analysis_results.get('summary', {}):
                report += "\n## Categorical Data Summary\n\n"
                for col, data in self.analysis_results['summary']['categorical'].items():
                    report += f"### {col}\n\n"
                    report += f"- **Unique Values**: {data['unique_values']}\n"
                    report += "- **Top Values**:\n"
                    for val, count in list(data['top_5_values'].items())[:5]:
                        report += f"  - {val}: {count}\n"
                    report += "\n"
            
            # Recommendations
            report += "## Recommendations\n\n"
            report += "Based on the analysis, here are some recommendations:\n\n"
            
            # Check for missing values
            missing_values = sum(self.analysis_results.get('missing_values', {}).values())
            if missing_values > 0:
                report += "- **Handle Missing Values**: The dataset contains missing values that should be addressed before further analysis.\n"
            
            # Check for potential outliers in numeric data
            if 'numeric' in self.analysis_results.get('summary', {}):
                for col in self.analysis_results['summary']['numeric'].keys():
                    col_data = self.analysis_results['summary']['numeric'][col]
                    # Simple outlier check: if max is much larger than 75th percentile or min is much smaller than 25th percentile
                    if col_data.get('max', 0) > col_data.get('75%', 0) * 1.5 or col_data.get('min', 0) < col_data.get('25%', 0) * 0.5:
                        report += f"- **Check for Outliers**: Column '{col}' may contain outliers that could affect analysis.\n"
                        break
            
            # General recommendations
            report += "- **Visualize Relationships**: Create visualizations to explore relationships between variables.\n"
            report += "- **Feature Engineering**: Consider creating new features that might better capture patterns in the data.\n"
            report += "- **Data Transformation**: For skewed numeric variables, consider applying transformations like log or square root.\n"
            
            return report
        
        except Exception as e:
            return f"Error generating report: {str(e)}"
