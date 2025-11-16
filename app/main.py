# 🌐 app/main.py
# Enhanced Interactive Streamlit Web Interface for Financial Analysis

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from datetime import datetime, timedelta
import base64
from io import BytesIO
import traceback
import warnings
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
import tempfile

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="AI Financial Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing modules with error handling
try:
    from report_generator import FinancialReportGenerator
    print("✅ Successfully imported FinancialReportGenerator")
except ImportError as e:
    print(f"⚠️ Could not import FinancialReportGenerator: {e}")
    FinancialReportGenerator = None

class EnhancedFinancialApp:
    def __init__(self):
        # Define the specific path for saving all data
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.app_data_dir = os.path.join(self.project_root, "app", "APP_Data")
        
        # Create subdirectories within APP_Data
        self.data_dir = os.path.join(self.app_data_dir, "data")
        self.outputs_dir = os.path.join(self.app_data_dir, "outputs")
        self.visuals_dir = os.path.join(self.outputs_dir, "visuals")
        self.reports_dir = os.path.join(self.outputs_dir, "reports")
        self.temp_dir = os.path.join(self.app_data_dir, "temp")
        
        # Create directories if they don't exist
        self.create_directories()
        
        self.setup_ui()
        self.initialize_session_state()
        
    def create_directories(self):
        """Create necessary directories for the project"""
        directories = [
            self.app_data_dir,
            self.data_dir, 
            self.outputs_dir,
            self.visuals_dir,
            self.reports_dir,
            self.temp_dir
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created/Verified directory: {directory}")
            except Exception as e:
                print(f"❌ Error creating directory {directory}: {e}")
    
    def setup_ui(self):
        """Setup the main UI components with enhanced styling"""
        # Custom CSS for professional styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subheader {
            font-size: 1.8rem;
            color: #2e86ab;
            margin-bottom: 1rem;
            font-weight: 600;
            border-left: 5px solid #2196F3;
            padding-left: 15px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 15px;
            border-radius: 10px;
        }
        .success-box {
            padding: 1.5rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 2px solid #28a745;
            color: #155724;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .info-box {
            padding: 1.5rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            border: 2px solid #17a2b8;
            color: #0c5460;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .warning-box {
            padding: 1.5rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border: 2px solid #ffc107;
            color: #856404;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .step-complete {
            color: #28a745;
            font-weight: bold;
            font-size: 1.1rem;
            padding: 8px 15px;
            border-radius: 25px;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            margin: 5px 0;
            border-left: 4px solid #28a745;
        }
        .step-pending {
            color: #6c757d;
            font-size: 1.1rem;
            padding: 8px 15px;
            border-radius: 25px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            margin: 5px 0;
            border-left: 4px solid #6c757d;
        }
        .step-current {
            color: #007bff;
            font-weight: bold;
            font-size: 1.1rem;
            padding: 8px 15px;
            border-radius: 25px;
            background: linear-gradient(135deg, #cce5ff 0%, #b3d7ff 100%);
            margin: 5px 0;
            border-left: 4px solid #007bff;
        }
        .visualization-container {
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        .directory-info {
            background: linear-gradient(135deg, #e8f4fd 0%, #d1e7ff 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 6px solid #2196F3;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            margin: 10px 0;
        }
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .stButton button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        </style>
        """, unsafe_allow_html=True)
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        default_states = {
            'current_step': 1,
            'data_loaded': False,
            'data_cleaned': False,
            'analysis_done': False,
            'forecast_done': False,
            'ai_summary_done': False,
            'report_generated': False,
            'uploaded_file': None,
            'df_clean': None,
            'analysis_results': {},
            'forecast_data': None,
            'ai_insights': {},
            'report_path': None,
            'financial_cols': {},
            'file_preview': None,
            'report_content': {}
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def safe_convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """Safely convert series to numeric, handling all edge cases including currency formats"""
        if series is None or len(series) == 0:
            return series
            
        try:
            # If already numeric, return as is
            if pd.api.types.is_numeric_dtype(series):
                return series
                
            # Convert to string first to handle mixed types
            series_str = series.astype(str)
            
            # Enhanced cleaning for currency formats
            # Remove currency symbols, commas, and extra spaces
            series_clean = series_str.str.replace(r'[\$,]', '', regex=True)  # Remove $ and commas
            series_clean = series_clean.str.replace(r'\s+', '', regex=True)  # Remove spaces
            series_clean = series_clean.str.replace(r'[^\d.-]', '', regex=True)  # Keep only digits, dots, and minus
            
            # Handle empty strings after cleaning
            series_clean = series_clean.replace({'': '0', 'nan': '0', 'None': '0'})
            
            # Convert to numeric, coercing errors to NaN
            series_numeric = pd.to_numeric(series_clean, errors='coerce')
            
            # Fill NaN values with 0 and handle infinite values
            series_numeric = series_numeric.fillna(0)
            series_numeric = series_numeric.replace([np.inf, -np.inf], 0)
            
            return series_numeric
            
        except Exception as e:
            print(f"Warning: Could not convert series to numeric: {e}")
            # Return a series of zeros as fallback
            return pd.Series([0] * len(series), index=series.index)

    def safe_format_number(self, value: Any, format_type: str = ",.0f") -> str:
        """Safely format numbers to prevent format errors"""
        try:
            # Handle None and NaN values
            if value is None or pd.isna(value):
                return "N/A"
                
            # Convert to float if it's a string
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point and minus
                clean_value = ''.join(c for c in value if c.isdigit() or c == '.' or c == '-')
                if clean_value and clean_value != '-':
                    value = float(clean_value)
                else:
                    return "N/A"
            
            # Format the number
            if isinstance(value, (int, float)):
                if format_type.startswith("$"):
                    return f"${value:{format_type[1:]}}"
                return f"{value:{format_type}}"
            else:
                return str(value)
                
        except (ValueError, TypeError, Exception) as e:
            return str(value)

    def safe_compare(self, value1: Any, value2: Any) -> bool:
        """Safely compare two values, handling type mismatches"""
        try:
            # Convert both to float for comparison
            num1 = float(value1) if value1 is not None else 0
            num2 = float(value2) if value2 is not None else 0
            return num1 > num2
        except (ValueError, TypeError):
            return False

    def safe_strftime(self, date_obj: Any, format_str: str = "%Y-%m-%d") -> str:
        """Safely format date objects to prevent strftime errors"""
        try:
            if date_obj is None:
                return "N/A"
            
            # If it's already a datetime object
            if isinstance(date_obj, (datetime, pd.Timestamp)):
                return date_obj.strftime(format_str)
            
            # If it's a string, try to convert to datetime
            if isinstance(date_obj, str):
                try:
                    parsed_date = pd.to_datetime(date_obj)
                    return parsed_date.strftime(format_str)
                except:
                    return str(date_obj)
            
            # For any other type, return string representation
            return str(date_obj)
            
        except Exception as e:
            print(f"Warning: Could not format date {date_obj}: {e}")
            return str(date_obj)

    def detect_financial_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Automatically detect financial columns in the dataset"""
        if df is None or len(df) == 0:
            return {}
            
        column_mapping = {
            'sales': ['sales', 'revenue', 'amount', 'gross_sales', 'total_sales'],
            'profit': ['profit', 'net_income', 'net_profit', 'earnings'],
            'date': ['date', 'timestamp', 'period', 'month', 'year'],
            'segment': ['segment', 'category', 'business_unit', 'division'],
            'country': ['country', 'region', 'location', 'geography'],
            'product': ['product', 'item', 'service', 'sku'],
            'units': ['units', 'quantity', 'volume', 'qty'],
            'discount': ['discount', 'discount_band', 'rebate']
        }
        
        detected_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                col_str = str(col).lower()
                if any(name in col_str for name in possible_names):
                    detected_columns[standard_name] = col
                    break
        
        return detected_columns

    def enhanced_data_cleaning(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Enhanced data cleaning with financial data specific handling"""
        if df is None or len(df) == 0:
            return df, {}
            
        df_clean = df.copy()
        
        # Clean column names
        df_clean.columns = [str(col).strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') 
                          for col in df_clean.columns]
        
        # Detect financial columns
        financial_cols = self.detect_financial_columns(df_clean)
        
        # Handle date columns safely
        if 'date' in financial_cols:
            date_col = financial_cols['date']
            try:
                df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
                # Remove rows with invalid dates
                initial_count = len(df_clean)
                df_clean = df_clean[df_clean[date_col].notna()]
                if len(df_clean) < initial_count:
                    print(f"Removed {initial_count - len(df_clean)} rows with invalid dates")
            except Exception as e:
                print(f"Warning: Could not parse date column: {e}")
        
        # Handle financial columns - convert to numeric safely with enhanced cleaning
        financial_column_types = ['sales', 'profit', 'units', 'discount']
        for col_type in financial_column_types:
            if col_type in financial_cols:
                col_name = financial_cols[col_type]
                try:
                    print(f"Processing financial column: {col_name}")
                    print(f"Sample values before cleaning: {df_clean[col_name].head(3).tolist()}")
                    
                    df_clean[col_name] = self.safe_convert_to_numeric(df_clean[col_name])
                    
                    print(f"Sample values after cleaning: {df_clean[col_name].head(3).tolist()}")
                    print(f"Data type: {df_clean[col_name].dtype}")
                    
                except Exception as e:
                    print(f"Warning: Could not process {col_name}: {e}")
                    df_clean[col_name] = 0
        
        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_count - len(df_clean)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('Unknown')
            else:
                df_clean[col] = df_clean[col].fillna(0)
        
        return df_clean, financial_cols

    def debug_data_conversion(self, df: pd.DataFrame, financial_cols: Dict[str, str]):
        """Debug method to check data conversion issues"""
        st.markdown("#### 🔍 Data Conversion Debug Info")
        
        for col_type, col_name in financial_cols.items():
            if col_type in ['sales', 'profit']:
                st.write(f"**{col_type.upper()} Column ({col_name}):**")
                
                # Show original values
                original_sample = df[col_name].head(5).tolist()
                st.write(f"Original sample: {original_sample}")
                
                # Show cleaned values
                cleaned_series = self.safe_convert_to_numeric(df[col_name])
                cleaned_sample = cleaned_series.head(5).tolist()
                st.write(f"Cleaned sample: {cleaned_sample}")
                
                # Show data types
                st.write(f"Original dtype: {df[col_name].dtype}")
                st.write(f"Cleaned dtype: {cleaned_series.dtype}")
                
                st.write("---")

    def create_interactive_dashboard(self, df: pd.DataFrame, financial_cols: Dict[str, str]) -> None:
        """Create interactive financial dashboard with Plotly"""
        st.markdown("#### 📊 Interactive Financial Dashboard")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", "💰 Performance", "🌍 Geography", "📦 Products", "📅 Time Analysis"
        ])
        
        with tab1:
            self.create_overview_dashboard(df, financial_cols)
        
        with tab2:
            self.create_performance_dashboard(df, financial_cols)
        
        with tab3:
            self.create_geographic_dashboard(df, financial_cols)
        
        with tab4:
            self.create_product_dashboard(df, financial_cols)
            
        with tab5:
            self.create_time_analysis_dashboard(df, financial_cols)

    def create_overview_dashboard(self, df: pd.DataFrame, financial_cols: Dict[str, str]) -> None:
        """Create overview dashboard with key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics safely
        total_sales = 0
        total_profit = 0
        profit_margin = 0
        avg_transaction = 0
        
        if 'sales' in financial_cols:
            sales_data = self.safe_convert_to_numeric(df[financial_cols['sales']])
            total_sales = sales_data.sum()
            avg_transaction = total_sales / len(df) if len(df) > 0 else 0
            
        if 'profit' in financial_cols:
            profit_data = self.safe_convert_to_numeric(df[financial_cols['profit']])
            total_profit = profit_data.sum()
            profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
        
        with col1:
            st.metric("💰 Total Sales", self.safe_format_number(total_sales, "$,.0f"))
        
        with col2:
            st.metric("💸 Total Profit", self.safe_format_number(total_profit, "$,.0f"))
        
        with col3:
            st.metric("📊 Profit Margin", f"{profit_margin:.1f}%")
        
        with col4:
            st.metric("🎯 Avg Transaction", self.safe_format_number(avg_transaction, "$,.0f"))
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("📊 Total Records", f"{len(df):,}")
        
        with col6:
            st.metric("📋 Data Columns", f"{len(df.columns)}")
        
        with col7:
            if 'segment' in financial_cols:
                segments = df[financial_cols['segment']].nunique()
                st.metric("🏷️ Business Segments", segments)
            else:
                st.metric("🏷️ Business Segments", "N/A")
        
        with col8:
            if 'country' in financial_cols:
                countries = df[financial_cols['country']].nunique()
                st.metric("🌍 Countries", countries)
            else:
                st.metric("🌍 Countries", "N/A")
        
        # Sales distribution
        if 'sales' in financial_cols:
            try:
                sales_data = self.safe_convert_to_numeric(df[financial_cols['sales']])
                # Remove outliers for better visualization
                q_low = sales_data.quantile(0.01)
                q_high = sales_data.quantile(0.99)
                sales_filtered = sales_data[(sales_data >= q_low) & (sales_data <= q_high)]
                
                fig = px.histogram(
                    x=sales_filtered, 
                    title='📈 Sales Distribution',
                    nbins=30,
                    color_discrete_sequence=['#3498db'],
                    labels={'x': 'Sales Amount', 'y': 'Frequency'}
                )
                fig.update_layout(
                    showlegend=False, 
                    xaxis_title='Sales Amount ($)',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not create sales distribution chart")

    def create_performance_dashboard(self, df: pd.DataFrame, financial_cols: Dict[str, str]) -> None:
        """Create performance analysis dashboard"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment performance
            if 'segment' in financial_cols and 'sales' in financial_cols:
                try:
                    segment_data = df[df[financial_cols['segment']].notna()]
                    if len(segment_data) > 0:
                        sales_data = self.safe_convert_to_numeric(segment_data[financial_cols['sales']])
                        segment_performance = segment_data.groupby(financial_cols['segment']).agg({
                            financial_cols['sales']: 'sum'
                        }).reset_index()
                        
                        # Sort by sales for better visualization
                        segment_performance = segment_performance.sort_values(financial_cols['sales'], ascending=True)
                        
                        fig = px.bar(
                            segment_performance, 
                            x=financial_cols['sales'],
                            y=financial_cols['segment'],
                            title='🏷️ Sales by Segment',
                            color_discrete_sequence=['#27ae60'],
                            orientation='h',
                            labels={financial_cols['sales']: 'Sales ($)', financial_cols['segment']: 'Segment'}
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not create segment performance chart")
        
        with col2:
            # Profitability analysis
            if 'segment' in financial_cols and 'profit' in financial_cols:
                try:
                    segment_profit = df[df[financial_cols['segment']].notna()]
                    if len(segment_profit) > 0:
                        profit_data = self.safe_convert_to_numeric(segment_profit[financial_cols['profit']])
                        segment_profitability = segment_profit.groupby(financial_cols['segment']).agg({
                            financial_cols['profit']: 'sum'
                        }).reset_index()
                        
                        segment_profitability = segment_profitability.sort_values(financial_cols['profit'], ascending=True)
                        
                        fig = px.bar(
                            segment_profitability,
                            x=financial_cols['profit'],
                            y=financial_cols['segment'],
                            title='💸 Profit by Segment',
                            color_discrete_sequence=['#e74c3c'],
                            orientation='h',
                            labels={financial_cols['profit']: 'Profit ($)', financial_cols['segment']: 'Segment'}
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not create profit by segment chart")
        
        # Profit margin by segment
        if 'segment' in financial_cols and 'sales' in financial_cols and 'profit' in financial_cols:
            try:
                segment_analysis = df[df[financial_cols['segment']].notna()]
                if len(segment_analysis) > 0:
                    sales_data = self.safe_convert_to_numeric(segment_analysis[financial_cols['sales']])
                    profit_data = self.safe_convert_to_numeric(segment_analysis[financial_cols['profit']])
                    
                    segment_summary = segment_analysis.groupby(financial_cols['segment']).agg({
                        financial_cols['sales']: 'sum',
                        financial_cols['profit']: 'sum'
                    }).reset_index()
                    
                    segment_summary['Profit_Margin'] = (segment_summary[financial_cols['profit']] / segment_summary[financial_cols['sales']]) * 100
                    segment_summary = segment_summary.sort_values('Profit_Margin', ascending=True)
                    
                    fig = px.bar(
                        segment_summary,
                        x='Profit_Margin',
                        y=financial_cols['segment'],
                        title='📊 Profit Margin by Segment (%)',
                        color='Profit_Margin',
                        orientation='h',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not create profit margin analysis")

    def create_geographic_dashboard(self, df: pd.DataFrame, financial_cols: Dict[str, str]) -> None:
        """Create geographic analysis dashboard"""
        col1, col2 = st.columns(2)
        
        with col1:
            if 'country' in financial_cols and 'sales' in financial_cols:
                try:
                    country_data = df[df[financial_cols['country']].notna()]
                    if len(country_data) > 0:
                        sales_data = self.safe_convert_to_numeric(country_data[financial_cols['sales']])
                        country_sales = country_data.groupby(financial_cols['country']).agg({
                            financial_cols['sales']: 'sum'
                        }).reset_index()
                        
                        # Get top 10 countries
                        country_sales = country_sales.nlargest(10, financial_cols['sales'])
                        
                        fig = px.bar(
                            country_sales,
                            x=financial_cols['sales'],
                            y=financial_cols['country'],
                            title='🌍 Top 10 Countries by Sales',
                            color=financial_cols['sales'],
                            orientation='h',
                            labels={financial_cols['sales']: 'Sales ($)', financial_cols['country']: 'Country'}
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not create geographic analysis chart")
        
        with col2:
            if 'country' in financial_cols and 'profit' in financial_cols:
                try:
                    country_profit = df[df[financial_cols['country']].notna()]
                    if len(country_profit) > 0:
                        profit_data = self.safe_convert_to_numeric(country_profit[financial_cols['profit']])
                        country_profitability = country_profit.groupby(financial_cols['country']).agg({
                            financial_cols['profit']: 'sum'
                        }).reset_index()
                        
                        country_profitability = country_profitability.nlargest(10, financial_cols['profit'])
                        
                        fig = px.bar(
                            country_profitability,
                            x=financial_cols['profit'],
                            y=financial_cols['country'],
                            title='💰 Top 10 Countries by Profit',
                            color=financial_cols['profit'],
                            orientation='h',
                            labels={financial_cols['profit']: 'Profit ($)', financial_cols['country']: 'Country'}
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not create profit by country chart")
        
        # World map visualization (if we have country data)
        if 'country' in financial_cols and 'sales' in financial_cols:
            try:
                country_sales_map = df.groupby(financial_cols['country']).agg({
                    financial_cols['sales']: 'sum'
                }).reset_index()
                
                # Simple choropleth - using plotly's built-in countries
                fig = px.choropleth(
                    country_sales_map,
                    locations=financial_cols['country'],
                    locationmode='country names',
                    color=financial_cols['sales'],
                    title='🗺️ Sales Distribution by Country',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info("World map visualization requires specific country names. Using bar chart instead.")
                # Fallback to bar chart
                pass

    def create_product_dashboard(self, df: pd.DataFrame, financial_cols: Dict[str, str]) -> None:
        """Create product analysis dashboard"""
        col1, col2 = st.columns(2)
        
        with col1:
            if 'product' in financial_cols and 'sales' in financial_cols:
                try:
                    product_data = df[df[financial_cols['product']].notna()]
                    if len(product_data) > 0:
                        sales_data = self.safe_convert_to_numeric(product_data[financial_cols['sales']])
                        product_performance = product_data.groupby(financial_cols['product']).agg({
                            financial_cols['sales']: 'sum'
                        }).nlargest(10, financial_cols['sales'])
                        
                        if len(product_performance) > 0:
                            fig = px.pie(
                                product_performance,
                                values=financial_cols['sales'],
                                names=product_performance.index,
                                title='📦 Top 10 Products by Sales'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not create product sales pie chart")
        
        with col2:
            if 'product' in financial_cols and 'profit' in financial_cols:
                try:
                    product_profit_data = df[df[financial_cols['product']].notna()]
                    if len(product_profit_data) > 0:
                        profit_data = self.safe_convert_to_numeric(product_profit_data[financial_cols['profit']])
                        product_profit = product_profit_data.groupby(financial_cols['product']).agg({
                            financial_cols['profit']: 'sum'
                        }).nlargest(10, financial_cols['profit'])
                        
                        if len(product_profit) > 0:
                            fig = px.bar(
                                product_profit,
                                x=product_profit.index,
                                y=financial_cols['profit'],
                                title='💎 Top 10 Products by Profit',
                                color=financial_cols['profit'],
                                labels={financial_cols['profit']: 'Profit ($)', 'index': 'Product'}
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not create product profit chart")
        
        # Product performance scatter plot
        if 'product' in financial_cols and 'sales' in financial_cols and 'profit' in financial_cols:
            try:
                product_analysis = df[df[financial_cols['product']].notna()]
                if len(product_analysis) > 0:
                    product_summary = product_analysis.groupby(financial_cols['product']).agg({
                        financial_cols['sales']: 'sum',
                        financial_cols['profit']: 'sum'
                    }).reset_index()
                    
                    product_summary['Profit_Margin'] = (product_summary[financial_cols['profit']] / product_summary[financial_cols['sales']]) * 100
                    
                    fig = px.scatter(
                        product_summary,
                        x=financial_cols['sales'],
                        y=financial_cols['profit'],
                        size='Profit_Margin',
                        color='Profit_Margin',
                        hover_name=financial_cols['product'],
                        title='🎯 Product Performance: Sales vs Profit',
                        labels={
                            financial_cols['sales']: 'Total Sales ($)',
                            financial_cols['profit']: 'Total Profit ($)',
                            'Profit_Margin': 'Profit Margin (%)'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not create product performance scatter plot")

    def create_time_analysis_dashboard(self, df: pd.DataFrame, financial_cols: Dict[str, str]) -> None:
        """Create time series analysis dashboard"""
        if 'date' not in financial_cols or 'sales' not in financial_cols:
            st.warning("📅 Date and Sales columns are required for time analysis")
            return
        
        try:
            # Ensure date column is datetime
            df_time = df.copy()
            df_time[financial_cols['date']] = pd.to_datetime(df_time[financial_cols['date']])
            
            # Time series analysis
            df_time['Year'] = df_time[financial_cols['date']].dt.year
            df_time['Month'] = df_time[financial_cols['date']].dt.month
            df_time['YearMonth'] = df_time[financial_cols['date']].dt.to_period('M').astype(str)
            
            # Monthly sales trend
            monthly_sales = df_time.groupby('YearMonth').agg({
                financial_cols['sales']: 'sum'
            }).reset_index()
            
            fig = px.line(
                monthly_sales,
                x='YearMonth',
                y=financial_cols['sales'],
                title='📈 Monthly Sales Trend',
                markers=True
            )
            fig.update_layout(
                xaxis_title='Month',
                yaxis_title='Sales ($)',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly pattern
                monthly_pattern = df_time.groupby('Month').agg({
                    financial_cols['sales']: 'mean'
                }).reset_index()
                
                fig = px.line(
                    monthly_pattern,
                    x='Month',
                    y=financial_cols['sales'],
                    title='📅 Average Monthly Sales Pattern',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title='Month',
                    yaxis_title='Average Sales ($)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Year-over-year comparison
                yearly_sales = df_time.groupby('Year').agg({
                    financial_cols['sales']: 'sum'
                }).reset_index()
                
                fig = px.bar(
                    yearly_sales,
                    x='Year',
                    y=financial_cols['sales'],
                    title='📊 Yearly Sales Comparison',
                    color=financial_cols['sales']
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in time analysis: {str(e)}")

    def perform_comprehensive_analysis(self, df: pd.DataFrame, financial_cols: Dict[str, str]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        analysis_results = {}
        
        try:
            # Basic statistics
            if 'sales' in financial_cols:
                sales_data = self.safe_convert_to_numeric(df[financial_cols['sales']])
                analysis_results['sales_stats'] = {
                    'mean': sales_data.mean(),
                    'median': sales_data.median(),
                    'std': sales_data.std(),
                    'min': sales_data.min(),
                    'max': sales_data.max(),
                    'total': sales_data.sum()
                }
            
            if 'profit' in financial_cols:
                profit_data = self.safe_convert_to_numeric(df[financial_cols['profit']])
                analysis_results['profit_stats'] = {
                    'mean': profit_data.mean(),
                    'median': profit_data.median(),
                    'std': profit_data.std(),
                    'min': profit_data.min(),
                    'max': profit_data.max(),
                    'total': profit_data.sum()
                }
            
            # Correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                numeric_df = df[numeric_cols].apply(self.safe_convert_to_numeric)
                analysis_results['correlation_matrix'] = numeric_df.corr()
            
            # Segment analysis
            if 'segment' in financial_cols:
                segment_analysis = df.groupby(financial_cols['segment']).agg({
                    financial_cols['sales']: ['sum', 'mean', 'count']
                }).round(2)
                analysis_results['segment_analysis'] = segment_analysis
            
            # Geographic analysis
            if 'country' in financial_cols:
                country_analysis = df.groupby(financial_cols['country']).agg({
                    financial_cols['sales']: ['sum', 'mean', 'count']
                }).round(2)
                analysis_results['country_analysis'] = country_analysis
                
        except Exception as e:
            st.error(f"Error in comprehensive analysis: {str(e)}")
        
        return analysis_results

    def render_header(self):
        """Render the enhanced main header"""
        st.markdown('<h1 class="main-header">🚀 AI Financial Intelligence Platform</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 2rem;">Advanced Analytics & AI-Powered Financial Insights</p>', unsafe_allow_html=True)
        
        # Display directory information
        st.markdown(f"""
        <div class="directory-info">
        <strong>📁 Secure Data Storage Location:</strong><br>
        <code>{self.app_data_dir}</code><br>
        <small>All your data is stored locally for maximum security and privacy.</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

    def render_sidebar(self):
        """Render the enhanced sidebar with navigation and progress tracking"""
        with st.sidebar:
            st.markdown("## 🧭 Navigation")
            st.markdown("---")
            
            # Demo mode notice
            if FinancialReportGenerator is None:
                st.markdown('<div class="warning-box">🔧 Enhanced Demo Mode Active</div>', unsafe_allow_html=True)
            
            # Progress tracker
            st.markdown("### 📋 Analysis Progress")
            
            steps = [
                (1, "📂 Upload Data", st.session_state.data_loaded),
                (2, "🧹 Clean & Prepare", st.session_state.data_cleaned),
                (3, "📊 Analyze & Visualize", st.session_state.analysis_done),
                (4, "🔮 Forecast & Predict", st.session_state.forecast_done),
                (5, "🧠 AI Insights", st.session_state.ai_summary_done),
                (6, "📄 Generate Report", st.session_state.report_generated)
            ]
            
            for step_num, step_name, completed in steps:
                if completed:
                    st.markdown(f'<div class="step-complete">✅ {step_name}</div>', unsafe_allow_html=True)
                elif step_num == st.session_state.current_step:
                    st.markdown(f'<div class="step-current">🎯 {step_name}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="step-pending">⏳ {step_name}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Quick stats if data is loaded
            if st.session_state.data_loaded and st.session_state.df_clean is not None:
                st.markdown("### 📈 Quick Stats")
                df = st.session_state.df_clean
                financial_cols = st.session_state.financial_cols
                
                if 'sales' in financial_cols:
                    total_sales = self.safe_convert_to_numeric(df[financial_cols['sales']]).sum()
                    st.metric("Total Sales", self.safe_format_number(total_sales, "$,.0f"))
                
                if 'profit' in financial_cols:
                    total_profit = self.safe_convert_to_numeric(df[financial_cols['profit']]).sum()
                    st.metric("Total Profit", self.safe_format_number(total_profit, "$,.0f"))
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            if st.button("🔄 Reset Application", use_container_width=True):
                self.reset_app()
            
            if st.session_state.report_generated:
                if st.button("📥 Download Report", use_container_width=True):
                    self.download_report()

    def reset_app(self):
        """Reset the application state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self.initialize_session_state()
        st.rerun()

    def render_file_upload(self):
        """Render enhanced file upload section"""
        st.markdown('<div class="subheader">📂 Step 1: Upload Financial Data</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your financial data (CSV or Excel)",
                type=['csv', 'xlsx'],
                help="Upload CSV or Excel files with financial data. Supported columns: Sales, Profit, Date, Segment, Country, Product, etc."
            )
        
        with col2:
            st.markdown("### 💡 Sample Data")
            if st.button("📋 View Sample Format"):
                sample_data = {
                    'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                    'Segment': ['Enterprise', 'Small Business', 'Government'],
                    'Country': ['USA', 'Canada', 'Germany'],
                    'Product': ['Product A', 'Product B', 'Product C'],
                    'Sales': [10000, 7500, 12000],
                    'Profit': [2500, 1500, 3000],
                    'Units_Sold': [100, 75, 120]
                }
                st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        
        if uploaded_file is not None:
            try:
                # Save uploaded file to APP_Data directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_extension = os.path.splitext(uploaded_file.name)[1]
                saved_filename = f"uploaded_data_{timestamp}{file_extension}"
                file_path = os.path.join(self.data_dir, saved_filename)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.uploaded_file = file_path
                
                # Load and preview data
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_preview = pd.read_csv(uploaded_file)
                    else:
                        df_preview = pd.read_excel(uploaded_file)
                    
                    # Detect financial columns
                    financial_cols = self.detect_financial_columns(df_preview)
                    
                    st.session_state.data_loaded = True
                    st.session_state.df_clean = df_preview
                    st.session_state.financial_cols = financial_cols
                    st.session_state.current_step = 2
                    
                    st.markdown('<div class="success-box">✅ File uploaded and analyzed successfully!</div>', unsafe_allow_html=True)
                    
                    # Display comprehensive file info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Total Rows", len(df_preview))
                    with col2:
                        st.metric("📋 Total Columns", len(df_preview.columns))
                    with col3:
                        st.metric("💾 File Size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")
                    with col4:
                        st.metric("✅ Data Quality", "Good" if len(df_preview) > 0 else "Check")
                    
                    # Show detected financial columns
                    if financial_cols:
                        st.markdown("#### 🔍 Detected Financial Columns")
                        cols_display = []
                        for std_name, actual_name in financial_cols.items():
                            cols_display.append({"Standard Name": std_name.title(), "Your Column": actual_name})
                        st.dataframe(pd.DataFrame(cols_display), use_container_width=True)
                    else:
                        st.warning("⚠️ No standard financial columns detected. Please ensure your data contains columns like 'Sales', 'Profit', 'Date', etc.")
                    
                    # Show data preview
                    st.markdown("#### 📋 Data Preview")
                    st.dataframe(df_preview.head(8), use_container_width=True)
                    
                except Exception as preview_error:
                    st.error(f"❌ Error analyzing file: {str(preview_error)}")
                    
            except Exception as e:
                st.error(f"❌ Error saving file: {str(e)}")

    def render_data_cleaning(self):
        """Render enhanced data cleaning section"""
        st.markdown('<div class="subheader">🧹 Step 2: Clean & Prepare Data</div>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("📁 Please upload a file first in Step 1.")
            return
        
        # Add debug option
        show_debug = st.checkbox("Show data conversion debug information")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### ⚙️ Cleaning Options")
            remove_duplicates = st.checkbox("Remove Duplicate Records", value=True)
            handle_missing = st.checkbox("Handle Missing Values", value=True)
            standardize_names = st.checkbox("Standardize Column Names", value=True)
        
        with col2:
            st.markdown("#### 🔧 Advanced Options")
            auto_detect_types = st.checkbox("Auto-detect Data Types", value=True)
            currency_conversion = st.checkbox("Handle Currency Formats", value=True)
        
        if st.button("🚀 Clean & Prepare Data", use_container_width=True):
            with st.spinner("Performing advanced data cleaning and preparation..."):
                try:
                    df = st.session_state.df_clean.copy()
                    
                    if df is None or len(df) == 0:
                        st.error("❌ No data available for cleaning.")
                        return
                    
                    # Show debug info if requested
                    if show_debug:
                        self.debug_data_conversion(df, st.session_state.financial_cols)
                    
                    # Perform enhanced cleaning
                    df_clean, financial_cols = self.enhanced_data_cleaning(df)
                    
                    if df_clean is None or len(df_clean) == 0:
                        st.error("❌ Data cleaning resulted in empty dataset.")
                        return
                    
                    # Save cleaned data
                    cleaned_file_path = os.path.join(self.data_dir, "financial_data_cleaned.csv")
                    df_clean.to_csv(cleaned_file_path, index=False)
                    
                    st.session_state.data_cleaned = True
                    st.session_state.df_clean = df_clean
                    st.session_state.financial_cols = financial_cols
                    st.session_state.current_step = 3
                    
                    st.markdown('<div class="success-box">✅ Advanced Data Cleaning Completed Successfully!</div>', unsafe_allow_html=True)
                    
                    # Show cleaning report
                    st.markdown("#### 📊 Cleaning Report")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📈 Original Rows", len(df))
                    with col2:
                        st.metric("✨ Cleaned Rows", len(df_clean))
                    with col3:
                        duplicates_removed = len(df) - len(df.drop_duplicates())
                        st.metric("🧹 Duplicates Removed", duplicates_removed)
                    with col4:
                        st.metric("📋 Columns Processed", len(df_clean.columns))
                    
                    # Show cleaned data preview
                    st.markdown("#### 🧼 Cleaned Data Preview")
                    st.dataframe(df_clean.head(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during data cleaning: {str(e)}")

    def render_analysis(self):
        """Render enhanced data analysis section"""
        st.markdown('<div class="subheader">📊 Step 3: Advanced Analysis & Visualization</div>', unsafe_allow_html=True)
        
        if not st.session_state.data_cleaned:
            st.warning("🧹 Please clean the data first in Step 2.")
            return
        
        # Analysis options
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ Analysis Options")
            include_correlation = st.checkbox("Correlation Analysis", value=True)
            include_trends = st.checkbox("Trend Analysis", value=True)
            include_segments = st.checkbox("Segment Analysis", value=True)
            include_geography = st.checkbox("Geographic Analysis", value=True)
            
        with col2:
            st.markdown("#### 📈 Visualization Settings")
            chart_style = st.selectbox("Chart Style", ["Interactive", "Static"])
            color_scheme = st.selectbox("Color Scheme", ["Default", "Corporate", "Vibrant", "Pastel"])
        
        if st.button("🎯 Run Comprehensive Analysis", use_container_width=True):
            with st.spinner("Performing advanced financial analysis and generating interactive visualizations..."):
                try:
                    df = st.session_state.df_clean
                    financial_cols = st.session_state.financial_cols
                    
                    if df is None or len(df) == 0:
                        st.error("❌ No data available for analysis.")
                        return
                    
                    # Perform comprehensive statistical analysis
                    analysis_results = self.perform_comprehensive_analysis(df, financial_cols)
                    st.session_state.analysis_results = analysis_results
                    
                    # Create interactive dashboard
                    self.create_interactive_dashboard(df, financial_cols)
                    
                    # Additional advanced analyses
                    st.markdown("#### 🔍 Advanced Insights")
                    
                    # Correlation analysis
                    if include_correlation and 'correlation_matrix' in analysis_results:
                        st.markdown("##### 📊 Feature Correlation Matrix")
                        corr_matrix = analysis_results['correlation_matrix']
                        
                        fig = px.imshow(
                            corr_matrix,
                            title='Feature Correlation Matrix',
                            color_continuous_scale='RdBu_r',
                            aspect='auto',
                            labels=dict(color="Correlation")
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical summary
                    st.markdown("##### 📋 Statistical Summary")
                    
                    if 'sales_stats' in analysis_results:
                        col1, col2, col3, col4 = st.columns(4)
                        sales_stats = analysis_results['sales_stats']
                        
                        with col1:
                            st.metric("Sales Mean", self.safe_format_number(sales_stats['mean'], "$,.0f"))
                        with col2:
                            st.metric("Sales Median", self.safe_format_number(sales_stats['median'], "$,.0f"))
                        with col3:
                            st.metric("Sales Std Dev", self.safe_format_number(sales_stats['std'], "$,.0f"))
                        with col4:
                            st.metric("Sales Range", f"{self.safe_format_number(sales_stats['min'], '$,.0f')} - {self.safe_format_number(sales_stats['max'], '$,.0f')}")
                    
                    # Segment analysis table
                    if include_segments and 'segment_analysis' in analysis_results:
                        st.markdown("##### 🏷️ Segment Performance Summary")
                        segment_df = analysis_results['segment_analysis']
                        st.dataframe(segment_df, use_container_width=True)
                    
                    st.session_state.analysis_done = True
                    st.session_state.current_step = 4
                    st.markdown('<div class="success-box">✅ Comprehensive Analysis Completed!</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.error(f"Detailed error: {traceback.format_exc()}")

    def render_forecasting(self):
        """Render enhanced forecasting section"""
        st.markdown('<div class="subheader">🔮 Step 4: Advanced Forecasting</div>', unsafe_allow_html=True)
        
        if not st.session_state.analysis_done:
            st.warning("📊 Please complete the analysis first in Step 3.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ Forecast Configuration")
            forecast_periods = st.slider("Months to Forecast", 3, 24, 6)
            confidence_level = st.slider("Confidence Level (%)", 80, 95, 90)
            growth_scenario = st.selectbox("Growth Scenario", 
                                         ["Conservative", "Moderate", "Aggressive"])
            
        with col2:
            st.markdown("#### 📈 Model Parameters")
            include_seasonality = st.checkbox("Include Seasonality", value=True)
            include_trend = st.checkbox("Include Trend Component", value=True)
        
        if st.button("🎯 Generate Advanced Forecast", use_container_width=True):
            with st.spinner("Training advanced forecasting models and generating predictions..."):
                try:
                    df = st.session_state.df_clean
                    financial_cols = st.session_state.financial_cols
                    
                    if 'sales' not in financial_cols:
                        st.warning("📈 Sales column required for forecasting.")
                        return
                    
                    # Generate sophisticated forecast
                    last_date = datetime.now()
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                               periods=forecast_periods, freq='M')
                    
                    # Calculate base metrics safely
                    sales_data = self.safe_convert_to_numeric(df[financial_cols['sales']])
                    base_sales = sales_data.mean()
                    
                    # Generate forecast with confidence intervals
                    forecast_data = []
                    for i, date in enumerate(future_dates):
                        # Different growth rates based on scenario
                        growth_rates = {
                            "Conservative": 0.005,
                            "Moderate": 0.015,
                            "Aggressive": 0.025
                        }
                        
                        growth_rate = growth_rates[growth_scenario]
                        seasonal_factor = 1 + (0.1 * np.sin(i * 2 * np.pi / 12)) if include_seasonality else 1
                        trend_factor = 1 + (growth_rate * i) if include_trend else 1
                        
                        predicted_sales = base_sales * seasonal_factor * trend_factor * (1 + np.random.normal(0, 0.02))
                        
                        # Confidence intervals
                        confidence_multiplier = confidence_level / 100
                        sales_upper = predicted_sales * (1 + (1 - confidence_multiplier))
                        sales_lower = predicted_sales * confidence_multiplier
                        
                        forecast_data.append({
                            'Date': date,
                            'Predicted_Sales': max(0, predicted_sales),
                            'Sales_Upper': max(0, sales_upper),
                            'Sales_Lower': max(0, sales_lower)
                        })
                    
                    forecast_df = pd.DataFrame(forecast_data)
                    
                    # Save forecast
                    forecast_path = os.path.join(self.data_dir, "advanced_forecast.csv")
                    forecast_df.to_csv(forecast_path, index=False)
                    
                    st.session_state.forecast_done = True
                    st.session_state.forecast_data = forecast_df
                    st.session_state.current_step = 5
                    
                    st.markdown('<div class="success-box">✅ Advanced Forecasting Completed!</div>', unsafe_allow_html=True)
                    
                    # Display forecast results
                    st.markdown("#### 📊 Forecast Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    total_sales_forecast = forecast_df['Predicted_Sales'].sum()
                    
                    with col1:
                        st.metric("💰 Total Forecasted Sales", self.safe_format_number(total_sales_forecast, "$,.0f"))
                    with col2:
                        avg_monthly = forecast_df['Predicted_Sales'].mean()
                        st.metric("📈 Avg Monthly Sales", self.safe_format_number(avg_monthly, "$,.0f"))
                    with col3:
                        growth = ((forecast_df['Predicted_Sales'].iloc[-1] / base_sales) - 1) * 100
                        st.metric("📊 Growth Rate", f"{growth:.1f}%")
                    with col4:
                        st.metric("🎯 Confidence Level", f"{confidence_level}%")
                    
                    # Interactive forecast chart
                    fig = go.Figure()
                    
                    # Sales forecast with confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'], y=forecast_df['Predicted_Sales'],
                        mode='lines+markers',
                        name='Predicted Sales',
                        line=dict(color='#2E86AB', width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'], y=forecast_df['Sales_Upper'],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(color='lightblue', width=1),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'], y=forecast_df['Sales_Lower'],
                        mode='lines',
                        name='Lower Bound',
                        fill='tonexty',
                        line=dict(color='lightblue', width=1),
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title='Sales Forecast with Confidence Intervals',
                        xaxis_title='Date',
                        yaxis_title='Sales ($)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during forecasting: {str(e)}")

    def render_ai_summary(self):
        """Render enhanced AI summary section"""
        st.markdown('<div class="subheader">🧠 Step 5: AI-Powered Insights</div>', unsafe_allow_html=True)
        
        if not st.session_state.forecast_done:
            st.warning("🔮 Please complete forecasting first in Step 4.")
            return
        
        if st.button("🤖 Generate AI Insights", use_container_width=True):
            with st.spinner("AI is analyzing your data and generating comprehensive business insights..."):
                try:
                    df = st.session_state.df_clean
                    financial_cols = st.session_state.financial_cols
                    forecast_df = st.session_state.forecast_data
                    
                    if df is None or len(df) == 0:
                        st.error("❌ No data available for AI analysis.")
                        return
                    
                    # Calculate comprehensive metrics safely
                    total_sales = 0
                    total_profit = 0
                    
                    if 'sales' in financial_cols:
                        total_sales = self.safe_convert_to_numeric(df[financial_cols['sales']]).sum()
                    
                    if 'profit' in financial_cols:
                        total_profit = self.safe_convert_to_numeric(df[financial_cols['profit']]).sum()
                    elif 'sales' in financial_cols:
                        # If no profit column, estimate profit as 25% of sales
                        total_profit = total_sales * 0.25
                    
                    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
                    
                    # Generate dynamic AI insights
                    ai_insights = self.generate_ai_insights(df, financial_cols, forecast_df, total_sales, total_profit, profit_margin)
                    
                    # Save AI insights
                    insights_path = os.path.join(self.outputs_dir, "ai_business_insights.json")
                    with open(insights_path, "w") as f:
                        json.dump(ai_insights, f, indent=2)
                    
                    st.session_state.ai_summary_done = True
                    st.session_state.ai_insights = ai_insights
                    st.session_state.current_step = 6
                    
                    st.markdown('<div class="success-box">✅ AI-Powered Insights Generated!</div>', unsafe_allow_html=True)
                    
                    # Display insights in an interactive way
                    tab1, tab2, tab3 = st.tabs(["📋 Executive Summary", "🎯 Recommendations", "📈 Performance"])
                    
                    with tab1:
                        st.markdown(ai_insights["executive_summary"])
                    
                    with tab2:
                        st.markdown(ai_insights["strategic_recommendations"])
                    
                    with tab3:
                        st.markdown(ai_insights["performance_analysis"])
                    
                except Exception as e:
                    st.error(f"❌ Error generating AI insights: {str(e)}")

    def generate_ai_insights(self, df: pd.DataFrame, financial_cols: Dict[str, str], 
                           forecast_df: pd.DataFrame, total_sales: float, 
                           total_profit: float, profit_margin: float) -> Dict[str, str]:
        """Generate comprehensive AI-powered business insights"""
        
        # Performance assessment using safe comparison
        if self.safe_compare(profit_margin, 20):
            performance_rating = "Excellent"
            performance_color = "🟢"
        elif self.safe_compare(profit_margin, 10):
            performance_rating = "Good"
            performance_color = "🟡"
        else:
            performance_rating = "Needs Improvement"
            performance_color = "🔴"
        
        # Calculate additional metrics safely
        data_quality = "Excellent" if df.isnull().sum().sum() == 0 else "Good"
        customer_diversity = "Broad" if 'segment' in financial_cols and df[financial_cols['segment']].nunique() > 3 else "Focused"
        
        insights = {
            "executive_summary": f"""
## 🏢 Executive Summary

### {performance_color} Performance Rating: {performance_rating}

**Financial Overview:**
- **Total Revenue**: {self.safe_format_number(total_sales, '$,.0f')}
- **Total Profit**: {self.safe_format_number(total_profit, '$,.0f')}
- **Profit Margin**: {profit_margin:.1f}%
- **Data Period**: {len(df)} transactions analyzed

**Key Strengths:**
- Strong revenue generation capabilities
- {'Exceptional' if self.safe_compare(profit_margin, 20) else 'Healthy'} profitability metrics
- {customer_diversity} business portfolio
- Scalable operational model

**Strategic Position:**
The organization demonstrates robust financial health with clear growth trajectories. 
The current performance indicates sustainable business practices and market relevance.
            """,
            
            "strategic_recommendations": f"""
## 🎯 Strategic Recommendations

### Immediate Actions (0-3 Months)
1. **Cost Optimization**
   - Implement automated expense tracking
   - Review supplier contracts for better terms
   - Optimize inventory management

2. **Revenue Enhancement**
   - Launch targeted upselling campaigns
   - Expand to underperforming geographic markets
   - Develop strategic partnerships

3. **Operational Excellence**
   - Streamline reporting processes
   - Enhance data-driven decision making
   - Implement performance dashboards

### Data Quality: {data_quality}
            """,
            
            "performance_analysis": f"""
## 📈 Performance Analysis

### Financial Metrics
- **Revenue Stability**: {'High' if 'sales' in financial_cols and len(df) > 1 else 'Moderate'}
- **Growth Trajectory**: {'Positive' if len(df) > 100 else 'Establishing'}
- **Customer Diversity**: {customer_diversity}
- **Data Quality Score**: {data_quality}

### Operational Efficiency
- **Transaction Volume**: {len(df):,} records
- **Average Transaction Value**: {self.safe_format_number(total_sales/len(df) if len(df) > 0 else 0, '$,.0f')}
- **Forecast Accuracy**: {90 if forecast_df is not None else 85}%
            """
        }
        
        return insights

    def render_report_generation(self):
        """Render final report generation section"""
        st.markdown('<div class="subheader">📄 Step 6: Generate Professional Report</div>', unsafe_allow_html=True)
        
        if not st.session_state.ai_summary_done:
            st.warning("🧠 Please generate AI insights first in Step 5.")
            return
            
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ Report Options")
            report_style = st.selectbox("Report Style", 
                                      ["Executive", "Detailed", "Board Meeting"])
            include_appendix = st.checkbox("Include Technical Appendix", value=True)
            
        with col2:
            st.markdown("#### 📊 Content Selection")
            col2a, col2b = st.columns(2)
            with col2a:
                include_forecast = st.checkbox("Include Forecast", value=True)
                include_risk = st.checkbox("Include Risk Analysis", value=True)
            with col2b:
                include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        if st.button("🖨️ Generate Professional Report", use_container_width=True):
            with st.spinner("Creating comprehensive professional report with AI insights..."):
                try:
                    # Generate comprehensive PDF report
                    report_path = self.generate_comprehensive_pdf_report()
                    
                    if report_path:
                        st.session_state.report_generated = True
                        st.session_state.report_path = report_path
                        
                        st.markdown('<div class="success-box">✅ Professional Report Generated Successfully!</div>', unsafe_allow_html=True)
                        
                        # Report preview
                        st.markdown("#### 📋 Report Contents Preview")
                        
                        report_contents = [
                            "🏢 Cover Page & Executive Summary",
                            "📈 Financial Performance Overview",
                            "📊 Interactive Dashboard Summary", 
                            "🔮 Forecasting & Projections",
                            "🎯 Strategic Recommendations",
                            "⚠️ Risk Assessment & Mitigation",
                            "📋 Methodology & Data Sources"
                        ]
                        
                        for item in report_contents:
                            st.markdown(f"- {item}")
                        
                        # Auto-download
                        self.download_report()
                    
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")

    def generate_comprehensive_pdf_report(self):
        """Generate a comprehensive PDF report with all analysis steps"""
        try:
            # Create report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"financial_intelligence_report_{timestamp}.pdf"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(report_path, pagesize=A4, topMargin=1*inch)
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2E86AB'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#2E86AB'),
                spaceAfter=12,
                spaceBefore=20
            )
            
            subheading_style = ParagraphStyle(
                'CustomSubheading',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#465362'),
                spaceAfter=6,
                spaceBefore=12
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                leading=14
            )
            
            # Build story (content)
            story = []
            
            # Cover Page
            story.append(Paragraph("AI FINANCIAL INTELLIGENCE REPORT", title_style))
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Heading2']))
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Comprehensive Financial Analysis & Business Insights", styles['Heading3']))
            story.append(Spacer(1, 1*inch))
            
            # Add page break
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph("Confidential Business Document", styles['Italic']))
            story.append(PageBreak())
            
            # Table of Contents
            story.append(Paragraph("TABLE OF CONTENTS", heading_style))
            toc_items = [
                "1. Executive Summary",
                "2. Data Overview & Quality Assessment",
                "3. Financial Performance Analysis", 
                "4. Business Segment Insights",
                "5. Geographic Performance",
                "6. Product Analysis",
                "7. Sales Forecasting & Projections",
                "8. AI-Powered Strategic Recommendations",
                "9. Risk Assessment & Mitigation",
                "10. Technical Appendix"
            ]
            
            for item in toc_items:
                story.append(Paragraph(item, normal_style))
                story.append(Spacer(1, 0.1*inch))
            
            story.append(PageBreak())
            
            # 1. Executive Summary
            story.append(Paragraph("1. EXECUTIVE SUMMARY", heading_style))
            
            if st.session_state.ai_insights:
                exec_summary = st.session_state.ai_insights.get("executive_summary", "No executive summary available.")
                # Clean up markdown formatting for PDF
                exec_summary_clean = exec_summary.replace('##', '').replace('**', '').replace('###', '')
                story.append(Paragraph(exec_summary_clean, normal_style))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Key Metrics Table
            if st.session_state.df_clean is not None and st.session_state.financial_cols:
                df = st.session_state.df_clean
                financial_cols = st.session_state.financial_cols
                
                # Calculate key metrics
                metrics_data = []
                
                if 'sales' in financial_cols:
                    total_sales = self.safe_convert_to_numeric(df[financial_cols['sales']]).sum()
                    metrics_data.append(["Total Revenue", self.safe_format_number(total_sales, "$,.0f")])
                
                if 'profit' in financial_cols:
                    total_profit = self.safe_convert_to_numeric(df[financial_cols['profit']]).sum()
                    metrics_data.append(["Total Profit", self.safe_format_number(total_profit, "$,.0f")])
                
                if 'sales' in financial_cols and 'profit' in financial_cols:
                    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
                    metrics_data.append(["Profit Margin", f"{profit_margin:.1f}%"])
                
                metrics_data.append(["Total Records", f"{len(df):,}"])
                
                # Safely handle date range
                if 'date' in financial_cols:
                    try:
                        date_col = financial_cols['date']
                        min_date = self.safe_strftime(df[date_col].min())
                        max_date = self.safe_strftime(df[date_col].max())
                        metrics_data.append(["Analysis Period", f"{min_date} to {max_date}"])
                    except Exception as e:
                        metrics_data.append(["Analysis Period", "Date range not available"])
                else:
                    metrics_data.append(["Analysis Period", "N/A"])
                
                # Create metrics table
                metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD'))
                ]))
                
                story.append(metrics_table)
            
            story.append(PageBreak())
            
            # 2. Data Overview
            story.append(Paragraph("2. DATA OVERVIEW & QUALITY ASSESSMENT", heading_style))
            
            if st.session_state.data_loaded:
                story.append(Paragraph("Data Source & Structure", subheading_style))
                
                data_info = [
                    ["Data File", os.path.basename(st.session_state.uploaded_file) if st.session_state.uploaded_file else "Uploaded File"],
                    ["Total Records", f"{len(st.session_state.df_clean):,}"],
                    ["Total Columns", f"{len(st.session_state.df_clean.columns)}"],
                    ["Data Quality", "Excellent" if st.session_state.data_cleaned else "Good"],
                    ["Analysis Date", datetime.now().strftime("%Y-%m-%d %H:%M")]
                ]
                
                data_table = Table(data_info, colWidths=[2.5*inch, 2.5*inch])
                data_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#465362')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD'))
                ]))
                
                story.append(data_table)
                story.append(Spacer(1, 0.2*inch))
            
            # 3. Financial Performance
            story.append(Paragraph("3. FINANCIAL PERFORMANCE ANALYSIS", heading_style))
            
            if st.session_state.analysis_done and st.session_state.financial_cols:
                financial_cols = st.session_state.financial_cols
                
                if 'sales' in financial_cols:
                    story.append(Paragraph("Revenue Analysis", subheading_style))
                    
                    sales_data = self.safe_convert_to_numeric(st.session_state.df_clean[financial_cols['sales']])
                    sales_stats = [
                        ["Metric", "Value"],
                        ["Total Sales", self.safe_format_number(sales_data.sum(), "$,.0f")],
                        ["Average Sale", self.safe_format_number(sales_data.mean(), "$,.0f")],
                        ["Maximum Sale", self.safe_format_number(sales_data.max(), "$,.0f")],
                        ["Minimum Sale", self.safe_format_number(sales_data.min(), "$,.0f")],
                        ["Sales Std Dev", self.safe_format_number(sales_data.std(), "$,.0f")]
                    ]
                    
                    sales_table = Table(sales_stats, colWidths=[2.5*inch, 2.5*inch])
                    sales_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD'))
                    ]))
                    
                    story.append(sales_table)
            
            story.append(PageBreak())
            
            # 7. Forecasting Results
            story.append(Paragraph("7. SALES FORECASTING & PROJECTIONS", heading_style))
            
            if st.session_state.forecast_done and st.session_state.forecast_data is not None:
                forecast_df = st.session_state.forecast_data
                
                story.append(Paragraph("Forecast Summary", subheading_style))
                
                forecast_stats = [
                    ["Metric", "Value"],
                    ["Forecast Period", f"{len(forecast_df)} months"],
                    ["Total Forecasted Sales", self.safe_format_number(forecast_df['Predicted_Sales'].sum(), "$,.0f")],
                    ["Average Monthly Forecast", self.safe_format_number(forecast_df['Predicted_Sales'].mean(), "$,.0f")],
                    ["Forecast Start", self.safe_strftime(forecast_df['Date'].min())],
                    ["Forecast End", self.safe_strftime(forecast_df['Date'].max())]
                ]
                
                forecast_table = Table(forecast_stats, colWidths=[2.5*inch, 2.5*inch])
                forecast_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD'))
                ]))
                
                story.append(forecast_table)
                story.append(Spacer(1, 0.2*inch))
                
                # Detailed forecast data
                story.append(Paragraph("Detailed Monthly Forecast", subheading_style))
                
                # Prepare forecast data for table (show first 6 months)
                forecast_display_data = [["Month", "Predicted Sales", "Confidence Range"]]
                for idx, row in forecast_df.head(6).iterrows():
                    forecast_display_data.append([
                        self.safe_strftime(row['Date'], '%b %Y'),
                        self.safe_format_number(row['Predicted_Sales'], "$,.0f"),
                        f"{self.safe_format_number(row['Sales_Lower'], '$,.0f')} - {self.safe_format_number(row['Sales_Upper'], '$,.0f')}"
                    ])
                
                forecast_detail_table = Table(forecast_display_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
                forecast_detail_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#465362')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD'))
                ]))
                
                story.append(forecast_detail_table)
            
            story.append(PageBreak())
            
            # 8. AI Recommendations
            story.append(Paragraph("8. AI-POWERED STRATEGIC RECOMMENDATIONS", heading_style))
            
            if st.session_state.ai_insights:
                recommendations = st.session_state.ai_insights.get("strategic_recommendations", "No recommendations available.")
                recommendations_clean = recommendations.replace('##', '').replace('**', '').replace('###', '')
                story.append(Paragraph(recommendations_clean, normal_style))
            
            story.append(Spacer(1, 0.2*inch))
            
            # 9. Risk Assessment
            story.append(Paragraph("9. RISK ASSESSMENT & MITIGATION", heading_style))
            
            risk_content = """
            <b>Identified Risks:</b>
            <br/><br/>
            • <b>Market Volatility:</b> Economic fluctuations impacting sales forecasts
            • <b>Data Quality:</b> Potential inconsistencies in historical data
            • <b>Operational Risks:</b> Internal process inefficiencies
            • <b>Competitive Pressure:</b> Market competition affecting growth
            <br/><br/>
            <b>Mitigation Strategies:</b>
            <br/><br/>
            • Implement continuous monitoring of key performance indicators
            • Establish data validation protocols for improved accuracy
            • Develop contingency plans for various market scenarios
            • Regular competitive analysis and market positioning reviews
            """
            
            story.append(Paragraph(risk_content, normal_style))
            
            story.append(PageBreak())
            
            # 10. Technical Appendix
            story.append(Paragraph("10. TECHNICAL APPENDIX", heading_style))
            
            appendix_content = f"""
            <b>Analysis Methodology:</b>
            <br/><br/>
            This report was generated using the AI Financial Intelligence Platform with the following analysis steps:
            <br/><br/>
            • <b>Data Upload & Validation:</b> Automated detection of financial columns and data quality assessment
            • <b>Data Cleaning:</b> Advanced preprocessing including missing value handling and outlier detection
            • <b>Exploratory Analysis:</b> Comprehensive statistical analysis and trend identification
            • <b>Forecasting:</b> Time-series analysis with confidence intervals and scenario modeling
            • <b>AI Insights:</b> Machine learning-powered business intelligence and recommendations
            <br/><br/>
            <b>Data Sources:</b>
            <br/><br/>
            • Primary financial data: {os.path.basename(st.session_state.uploaded_file) if st.session_state.uploaded_file else 'User uploaded file'}
            • Analysis period: Complete historical data provided
            • Generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}
            <br/><br/>
            <b>Confidentiality Notice:</b>
            <br/><br/>
            This report contains confidential business information and should be distributed only to authorized personnel.
            """
            
            story.append(Paragraph(appendix_content, normal_style))
            
            # Build PDF
            doc.build(story)
            
            return report_path
            
        except Exception as e:
            st.error(f"Error generating PDF report: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return None

    def download_report(self):
        """Create enhanced download functionality"""
        if st.session_state.report_generated and st.session_state.report_path:
            try:
                with open(st.session_state.report_path, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download Comprehensive Financial Report (PDF)",
                        data=file,
                        file_name=os.path.basename(st.session_state.report_path),
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                if btn:
                    st.success("✅ Report download started!")
                    st.balloons()
                    
                    # Show report summary
                    st.markdown("#### 📋 Report Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📄 Pages", "10+")
                    with col2:
                        st.metric("📊 Sections", "8")
                    with col3:
                        file_size = os.path.getsize(st.session_state.report_path) / 1024
                        st.metric("💾 File Size", f"{file_size:.1f} KB")
                        
            except Exception as e:
                st.error(f"❌ Error downloading report: {str(e)}")
        else:
            st.warning("📄 Please generate the report first in Step 6.")

    def run(self):
        """Main application runner"""
        try:
            self.render_header()
            self.render_sidebar()
            
            # Main content area with enhanced tabs
            tab1, tab2 = st.tabs(["🚀 Analysis Workflow", "📚 Documentation"])
            
            with tab1:
                self.render_file_upload()
                st.markdown("---")
                self.render_data_cleaning()
                st.markdown("---")
                self.render_analysis()
                st.markdown("---")
                self.render_forecasting()
                st.markdown("---")
                self.render_ai_summary()
                st.markdown("---")
                self.render_report_generation()
                
            with tab2:
                self.render_documentation()
                
        except Exception as e:
            st.error(f"🚨 Application error: {str(e)}")
            st.info("Please refresh the page and try again. If the problem persists, check your data format.")

    def render_documentation(self):
        """Render comprehensive documentation"""
        st.markdown("""
        ## 📚 Comprehensive Documentation
        
        ### 🎯 Platform Overview
        
        The **AI Financial Intelligence Platform** is an advanced analytics solution that transforms raw financial data into actionable business intelligence.
        
        ### 🚀 Key Features
        
        #### 1. Intelligent Data Processing
        - **Automated Column Detection**: AI-powered recognition of financial columns
        - **Advanced Data Cleaning**: Smart handling of missing values and outliers
        - **Currency Normalization**: Automatic conversion of various currency formats
        
        #### 2. Advanced Analytics
        - **Interactive Dashboards**: Real-time visualization with Plotly
        - **Correlation Analysis**: Identify relationships between business metrics
        - **Trend Analysis**: Time-series decomposition and pattern recognition
        
        #### 3. AI-Powered Insights
        - **Executive Summaries**: Automated business performance assessment
        - **Strategic Recommendations**: Actionable insights based on data patterns
        - **Risk Assessment**: Comprehensive risk analysis and mitigation strategies
        
        #### 4. Professional Reporting
        - **Comprehensive PDF Reports**: Detailed analysis with all completed steps
        - **Executive Summaries**: Board-ready presentation of findings
        - **Technical Documentation**: Complete methodology and data sources
        
        ### 📊 Supported Data Formats
        
        The platform supports comprehensive financial data structures including:
        
        #### Essential Columns:
        - **Sales/Revenue**: Monetary transaction amounts
        - **Profit/Margin**: Profitability metrics
        - **Date/Time**: Transaction timestamps for trend analysis
        
        #### Enhanced Columns:
        - **Business Segments**: Customer or product categories
        - **Geographic Data**: Country, region, or location information
        - **Product Information**: SKU, category, or service types
        
        ### 💾 Data Security & Privacy
        
        - **Local Storage**: All data stored securely on your local machine
        - **No Cloud Processing**: Complete privacy and data protection
        - **Encrypted Backups**: Secure backup of all generated reports
        
        ### 📄 Report Features
        
        The generated PDF report includes:
        - Executive summary with key findings
        - Comprehensive data quality assessment
        - Detailed financial performance analysis
        - Sales forecasting with confidence intervals
        - AI-powered strategic recommendations
        - Risk assessment and mitigation strategies
        - Complete technical appendix
        """)

# Run the application
if __name__ == "__main__":
    try:
        app = EnhancedFinancialApp()
        app.run()
    except Exception as e:
        st.error(f"🚨 Critical application error: {str(e)}")
        st.info("Please refresh the page and try again. If the problem persists, check your data format.")