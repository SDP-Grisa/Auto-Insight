import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import holidays
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import json

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #000000;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        color: #000000;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #000000;
    }
    .danger-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)


class DataAnalyzer:
    """Intelligent data analyzer that automatically detects patterns and generates insights"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.date_col = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.insights = []
        self._analyze_structure()
    
    def _analyze_structure(self):
        """Automatically detect column types and structure"""
        # Detect date columns
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    if self.date_col is None:
                        self.date_col = col
                except:
                    pass
        
        # Classify columns
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_cols.append(col)
            elif pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                self.categorical_cols.append(col)
    
    def enrich_with_time_features(self):
        """Add time-based features for deeper analysis"""
        if self.date_col:
            df = self.df.copy()
            df['Year'] = df[self.date_col].dt.year
            df['Month'] = df[self.date_col].dt.month
            df['Month_Name'] = df[self.date_col].dt.strftime('%B')
            df['Week'] = df[self.date_col].dt.isocalendar().week
            df['Day'] = df[self.date_col].dt.day
            df['DayOfWeek'] = df[self.date_col].dt.dayofweek
            df['DayName'] = df[self.date_col].dt.strftime('%A')
            df['Quarter'] = df[self.date_col].dt.quarter
            df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
            
            # Add US holidays
            us_holidays = holidays.US(years=df['Year'].unique().tolist())
            df['IsHoliday'] = df[self.date_col].dt.date.isin(us_holidays).astype(int)
            df['HolidayName'] = df[self.date_col].dt.date.map(
                lambda x: us_holidays.get(x, '')
            )
            
            # Add week type
            df['WeekType'] = df.apply(
                lambda row: 'Holiday' if row['IsHoliday'] else ('Weekend' if row['IsWeekend'] else 'Weekday'),
                axis=1
            )
            
            return df
        return self.df
    
    def detect_sales_metrics(self):
        """Automatically detect and calculate sales-related metrics"""
        metrics = {}
        df = self.df
        
        # Try to identify common metric patterns
        sales_cols = [col for col in df.columns if any(x in col.lower() for x in ['sale', 'revenue', 'amount'])]
        click_cols = [col for col in df.columns if 'click' in col.lower()]
        impression_cols = [col for col in df.columns if any(x in col.lower() for x in ['impr', 'impression', 'view'])]
        cart_cols = [col for col in df.columns if any(x in col.lower() for x in ['cart', 'abandon'])]
        cost_cols = [col for col in df.columns if 'cost' in col.lower() and 'ad' not in col.lower()]
        price_cols = [col for col in df.columns if 'price' in col.lower() and 'avg' not in col.lower()]
        ad_cost_cols = [col for col in df.columns if 'ad' in col.lower() and 'cost' in col.lower()]
        inventory_cols = [col for col in df.columns if any(x in col.lower() for x in ['inventory', 'stock'])]
        
        # Calculate CTR if possible
        if click_cols and impression_cols:
            total_clicks = df[click_cols[0]].sum()
            total_impressions = df[impression_cols[0]].sum()
            if total_impressions > 0:
                metrics['CTR'] = (total_clicks / total_impressions) * 100
        
        # Calculate conversion rate if possible
        if sales_cols and click_cols:
            total_sales = df[sales_cols[0]].sum()
            total_clicks = df[click_cols[0]].sum()
            if total_clicks > 0:
                metrics['Conversion_Rate'] = (total_sales / total_clicks) * 100
        
        # Calculate cart abandonment rate
        if cart_cols and sales_cols:
            total_abandoned = df[cart_cols[0]].sum()
            total_sales = df[sales_cols[0]].sum()
            total_carts = total_abandoned + total_sales
            if total_carts > 0:
                metrics['Cart_Abandonment_Rate'] = (total_abandoned / total_carts) * 100
        
        # Calculate ROAS (Return on Ad Spend)
        if sales_cols and ad_cost_cols and (cost_cols or price_cols):
            revenue_col = price_cols[0] if price_cols else cost_cols[0]
            total_revenue = (df[sales_cols[0]] * df[revenue_col]).sum()
            total_ad_cost = df[ad_cost_cols[0]].sum()
            if total_ad_cost > 0:
                metrics['ROAS'] = total_revenue / total_ad_cost
        
        # Calculate Average Order Value
        if sales_cols and (cost_cols or price_cols):
            revenue_col = price_cols[0] if price_cols else cost_cols[0]
            total_revenue = (df[sales_cols[0]] * df[revenue_col]).sum()
            total_orders = df[sales_cols[0]].sum()
            if total_orders > 0:
                metrics['AOV'] = total_revenue / total_orders
        
        # Calculate Average Margin
        if price_cols and cost_cols:
            avg_margin = ((df[price_cols[0]] - df[cost_cols[0]]) / df[price_cols[0]]) * 100
            metrics['Avg_Margin'] = avg_margin.mean()
        
        return metrics
    
    def analyze_pricing_impact(self):
        """Analyze how pricing affects cart abandonment and sales - with safe handling"""
        analysis = {}
        
        price_col = self._find_column_by_keyword(['price'])
        cart_col  = self._find_column_by_keyword(['cart', 'abandon'])
        sales_col = self._find_column_by_keyword(['sale', 'revenue', 'amount'])
        
        if not price_col:
            return analysis
        
        df = self.df.copy()
        
        # Make sure price is numeric
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df = df.dropna(subset=[price_col])  # drop rows where price is invalid
        
        if len(df) == 0 or df[price_col].nunique() < 2:
            analysis['note'] = "Not enough valid price variation for binning"
            return analysis
        
        # Safe quantile binning
        try:
            df['Price_Bin'] = pd.qcut(
                df[price_col],
                q=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
        except ValueError:
            # Fallback: equal-width bins if quantiles fail
            df['Price_Bin'] = pd.cut(
                df[price_col],
                bins=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                include_lowest=True,
                duplicates='drop'
            )
        
        # Cart abandonment by price bin
        if cart_col and sales_col:
            price_cart = df.groupby('Price_Bin', observed=True).agg({
                cart_col: 'sum',
                sales_col: 'sum'
            }).rename(columns={
                cart_col: 'Abandoned_Carts',
                sales_col: 'Completed_Sales'
            })
            price_cart['Total_Carts'] = price_cart['Abandoned_Carts'] + price_cart['Completed_Sales']
            price_cart['Abandonment_Rate_%'] = (price_cart['Abandoned_Carts'] / price_cart['Total_Carts'].replace(0, np.nan)) * 100
            analysis['cart_abandonment_by_price'] = price_cart.reset_index()
        
        # Sales performance by price bin
        if sales_col:
            agg_dict = {sales_col: ['sum', 'mean', 'count']}
            price_sales = df.groupby('Price_Bin', observed=True).agg(agg_dict)
            
            # Flatten columns immediately
            price_sales.columns = [
                f"{sales_col}_{agg}" for agg in ['sum', 'mean', 'count']
            ]
            
            price_sales = price_sales.reset_index()
            
            # Find best price range (highest total sales)
            total_sales_col = f"{sales_col}_sum"
            if total_sales_col in price_sales.columns and not price_sales.empty:
                best_bin_label = price_sales[total_sales_col].idxmax()
                analysis['best_price_range'] = price_sales.loc[best_bin_label, 'Price_Bin']
                # Optional: also store the value
                analysis['best_price_range_sales'] = price_sales.loc[best_bin_label, total_sales_col]
            
            analysis['sales_by_price'] = price_sales
        
        return analysis

    def analyze_brand_margins(self):
        """Updated with flattened columns"""
        analysis = {}
        
        price_col = self._find_column_by_keyword(['price'])
        cost_col  = self._find_column_by_keyword(['cost'])
        brand_col = 'Brand' if 'Brand' in self.df.columns else None
        sales_col = self._find_column_by_keyword(['sale', 'revenue'])
        
        if not (price_col and cost_col):
            return analysis
        
        df = self.df.copy()
        df['Margin']     = df[price_col] - df[cost_col]
        df['Margin_Pct'] = (df['Margin'] / df[price_col].replace(0, np.nan)) * 100
        
        if brand_col:
            agg_dict = {
                'Margin': 'mean',
                'Margin_Pct': 'mean',
                price_col: 'mean',
                cost_col: 'mean'
            }
            if sales_col:
                agg_dict[sales_col] = 'sum'
            
            brand_margins = df.groupby(brand_col).agg(agg_dict)
            
            # Flatten
            brand_margins.columns = [
                col if not isinstance(col, tuple) else f"{col[0]}_{col[1]}"
                for col in brand_margins.columns
            ]
            
            if sales_col:
                brand_margins['Total_Revenue'] = df.groupby(brand_col).apply(
                    lambda x: (x[sales_col] * x[price_col]).sum()
                )
                brand_margins['Total_Margin_Amount'] = df.groupby(brand_col).apply(
                    lambda x: (x[sales_col] * x['Margin']).sum()
                )
            
            brand_margins = brand_margins.sort_values('Margin_Pct', ascending=False)
            analysis['brand_margins'] = brand_margins.reset_index()
        
        analysis['avg_margin']     = df['Margin'].mean()
        analysis['avg_margin_pct'] = df['Margin_Pct'].mean()
        analysis['median_margin_pct'] = df['Margin_Pct'].median()
        
        return analysis


    def analyze_ad_impact(self):
        """Analyze how ad spend impacts sales - with safe binning & flattened columns"""
        analysis = {}
        
        ad_cost_col = self._find_column_by_keyword(['adcost', 'ad cost', 'ad_spend', 'spend'])
        sales_col   = self._find_column_by_keyword(['sale', 'revenue', 'amount'])
        click_col   = self._find_column_by_keyword(['click'])
        
        if not (ad_cost_col and sales_col):
            analysis['error'] = "Missing required columns: ad cost and sales/revenue"
            return analysis
        
        df = self.df.copy()
        df[ad_cost_col] = pd.to_numeric(df[ad_cost_col], errors='coerce').fillna(0)
        df[sales_col]   = pd.to_numeric(df[sales_col], errors='coerce').fillna(0)
        
        total_ad_spend = df[ad_cost_col].sum()
        if total_ad_spend <= 0:
            analysis['warning'] = "Total ad spend is zero or negative — no meaningful analysis"
            return analysis
        
        # Safe binning
        positive_ad = df[ad_cost_col][df[ad_cost_col] > 0]
        n_unique_ad = positive_ad.nunique()
        
        if n_unique_ad >= 5:
            try:
                df['Ad_Spend_Level'] = pd.qcut(
                    df[ad_cost_col],
                    q=4,
                    labels=['Low', 'Medium', 'High', 'Very High'],
                    duplicates='drop'
                )
            except ValueError:
                df['Ad_Spend_Level'] = pd.cut(
                    df[ad_cost_col],
                    bins=4,
                    labels=['Low', 'Medium', 'High', 'Very High'],
                    include_lowest=True
                )
        else:
            if n_unique_ad >= 2:
                med = df[ad_cost_col].median()
                df['Ad_Spend_Level'] = pd.cut(
                    df[ad_cost_col],
                    bins=[-np.inf, 0, med, df[ad_cost_col].max()],
                    labels=['Zero/Low', 'Above Median'],
                    include_lowest=True
                )
            else:
                df['Ad_Spend_Level'] = 'All'
            analysis['note'] = f"Low variation in ad spend ({n_unique_ad} unique positive values) — simplified grouping"

        # Aggregation with flattened names
        agg_dict = {
            sales_col: ['sum', 'mean', 'count'],
            ad_cost_col: ['sum', 'mean']
        }
        if click_col in df.columns:
            agg_dict[click_col] = 'sum'

        ad_performance = df.groupby('Ad_Spend_Level', observed=True).agg(agg_dict)
        
        # Flatten columns right away
        ad_performance.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] else col
            for col in ad_performance.columns
        ]
        
        # Rename for clarity
        rename_map = {
            f'{sales_col}_sum':   'Total_Sales',
            f'{sales_col}_mean':  'Avg_Sales',
            f'{sales_col}_count': 'Record_Count',
            f'{ad_cost_col}_sum': 'Total_Ad_Spend',
            f'{ad_cost_col}_mean':'Avg_Ad_Spend',
        }
        if click_col in df.columns:
            rename_map[f'{click_col}_sum'] = 'Total_Clicks'
        
        ad_performance = ad_performance.rename(columns=rename_map)
        
        # ROAS
        if 'Total_Sales' in ad_performance.columns and 'Total_Ad_Spend' in ad_performance.columns:
            ad_performance['ROAS'] = ad_performance['Total_Sales'] / ad_performance['Total_Ad_Spend'].replace(0, np.nan)
        
        analysis['ad_performance'] = ad_performance.reset_index()
        
        # Overall ROAS
        total_sales = df[sales_col].sum()
        if total_ad_spend > 0:
            analysis['overall_roas'] = total_sales / total_ad_spend
        
        # Daily correlation if date available
        if self.date_col:
            daily = df.groupby(df[self.date_col].dt.date).agg({
                ad_cost_col: 'sum',
                sales_col: 'sum'
            }).rename(columns={ad_cost_col: 'Ad_Spend', sales_col: 'Sales'})
            
            if len(daily) >= 5:
                analysis['ad_sales_correlation'] = daily['Ad_Spend'].corr(daily['Sales'])
                daily['ROAS'] = daily['Sales'] / daily['Ad_Spend'].replace(0, np.nan)
                best_day = daily['ROAS'].idxmax()
                analysis['best_daily_roas']   = daily.loc[best_day, 'ROAS']
                analysis['best_daily_spend']  = daily.loc[best_day, 'Ad_Spend']
                analysis['best_day_date']     = best_day

        return analysis
    
    def generate_ai_insights(self, enriched_df):
        """Generate AI-powered insights from the data"""
        insights = []
        
        # Time-based insights
        if self.date_col and 'WeekType' in enriched_df.columns:
            sales_col = self._find_column_by_keyword(['sale', 'revenue'])
            if sales_col:
                week_performance = enriched_df.groupby('WeekType')[sales_col].agg(['sum', 'mean', 'count'])
                
                if 'Weekday' in week_performance.index and 'Weekend' in week_performance.index:
                    weekday_avg = week_performance.loc['Weekday', 'mean']
                    weekend_avg = week_performance.loc['Weekend', 'mean']
                    diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
                    
                    if abs(diff_pct) > 5:
                        trend = "higher" if diff_pct > 0 else "lower"
                        insights.append({
                            'type': 'success' if diff_pct > 0 else 'warning',
                            'title': 'Weekend Performance',
                            'message': f"Weekend sales are {abs(diff_pct):.1f}% {trend} than weekdays. Average: ${weekend_avg:,.2f} vs ${weekday_avg:,.2f}",
                            'priority': 'high'
                        })
                
                if 'Holiday' in week_performance.index:
                    holiday_avg = week_performance.loc['Holiday', 'mean']
                    weekday_avg = week_performance.loc['Weekday', 'mean']
                    diff_pct = ((holiday_avg - weekday_avg) / weekday_avg) * 100
                    
                    if abs(diff_pct) > 10:
                        trend = "spike" if diff_pct > 0 else "drop"
                        insights.append({
                            'type': 'success' if diff_pct > 0 else 'danger',
                            'title': 'Holiday Impact',
                            'message': f"Holiday sales show a {abs(diff_pct):.1f}% {trend} compared to regular weekdays",
                            'priority': 'high'
                        })
        
        # Trend analysis
        if self.date_col:
            sales_col = self._find_column_by_keyword(['sale', 'revenue'])
            if sales_col:
                daily_sales = enriched_df.groupby(enriched_df[self.date_col].dt.date)[sales_col].sum()
                
                # Calculate growth trend
                if len(daily_sales) >= 7:
                    recent_avg = daily_sales.tail(7).mean()
                    previous_avg = daily_sales.head(7).mean()
                    if previous_avg > 0:
                        growth = ((recent_avg - previous_avg) / previous_avg) * 100
                        if abs(growth) > 5:
                            trend = "growth" if growth > 0 else "decline"
                            insights.append({
                                'type': 'success' if growth > 0 else 'warning',
                                'title': 'Sales Trend',
                                'message': f"Recent 7-day average shows {abs(growth):.1f}% {trend} compared to the first week",
                                'priority': 'medium'
                            })
        
        # Inventory insights
        inventory_col = self._find_column_by_keyword(['inventory', 'stock'])
        if inventory_col:
            low_stock = enriched_df[enriched_df[inventory_col] < enriched_df[inventory_col].quantile(0.1)]
            if len(low_stock) > 0:
                pct_low = (len(low_stock) / len(enriched_df)) * 100
                insights.append({
                    'type': 'danger',
                    'title': 'Low Inventory Alert',
                    'message': f"{pct_low:.1f}% of products are running low on inventory (bottom 10%)",
                    'priority': 'high'
                })
        
        # Performance insights
        if 'Brand' in enriched_df.columns or 'Category' in enriched_df.columns:
            group_col = 'Brand' if 'Brand' in enriched_df.columns else 'Category'
            sales_col = self._find_column_by_keyword(['sale', 'revenue'])
            
            if sales_col:
                performance = enriched_df.groupby(group_col)[sales_col].sum().sort_values(ascending=False)
                top_performer = performance.index[0]
                top_sales = performance.iloc[0]
                total_sales = performance.sum()
                contribution = (top_sales / total_sales) * 100
                
                insights.append({
                    'type': 'success',
                    'title': 'Top Performer',
                    'message': f"{top_performer} leads with {contribution:.1f}% of total sales",
                    'priority': 'medium'
                })
        
        return insights
    
    def _find_column_by_keyword(self, keywords):
        """Helper to find column by keywords"""
        for col in self.df.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None


def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def create_overview_metrics(df, analyzer):
    """Create overview metrics cards"""
    metrics = analyzer.detect_sales_metrics()
    
    cols = st.columns(4)
    
    # Display detected metrics
    metric_configs = [
        ('CTR', 'Click-Through Rate', '%', 'success'),
        ('Conversion_Rate', 'Conversion Rate', '%', 'success'),
        ('Cart_Abandonment_Rate', 'Cart Abandonment', '%', 'danger'),
        ('ROAS', 'Return on Ad Spend', 'x', 'success'),
        ('AOV', 'Avg Order Value', '$', 'success'),
    ]
    
    col_idx = 0
    for key, label, suffix, type_color in metric_configs:
        if key in metrics:
            with cols[col_idx % 4]:
                value = metrics[key]
                if suffix == '$':
                    display_value = f"${value:,.2f}"
                elif suffix == '%':
                    display_value = f"{value:.2f}%"
                else:
                    display_value = f"{value:.2f}{suffix}"
                
                st.metric(label, display_value)
            col_idx += 1


def create_time_analysis(enriched_df, analyzer):
    """Create comprehensive time-based analysis"""
    st.header("📅 Temporal Analysis")
    
    if analyzer.date_col is None:
        st.warning("No date column detected for time-based analysis")
        return
    
    sales_col = analyzer._find_column_by_keyword(['sale', 'revenue', 'amount'])
    if not sales_col:
        st.warning("No sales column detected")
        return
    
    # Monthly trend
    # st.subheader("Monthly Sales Trend")
    # monthly_sales = enriched_df.groupby(['Year', 'Month_Name'])[sales_col].sum().reset_index()
    # monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month_Name']
    
    # fig = px.line(monthly_sales, x='YearMonth', y=sales_col, 
    #               title='Sales Trend Over Time',
    #               labels={sales_col: 'Total Sales', 'YearMonth': 'Month'})
    # fig.update_traces(line_color='#667eea', line_width=3)
    # st.plotly_chart(fig, use_container_width=True)

    # Monthly trend – sorted chronologically
    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        enriched_df
        .groupby(['Year', 'Month', 'Month_Name'])[sales_col]
        .sum()
        .reset_index()
    )

    # Create sortable year-month string (YYYY-MM format)
    monthly_sales['YearMonth'] = (
        monthly_sales['Year'].astype(str) + '-' +
        monthly_sales['Month'].astype(str).str.zfill(2)
    )

    # Sort chronologically
    monthly_sales = monthly_sales.sort_values('YearMonth')

    # Optional: nicer display label
    monthly_sales['DisplayMonth'] = monthly_sales['Month_Name'] + ' ' + monthly_sales['Year'].astype(str)

    fig = px.line(
        monthly_sales,
        x='YearMonth',           # invisible but correctly ordered
        y=sales_col,
        text='DisplayMonth',     # what actually shows in hover / labels
        title='Sales Trend Over Time',
        labels={sales_col: 'Total Sales', 'YearMonth': 'Month'}
    )

    # Make x-axis show friendly month-year
    fig.update_xaxes(
        tickmode='array',
        tickvals=monthly_sales['YearMonth'],
        ticktext=monthly_sales['DisplayMonth']
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Day of Week")
        dow_sales = enriched_df.groupby('DayName')[sales_col].agg(['sum', 'mean']).reset_index()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_sales['DayName'] = pd.Categorical(dow_sales['DayName'], categories=dow_order, ordered=True)
        dow_sales = dow_sales.sort_values('DayName')
        
        fig = px.bar(dow_sales, x='DayName', y='mean',
                     title='Average Daily Sales by Weekday',
                     labels={'mean': 'Average Sales', 'DayName': 'Day'})
        fig.update_traces(marker_color='#764ba2')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Weekend vs Weekday vs Holiday")
        week_type_sales = enriched_df.groupby('WeekType')[sales_col].agg(['sum', 'mean', 'count']).reset_index()
        
        fig = px.bar(week_type_sales, x='WeekType', y='mean',
                     title='Average Sales by Day Type',
                     labels={'mean': 'Average Sales', 'WeekType': 'Day Type'})
        fig.update_traces(marker_color='#667eea')
        st.plotly_chart(fig, use_container_width=True)


def create_category_analysis(enriched_df, analyzer):
    """Create category and brand analysis"""
    st.header("🏷️ Category & Brand Analysis")
    
    sales_col = analyzer._find_column_by_keyword(['sale', 'revenue'])
    if not sales_col:
        return
    
    # Check for categorical columns
    cat_cols = [col for col in enriched_df.columns if col in ['Category', 'Brand', 'SubCategory']]
    
    if not cat_cols:
        st.warning("No category or brand columns detected")
        return
    
    tabs = st.tabs([col for col in cat_cols])
    
    for idx, col in enumerate(cat_cols):
        with tabs[idx]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top performers
                top_perf = enriched_df.groupby(col)[sales_col].sum().sort_values(ascending=False).head(10)
                fig = px.bar(x=top_perf.values, y=top_perf.index, orientation='h',
                            title=f'Top 10 {col} by Sales',
                            labels={'x': 'Total Sales', 'y': col})
                fig.update_traces(marker_color='#667eea')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sales distribution
                dist_data = enriched_df.groupby(col)[sales_col].sum()
                fig = px.pie(values=dist_data.values, names=dist_data.index,
                            title=f'{col} Sales Distribution',
                            hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            
            # Time-based performance for this category
            if analyzer.date_col and 'Month_Name' in enriched_df.columns:
                st.subheader(f"{col} Performance Over Time")
                top_items = enriched_df.groupby(col)[sales_col].sum().nlargest(5).index
                filtered_df = enriched_df[enriched_df[col].isin(top_items)]
                monthly_cat = filtered_df.groupby(['Month_Name', col])[sales_col].sum().reset_index()
                
                fig = px.line(monthly_cat, x='Month_Name', y=sales_col, color=col,
                             title=f'Top 5 {col} - Monthly Trend',
                             labels={sales_col: 'Sales', 'Month_Name': 'Month'})
                st.plotly_chart(fig, use_container_width=True)


def create_marketing_analysis(enriched_df, analyzer):
    """Create marketing and advertising analysis"""
    st.header("📢 Marketing Performance Analysis")
    
    # Check for marketing columns
    has_clicks = 'Clicks' in enriched_df.columns
    has_impr = 'Impr' in enriched_df.columns
    has_ad_cost = 'AdCost' in enriched_df.columns
    has_cpc = 'AvgCPC' in enriched_df.columns
    
    if not (has_clicks or has_impr or has_ad_cost):
        st.warning("No marketing data columns detected")
        return
    
    tabs = st.tabs(["📊 Overview", "💰 Ad Impact Analysis", "📈 Campaign Trends"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            if has_clicks and has_impr:
                st.subheader("CTR Trend Over Time")
                if analyzer.date_col:
                    daily_marketing = enriched_df.groupby(enriched_df[analyzer.date_col].dt.date).agg({
                        'Clicks': 'sum',
                        'Impr': 'sum'
                    }).reset_index()
                    daily_marketing['CTR'] = (daily_marketing['Clicks'] / daily_marketing['Impr']) * 100
                    
                    fig = px.line(daily_marketing, x=analyzer.date_col, y='CTR',
                                 title='Click-Through Rate Trend',
                                 labels={'CTR': 'CTR (%)', analyzer.date_col: 'Date'})
                    fig.update_traces(line_color='#667eea', line_width=2)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if has_ad_cost and has_clicks:
                st.subheader("Ad Efficiency")
                if analyzer.date_col:
                    daily_ad = enriched_df.groupby(enriched_df[analyzer.date_col].dt.date).agg({
                        'AdCost': 'sum',
                        'Clicks': 'sum'
                    }).reset_index()
                    daily_ad['CostPerClick'] = daily_ad['AdCost'] / daily_ad['Clicks']
                    
                    fig = px.line(daily_ad, x=analyzer.date_col, y='CostPerClick',
                                 title='Cost Per Click Trend',
                                 labels={'CostPerClick': 'CPC ($)', analyzer.date_col: 'Date'})
                    fig.update_traces(line_color='#764ba2', line_width=2)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Campaign performance by category
        if 'Brand' in enriched_df.columns and has_clicks and has_impr:
            st.subheader("Campaign Performance by Brand")
            brand_perf = enriched_df.groupby('Brand').agg({
                'Clicks': 'sum',
                'Impr': 'sum',
                'AdCost': 'sum' if has_ad_cost else 'count'
            }).reset_index()
            brand_perf['CTR'] = (brand_perf['Clicks'] / brand_perf['Impr']) * 100
            brand_perf = brand_perf.nlargest(10, 'Clicks')
            
            fig = px.scatter(brand_perf, x='Impr', y='Clicks', size='CTR', color='Brand',
                            hover_data=['CTR'],
                            title='Brand Advertising Performance',
                            labels={'Impr': 'Impressions', 'Clicks': 'Clicks'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:  # Ad Impact tab
        ad_analysis = analyzer.analyze_ad_impact()
        
        if 'ad_performance' in ad_analysis and not ad_analysis['ad_performance'].empty:
            st.subheader("Performance by Ad Spend Level")
            ad_perf_df = ad_analysis['ad_performance']
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Sales by Ad Spend Level', 'Ad Spend vs Sales')
            )
            
            if 'Total_Sales' in ad_perf_df.columns:
                fig.add_trace(
                    go.Bar(x=ad_perf_df['Ad_Spend_Level'], 
                           y=ad_perf_df['Total_Sales'],
                           name='Total Sales',
                           marker_color='#667eea'),
                    row=1, col=1
                )
            
            if 'Total_Ad_Spend' in ad_perf_df.columns:
                fig.add_trace(
                    go.Bar(x=ad_perf_df['Ad_Spend_Level'], 
                           y=ad_perf_df['Total_Ad_Spend'],
                           name='Ad Spend',
                           marker_color='#ffc107'),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            if 'ROAS' in ad_perf_df.columns:
                st.subheader("📊 ROI by Ad Spend Level")
                roi_df = ad_perf_df[['Ad_Spend_Level', 'ROAS']].copy()
                fig = px.bar(roi_df, x='Ad_Spend_Level', y='ROAS',
                             title='Return on Investment by Ad Spend Category',
                             color='ROAS', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
                
                if not roi_df.empty:
                    best = roi_df.loc[roi_df['ROAS'].idxmax()]
                    st.success(f"Best ROI: {best['ROAS']:.2f}x at **{best['Ad_Spend_Level']}** level")
        else:
            st.info("Not enough data for ad spend level analysis")


    with tabs[2]:
        if analyzer.date_col and has_ad_cost:
            st.subheader("📈 Campaign Performance Over Time")
            
            sales_col = analyzer._find_column_by_keyword(['sale', 'revenue'])
            if sales_col:
                daily_campaign = enriched_df.groupby(enriched_df[analyzer.date_col].dt.date).agg({
                    'AdCost': 'sum',
                    sales_col: 'sum',
                    'Clicks': 'sum' if has_clicks else 'count'
                }).reset_index()
                
                daily_campaign['ROAS'] = daily_campaign[sales_col] / daily_campaign['AdCost']
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Ad Spend vs Sales Over Time', 'ROAS Trend'),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_campaign[analyzer.date_col], y=daily_campaign['AdCost'],
                              name='Ad Spend', line=dict(color='#ffc107')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_campaign[analyzer.date_col], y=daily_campaign[sales_col],
                              name='Sales', line=dict(color='#28a745')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_campaign[analyzer.date_col], y=daily_campaign['ROAS'],
                              name='ROAS', line=dict(color='#667eea', width=3),
                              fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)'),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
                fig.update_yaxes(title_text="ROAS", row=2, col=1)
                
                fig.update_layout(height=700, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)


def create_pricing_margin_analysis(enriched_df, analyzer):
    st.header("💰 Pricing & Margin Intelligence")
    
    tabs = st.tabs(["📊 Margin Analysis", "💵 Pricing Impact", "🎯 Optimization"])
    
    with tabs[1]:  # Pricing Impact
        pricing_analysis = analyzer.analyze_pricing_impact()
        
        if 'sales_by_price' in pricing_analysis and not pricing_analysis['sales_by_price'].empty:
            st.subheader("📊 Sales Performance by Price Range")
            
            sales_data = pricing_analysis['sales_by_price']
            sales_col = analyzer._find_column_by_keyword(['sale', 'revenue'])
            
            total_col = f"{sales_col}_sum"
            mean_col  = f"{sales_col}_mean"
            
            col1, col2 = st.columns(2)
            
            with col1:
                if total_col in sales_data.columns:
                    fig = px.bar(
                        sales_data, x='Price_Bin', y=total_col,
                        title='Total Sales by Price Range',
                        labels={total_col: 'Total Sales', 'Price_Bin': 'Price Range'},
                        color=total_col, color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if mean_col in sales_data.columns:
                    fig = px.bar(
                        sales_data, x='Price_Bin', y=mean_col,
                        title='Average Sales by Price Range',
                        labels={mean_col: 'Avg Sales', 'Price_Bin': 'Price Range'},
                        color=mean_col, color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)

def create_inventory_analysis(enriched_df, analyzer):
    """Create inventory analysis"""
    st.header("📦 Inventory Analysis")
    
    inventory_col = analyzer._find_column_by_keyword(['inventory', 'stock'])
    sales_col = analyzer._find_column_by_keyword(['sale', 'revenue'])
    
    if not inventory_col:
        st.warning("No inventory column detected")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Inventory Distribution")
        fig = px.histogram(enriched_df, x=inventory_col, nbins=50,
                          title='Inventory Level Distribution',
                          labels={inventory_col: 'Inventory Units'})
        fig.update_traces(marker_color='#667eea')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Low Stock Products")
        threshold = enriched_df[inventory_col].quantile(0.1)
        low_stock = enriched_df[enriched_df[inventory_col] <= threshold]
        
        if 'Brand' in enriched_df.columns:
            low_stock_brands = low_stock.groupby('Brand').size().sort_values(ascending=False).head(10)
            fig = px.bar(x=low_stock_brands.values, y=low_stock_brands.index, orientation='h',
                        title=f'Brands with Most Low-Stock Items (< {threshold:.0f} units)',
                        labels={'x': 'Number of Products', 'y': 'Brand'})
            fig.update_traces(marker_color='#dc3545')
            st.plotly_chart(fig, use_container_width=True)
    
    # Inventory turnover
    if sales_col and analyzer.date_col:
        st.subheader("Inventory Turnover Analysis")
        
        # Calculate days of data
        date_range = (enriched_df[analyzer.date_col].max() - enriched_df[analyzer.date_col].min()).days
        
        if 'Brand' in enriched_df.columns:
            turnover = enriched_df.groupby('Brand').agg({
                sales_col: 'sum',
                inventory_col: 'mean'
            }).reset_index()
            turnover['Turnover_Rate'] = (turnover[sales_col] / turnover[inventory_col]) if date_range > 0 else 0
            turnover = turnover.nlargest(15, 'Turnover_Rate')
            
            fig = px.bar(turnover, x='Brand', y='Turnover_Rate',
                        title='Top 15 Brands by Inventory Turnover',
                        labels={'Turnover_Rate': 'Turnover Rate', 'Brand': 'Brand'})
            fig.update_traces(marker_color='#28a745')
            st.plotly_chart(fig, use_container_width=True)


def create_advanced_insights(enriched_df, analyzer):
    """Create advanced statistical insights"""
    st.header("🔬 Advanced Analytics")
    
    tabs = st.tabs(["Correlation Analysis", "Anomaly Detection", "Time Series Forecasting"])
    
    with tabs[0]:
        st.subheader("Feature Correlation Matrix")
        numeric_df = enriched_df[analyzer.numeric_cols].select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, 
                           text_auto='.2f',
                           aspect='auto',
                           color_continuous_scale='RdBu_r',
                           title='Correlation Heatmap')
            st.plotly_chart(fig, use_container_width=True)
            
            # Key correlations
            st.subheader("Key Correlations")
            sales_col = analyzer._find_column_by_keyword(['sale', 'revenue'])
            if sales_col and sales_col in corr_matrix.columns:
                correlations = corr_matrix[sales_col].sort_values(ascending=False)[1:6]
                for feature, corr_value in correlations.items():
                    direction = "positively" if corr_value > 0 else "negatively"
                    st.info(f"**{feature}** is {direction} correlated with sales (r = {corr_value:.3f})")
    
    with tabs[1]:
        st.subheader("Anomaly Detection")
        sales_col = analyzer._find_column_by_keyword(['sale', 'revenue'])
        
        if sales_col and analyzer.date_col:
            daily_sales = enriched_df.groupby(enriched_df[analyzer.date_col].dt.date)[sales_col].sum().reset_index()
            daily_sales.columns = ['Date', 'Sales']
            
            # Calculate Z-scores
            daily_sales['Z_Score'] = np.abs(stats.zscore(daily_sales['Sales']))
            anomalies = daily_sales[daily_sales['Z_Score'] > 2.5]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['Sales'],
                                   mode='lines', name='Daily Sales',
                                   line=dict(color='#667eea')))
            fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Sales'],
                                   mode='markers', name='Anomalies',
                                   marker=dict(color='red', size=10)))
            fig.update_layout(title='Sales Anomalies Detected',
                            xaxis_title='Date', yaxis_title='Sales')
            st.plotly_chart(fig, use_container_width=True)
            
            if len(anomalies) > 0:
                st.warning(f"Detected {len(anomalies)} unusual sales days")
                with st.expander("View Anomaly Details"):
                    st.dataframe(anomalies[['Date', 'Sales']].sort_values('Date', ascending=False))
    
    with tabs[2]:
        create_time_series_forecasting(enriched_df, analyzer)


def create_time_series_forecasting(enriched_df, analyzer):
    """Create comprehensive time series forecasting with ARIMA/SARIMA"""
    st.subheader("📈 Time Series Forecasting (ARIMA/SARIMA)")
    
    sales_col = analyzer._find_column_by_keyword(['sale', 'revenue'])
    
    if not (sales_col and analyzer.date_col):
        st.warning("Need both sales and date columns for forecasting")
        return
    
    if len(enriched_df) < 30:
        st.warning("Need at least 30 days of data for reliable forecasting")
        return
    
    try:
        # Prepare data
        daily_sales = enriched_df.groupby(enriched_df[analyzer.date_col].dt.date)[sales_col].sum().reset_index()
        daily_sales.columns = ['Date', 'Sales']
        daily_sales = daily_sales.set_index('Date')
        daily_sales.index = pd.to_datetime(daily_sales.index)
        
        # Remove any zeros to avoid log issues
        daily_sales = daily_sales[daily_sales['Sales'] > 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_days = st.slider("Forecast Period (days)", 7, 60, 30)
        
        with col2:
            model_type = st.selectbox("Model Type", ["Auto ARIMA", "SARIMA (Seasonal)", "Simple ARIMA"])
        
        # Stationarity test
        st.subheader("📊 Stationarity Test (ADF Test)")
        adf_result = adfuller(daily_sales['Sales'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
        with col2:
            st.metric("P-Value", f"{adf_result[1]:.4f}")
        with col3:
            is_stationary = adf_result[1] < 0.05
            st.metric("Stationary?", "Yes ✅" if is_stationary else "No ❌")
        
        if not is_stationary:
            st.info("💡 Data is non-stationary. The model will apply differencing automatically.")
        
        # Fit model and forecast
        with st.spinner("Building forecast model... This may take 30-60 seconds..."):
            
            if model_type == "Auto ARIMA":
                # Auto ARIMA - let statsmodels find best parameters
                st.info("🤖 Using Auto ARIMA to find optimal parameters...")
                
                # Use simple ARIMA with auto parameters
                best_aic = float('inf')
                best_order = (1, 1, 1)
                
                # Quick search for best parameters
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model = ARIMA(daily_sales['Sales'], order=(p, d, q))
                                fitted = model.fit()
                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                st.success(f"✅ Best ARIMA order found: {best_order} (AIC: {best_aic:.2f})")
                
                model = ARIMA(daily_sales['Sales'], order=best_order)
                fitted_model = model.fit()
                
            elif model_type == "SARIMA (Seasonal)":
                st.info("📅 Using SARIMA with seasonal component (7-day cycle)...")
                
                # SARIMA with weekly seasonality
                model = SARIMAX(daily_sales['Sales'], 
                               order=(1, 1, 1),
                               seasonal_order=(1, 1, 1, 7))
                fitted_model = model.fit(disp=False)
                
            else:  # Simple ARIMA
                st.info("📊 Using ARIMA(1,1,1) model...")
                model = ARIMA(daily_sales['Sales'], order=(1, 1, 1))
                fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=forecast_days)
            forecast_index = pd.date_range(start=daily_sales.index[-1] + timedelta(days=1), 
                                          periods=forecast_days, freq='D')
            
            # Calculate confidence intervals
            forecast_df = fitted_model.get_forecast(steps=forecast_days)
            forecast_ci = forecast_df.conf_int()
            
            # Plot results
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=daily_sales.index,
                y=daily_sales['Sales'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='#667eea', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color='#28a745', width=2, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_index.tolist() + forecast_index.tolist()[::-1],
                y=forecast_ci.iloc[:, 1].tolist() + forecast_ci.iloc[:, 0].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(40, 167, 69, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f'{model_type} Forecast - Next {forecast_days} Days',
                xaxis_title='Date',
                yaxis_title='Sales',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model summary
            with st.expander("📊 Model Statistics"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("AIC", f"{fitted_model.aic:.2f}")
                with col2:
                    st.metric("BIC", f"{fitted_model.bic:.2f}")
                with col3:
                    if hasattr(fitted_model, 'mse'):
                        st.metric("MSE", f"{fitted_model.mse:.2f}")
                
                # Forecast summary
                st.subheader("Forecast Summary")
                forecast_summary = pd.DataFrame({
                    'Date': forecast_index,
                    'Forecasted Sales': forecast.values,
                    'Lower Bound (95%)': forecast_ci.iloc[:, 0].values,
                    'Upper Bound (95%)': forecast_ci.iloc[:, 1].values
                })
                st.dataframe(forecast_summary.head(10), use_container_width=True)
                
                # Key insights
                avg_forecast = forecast.mean()
                avg_historical = daily_sales['Sales'].tail(30).mean()
                trend = "increasing" if avg_forecast > avg_historical else "decreasing"
                change_pct = ((avg_forecast - avg_historical) / avg_historical) * 100
                
                st.info(f"""
                **📈 Forecast Insights:**
                - Average forecasted sales: **${avg_forecast:,.2f}**
                - Recent 30-day average: **${avg_historical:,.2f}**
                - Trend: **{trend}** ({change_pct:+.1f}%)
                - Total forecasted revenue ({forecast_days} days): **${forecast.sum():,.2f}**
                """)
    
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        st.info("💡 Tip: Ensure you have enough data points and no missing dates for accurate forecasting.")


def display_ai_insights(insights):
    """Display AI-generated insights"""
    st.header("🤖 Data Insights")
    
    if not insights:
        st.info("Analyzing data for insights...")
        return
    
    # Sort by priority
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_insights = sorted(insights, key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
    
    for insight in sorted_insights:
        box_class = f"{insight['type']}-box" if insight['type'] in ['success', 'warning', 'danger'] else ''
        
        st.markdown(f"""
        <div class="insight-box {box_class}">
            <h4>💡 {insight['title']}</h4>
            <p>{insight['message']}</p>
        </div>
        """, unsafe_allow_html=True)


def create_export_section(enriched_df):
    """Create data export options"""
    st.header("📥 Export Enhanced Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = enriched_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="enhanced_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create summary report
        summary_data = {
            'Total Records': len(enriched_df),
            'Date Range': f"{enriched_df['Date'].min()} to {enriched_df['Date'].max()}" if 'Date' in enriched_df.columns else 'N/A',
            'Columns': len(enriched_df.columns)
        }
        summary_json = json.dumps(summary_data, indent=2)
        st.download_button(
            label="Download Summary",
            data=summary_json,
            file_name="data_summary.json",
            mime="application/json"
        )


def main():
    st.markdown('<h1 class="main-header">🚀 Data Insights Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to your intelligent analytics platform!
    Upload your data (CSV or Excel) and get comprehensive insights including:
    - 📊 Automatic metric detection (CTR, conversion rates, ROAS, margins, etc.)
    - 📅 Time-based analysis (weekday/weekend/holiday trends)
    - 🏷️ Category and brand performance with margin analysis
    - 📢 Marketing campaign analysis with ad impact insights
    - 📦 Inventory optimization
    - 💰 Pricing intelligence and cart abandonment analysis
    - 📈 Advanced forecasting (ARIMA, SARIMA models)
    - 🤖 AI-powered actionable insights
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        with st.spinner("Loading and analyzing data..."):
            df = load_data(uploaded_file)
            
            if df is not None:
                # Initialize analyzer
                analyzer = DataAnalyzer(df)
                
                # Enrich data with time features
                enriched_df = analyzer.enrich_with_time_features()
                
                # Generate AI insights
                ai_insights = analyzer.generate_ai_insights(enriched_df)
                
                # Display insights first
                display_ai_insights(ai_insights)
                
                st.success(f"✅ Data loaded successfully! {len(df):,} records with {len(df.columns)} columns")
                
                # Overview metrics
                st.header("📈 Key Metrics Overview")
                create_overview_metrics(enriched_df, analyzer)
                
                # Create main navigation tabs
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "📅 Temporal Analysis",
                    "🏷️ Category & Brand",
                    "📢 Marketing",
                    "💰 Pricing & Margins",
                    "📦 Inventory",
                    "🔬 Advanced Analytics",
                    "📥 Export"
                ])
                
                with tab1:
                    create_time_analysis(enriched_df, analyzer)
                
                with tab2:
                    create_category_analysis(enriched_df, analyzer)
                
                with tab3:
                    create_marketing_analysis(enriched_df, analyzer)
                
                with tab4:
                    create_pricing_margin_analysis(enriched_df, analyzer)
                
                with tab5:
                    create_inventory_analysis(enriched_df, analyzer)
                
                with tab6:
                    create_advanced_insights(enriched_df, analyzer)
                
                with tab7:
                    # Export section
                    create_export_section(enriched_df)
                    
                    # Raw data viewer
                    with st.expander("🔍 View Enhanced Data"):
                        st.dataframe(enriched_df, use_container_width=True)
    
    else:
        st.info("👆 Please upload a data file to begin analysis")
        
        # Show example of what the app can do
        st.markdown("""
        ---
        ### 🎯 What This App Does:
        
        **Automatic Insights:**
        - Detects sales patterns across weekdays, weekends, and holidays
        - Identifies top-performing products, brands, and categories
        - Calculates marketing metrics (CTR, conversion rate, ROAS)
        - Analyzes pricing impact on cart abandonment
        - Calculates brand margins and profitability
        - Monitors inventory levels and turnover rates
        
        **Advanced Analytics:**
        - **ARIMA/SARIMA Forecasting**: Predict future sales with statistical models
        - **Ad Impact Analysis**: Understand how ad spend affects sales
        - **Pricing Intelligence**: Discover optimal price points
        - **Margin Analysis**: Track profitability by brand/category
        - Correlation analysis to find what drives sales
        - Anomaly detection to spot unusual patterns
        - Time-series decomposition
        
        **Marketing Intelligence:**
        - Campaign performance tracking
        - Ad spend efficiency analysis (ROAS by ad level)
        - Customer engagement metrics
        - Cart abandonment insights by price range
        
        **Business Recommendations:**
        - AI-powered actionable insights
        - Performance benchmarking
        - Optimization opportunities (pricing, inventory, marketing)
        - Risk alerts and warnings
        """)


if __name__ == "__main__":
    main()