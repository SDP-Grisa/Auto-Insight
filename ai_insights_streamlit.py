import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import json
import os
from groq import Groq
from dotenv import load_dotenv
from scipy import stats

# Try to load local .env
load_dotenv()

warnings.filterwarnings('ignore')

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Page configuration
st.set_page_config(
    page_title="AI Data Insights Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .ai-insight {
        background: #f0f7ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .priority-high { border-left-color: #dc3545; background: #ffe6e6; color: #000000 }
    .priority-medium { border-left-color: #ffc107; background: #fff9e6; color: #000000 }
    .priority-low { border-left-color: #28a745; background: #e6f9e6; color: #000000 }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class DeepDataAnalyzer:
    """Enhanced AI-powered data analyzer with deep metrics"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.analysis = {}
        self.deep_metrics = {}
        self._analyze_structure()
        self._detect_business_metrics()
    
    def _analyze_structure(self):
        """Automatically detect column types with unique value details"""
        self.analysis = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'column_info': {}
        }
        
        for col in self.df.columns:
            unique_values = self.df[col].dropna().unique()
            value_counts = self.df[col].value_counts()
            
            col_info = {
                'dtype': str(self.df[col].dtype),
                'null_count': int(self.df[col].isnull().sum()),
                'null_percentage': float(self.df[col].isnull().sum() / len(self.df) * 100),
                'unique_count': int(self.df[col].nunique()),
                'sample_values': [str(x) for x in self.df[col].dropna().head(3).tolist()],
                'all_unique_values': [str(x) for x in unique_values[:50]],  # First 50 unique values
                'value_distribution': dict(value_counts.head(10))
            }
            
            # Try to detect date columns
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.analysis['datetime_columns'].append(col)
                except:
                    pass
            
            # Classify columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.analysis['numeric_columns'].append(col)
                if not self.df[col].isna().all():
                    col_info['min'] = float(self.df[col].min())
                    col_info['max'] = float(self.df[col].max())
                    col_info['mean'] = float(self.df[col].mean())
                    col_info['median'] = float(self.df[col].median())
                    col_info['std'] = float(self.df[col].std())
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                if col not in self.analysis['datetime_columns']:
                    self.analysis['datetime_columns'].append(col)
            elif self.df[col].dtype == 'object':
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.5:
                    self.analysis['categorical_columns'].append(col)
            
            self.analysis['column_info'][col] = col_info
    
    def _detect_business_metrics(self):
        """Detect and calculate business metrics"""
        metrics = {}
        
        # Find relevant columns
        sales_col = self._find_column(['sale', 'sales', 'sold', 'quantity', 'qty'])
        revenue_col = self._find_column(['revenue', 'amount', 'total', 'price'])
        clicks_col = self._find_column(['click', 'clicks'])
        impressions_col = self._find_column(['impression', 'impr', 'view'])
        cart_col = self._find_column(['cart', 'abandon'])
        cost_col = self._find_column(['cost', 'cogs'])
        ad_cost_col = self._find_column(['ad_cost', 'adcost', 'ad_spend'])
        
        # Calculate CTR
        if clicks_col and impressions_col:
            total_clicks = self.df[clicks_col].sum()
            total_impressions = self.df[impressions_col].sum()
            if total_impressions > 0:
                metrics['CTR'] = (total_clicks / total_impressions) * 100
        
        # Calculate Conversion Rate
        if sales_col and clicks_col:
            total_sales = self.df[sales_col].sum()
            total_clicks = self.df[clicks_col].sum()
            if total_clicks > 0:
                metrics['Conversion_Rate'] = (total_sales / total_clicks) * 100
        
        # Calculate Cart Abandonment Rate
        if cart_col and sales_col:
            total_abandoned = self.df[cart_col].sum()
            total_sales = self.df[sales_col].sum()
            total_carts = total_abandoned + total_sales
            if total_carts > 0:
                metrics['Cart_Abandonment_Rate'] = (total_abandoned / total_carts) * 100
        
        # Calculate ROAS
        if revenue_col and ad_cost_col:
            total_revenue = self.df[revenue_col].sum()
            total_ad_cost = self.df[ad_cost_col].sum()
            if total_ad_cost > 0:
                metrics['ROAS'] = total_revenue / total_ad_cost
        
        # Average Order Value
        if revenue_col and sales_col:
            total_revenue = self.df[revenue_col].sum()
            total_orders = self.df[sales_col].sum()
            if total_orders > 0:
                metrics['AOV'] = total_revenue / total_orders
        
        # Profit Margin
        if revenue_col and cost_col:
            total_revenue = self.df[revenue_col].sum()
            total_cost = self.df[cost_col].sum()
            if total_revenue > 0:
                metrics['Profit_Margin'] = ((total_revenue - total_cost) / total_revenue) * 100
        
        self.deep_metrics = metrics
    
    def _find_column(self, keywords):
        """Find column matching keywords"""
        for col in self.df.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None
    
    def get_stage1_prompt(self, sample_size=5):
        """Stage 1: Data Understanding and Pattern Detection"""
        sample_data = self.df.head(sample_size).to_dict(orient='records')
        
        # Enhanced column details with unique values
        column_details = {}
        for col, info in self.analysis['column_info'].items():
            column_details[col] = {
                'type': info['dtype'],
                'unique_count': info['unique_count'],
                'unique_values': info.get('all_unique_values', []),
                'null_percentage': info['null_percentage'],
                'sample_values': info['sample_values']
            }
            if col in self.analysis['numeric_columns']:
                column_details[col].update({
                    'min': info.get('min'),
                    'max': info.get('max'),
                    'mean': info.get('mean'),
                    'median': info.get('median')
                })
        
        prompt = f"""You are a senior data analyst. Analyze this dataset deeply.

DATASET OVERVIEW:
- Total Rows: {self.analysis['total_rows']:,}
- Total Columns: {self.analysis['total_columns']}
- Numeric Columns: {', '.join(self.analysis['numeric_columns']) or 'None'}
- Categorical Columns: {', '.join(self.analysis['categorical_columns']) or 'None'}
- DateTime Columns: {', '.join(self.analysis['datetime_columns']) or 'None'}

DETAILED COLUMN INFORMATION (with unique values):
{json.dumps(column_details, indent=2, default=str)}

SAMPLE DATA (first {sample_size} rows):
{json.dumps(sample_data, indent=2, default=str)}

DETECTED BUSINESS METRICS:
{json.dumps(self.deep_metrics, indent=2, default=str)}

TASK 1: Analyze the data structure and identify:
1. What type of business data is this? (e.g., e-commerce sales, marketing campaign, customer behavior)
2. What are the key relationships between columns?
3. Are there any data quality issues?
4. What advanced analyses are possible? (time series, cohort analysis, segmentation, etc.)

Respond ONLY with valid JSON:
{{
    "data_type": "e-commerce|marketing|finance|customer|operations|other",
    "business_context": "2-3 sentence description of what this data represents",
    "key_relationships": ["relationship 1", "relationship 2", "relationship 3"],
    "data_quality_issues": ["issue 1", "issue 2"],
    "possible_analyses": ["analysis 1", "analysis 2", "analysis 3"],
    "important_segments": ["segment 1", "segment 2"]
}}"""
        
        return prompt
    
    def get_stage2_prompt(self, stage1_result, sample_size=5):
        """Stage 2: Deep Insights and Advanced Metrics"""
        
        stats_summary = {}
        if self.analysis['numeric_columns']:
            stats_df = self.df[self.analysis['numeric_columns']].describe()
            stats_summary = {col: stats_df[col].to_dict() for col in stats_df.columns}
        
        prompt = f"""You are an expert data scientist. Based on the initial analysis, provide deep insights.

INITIAL ANALYSIS:
{json.dumps(stage1_result, indent=2)}

STATISTICAL SUMMARY:
{json.dumps(stats_summary, indent=2, default=str)}

BUSINESS METRICS DETECTED:
{json.dumps(self.deep_metrics, indent=2, default=str)}

TASK 2: Provide 5-7 deep, actionable insights that include:
- Statistical findings (correlations, distributions, outliers)
- Business performance metrics (CTR, conversion rate, ROAS, cart abandonment if applicable)
- Trend analysis (if time data available)
- Segmentation insights (based on categories)
- Recommendations for improvement

Respond ONLY with valid JSON:
{{
    "insights": [
        {{
            "type": "statistical|business|trend|segmentation|recommendation",
            "title": "Specific, data-driven title",
            "description": "Detailed finding with exact numbers and percentages",
            "impact": "high|medium|low",
            "action": "Specific actionable recommendation"
        }}
    ]
}}"""
        
        return prompt
    
    def get_stage3_prompt(self, stage1_result, stage2_result):
        """Stage 3: Visualization Recommendations"""
        
        prompt = f"""You are a data visualization expert. Create optimal visualization recommendations.

DATA CONTEXT:
{json.dumps(stage1_result, indent=2)}

INSIGHTS DISCOVERED:
{json.dumps(stage2_result, indent=2)}

AVAILABLE COLUMNS:
- Numeric: {', '.join(self.analysis['numeric_columns'])}
- Categorical: {', '.join(self.analysis['categorical_columns'])}
- DateTime: {', '.join(self.analysis['datetime_columns'])}

TASK 3: Recommend 8-10 visualizations that best reveal the insights. Include:
- Trend charts (if time data exists)
- Distribution analysis (histograms, box plots)
- Correlation analysis (scatter plots, heatmaps)
- Segmentation charts (bar charts by category)
- Performance metrics (if business metrics exist)
- Advanced: funnel charts, cohort analysis, forecasting charts

Respond ONLY with valid JSON:
{{
    "visualizations": [
        {{
            "chart_type": "scatter|line|bar|histogram|box|heatmap|pie|funnel|area",
            "x_axis": "exact_column_name",
            "y_axis": "exact_column_name_or_null",
            "color": "exact_column_name_or_null",
            "aggregation": "sum|mean|count|none",
            "title": "Clear, descriptive title",
            "insight": "What specific pattern this reveals",
            "priority": "high|medium|low",
            "advanced_analysis": "timeseries|correlation|distribution|segmentation|performance"
        }}
    ]
}}

CRITICAL: Use ONLY column names that exist in the data. Match chart types to data types."""
        
        return prompt
    
    def generate_fallback_analysis(self):
        """Enhanced fallback analysis with business metrics"""
        insights = []
        visualizations = []
        
        nc = self.analysis['numeric_columns']
        cc = self.analysis['categorical_columns']
        dc = self.analysis['datetime_columns']
        
        # Business metric insights
        if self.deep_metrics:
            for metric, value in self.deep_metrics.items():
                insights.append({
                    "type": "business",
                    "title": f"{metric}: {value:.2f}{'%' if 'Rate' in metric or 'CTR' in metric or 'Margin' in metric else 'x' if metric == 'ROAS' else ''}",
                    "description": f"Current {metric.replace('_', ' ')} stands at {value:.2f}. This indicates {'strong' if value > 50 else 'moderate' if value > 20 else 'needs improvement'} performance.",
                    "priority": "high"
                })
        
        # Data structure insights
        if nc:
            insights.append({
                "type": "distribution",
                "title": f"Dataset contains {len(nc)} measurable metrics",
                "description": f"Key numeric columns: {', '.join(nc[:3])}. Range from {self.df[nc[0]].min():.2f} to {self.df[nc[0]].max():.2f} in {nc[0]}.",
                "priority": "high"
            })
        
        # Category insights with unique values
        if cc:
            for cat_col in cc[:2]:
                unique_count = self.df[cat_col].nunique()
                top_value = self.df[cat_col].value_counts().index[0]
                top_count = self.df[cat_col].value_counts().iloc[0]
                all_values = ', '.join([str(x) for x in self.df[cat_col].unique()[:5]])
                
                insights.append({
                    "type": "pattern",
                    "title": f"{cat_col} has {unique_count} categories",
                    "description": f"Categories: {all_values}{'...' if unique_count > 5 else ''}. Top category '{top_value}' appears {top_count} times ({top_count/len(self.df)*100:.1f}% of data).",
                    "priority": "high"
                })
        
        # Time series insight
        if dc and nc:
            date_col = dc[0]
            date_range = (self.df[date_col].max() - self.df[date_col].min()).days
            insights.append({
                "type": "trend",
                "title": f"Time series data spans {date_range} days",
                "description": f"Date range: {self.df[date_col].min()} to {self.df[date_col].max()}. Enables trend analysis, seasonality detection, and forecasting.",
                "priority": "high"
            })
        
        # Correlation insights
        if len(nc) >= 2:
            corr = self.df[nc[:2]].corr().iloc[0, 1]
            insights.append({
                "type": "correlation",
                "title": f"{nc[0]} vs {nc[1]} correlation: {corr:.2f}",
                "description": f"{'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'} {'positive' if corr > 0 else 'negative'} relationship. This suggests {'direct' if corr > 0 else 'inverse'} relationship between variables.",
                "priority": "medium"
            })
        
        # Generate visualizations
        # 1. Business metrics dashboard
        if self.deep_metrics:
            for metric in list(self.deep_metrics.keys())[:2]:
                metric_col = self._find_column([metric.lower().replace('_', '')])
                if metric_col and cc:
                    visualizations.append({
                        "chart_type": "bar",
                        "x_axis": cc[0],
                        "y_axis": metric_col,
                        "color": None,
                        "title": f"{metric} by {cc[0]}",
                        "insight": f"Compare {metric} performance across segments",
                        "priority": "high"
                    })
        
        # 2. Distribution analysis
        for col in nc[:2]:
            visualizations.append({
                "chart_type": "histogram",
                "x_axis": col,
                "y_axis": None,
                "color": cc[0] if cc else None,
                "title": f"Distribution of {col}",
                "insight": f"Shows spread, outliers, and central tendency",
                "priority": "high"
            })
        
        # 3. Correlation scatter
        if len(nc) >= 2:
            visualizations.append({
                "chart_type": "scatter",
                "x_axis": nc[0],
                "y_axis": nc[1],
                "color": cc[0] if cc else None,
                "title": f"{nc[0]} vs {nc[1]} Relationship",
                "insight": "Reveals correlation and clustering patterns",
                "priority": "high"
            })
        
        # 4. Category analysis
        if cc and nc:
            visualizations.append({
                "chart_type": "box",
                "x_axis": cc[0],
                "y_axis": nc[0],
                "color": None,
                "title": f"{nc[0]} Distribution by {cc[0]}",
                "insight": f"Compare {nc[0]} across different {cc[0]} segments",
                "priority": "high"
            })
        
        # 5. Time series
        if dc and nc:
            visualizations.append({
                "chart_type": "line",
                "x_axis": dc[0],
                "y_axis": nc[0],
                "color": cc[0] if cc else None,
                "title": f"{nc[0]} Trend Over Time",
                "insight": "Identify seasonality, trends, and anomalies",
                "priority": "high"
            })
        
        # 6. Correlation heatmap
        if len(nc) >= 3:
            visualizations.append({
                "chart_type": "heatmap",
                "x_axis": "all_numeric",
                "y_axis": "all_numeric",
                "color": None,
                "title": "Correlation Matrix",
                "insight": "Discover relationships between all metrics",
                "priority": "medium"
            })
        
        # 7. Category distribution
        if cc:
            visualizations.append({
                "chart_type": "pie",
                "x_axis": cc[0],
                "y_axis": None,
                "color": None,
                "title": f"{cc[0]} Distribution",
                "insight": "Visualize proportions of categories",
                "priority": "low"
            })
        
        return {
            "insights": insights[:7],
            "visualizations": visualizations[:10]
        }


def call_groq_api(prompt, model="llama-3.3-70b-versatile"):
    """Call Groq API with error handling"""
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=4000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Clean markdown
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        return json.loads(response_text)
    
    except Exception as e:
        st.warning(f"API call error: {str(e)}")
        return None


def multi_stage_analysis(analyzer, sample_size=5):
    """Perform multi-stage AI analysis"""
    results = {}
    
    # Stage 1: Data Understanding
    with st.status("🔍 Stage 1: Understanding your data...", expanded=True) as status:
        st.write("Analyzing data structure and patterns...")
        prompt1 = analyzer.get_stage1_prompt(sample_size)
        stage1_result = call_groq_api(prompt1)
        
        if stage1_result:
            st.write(f"✅ Identified as: **{stage1_result.get('data_type', 'unknown')}** data")
            st.write(f"📝 {stage1_result.get('business_context', '')}")
            results['stage1'] = stage1_result
        else:
            status.update(label="⚠️ Stage 1 failed, using fallback", state="error")
            return None
    
    # Stage 2: Deep Insights
    with st.status("💡 Stage 2: Generating deep insights...", expanded=True) as status:
        st.write("Calculating business metrics and statistical patterns...")
        prompt2 = analyzer.get_stage2_prompt(stage1_result, sample_size)
        stage2_result = call_groq_api(prompt2)
        
        if stage2_result:
            st.write(f"✅ Generated {len(stage2_result.get('insights', []))} actionable insights")
            results['stage2'] = stage2_result
        else:
            status.update(label="⚠️ Stage 2 failed, using fallback", state="error")
            return None
    
    # Stage 3: Visualizations
    with st.status("📊 Stage 3: Creating visualization plan...", expanded=True) as status:
        st.write("Designing optimal charts and graphs...")
        prompt3 = analyzer.get_stage3_prompt(stage1_result, stage2_result)
        stage3_result = call_groq_api(prompt3)
        
        if stage3_result:
            st.write(f"✅ Planned {len(stage3_result.get('visualizations', []))} visualizations")
            results['stage3'] = stage3_result
            status.update(label="✅ All stages complete!", state="complete")
        else:
            status.update(label="⚠️ Stage 3 failed, using fallback", state="error")
            return None
    
    # Combine results
    combined = {
        "insights": stage2_result.get('insights', []),
        "visualizations": stage3_result.get('visualizations', []),
        "context": stage1_result
    }
    
    return combined


def create_visualization(df, viz_config):
    """Create visualization with enhanced handling"""
    try:
        chart_type = viz_config['chart_type']
        title = viz_config['title']
        x_axis = viz_config.get('x_axis')
        y_axis = viz_config.get('y_axis')
        
        if chart_type == 'histogram':
            fig = px.histogram(df, x=x_axis, color=viz_config.get('color'),
                             title=title, marginal='box', nbins=30)
        
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis,
                           color=viz_config.get('color'), title=title,
                           trendline='ols' if not viz_config.get('color') else None)
        
        elif chart_type == 'line':
            agg = viz_config.get('aggregation', 'mean')
            if pd.api.types.is_datetime64_any_dtype(df[x_axis]):
                if agg == 'sum':
                    agg_df = df.groupby(x_axis)[y_axis].sum().reset_index()
                elif agg == 'count':
                    agg_df = df.groupby(x_axis)[y_axis].count().reset_index()
                else:
                    agg_df = df.groupby(x_axis)[y_axis].mean().reset_index()
            else:
                agg_df = df.groupby(x_axis)[y_axis].mean().reset_index()
            
            fig = px.line(agg_df, x=x_axis, y=y_axis, title=title)
        
        elif chart_type == 'bar':
            agg = viz_config.get('aggregation', 'mean')
            if y_axis and y_axis != 'count':
                if agg == 'sum':
                    agg_df = df.groupby(x_axis)[y_axis].sum().reset_index()
                elif agg == 'count':
                    agg_df = df.groupby(x_axis)[y_axis].count().reset_index()
                else:
                    agg_df = df.groupby(x_axis)[y_axis].mean().reset_index()
                fig = px.bar(agg_df.head(20), x=x_axis, y=y_axis, title=title)
            else:
                vc = df[x_axis].value_counts().reset_index()
                vc.columns = [x_axis, 'count']
                fig = px.bar(vc.head(20), x=x_axis, y='count', title=title)
        
        elif chart_type == 'box':
            fig = px.box(df, x=x_axis, y=y_axis, title=title, color=viz_config.get('color'))
        
        elif chart_type == 'heatmap':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                          color_continuous_scale='RdBu_r', title=title)
        
        elif chart_type == 'pie':
            vc = df[x_axis].value_counts().head(10)
            fig = px.pie(values=vc.values, names=vc.index, title=title, hole=0.3)
        
        elif chart_type == 'area':
            agg_df = df.groupby(x_axis)[y_axis].mean().reset_index()
            fig = px.area(agg_df, x=x_axis, y=y_axis, title=title)
        
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis or x_axis, title=title)
        
        fig.update_layout(height=400, template='plotly_white')
        return fig
    
    except Exception as e:
        st.error(f"Error creating {chart_type} chart: {str(e)}")
        return None


def display_business_metrics(metrics):
    """Display business metrics in cards"""
    if not metrics:
        return
    
    st.subheader("📈 Business Performance Metrics")
    
    cols = st.columns(min(4, len(metrics)))
    for idx, (metric, value) in enumerate(metrics.items()):
        with cols[idx % 4]:
            suffix = '%' if any(x in metric for x in ['Rate', 'CTR', 'Margin']) else 'x' if metric == 'ROAS' else ''
            st.markdown(f"""
            <div class="metric-card">
                <h3>{value:.2f}{suffix}</h3>
                <p>{metric.replace('_', ' ')}</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">🤖 AI Data Insights Generator Pro</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Deep Intelligence Analysis - Multi-Stage AI Processing
    Upload your data for comprehensive analysis including:
    - **Business Metrics**: CTR, Conversion Rate, ROAS, Cart Abandonment
    - **Statistical Analysis**: Correlations, Distributions, Outliers
    - **Time Series**: Trends, Seasonality, Forecasting insights
    - **3-Stage AI**: Data Understanding → Deep Insights → Smart Visualizations
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if GROQ_API_KEY:
            use_ai = st.checkbox("Use Multi-Stage AI Analysis", value=True,
                               help="Performs 3-stage deep analysis with Groq Llama 3.3 70B")
            st.success("✅ Groq API Key detected")
        else:
            use_ai = False
            st.warning("⚠️ No GROQ_API_KEY found")
            st.info("Set GROQ_API_KEY environment variable")
        
        sample_size = st.slider("Sample rows for AI", 3, 10, 5)
        
        st.markdown("---")
        st.markdown("### 🎯 Analysis Stages")
        st.markdown("""
        **Stage 1**: Data Understanding
        - Identify data type
        - Detect relationships
        
        **Stage 2**: Deep Insights
        - Business metrics
        - Statistical patterns
        - Recommendations
        
        **Stage 3**: Visualizations
        - Optimal chart selection
        - Advanced analytics
        """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        
        analyzer = DeepDataAnalyzer(df)
        
        st.success(f"✅ Loaded **{len(df):,}** rows × **{len(df.columns)}** columns")
        
        # Display business metrics if detected
        if analyzer.deep_metrics:
            display_business_metrics(analyzer.deep_metrics)
        
        # Overview
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Numeric", len(analyzer.analysis['numeric_columns']))
        col4.metric("Categorical", len(analyzer.analysis['categorical_columns']))
        
        # Enhanced data preview with unique values
        with st.expander("👀 Preview Data & Column Details", expanded=False):
            tab1, tab2 = st.tabs(["📋 Data Sample", "📊 Column Analysis"])
            
            with tab1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with tab2:
                for col in df.columns[:5]:  # Show first 5 columns
                    col_info = analyzer.analysis['column_info'][col]
                    st.markdown(f"**{col}** ({col_info['dtype']})")
                    st.write(f"Unique values ({col_info['unique_count']}): {', '.join(map(str, col_info['all_unique_values'][:10]))}{'...' if col_info['unique_count'] > 10 else ''}")
                    st.markdown("---")
        
        # Generate Analysis
        if st.button("🚀 Generate Deep AI Insights", type="primary", use_container_width=True):
            
            if use_ai and client:
                # Multi-stage analysis
                analysis_result = multi_stage_analysis(analyzer, sample_size)
                
                if not analysis_result:
                    st.warning("Multi-stage analysis failed. Using enhanced fallback...")
                    analysis_result = analyzer.generate_fallback_analysis()
            else:
                with st.spinner("🤖 Generating insights..."):
                    analysis_result = analyzer.generate_fallback_analysis()
            
            if analysis_result:
                st.session_state['analysis'] = analysis_result
                st.balloons()
                st.rerun()
        
        # Display Results
        if 'analysis' in st.session_state:
            result = st.session_state['analysis']
            
            # Show context if available
            if 'context' in result:
                with st.expander("🎯 Data Context", expanded=True):
                    ctx = result['context']
                    st.info(f"**Data Type**: {ctx.get('data_type', 'Unknown')}")
                    st.write(f"**Context**: {ctx.get('business_context', '')}")
            
            # Insights
            st.header("💡 Deep Insights & Recommendations")
            for insight in result.get('insights', []):
                priority = insight.get('priority', insight.get('impact', 'low'))
                priority_class = f"priority-{priority}"
                emoji = "🔴" if priority == 'high' else "🟡" if priority == 'medium' else "🟢"
                
                st.markdown(f"""
                <div class="ai-insight {priority_class}">
                    <h4>{emoji} {insight['title']}</h4>
                    <p><strong>Type:</strong> {insight['type'].title()}</p>
                    <p>{insight['description']}</p>
                    {f"<p><strong>Action:</strong> {insight.get('action', '')}</p>" if insight.get('action') else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            st.header("📊 Smart Visualizations")
            
            for i, viz in enumerate(result.get('visualizations', [])):
                with st.container():
                    st.subheader(f"📈 {viz['title']}")
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        fig = create_visualization(df, viz)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        priority_badge = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}
                        st.markdown(f"**Priority:** {priority_badge.get(viz.get('priority', 'low'), 'N/A')}")
                        st.info(f"💡 {viz['insight']}")
                        st.markdown(f"**Type:** {viz['chart_type']}")
                        if viz.get('advanced_analysis'):
                            st.markdown(f"**Analysis:** {viz['advanced_analysis']}")
                    
                    st.markdown("---")
            
            # Export
            st.header("📥 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button("📄 Download Data", csv, "data.csv", "text/csv", use_container_width=True)
            
            with col2:
                insights_json = json.dumps(result, indent=2)
                st.download_button("📋 Download Insights", insights_json,
                                 "insights.json", "application/json", use_container_width=True)
            
            with col3:
                if analyzer.deep_metrics:
                    metrics_json = json.dumps(analyzer.deep_metrics, indent=2)
                    st.download_button("📊 Download Metrics", metrics_json,
                                     "metrics.json", "application/json", use_container_width=True)


if __name__ == "__main__":
    main()