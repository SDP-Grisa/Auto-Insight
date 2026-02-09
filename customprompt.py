import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
import json
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Page config
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
    .priority-high { border-left-color: #dc3545; background: #ffe6e6; color: #000000}
    .priority-medium { border-left-color: #ffc107; background: #fff9e6; color: #000000}
    .priority-low { border-left-color: #28a745; background: #e6f9e6; color: #000000}
</style>
""", unsafe_allow_html=True)

class AIDataAnalyzer:
    """AI-powered data analyzer that works with any dataset"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.analysis = {}
        self._analyze_structure()
    
    def _analyze_structure(self):
        self.analysis = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'column_info': {}
        }
        
        for col in self.df.columns:
            col_info = {
                'dtype': str(self.df[col].dtype),
                'null_count': int(self.df[col].isnull().sum()),
                'unique_count': int(self.df[col].nunique()),
                'sample_values': [str(x) for x in self.df[col].dropna().head(3).tolist()]
            }
            
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.analysis['datetime_columns'].append(col)
                except:
                    pass
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.analysis['numeric_columns'].append(col)
                if not self.df[col].isna().all():
                    col_info['min'] = float(self.df[col].min())
                    col_info['max'] = float(self.df[col].max())
                    col_info['mean'] = float(self.df[col].mean())
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                if col not in self.analysis['datetime_columns']:
                    self.analysis['datetime_columns'].append(col)
            elif self.df[col].dtype == 'object':
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.5:
                    self.analysis['categorical_columns'].append(col)
                    if self.df[col].nunique() < 20:
                        col_info['top_categories'] = dict(self.df[col].value_counts().head(5))
            
            self.analysis['column_info'][col] = col_info
    
    def get_default_prompt(self, sample_size=5):
        """Default prompt that ensures valid JSON output"""
        sample_data = self.df.head(sample_size).to_dict(orient='records')
        
        stats_summary = {}
        if self.analysis['numeric_columns']:
            stats_df = self.df[self.analysis['numeric_columns']].describe()
            stats_summary = {col: stats_df[col].to_dict() for col in stats_df.columns}
        
        prompt = f"""You are an expert data analyst. Analyze this dataset and provide actionable insights.

DATASET STRUCTURE:
- Rows: {self.analysis['total_rows']:,}
- Columns: {self.analysis['total_columns']}
- Numeric: {', '.join(self.analysis['numeric_columns']) or 'None'}
- Categorical: {', '.join(self.analysis['categorical_columns']) or 'None'}
- DateTime: {', '.join(self.analysis['datetime_columns']) or 'None'}

SAMPLE DATA (first {sample_size} rows):
{json.dumps(sample_data, indent=2, default=str)}

STATISTICAL SUMMARY:
{json.dumps(stats_summary, indent=2, default=str)}

TASK: Provide exactly 4-6 key insights and 6-8 visualization recommendations.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "insights": [
        {{
            "type": "trend|pattern|correlation|anomaly|distribution",
            "title": "Short punchy title",
            "description": "1-2 sentence insight with specific numbers/findings",
            "priority": "high|medium|low"
        }}
    ],
    "visualizations": [
        {{
            "chart_type": "scatter|line|bar|histogram|box|heatmap|pie",
            "x_axis": "exact_column_name",
            "y_axis": "exact_column_name_or_null",
            "color": "exact_column_name_or_null",
            "title": "Descriptive chart title",
            "insight": "What pattern this will reveal",
            "priority": "high|medium|low"
        }}
    ]
}}

Rules:
1. Use ONLY column names that exist in the data
2. Match chart types to data types (scatter for numeric pairs, bar for categorical, etc)
3. Provide specific insights with numbers when possible
4. Priority high = critical business insights, medium = supporting analysis, low = nice to have
5. NO markdown formatting, just pure JSON"""
        
        return prompt
    
    def generate_fallback_analysis(self):
        """Intelligent fallback when AI fails"""
        insights = []
        visualizations = []
        
        nc = self.analysis['numeric_columns']
        cc = self.analysis['categorical_columns']
        dc = self.analysis['datetime_columns']
        
        if nc:
            insights.append({
                "type": "distribution",
                "title": f"{len(nc)} measurable metrics",
                "description": f"Key columns: {', '.join(nc[:3])}. Range in {nc[0]}: {self.df[nc[0]].min():.2f} to {self.df[nc[0]].max():.2f}.",
                "priority": "high"
            })
        
        if cc:
            top_cat = cc[0]
            n_unique = self.df[top_cat].nunique()
            top_val = self.df[top_cat].value_counts().index[0]
            insights.append({
                "type": "pattern",
                "title": f"{n_unique} categories in {top_cat}",
                "description": f"Top: {top_val} ({self.df[top_cat].value_counts().iloc[0]} records)",
                "priority": "high"
            })
        
        if dc and nc:
            insights.append({
                "type": "trend",
                "title": "Time-series ready",
                "description": f"Range: {self.df[dc[0]].min()} to {self.df[dc[0]].max()}. Analyze {nc[0]} trends.",
                "priority": "high"
            })
        
        if len(nc) >= 2:
            corr = self.df[nc[:2]].corr().iloc[0, 1]
            insights.append({
                "type": "correlation",
                "title": f"{nc[0]} vs {nc[1]} corr: {corr:.2f}",
                "description": f"{'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'} {'positive' if corr > 0 else 'negative'} relationship.",
                "priority": "medium"
            })
        
        # Visualizations (same as before)
        for col in nc[:2]:
            visualizations.append({
                "chart_type": "histogram",
                "x_axis": col,
                "y_axis": None,
                "color": cc[0] if cc else None,
                "title": f"Distribution of {col}",
                "insight": f"Spread and outliers in {col}",
                "priority": "high"
            })
        
        if len(nc) >= 2:
            visualizations.append({
                "chart_type": "scatter",
                "x_axis": nc[0],
                "y_axis": nc[1],
                "color": cc[0] if cc else None,
                "title": f"{nc[0]} vs {nc[1]}",
                "insight": "Correlation and clusters",
                "priority": "high"
            })
        
        if cc and nc:
            visualizations.append({
                "chart_type": "box",
                "x_axis": cc[0],
                "y_axis": nc[0],
                "title": f"{nc[0]} by {cc[0]}",
                "insight": "Compare across segments",
                "priority": "high"
            })
            visualizations.append({
                "chart_type": "bar",
                "x_axis": cc[0],
                "y_axis": nc[0],
                "title": f"Avg {nc[0]} by {cc[0]}",
                "insight": "Top/bottom performers",
                "priority": "medium"
            })
        
        if dc and nc:
            visualizations.append({
                "chart_type": "line",
                "x_axis": dc[0],
                "y_axis": nc[0],
                "title": f"{nc[0]} Trend",
                "insight": "Trends and seasonality",
                "priority": "high"
            })
        
        if len(nc) >= 3:
            visualizations.append({
                "chart_type": "heatmap",
                "x_axis": "all_numeric",
                "y_axis": "all_numeric",
                "title": "Correlation Matrix",
                "insight": "All variable relationships",
                "priority": "medium"
            })
        
        if cc:
            visualizations.append({
                "chart_type": "pie",
                "x_axis": cc[0],
                "title": f"{cc[0]} Distribution",
                "insight": "Category proportions",
                "priority": "low"
            })
        
        return {"insights": insights[:6], "visualizations": visualizations[:8]}

def get_ai_analysis(analyzer, sample_size=5, custom_instructions=""):
    """Call Groq with default prompt + appended custom instructions"""
    try:
        default_prompt = analyzer.get_default_prompt(sample_size)
        
        if custom_instructions.strip():
            full_prompt = f"""{default_prompt}

Additional / override instructions from user — follow these carefully:
──────────────────────────────────────────────────────────────
{custom_instructions.strip()}
──────────────────────────────────────────────────────────────

Remember: respond ONLY with valid JSON. Do NOT add any explanation, markdown, or text outside the JSON object."""
        else:
            full_prompt = default_prompt
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=4000,
            temperature=0.7,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        text = response.choices[0].message.content.strip()
        text = text.replace('```json', '').replace('```', '').strip()
        
        return json.loads(text), full_prompt  # Return both result and prompt for debugging
    
    except Exception as e:
        st.warning(f"API error: {str(e)[:150]}...")
        return None, None

def create_visualization(df, viz_config):
    """Create Plotly chart from config"""
    try:
        chart_type = viz_config['chart_type']
        title = viz_config['title']
        
        if chart_type == 'histogram':
            fig = px.histogram(df, x=viz_config['x_axis'], color=viz_config.get('color'), title=title, marginal='box', nbins=30)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=viz_config['x_axis'], y=viz_config['y_axis'], color=viz_config.get('color'), title=title,
                             trendline='ols' if not viz_config.get('color') else None)
        elif chart_type == 'line':
            agg_df = df.groupby(viz_config['x_axis'])[viz_config['y_axis']].mean().reset_index()
            fig = px.line(agg_df, x=viz_config['x_axis'], y=viz_config['y_axis'], title=title)
        elif chart_type == 'bar':
            if viz_config.get('y_axis') and viz_config['y_axis'] != 'count':
                agg_df = df.groupby(viz_config['x_axis'])[viz_config['y_axis']].mean().reset_index()
                fig = px.bar(agg_df.head(15), x=viz_config['x_axis'], y=viz_config['y_axis'], title=title)
            else:
                vc = df[viz_config['x_axis']].value_counts().reset_index()
                vc.columns = [viz_config['x_axis'], 'count']
                fig = px.bar(vc.head(15), x=viz_config['x_axis'], y='count', title=title)
        elif chart_type == 'box':
            fig = px.box(df, x=viz_config['x_axis'], y=viz_config['y_axis'], title=title)
        elif chart_type == 'heatmap':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r', title=title)
        elif chart_type == 'pie':
            vc = df[viz_config['x_axis']].value_counts().head(10)
            fig = px.pie(values=vc.values, names=vc.index, title=title, hole=0.3)
        else:
            fig = px.scatter(df, x=viz_config['x_axis'], y=viz_config.get('y_axis', viz_config['x_axis']), title=title)
        
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        st.error(f"Chart error: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🤖 AI Data Insights Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Intelligent Data Analysis - Works with ANY Dataset
    Upload your data and let AI discover insights + visualizations automatically.
    **Powered by Groq Llama 3.3 70B**
    """)
    
    # ────────────────────────────────────────────────
    # Sidebar with Custom Prompt Support
    # ────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if GROQ_API_KEY:
            use_ai = st.checkbox("Use Groq AI Analysis", value=True)
            st.success("✅ Groq API Key detected")
        else:
            use_ai = False
            st.warning("⚠️ No GROQ_API_KEY found")
        
        sample_size = st.slider("Sample rows for AI", 3, 10, 5)
        
        st.markdown("---")
        st.subheader("🛠️ Custom AI Instructions")
        
        use_custom = st.checkbox(
            "Add custom instructions (append to default)",
            value=False,
            help="Your text will be added to the default prompt — keeps JSON format safe"
        )
        
        custom_instructions = ""
        if use_custom:
            custom_instructions = st.text_area(
                "Your custom instructions",
                height=180,
                placeholder="Examples:\n"
                            "- Focus on marketing funnel and conversion rates\n"
                            "- Include statistical significance where possible\n"
                            "- Be very concise: max 4 insights\n"
                            "- Prioritize trends over distributions\n"
                            "- Always recommend business actions",
                help="These instructions are appended to the default prompt. The AI will still return valid JSON."
            )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("- Auto-detects columns\n- AI insights + charts\n- Customizable via instructions")

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
        
        analyzer = AIDataAnalyzer(df)
        
        st.success(f"✅ Loaded **{len(df):,}** rows × **{len(df.columns)}** columns")
        
        # with st.expander("🔧 Column Type Fix (optional)", expanded=False):
        #     st.write("Select columns that should be numeric but are shown as text:")
            
        #     object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        #     if object_cols:
        #         selected = st.multiselect(
        #             "Columns to force-convert to number",
        #             object_cols,
        #             help="Choose columns that contain numbers but are detected as text (prices, quantities, etc.)"
        #         )
                
        #         if selected and st.button("Convert selected → numeric", use_container_width=True):
        #             progress = st.progress(0)
        #             for i, col in enumerate(selected):
        #                 cleaned = df[col].replace(r'[\$,₹,€,£,%]|[\s,]', '', regex=True)
        #                 df[col] = pd.to_numeric(cleaned, errors='coerce')
        #                 progress.progress((i + 1) / len(selected))
        #             st.success(f"Converted {len(selected)} columns → charts should improve now")
        #             if st.button("Refresh page", type="primary"):
        #                 st.rerun()
        #     else:
        #         st.info("No object/string columns found that might need conversion.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Numeric", len(analyzer.analysis['numeric_columns']))
        col4.metric("Categorical", len(analyzer.analysis['categorical_columns']))
        
        with st.expander("👀 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Generate Analysis Button
        if st.button("🚀 Generate AI Insights", type="primary", use_container_width=True):
            with st.spinner("🤖 AI analyzing data..."):
                if use_ai and client:
                    st.info("📡 Calling Groq with Llama 3.3 70B...")
                    analysis_result, sent_prompt = get_ai_analysis(analyzer, sample_size, custom_instructions)
                    
                    if not analysis_result:
                        st.warning("AI failed — using smart fallback")
                        analysis_result = analyzer.generate_fallback_analysis()
                        sent_prompt = None
                else:
                    analysis_result = analyzer.generate_fallback_analysis()
                    sent_prompt = None
                
                if analysis_result:
                    st.session_state['analysis'] = analysis_result
                    st.session_state['sent_prompt'] = sent_prompt
                    st.success("✅ Analysis complete!")
                    st.rerun()
        
        # Display Results
        if 'analysis' in st.session_state:
            result = st.session_state['analysis']
            
            # Show sent prompt for debugging (collapsed)
            if 'sent_prompt' in st.session_state and st.session_state['sent_prompt']:
                with st.expander("📤 Prompt sent to AI (for debugging)", expanded=False):
                    st.code(st.session_state['sent_prompt'][:3000] + "…", language="text")
            
            st.header("💡 Key Insights")
            for insight in result.get('insights', []):
                priority_class = f"priority-{insight.get('priority', 'low')}"
                emoji = "🔴" if insight['priority'] == 'high' else "🟡" if insight['priority'] == 'medium' else "🟢"
                
                st.markdown(f"""
                <div class="ai-insight {priority_class}">
                    <h4>{emoji} {insight['title']}</h4>
                    <p><strong>Type:</strong> {insight['type'].title()}</p>
                    <p>{insight['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.header("📊 Recommended Visualizations")
            for viz in result.get('visualizations', []):
                with st.container():
                    st.subheader(f"📈 {viz['title']}")
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        fig = create_visualization(df, viz)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        priority_badge = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}
                        st.markdown(f"**Priority:** {priority_badge.get(viz['priority'], 'N/A')}")
                        st.info(f"💡 {viz['insight']}")
                        st.markdown(f"**Chart Type:** {viz['chart_type']}")
                    
                    st.markdown("---")
            
            # Export
            st.header("📥 Export")
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button("📄 Download Data (CSV)", csv, "data.csv", "text/csv", use_container_width=True)
            with col2:
                insights_json = json.dumps(result, indent=2)
                st.download_button("📋 Download Insights (JSON)", insights_json, "insights.json", "application/json", use_container_width=True)

if __name__ == "__main__":
    main()