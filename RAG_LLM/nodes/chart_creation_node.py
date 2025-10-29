from typing import TypedDict, Dict, Any, List, Optional, Literal
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from decimal import Decimal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import psycopg2
from datetime import datetime, timedelta
import re

# Chart recommendation schema
class ChartRecommendation(BaseModel):
    """Structured output for chart type recommendation"""
    chart_type: Literal["bar", "line", "pie", "scatter", "area", "comparison_bar", "trend_line"] = Field(
        description="The most appropriate chart type"
    )
    x_column: str = Field(description="Column for x-axis")
    y_column: str = Field(description="Column for y-axis")
    title: str = Field(description="Chart title")
    reasoning: str = Field(description="Why this chart type")
    should_add_context: bool = Field(
        default=False,
        description="Whether to add comparative context (other products/distributors)"
    )
    context_type: Optional[Literal["top_products", "top_distributors", "top_superstockists", "trend_over_time"]] = Field(
        default=None,
        description="Type of comparative context to add"
    )
    time_granularity: Optional[Literal["day", "week", "month"]] = Field(
        default=None,
        description="Time granularity for trend analysis"
    )


def safe_sql_identifier(value: str) -> str:
    """Safely escape SQL string values"""
    return value.replace("'", "''")


def detect_time_granularity(filters: dict, user_query: str) -> str:
    """
    Detect appropriate time granularity based on time range and query
    """
    query_lower = user_query.lower()
    
    # Check for explicit granularity in query
    if any(word in query_lower for word in ['daily', 'day by day', 'each day']):
        return 'day'
    if any(word in query_lower for word in ['weekly', 'week by week', 'each week']):
        return 'week'
    if any(word in query_lower for word in ['monthly', 'month by month', 'each month']):
        return 'month'
    
    # Auto-detect based on time range
    time_range = filters.get("time_range")
    if time_range and len(time_range) == 2:
        try:
            start = datetime.strptime(time_range[0], '%Y-%m-%d')
            end = datetime.strptime(time_range[1], '%Y-%m-%d')
            days_diff = (end - start).days
            
            if days_diff <= 31:
                return 'day'
            elif days_diff <= 90:
                return 'week'
            else:
                return 'month'
        except:
            pass
    
    return 'month'


def enhance_data_with_context(df: pd.DataFrame, state: dict, recommendation: ChartRecommendation) -> pd.DataFrame:
    """
    Enhance single-row results with comparative context or trend data
    """
    resolved = state.get("resolved", {})
    entities = resolved.get("entities", {})
    filters = resolved.get("filters", {})
    table = resolved.get("table")
    user_query = state.get("user_query", "")
    
    conn = psycopg2.connect(
        host="localhost",
        dbname="haldiram",
        user="postgres",
        password="12345678"
    )
    
    try:
        # CASE 1: Time trend for specific entity
        if recommendation.should_add_context and recommendation.context_type == "trend_over_time":
            time_range = filters.get("time_range")
            
            if not time_range:
                print("⚠️ No time range for trend analysis")
                return df
            
            entity_type = None
            entity_value = None
            entity_column = None
            
            if "product" in entities:
                entity_type = "product"
                entity_data = entities["product"]
                entity_value = entity_data.get("value") if isinstance(entity_data, dict) else entity_data
                entity_column = "product_name"
            elif "distributor" in entities:
                entity_type = "distributor"
                entity_data = entities["distributor"]
                entity_value = entity_data.get("value") if isinstance(entity_data, dict) else entity_data
                entity_column = "distributor_name"
            elif "superstockist" in entities:
                entity_type = "superstockist"
                entity_data = entities["superstockist"]
                entity_value = entity_data.get("value") if isinstance(entity_data, dict) else entity_data
                entity_column = "superstockist_name"
            
            if entity_value and entity_column and table in ["tbl_primary", "tbl_shipment"]:
                granularity = detect_time_granularity(filters, user_query)
                print(f"📊 Creating {granularity}ly trend for {entity_type}: {entity_value}")
                
                date_column = "sales_order_date" if table == "tbl_primary" else "shipment_date"
                
                query = f"""
                SELECT 
                    DATE_TRUNC('{granularity}', {date_column}) as time_period,
                    SUM(invoiced_total_quantity) as total_sales,
                    COUNT(*) as order_count
                FROM {table}
                WHERE {entity_column} = '{safe_sql_identifier(entity_value)}'
                    AND {date_column} BETWEEN '{time_range[0]}' AND '{time_range[1]}'
                GROUP BY DATE_TRUNC('{granularity}', {date_column})
                ORDER BY time_period
                """
                
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                
                if results:
                    enhanced_df = pd.DataFrame(results, columns=['time_period', 'total_sales', 'order_count'])
                    enhanced_df['time_period'] = pd.to_datetime(enhanced_df['time_period'])
                    
                    if granularity == 'day':
                        enhanced_df['period_label'] = enhanced_df['time_period'].dt.strftime('%b %d')
                    elif granularity == 'week':
                        enhanced_df['period_label'] = enhanced_df['time_period'].dt.strftime('Week of %b %d')
                    else:
                        enhanced_df['period_label'] = enhanced_df['time_period'].dt.strftime('%B %Y')
                    
                    print(f"✅ Generated {len(enhanced_df)} time periods")
                    print(f"📊 Sample data:\n{enhanced_df.head()}")
                    return enhanced_df
        
        # CASE 2: Top products comparison
        elif recommendation.should_add_context and recommendation.context_type == "top_products":
            product_entity = entities.get("product", {})
            product_name = product_entity.get("value") if isinstance(product_entity, dict) else None
            
            if product_name and table in ["tbl_primary", "tbl_shipment"]:
                print(f"📊 Adding top products comparison for: {product_name}")
                
                time_filter = ""
                if filters.get("time_range"):
                    date_col = "sales_order_date" if table == "tbl_primary" else "shipment_date"
                    time_filter = f"AND {date_col} BETWEEN '{filters['time_range'][0]}' AND '{filters['time_range'][1]}'"
                
                query = f"""
                SELECT 
                    product_name, 
                    SUM(invoiced_total_quantity) as total_sales,
                    COUNT(*) as order_count
                FROM {table}
                WHERE 1=1 {time_filter}
                GROUP BY product_name
                ORDER BY total_sales DESC
                LIMIT 10
                """
                
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                
                enhanced_df = pd.DataFrame(results, columns=['product_name', 'total_sales', 'order_count'])
                
                if product_name in enhanced_df['product_name'].values:
                    enhanced_df['is_queried'] = enhanced_df['product_name'] == product_name
                    print(f"✅ {product_name} is in top 10")
                else:
                    query_specific = f"""
                    SELECT 
                        product_name, 
                        SUM(invoiced_total_quantity) as total_sales,
                        COUNT(*) as order_count
                    FROM {table}
                    WHERE product_name = '{safe_sql_identifier(product_name)}' {time_filter}
                    GROUP BY product_name
                    """
                    
                    cursor = conn.cursor()
                    cursor.execute(query_specific)
                    specific_result = cursor.fetchone()
                    cursor.close()
                    
                    if specific_result:
                        specific_df = pd.DataFrame([specific_result], 
                                                   columns=['product_name', 'total_sales', 'order_count'])
                        specific_df['is_queried'] = True
                        
                        enhanced_df = enhanced_df.head(5)
                        enhanced_df['is_queried'] = False
                        enhanced_df = pd.concat([specific_df, enhanced_df], ignore_index=True)
                        print(f"✅ Added {product_name} to comparison")
                
                print(f"📊 Sample data:\n{enhanced_df.head()}")
                return enhanced_df
        
        # CASE 3: Top distributors comparison
        elif recommendation.should_add_context and recommendation.context_type == "top_distributors":
            distributor_entity = entities.get("distributor", {})
            distributor_name = distributor_entity.get("value") if isinstance(distributor_entity, dict) else None
            
            if distributor_name and table in ["tbl_primary", "tbl_shipment"]:
                print(f"📊 Adding top distributors comparison for: {distributor_name}")
                
                time_filter = ""
                if filters.get("time_range"):
                    date_col = "sales_order_date" if table == "tbl_primary" else "shipment_date"
                    time_filter = f"AND {date_col} BETWEEN '{filters['time_range'][0]}' AND '{filters['time_range'][1]}'"
                
                query = f"""
                SELECT 
                    distributor_name, 
                    SUM(invoiced_total_quantity) as total_sales,
                    COUNT(*) as order_count
                FROM {table}
                WHERE 1=1 {time_filter}
                GROUP BY distributor_name
                ORDER BY total_sales DESC
                LIMIT 10
                """
                
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                
                enhanced_df = pd.DataFrame(results, columns=['distributor_name', 'total_sales', 'order_count'])
                
                if distributor_name in enhanced_df['distributor_name'].values:
                    enhanced_df['is_queried'] = enhanced_df['distributor_name'] == distributor_name
                    print(f"✅ {distributor_name} is in top 10")
                else:
                    query_specific = f"""
                    SELECT 
                        distributor_name, 
                        SUM(invoiced_total_quantity) as total_sales,
                        COUNT(*) as order_count
                    FROM {table}
                    WHERE distributor_name = '{safe_sql_identifier(distributor_name)}' {time_filter}
                    GROUP BY distributor_name
                    """
                    
                    cursor = conn.cursor()
                    cursor.execute(query_specific)
                    specific_result = cursor.fetchone()
                    cursor.close()
                    
                    if specific_result:
                        specific_df = pd.DataFrame([specific_result], 
                                                   columns=['distributor_name', 'total_sales', 'order_count'])
                        specific_df['is_queried'] = True
                        
                        enhanced_df = enhanced_df.head(5)
                        enhanced_df['is_queried'] = False
                        enhanced_df = pd.concat([specific_df, enhanced_df], ignore_index=True)
                        print(f"✅ Added {distributor_name} to comparison")
                
                print(f"📊 Sample data:\n{enhanced_df.head()}")
                return enhanced_df
        
        # CASE 4: Top superstockists comparison
        elif recommendation.should_add_context and recommendation.context_type == "top_superstockists":
            superstockist_entity = entities.get("superstockist", {})
            superstockist_name = superstockist_entity.get("value") if isinstance(superstockist_entity, dict) else None
            
            if superstockist_name and table in ["tbl_primary", "tbl_shipment"]:
                print(f"📊 Adding top superstockists comparison for: {superstockist_name}")
                
                time_filter = ""
                if filters.get("time_range"):
                    date_col = "sales_order_date" if table == "tbl_primary" else "shipment_date"
                    time_filter = f"AND {date_col} BETWEEN '{filters['time_range'][0]}' AND '{filters['time_range'][1]}'"
                
                query = f"""
                SELECT 
                    superstockist_name, 
                    SUM(invoiced_total_quantity) as total_sales,
                    COUNT(*) as order_count
                FROM {table}
                WHERE 1=1 {time_filter}
                GROUP BY superstockist_name
                ORDER BY total_sales DESC
                LIMIT 10
                """
                
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                
                enhanced_df = pd.DataFrame(results, columns=['superstockist_name', 'total_sales', 'order_count'])
                
                if superstockist_name in enhanced_df['superstockist_name'].values:
                    enhanced_df['is_queried'] = enhanced_df['superstockist_name'] == superstockist_name
                else:
                    query_specific = f"""
                    SELECT 
                        superstockist_name, 
                        SUM(invoiced_total_quantity) as total_sales,
                        COUNT(*) as order_count
                    FROM {table}
                    WHERE superstockist_name = '{safe_sql_identifier(superstockist_name)}' {time_filter}
                    GROUP BY superstockist_name
                    """
                    
                    cursor = conn.cursor()
                    cursor.execute(query_specific)
                    specific_result = cursor.fetchone()
                    cursor.close()
                    
                    if specific_result:
                        specific_df = pd.DataFrame([specific_result], 
                                                   columns=['superstockist_name', 'total_sales', 'order_count'])
                        specific_df['is_queried'] = True
                        
                        enhanced_df = enhanced_df.head(5)
                        enhanced_df['is_queried'] = False
                        enhanced_df = pd.concat([specific_df, enhanced_df], ignore_index=True)
                
                print(f"📊 Sample data:\n{enhanced_df.head()}")
                return enhanced_df
    
    except Exception as e:
        print(f"⚠️ Error enhancing data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
    
    return df


def get_smart_chart_recommendation(user_query: str, df: pd.DataFrame, state: dict) -> Optional[ChartRecommendation]:
    """
    Enhanced LLM recommendation with context awareness
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ChartRecommendation)
        
        resolved = state.get("resolved", {})
        entities = resolved.get("entities", {})
        filters = resolved.get("filters", {})
        
        has_product = "product" in entities
        has_distributor = "distributor" in entities
        has_superstockist = "superstockist" in entities
        has_time_filter = "time_range" in filters
        
        # Detect if we have multiple rows (aggregated data)
        multiple_rows = len(df) > 1
        
        query_lower = user_query.lower()
        asks_for_trend = any(word in query_lower for word in ['trend', 'over time', 'last', 'past', 'months', 'weeks'])
        asks_for_comparison = any(word in query_lower for word in ['compare', 'vs', 'versus', 'top', 'best'])
        
        # Detect what columns we have
        has_distributor_col = 'distributor_name' in df.columns
        has_product_col = 'product_name' in df.columns
        has_quantity_col = any(col in df.columns for col in ['invoiced_total_quantity', 'total_sales', 'quantity'])
        
        data_summary = {
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "multiple_rows": multiple_rows,
            "has_product_filter": has_product,
            "has_distributor_filter": has_distributor,
            "has_superstockist_filter": has_superstockist,
            "has_time_filter": has_time_filter,
            "asks_for_trend": asks_for_trend,
            "asks_for_comparison": asks_for_comparison,
            "has_distributor_column": has_distributor_col,
            "has_product_column": has_product_col,
            "sample_data": df.head(2).to_dict('records')
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. Recommend charts based on the DATA RETURNED, not the query intent.

CRITICAL RULES BASED ON DATA STRUCTURE:

1. **Multiple rows WITH distributor_name column** → should_add_context=False
   - Data already shows multiple distributors
   - Use: chart_type="bar", x_column="distributor_name", y_column="invoiced_total_quantity" (or whatever quantity column exists)

2. **Multiple rows WITH product_name column** → should_add_context=False
   - Data already shows multiple products
   - Use: chart_type="bar", x_column="product_name", y_column="invoiced_total_quantity"

3. **Single row only** → should_add_context=True
   - Add context to show comparison

4. **Time series data** → chart_type="line"

IMPORTANT: 
- Look at the ACTUAL columns in the data
- Use columns that EXIST in the data
- Don't recommend context if data already has multiple rows
- Match x_column and y_column to REAL column names in the data"""),
            ("user", """User Query: {query}

Data Structure:
- Rows: {row_count} (multiple_rows={multiple_rows})
- Has distributor_name column: {has_distributor_col}
- Has product_name column: {has_product_col}
- Product filter in query: {has_product}
- Distributor filter in query: {has_distributor}
- Time filter: {has_time}

Available Columns: {columns}
Sample Data: {sample_data}

Based on the DATA STRUCTURE (not the query), recommend the chart.""")
        ])
        
        formatted_prompt = prompt.format_messages(
            query=user_query,
            row_count=data_summary['row_count'],
            multiple_rows=data_summary['multiple_rows'],
            has_distributor_col=data_summary['has_distributor_column'],
            has_product_col=data_summary['has_product_column'],
            has_product=data_summary['has_product_filter'],
            has_distributor=data_summary['has_distributor_filter'],
            has_time=data_summary['has_time_filter'],
            columns=data_summary['columns'],
            sample_data=str(data_summary['sample_data'])
        )
        
        recommendation = structured_llm.invoke(formatted_prompt)
        
        print(f"\n📊 Chart Recommendation:")
        print(f"   Type: {recommendation.chart_type}")
        print(f"   X: {recommendation.x_column}, Y: {recommendation.y_column}")
        print(f"   Add Context: {recommendation.should_add_context}")
        if recommendation.should_add_context:
            print(f"   Context Type: {recommendation.context_type}")
        print(f"   Reasoning: {recommendation.reasoning}")
        
        return recommendation
        
    except Exception as e:
        print(f"⚠️ Error getting recommendation: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_enhanced_chart(df: pd.DataFrame, recommendation: ChartRecommendation, state: dict, output_path: str = "chart.png"):
    """
    FIXED: Create chart with proper column mapping after data enhancement
    """
    try:
        print(f"\n🎨 Initial chart setup:")
        print(f"   Original columns in df: {list(df.columns)}")
        print(f"   Recommended X: {recommendation.x_column}, Y: {recommendation.y_column}")
        
        # Enhance data if context requested
        original_df = df.copy()
        data_was_enhanced = False
        
        if recommendation.should_add_context:
            print(f"\n🔄 Attempting to enhance data with context: {recommendation.context_type}")
            enhanced_df = enhance_data_with_context(df, state, recommendation)
            
            if enhanced_df is not None and len(enhanced_df) > 0 and not enhanced_df.equals(df):
                df = enhanced_df
                data_was_enhanced = True
                print(f"✅ Data enhanced! New shape: {df.shape}")
                print(f"   New columns: {list(df.columns)}")
                
                # CRITICAL FIX: Map column names based on what we actually got back
                if recommendation.context_type == "trend_over_time":
                    if 'period_label' in df.columns and 'total_sales' in df.columns:
                        recommendation.x_column = 'period_label'
                        recommendation.y_column = 'total_sales'
                        print(f"   ✓ Mapped to trend columns: X={recommendation.x_column}, Y={recommendation.y_column}")
                    
                elif recommendation.context_type == "top_products":
                    if 'product_name' in df.columns and 'total_sales' in df.columns:
                        recommendation.x_column = 'product_name'
                        recommendation.y_column = 'total_sales'
                        print(f"   ✓ Mapped to product columns: X={recommendation.x_column}, Y={recommendation.y_column}")
                
                elif recommendation.context_type == "top_distributors":
                    if 'distributor_name' in df.columns and 'total_sales' in df.columns:
                        recommendation.x_column = 'distributor_name'
                        recommendation.y_column = 'total_sales'
                        print(f"   ✓ Mapped to distributor columns: X={recommendation.x_column}, Y={recommendation.y_column}")
                
                elif recommendation.context_type == "top_superstockists":
                    if 'superstockist_name' in df.columns and 'total_sales' in df.columns:
                        recommendation.x_column = 'superstockist_name'
                        recommendation.y_column = 'total_sales'
                        print(f"   ✓ Mapped to superstockist columns: X={recommendation.x_column}, Y={recommendation.y_column}")
            else:
                print("⚠️ Could not enhance data, using original")
                df = original_df
        
        # Verify columns exist BEFORE proceeding
        if recommendation.x_column not in df.columns:
            print(f"❌ X column '{recommendation.x_column}' not found!")
            print(f"   Available columns: {list(df.columns)}")
            
            # Try to find a suitable column
            possible_x = [col for col in df.columns if 'name' in col.lower() or 'label' in col.lower() or 'period' in col.lower()]
            if possible_x:
                recommendation.x_column = possible_x[0]
                print(f"   ✓ Auto-corrected to: {recommendation.x_column}")
            else:
                print(f"   ❌ Cannot create chart without valid X column")
                return False
        
        if recommendation.y_column not in df.columns:
            print(f"❌ Y column '{recommendation.y_column}' not found!")
            print(f"   Available columns: {list(df.columns)}")
            
            # Try to find a suitable numeric column
            possible_y = [col for col in df.columns if 'sales' in col.lower() or 'quantity' in col.lower() or 'total' in col.lower()]
            if possible_y:
                recommendation.y_column = possible_y[0]
                print(f"   ✓ Auto-corrected to: {recommendation.y_column}")
            else:
                print(f"   ❌ Cannot create chart without valid Y column")
                return False
        
        # Clean data
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == object:
                try:
                    df_clean[col] = df_clean[col].apply(
                        lambda x: float(x) if isinstance(x, Decimal) else x
                    )
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except:
                    pass
        
        print(f"\n🎨 Creating {recommendation.chart_type} chart...")
        print(f"   Final data shape: {df_clean.shape}")
        print(f"   Final X: {recommendation.x_column}, Y: {recommendation.y_column}")
        print(f"   Sample values - X: {df_clean[recommendation.x_column].head(3).tolist()}")
        print(f"   Sample values - Y: {df_clean[recommendation.y_column].head(3).tolist()}")
        
        # Create chart
        fig = None
        
        if recommendation.chart_type in ["bar", "comparison_bar"]:
            if 'is_queried' in df_clean.columns:
                colors = ['#FF6B6B' if q else '#4ECDC4' for q in df_clean['is_queried']]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_clean[recommendation.x_column],
                        y=df_clean[recommendation.y_column],
                        marker_color=colors,
                        text=df_clean[recommendation.y_column].apply(lambda x: f'{x:,.0f}'),
                        textposition='outside',
                        textfont=dict(size=11)
                    )
                ])
                
                queried_items = df_clean[df_clean['is_queried']][recommendation.x_column].tolist()
                if queried_items:
                    fig.add_annotation(
                        text=f"🔍 Queried: {queried_items[0]}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        bgcolor="#FF6B6B",
                        font=dict(color="white", size=12),
                        borderpad=8
                    )
            else:
                fig = px.bar(
                    df_clean, 
                    x=recommendation.x_column, 
                    y=recommendation.y_column,
                    text=recommendation.y_column
                )
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            
            fig.update_xaxes(tickangle=-45)
            fig.update_layout(
                title=recommendation.title, 
                title_font_size=16,
                xaxis_title=recommendation.x_column.replace('_', ' ').title(),
                yaxis_title=recommendation.y_column.replace('_', ' ').title()
            )
            
        elif recommendation.chart_type in ["line", "trend_line"]:
            fig = px.line(
                df_clean, 
                x=recommendation.x_column,
                y=recommendation.y_column,
                title=recommendation.title,
                markers=True
            )
            
            fig.update_traces(
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8, color='#FF6B6B'),
                text=df_clean[recommendation.y_column].apply(lambda x: f'{x:,.0f}'),
                textposition='top center',
                textfont=dict(size=10),
                mode='lines+markers+text'
            )
            
            fig.update_xaxes(
                tickangle=-45,
                title=recommendation.x_column.replace('_', ' ').title()
            )
            fig.update_yaxes(
                title=recommendation.y_column.replace('_', ' ').title()
            )
        
        elif recommendation.chart_type == "pie":
            fig = px.pie(
                df_clean, 
                names=recommendation.x_column,
                values=recommendation.y_column,
                title=recommendation.title
            )
        
        if fig is None:
            print("❌ Chart type not supported")
            return False
        
        fig.update_layout(
            margin=dict(b=150, l=80, r=80, t=100),
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            height=700,
            width=1200
        )
        
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        fig.write_image(output_path, width=1200, height=700)
        print(f"✅ Chart saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        return False


def chart_creation_node(state):
    """
    Enhanced chart node with smart context-aware visualizations
    """
    print("\n" + "="*70)
    print("📊 CHART CREATION NODE")
    print("="*70)
    
    execution_result = state.get("execution_result", [])
    execution_status = state.get("execution_status", "")
    
    if execution_status != "success" or not execution_result:
        print("⏭️  Skipping chart creation - no successful results")
        return {"final_output": state.get("final_output", "")}
    
    try:
        df = pd.DataFrame(execution_result)
        print(f"📋 Data: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📋 Sample data:\n{df.head(3)}")
        
        recommendation = get_smart_chart_recommendation(
            state.get("user_query", ""), 
            df, 
            state
        )
        
        if recommendation:
            success = create_enhanced_chart(df, recommendation, state)
            
            if success:
                current_output = state.get("final_output", "")
                
                chart_info = f"\n\n{'='*70}\n"
                chart_info += f"📊 **VISUALIZATION CREATED**\n"
                chart_info += f"{'='*70}\n"
                chart_info += f"**Chart Type**: {recommendation.chart_type.replace('_', ' ').title()}\n"
                chart_info += f"**Title**: {recommendation.title}\n"
                chart_info += f"**File**: chart.png\n"
                
                if recommendation.should_add_context:
                    context_desc = {
                        "trend_over_time": "Time-based trend analysis",
                        "top_products": "Comparison with top products",
                        "top_distributors": "Comparison with top distributors",
                        "top_superstockists": "Comparison with top superstockists"
                    }
                    chart_info += f"**Context**: {context_desc.get(recommendation.context_type, 'Comparative analysis')}\n"
                
                chart_info += f"**Insight**: {recommendation.reasoning}\n"
                chart_info += f"{'='*70}\n"
                
                return {"final_output": current_output + chart_info}
        else:
            print("⚠️ No recommendation generated")
        
    except Exception as e:
        print(f"❌ Error in chart creation: {e}")
        import traceback
        traceback.print_exc()
    
    return {"final_output": state.get("final_output", "")}