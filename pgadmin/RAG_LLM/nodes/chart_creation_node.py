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
import json
import os
from dotenv import load_dotenv

load_dotenv()

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
        description="Whether to add comparative context"
    )
    context_type: Optional[Literal["top_products", "top_distributors", "top_superstockists", "trend_over_time"]] = Field(
        default=None,
        description="Type of comparative context to add"
    )


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        dbname=os.getenv("PG_DBNAME", "haldiram"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "12345678")
    )


def safe_sql_identifier(value: str) -> str:
    """Safely escape SQL string values"""
    return value.replace("'", "''")


def shorten_label(label: str, max_length: int = 25) -> str:
    """Shorten long labels for better display"""
    if len(label) <= max_length:
        return label
    return label[:max_length-3] + '...'


def enhance_single_product_data(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """
    Enhance single product result with top products comparison
    """
    print("\n🔄 Enhancing single product data with top products...")
    
    resolved = state.get("resolved", {})
    entities = resolved.get("entities", {})
    filters = resolved.get("filters", {})
    
    # Get the product from entities
    product_entity = entities.get("product", {})
    if isinstance(product_entity, dict):
        product_name = product_entity.get("value")
    else:
        product_name = product_entity
    
    if not product_name:
        print("⚠️ No product entity found")
        return df
    
    conn = get_db_connection()
    
    try:
        # Build time filter
        time_filter = ""
        if filters and filters.get("time_range"):
            time_range = filters["time_range"]
            time_filter = f"AND sales_order_date BETWEEN '{time_range[0]}' AND '{time_range[1]}'"
        
        # Get top 10 products
        query = f"""
        SELECT 
            product_name,
            SUM(invoiced_total_quantity) as total_sales,
            COUNT(*) as order_count
        FROM tbl_primary
        WHERE 1=1 {time_filter}
        GROUP BY product_name
        ORDER BY total_sales DESC
        LIMIT 10
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        
        if results:
            # Create dataframe with top products
            enhanced_df = pd.DataFrame(results, columns=['product_name', 'total_sales', 'order_count'])
            
            # Add shortened names for display
            enhanced_df['product_display'] = enhanced_df['product_name'].apply(lambda x: shorten_label(x, 30))
            
            # Check if queried product is in top 10
            if product_name in enhanced_df['product_name'].values:
                enhanced_df['is_queried'] = enhanced_df['product_name'] == product_name
                print(f"✅ {product_name} is in top 10")
            else:
                # Add the queried product to comparison
                query_specific = f"""
                SELECT 
                    product_name,
                    SUM(invoiced_total_quantity) as total_sales,
                    COUNT(*) as order_count
                FROM tbl_primary
                WHERE product_name = '{safe_sql_identifier(product_name)}' {time_filter}
                GROUP BY product_name
                """
                
                cursor = conn.cursor()
                cursor.execute(query_specific)
                specific_result = cursor.fetchone()
                cursor.close()
                
                if specific_result:
                    # Add queried product at the top
                    specific_df = pd.DataFrame([specific_result], 
                                               columns=['product_name', 'total_sales', 'order_count'])
                    specific_df['product_display'] = specific_df['product_name'].apply(lambda x: shorten_label(x, 30))
                    specific_df['is_queried'] = True
                    
                    # Keep only top 5 others
                    enhanced_df = enhanced_df.head(5)
                    enhanced_df['is_queried'] = False
                    
                    # Combine
                    enhanced_df = pd.concat([specific_df, enhanced_df], ignore_index=True)
                    print(f"✅ Added {product_name} to comparison with top 5")
            
            print(f"📊 Enhanced data shape: {enhanced_df.shape}")
            return enhanced_df
        
    except Exception as e:
        print(f"⚠️ Error enhancing data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
    
    return df


def get_smart_chart_recommendation(user_query: str, df: pd.DataFrame, state: dict) -> Optional[ChartRecommendation]:
    """Get smart chart recommendation from LLM"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ChartRecommendation)
        
        resolved = state.get("resolved", {})
        entities = resolved.get("entities", {})
        
        has_product = "product" in entities
        has_distributor = "distributor" in entities
        
        # Detect if we need context
        single_row = len(df) == 1
        
        data_summary = {
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "single_row": single_row,
            "has_product_filter": has_product,
            "has_distributor_filter": has_distributor,
            "sample_data": df.head(2).to_dict('records')
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert.

CRITICAL RULES:
1. If data has ONLY 1 row → set should_add_context=True and context_type="top_products" (or top_distributors)
2. If data has multiple rows with name columns → bar chart, should_add_context=False
3. Use EXACT column names from the data

Single row means we should add comparison context to make the chart meaningful."""),
            ("user", """Query: {query}
Data: {row_count} rows (single={single_row})
Columns: {columns}
Has product filter: {has_product}
Sample: {sample_data}

Recommend chart type and whether to add context.""")
        ])
        
        formatted_prompt = prompt.format_messages(
            query=user_query,
            row_count=data_summary['row_count'],
            single_row=data_summary['single_row'],
            columns=data_summary['columns'],
            has_product=data_summary['has_product_filter'],
            sample_data=str(data_summary['sample_data'])
        )
        
        recommendation = structured_llm.invoke(formatted_prompt)
        
        print(f"\n📊 Chart Recommendation:")
        print(f"   Type: {recommendation.chart_type}")
        print(f"   X: {recommendation.x_column}, Y: {recommendation.y_column}")
        print(f"   Add Context: {recommendation.should_add_context}")
        print(f"   Context Type: {recommendation.context_type}")
        
        return recommendation
        
    except Exception as e:
        print(f"⚠️ Error getting recommendation: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_plotly_chart(df: pd.DataFrame, recommendation: ChartRecommendation, state: dict) -> Optional[Dict[str, Any]]:
    """
    Create optimized Plotly chart with perfect alignment
    """
    try:
        print(f"\n🎨 Creating chart...")
        print(f"   Original data shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Enhance single-row data if needed
        if recommendation.should_add_context and len(df) == 1:
            if recommendation.context_type == "top_products":
                df = enhance_single_product_data(df, state)
                
                # Update column mapping
                if 'product_display' in df.columns and 'total_sales' in df.columns:
                    recommendation.x_column = 'product_display'
                    recommendation.y_column = 'total_sales'
                    print(f"   ✓ Mapped to enhanced columns: X={recommendation.x_column}, Y={recommendation.y_column}")
        
        # Clean data - convert Decimals
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
        
        # Verify columns exist
        if recommendation.x_column not in df_clean.columns:
            print(f"❌ X column '{recommendation.x_column}' not found!")
            print(f"   Available: {list(df_clean.columns)}")
            
            # Try to find suitable column
            possible_x = [col for col in df_clean.columns if 'name' in col.lower() or 'product' in col.lower() or 'display' in col.lower()]
            if possible_x:
                recommendation.x_column = possible_x[0]
                print(f"   ✓ Auto-corrected to: {recommendation.x_column}")
            else:
                return None
        
        if recommendation.y_column not in df_clean.columns:
            print(f"❌ Y column '{recommendation.y_column}' not found!")
            print(f"   Available: {list(df_clean.columns)}")
            
            # Try to find suitable numeric column
            possible_y = [col for col in df_clean.columns if 'sales' in col.lower() or 'quantity' in col.lower() or 'total' in col.lower()]
            if possible_y:
                recommendation.y_column = possible_y[0]
                print(f"   ✓ Auto-corrected to: {recommendation.y_column}")
            else:
                return None
        
        print(f"\n🎨 Final chart config:")
        print(f"   Type: {recommendation.chart_type}")
        print(f"   X: {recommendation.x_column}")
        print(f"   Y: {recommendation.y_column}")
        
        # Sort by Y value for better visualization
        df_clean = df_clean.sort_values(by=recommendation.y_column, ascending=False)
        
        # Create chart
        fig = None
        
        if recommendation.chart_type in ["bar", "comparison_bar"]:
            # Check if we have highlighted data
            if 'is_queried' in df_clean.columns:
                colors = ['#FF6B6B' if q else '#4ECDC4' for q in df_clean['is_queried']]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_clean[recommendation.x_column],
                        y=df_clean[recommendation.y_column],
                        marker=dict(
                            color=colors,
                            line=dict(color='#333', width=1)
                        ),
                        text=df_clean[recommendation.y_column].apply(lambda x: f'{x:,.0f}'),
                        textposition='outside',
                        textfont=dict(size=11, color='#333'),
                        hovertemplate='<b>%{x}</b><br>Sales: %{y:,.0f}<extra></extra>'
                    )
                ])
                
                # Add annotation for queried item
                queried_items = df_clean[df_clean['is_queried']][recommendation.x_column].tolist()
                if queried_items:
                    fig.add_annotation(
                        text=f"🔍 Your Query",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        bgcolor="#FF6B6B",
                        font=dict(color="white", size=12, family="Arial"),
                        borderpad=10,
                        bordercolor="#FF6B6B",
                        borderwidth=2
                    )
            else:
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_clean[recommendation.x_column],
                        y=df_clean[recommendation.y_column],
                        marker=dict(
                            color='#4ECDC4',
                            line=dict(color='#333', width=1)
                        ),
                        text=df_clean[recommendation.y_column].apply(lambda x: f'{x:,.0f}'),
                        textposition='outside',
                        textfont=dict(size=11, color='#333'),
                        hovertemplate='<b>%{x}</b><br>Sales: %{y:,.0f}<extra></extra>'
                    )
                ])
            
            # CRITICAL: Perfect alignment settings for bar charts
            fig.update_layout(
                title={
                    'text': recommendation.title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#333', 'family': 'Arial, sans-serif'}
                },
                xaxis=dict(
                    title=dict(
                        text=recommendation.x_column.replace('_', ' ').replace('display', '').title(),
                        font=dict(size=14, color='#333', family='Arial')
                    ),
                    tickangle=-45,
                    tickfont=dict(size=10, color='#333'),
                    showgrid=False,
                    showline=True,
                    linewidth=2,
                    linecolor='#333',
                    mirror=True
                ),
                yaxis=dict(
                    title=dict(
                        text=recommendation.y_column.replace('_', ' ').title(),
                        font=dict(size=14, color='#333', family='Arial')
                    ),
                    tickfont=dict(size=11, color='#333'),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    showline=True,
                    linewidth=2,
                    linecolor='#333',
                    mirror=True,
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='rgba(128, 128, 128, 0.3)'
                ),
                plot_bgcolor='rgba(250, 250, 250, 0.5)',
                paper_bgcolor='white',
                font=dict(family='Arial, sans-serif', size=12, color='#333'),
                height=550,
                margin=dict(l=100, r=50, t=100, b=150),
                showlegend=False,
                hovermode='closest'
            )
            
        elif recommendation.chart_type in ["line", "trend_line"]:
            fig = go.Figure(data=[
                go.Scatter(
                    x=df_clean[recommendation.x_column],
                    y=df_clean[recommendation.y_column],
                    mode='lines+markers+text',
                    line=dict(color='#4ECDC4', width=3),
                    marker=dict(size=10, color='#FF6B6B', line=dict(color='white', width=2)),
                    text=df_clean[recommendation.y_column].apply(lambda x: f'{x:,.0f}'),
                    textposition='top center',
                    textfont=dict(size=10, color='#333'),
                    hovertemplate='<b>%{x}</b><br>Sales: %{y:,.0f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': recommendation.title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#333', 'family': 'Arial, sans-serif'}
                },
                xaxis=dict(
                    title=dict(
                        text=recommendation.x_column.replace('_', ' ').title(),
                        font=dict(size=14, color='#333', family='Arial')
                    ),
                    tickangle=-45,
                    tickfont=dict(size=11, color='#333'),
                    showgrid=False,
                    showline=True,
                    linewidth=2,
                    linecolor='#333',
                    mirror=True
                ),
                yaxis=dict(
                    title=dict(
                        text=recommendation.y_column.replace('_', ' ').title(),
                        font=dict(size=14, color='#333', family='Arial')
                    ),
                    tickfont=dict(size=11, color='#333'),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    showline=True,
                    linewidth=2,
                    linecolor='#333',
                    mirror=True,
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='rgba(128, 128, 128, 0.3)'
                ),
                plot_bgcolor='rgba(250, 250, 250, 0.5)',
                paper_bgcolor='white',
                font=dict(family='Arial, sans-serif', size=12, color='#333'),
                height=550,
                margin=dict(l=100, r=50, t=100, b=150),
                showlegend=False,
                hovermode='x unified'
            )
        
        elif recommendation.chart_type == "pie":
            fig = go.Figure(data=[
                go.Pie(
                    labels=df_clean[recommendation.x_column].apply(lambda x: shorten_label(str(x), 20)),
                    values=df_clean[recommendation.y_column],
                    hole=0.3,
                    marker=dict(
                        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'],
                        line=dict(color='white', width=2)
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=11, color='white'),
                    hovertemplate='<b>%{label}</b><br>Sales: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': recommendation.title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#333', 'family': 'Arial, sans-serif'}
                },
                font=dict(family='Arial, sans-serif', size=12, color='#333'),
                height=550,
                margin=dict(l=50, r=50, t=100, b=50),
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02
                )
            )
        
        if fig is None:
            print("❌ Chart type not supported")
            return None
        
        # Convert to JSON
        plotly_json = fig.to_json()
        
        # Prepare CSV - use original full names if available
        csv_columns = []
        if 'product_name' in df_clean.columns:
            csv_columns.append('product_name')
        elif recommendation.x_column in df_clean.columns:
            csv_columns.append(recommendation.x_column)
        
        if recommendation.y_column in df_clean.columns:
            csv_columns.append(recommendation.y_column)
        
        csv_data = df_clean[csv_columns].to_csv(index=False)
        
        # Chart metadata
        chart_info = {
            "type": recommendation.chart_type,
            "title": recommendation.title,
            "x_label": recommendation.x_column.replace('_', ' ').replace('display', '').title(),
            "y_label": recommendation.y_column.replace('_', ' ').title(),
            "reasoning": recommendation.reasoning,
            "data_points": len(df_clean)
        }
        
        print(f"✅ Chart created successfully!")
        
        return {
            "plotly_json": plotly_json,
            "csv_data": csv_data,
            "chart_info": chart_info
        }
        
    except Exception as e:
        print(f"❌ Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def chart_creation_node(state):
    """
    Enhanced chart node that creates meaningful visualizations
    """
    print("\n" + "="*70)
    print("📊 CHART CREATION NODE")
    print("="*70)
    
    execution_result = state.get("execution_result", [])
    execution_status = state.get("execution_status", "")
    
    print(f"📋 Execution status: {execution_status}")
    print(f"📋 Execution result type: {type(execution_result)}")
    print(f"📋 Execution result length: {len(execution_result) if execution_result else 0}")
    
    # Check for data
    if execution_status != "success":
        print(f"⏭️  Skipping chart - status is '{execution_status}' not 'success'")
        return {"final_output": state.get("final_output", "")}
    
    if not execution_result:
        print("⏭️  Skipping chart - no execution results")
        return {"final_output": state.get("final_output", "")}
    
    if len(execution_result) == 0:
        print("⏭️  Skipping chart - empty results")
        return {"final_output": state.get("final_output", "")}
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(execution_result)
        print(f"✅ Created DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📋 Sample data:\n{df.head(3)}")
        
        # Get recommendation
        recommendation = get_smart_chart_recommendation(
            state.get("user_query", ""), 
            df, 
            state
        )
        
        if not recommendation:
            print("⚠️ No recommendation generated")
            return {"final_output": state.get("final_output", "")}
        
        # Create chart
        chart_data = create_plotly_chart(df, recommendation, state)
        
        if chart_data:
            print("✅ Chart data prepared for frontend")
            
            return {
                "final_output": state.get("final_output", ""),
                "chart_data": chart_data
            }
        else:
            print("⚠️ Chart creation failed")
        
    except Exception as e:
        print(f"❌ Error in chart creation node: {e}")
        import traceback
        traceback.print_exc()
    
    return {"final_output": state.get("final_output", "")}