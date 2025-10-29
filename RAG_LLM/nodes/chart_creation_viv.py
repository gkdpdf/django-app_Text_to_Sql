from typing import Dict, Any, Optional
import pandas as pd
import plotly.express as px
import os
from decimal import Decimal
import traceback
from dotenv import load_dotenv

load_dotenv()

# --- STRUCTURED OUTPUT SCHEMA ---
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class ChartDecision(BaseModel):
    """Structured output schema for chart generation."""
    chart_type: str = Field(description="Either 'bar' or 'line'")
    x_column: str = Field(description="Column name for x-axis")
    y_column: str = Field(description="Column name for y-axis")
    title: str = Field(description="Short, clear chart title")
    # ✅ FIXED: Use explicit Optional with None default
    time_filter: Optional[dict] = Field(
        default=None,
        description="If query mentions a time period (e.g., 'March', 'last 2 months'), include {'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'}"
    )


def create_chart_from_decision(
    df: pd.DataFrame,
    decision: ChartDecision,
    chart_path: str = "charts/chart.png"
) -> None:
    """Render and save Plotly chart based on structured LLM decision."""
    if decision.x_column not in df.columns or decision.y_column not in df.columns:
        raise ValueError(f"Columns {decision.x_column} or {decision.y_column} missing in data")

    # Apply time filter if requested
    if decision.time_filter:
        start_date = decision.time_filter.get("start_date")
        end_date = decision.time_filter.get("end_date")
        if start_date and end_date:
            # Try to find a date column
            date_col = None
            for col in df.columns:
                if "date" in col.lower() or "month" in col.lower():
                    date_col = col
                    break
            
            if date_col:
                # Convert to datetime if needed
                if df[date_col].dtype == 'object':
                    month_to_num = {
                        'january': '01', 'february': '02', 'march': '03', 'april': '04',
                        'may': '05', 'june': '06', 'july': '07', 'august': '08',
                        'september': '09', 'october': '10', 'november': '11', 'december': '12'
                    }
                    df[date_col] = df[date_col].str.lower().map(month_to_num).apply(
                        lambda x: f"2024-{x}-01" if x else None
                    )
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # ✅ FIXED: Filter BEFORE creating df_clean
                mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
                df = df.loc[mask].copy()

    # ✅ FIXED: Create df_clean AFTER filtering
    df_clean = df.dropna(subset=[decision.x_column, decision.y_column]).copy()
    if df_clean.empty:
        raise ValueError("No valid data after cleaning")

    # Limit bar charts to top 10
    if decision.chart_type == "bar" and len(df_clean) > 10:
        df_clean = df_clean.nlargest(10, decision.y_column)

    # Sort line charts by x-axis
    if decision.chart_type == "line":
        df_clean = df_clean.sort_values(decision.x_column)

    # Create chart
    if decision.chart_type == "line":
        fig = px.line(df_clean, x=decision.x_column, y=decision.y_column, title=decision.title)
    else:  # bar
        fig = px.bar(df_clean, x=decision.x_column, y=decision.y_column, title=decision.title)
        fig.update_xaxes(tickangle=45)

    fig.update_layout(
        margin=dict(b=150, l=60, r=60, t=60),
        font=dict(size=10),
        showlegend=False
    )

    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    fig.write_image(chart_path, width=1000, height=700)
    print(f"DEBUG: Chart saved to {chart_path}")


def chart_creation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node that:
    1. Validates SQL execution result
    2. Uses LLM with structured output to decide chart type/columns + time filter
    3. Renders and saves chart
    """
    print("DEBUG: Chart creation node called!")

    user_query = state.get("user_query", "")
    execution_result = state.get("execution_result", [])
    execution_status = state.get("execution_status", "")

    os.makedirs("charts", exist_ok=True)

    if execution_status != "success" or not execution_result:
        print("DEBUG: Skipping chart - no results or failed execution")
        return {"final_output": state.get("final_output", "")}

    try:
        df = pd.DataFrame(execution_result)
        print(f"DEBUG: DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")

        # Convert numeric-like columns
        numeric_keywords = {
            'quantity', 'amount', 'sales', 'revenue', 'total', 'count',
            'price', 'value', 'number', 'sum', 'avg', 'mean', 'rate', 'billed', 'margin'
        }

        for col in df.columns:
            if df[col].empty:
                continue
            first_val = df[col].iloc[0]

            is_numeric_col = any(kw in col.lower() for kw in numeric_keywords)
            is_numeric_data = isinstance(first_val, (Decimal, int, float)) or (
                isinstance(first_val, str) and
                first_val.replace(',', '').replace('.', '').replace('-', '').isdigit()
            )

            if is_numeric_col or is_numeric_data:
                try:
                    if isinstance(first_val, Decimal):
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"DEBUG: Converted '{col}' to numeric")
                except Exception as e:
                    print(f"DEBUG: Failed to convert '{col}': {e}")

        df = df.dropna(how='all', axis=1)
        if df.empty or df.shape[0] == 0:
            return {"final_output": state.get("final_output", "")}

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

        if not numeric_cols:
            print("DEBUG: No numeric columns — skipping chart")
            return {"final_output": state.get("final_output", "")}

        # --- USE STRUCTURED OUTPUT ---
        all_cols = list(df.columns)
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        # ✅ FIXED: Use function calling to avoid OpenAI JSON Schema error
        structured_llm = llm.with_structured_output(ChartDecision, method="function_calling")

        prompt = f"""
You are a chart decision assistant. Your job is to decide how to visualize the data.

User query: "{user_query}"
Available columns in data: {all_cols}
Numeric columns: {numeric_cols}
Categorical columns: {categorical_cols}
Datetime columns: {datetime_cols}

Rules:
- If the query mentions time (e.g., "last month", "trend", "over time"), use 'line' chart.
- Otherwise, use 'bar' chart.
- Always pick one categorical/datetime column for x, one numeric for y.
- Title should be short and based on user intent.
- If the query mentions a specific time period (e.g., "March", "last 2 months"), include a 'time_filter' field with 'start_date' and 'end_date' (YYYY-MM-DD).
- Assume current year is 2024 unless specified.

Examples:
- Query: "Sales in March 2024" → time_filter: {{"start_date": "2024-03-01", "end_date": "2024-03-31"}}
- Query: "Top distributors in January" → time_filter: {{"start_date": "2024-01-01", "end_date": "2024-01-31"}}

Respond ONLY with the structured format.
        """.strip()

        try:
            decision: ChartDecision = structured_llm.invoke(prompt)
            print(f"DEBUG: LLM decided - {decision}")
        except Exception as e:
            print(f"DEBUG: LLM decision failed, falling back: {e}")
            if datetime_cols:
                decision = ChartDecision(
                    chart_type="line",
                    x_column=datetime_cols[0],
                    y_column=numeric_cols[0],
                    title=f"{numeric_cols[0].replace('_', ' ').title()} Over Time"
                )
            elif categorical_cols:
                decision = ChartDecision(
                    chart_type="bar",
                    x_column=categorical_cols[0],
                    y_column=numeric_cols[0],
                    title=f"{numeric_cols[0].replace('_', ' ').title()} by {categorical_cols[0].replace('_', ' ').title()}"
                )
            else:
                df = df.reset_index()
                decision = ChartDecision(
                    chart_type="bar",
                    x_column="index",
                    y_column=numeric_cols[0],
                    title="Result Distribution"
                )

        # Render chart (includes time filtering)
        create_chart_from_decision(df, decision, chart_path="charts/chart.png")

        current_output = state.get("final_output", "")
        chart_info = f"\n\n📊 Chart created! Saved as `charts/chart.png`\n**{decision.title}**"
        return {"final_output": current_output + chart_info}

    except Exception as e:
        print(f"DEBUG: Chart creation failed: {e}")
        print(traceback.format_exc())
        return {"final_output": state.get("final_output", "")}



if __name__ == "__main__":
    mock_state = {
        "user_query": "Top distributor by sales in March",
        "execution_result": [
            {"distributor": "A", "sales": 100, "month": "2024-01-15"},
            {"distributor": "B", "sales": 200, "month": "2024-02-20"},
            {"distributor": "C", "sales": 300, "month": "2024-03-10"},
            {"distributor": "D", "sales": 250, "month": "2024-03-25"},
        ],
        "execution_status": "success",
        "final_output": "Here are the results."
    }
    result = chart_creation_node(mock_state)
    print("✅ Final output:", result["final_output"])