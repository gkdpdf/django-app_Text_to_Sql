from typing import TypedDict, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    user_query: str
    execution_result: Any
    execution_status: str
    chart_data: Optional[Dict[str, Any]]
    final_output: str
    module_config: Optional[Dict[str, Any]]


def chart_creation_node(state: GraphState):
    """
    Create chart data from SQL execution results
    Supports: bar, line, pie, table
    """
    print("\n" + "="*70)
    print("📊 CHART CREATION NODE")
    print("="*70)
    
    execution_status = state.get("execution_status")
    execution_result = state.get("execution_result")
    user_query = state.get("user_query", "").lower()
    
    print(f"📋 Execution status: {execution_status}")
    print(f"📋 Execution result type: {type(execution_result)}")
    
    if execution_result:
        if isinstance(execution_result, list):
            print(f"📋 Execution result length: {len(execution_result)}")
        else:
            print(f"📋 Execution result: {execution_result}")
    
    # Skip if execution failed
    if execution_status != "success" or not execution_result:
        print("⏭️  Skipping chart - status is '{}' not 'success'".format(execution_status))
        return {"chart_data": None}
    
    # Skip if result is not a list or is empty
    if not isinstance(execution_result, list) or len(execution_result) == 0:
        print("⏭️  Skipping chart - result is not a valid list or is empty")
        return {"chart_data": None}
    
    # Get first row to analyze structure
    first_row = execution_result[0]
    columns = list(first_row.keys())
    
    print(f"📊 Columns found: {columns}")
    print(f"📊 Row count: {len(execution_result)}")
    
    # Determine chart type based on query and data structure
    chart_type = determine_chart_type(user_query, columns, execution_result)
    
    print(f"📈 Chart type selected: {chart_type}")
    
    # Generate chart data
    if chart_type == "table":
        chart_data = create_table_chart(execution_result, columns)
    elif chart_type == "bar":
        chart_data = create_bar_chart(execution_result, columns)
    elif chart_type == "line":
        chart_data = create_line_chart(execution_result, columns)
    elif chart_type == "pie":
        chart_data = create_pie_chart(execution_result, columns)
    else:
        chart_data = create_table_chart(execution_result, columns)
    
    print(f"✅ Chart data created: {chart_type}")
    
    return {"chart_data": chart_data}


def determine_chart_type(query, columns, data):
    """Determine appropriate chart type based on query and data"""
    
    # Keywords for different chart types
    trend_keywords = ['trend', 'over time', 'monthly', 'daily', 'yearly', 'timeline']
    ranking_keywords = ['top', 'bottom', 'best', 'worst', 'highest', 'lowest', 'rank']
    distribution_keywords = ['distribution', 'breakdown', 'share', 'percentage', 'proportion']
    
    # Check if data has date/time column
    has_date_column = any('date' in col.lower() or 'time' in col.lower() or 'month' in col.lower() 
                          for col in columns)
    
    # Check if data has numeric aggregations
    numeric_columns = []
    for col in columns:
        if len(data) > 0:
            first_value = data[0].get(col)
            if isinstance(first_value, (int, float)) and 'id' not in col.lower():
                numeric_columns.append(col)
    
    # Decision logic
    if any(keyword in query for keyword in trend_keywords) or has_date_column:
        return "line"
    elif any(keyword in query for keyword in ranking_keywords):
        return "bar"
    elif any(keyword in query for keyword in distribution_keywords) and len(data) <= 10:
        return "pie"
    elif len(numeric_columns) >= 1 and len(data) <= 20:
        return "bar"
    else:
        return "table"


def create_table_chart(data, columns):
    """Create table chart data"""
    return {
        "type": "table",
        "columns": columns,
        "data": data
    }


def create_bar_chart(data, columns):
    """Create bar chart data"""
    # Find label column (usually first non-numeric column)
    label_col = None
    for col in columns:
        first_value = data[0].get(col)
        if not isinstance(first_value, (int, float)):
            label_col = col
            break
    
    if not label_col:
        label_col = columns[0]
    
    # Find numeric columns for values
    value_cols = []
    for col in columns:
        if col != label_col:
            first_value = data[0].get(col)
            if isinstance(first_value, (int, float)):
                value_cols.append(col)
    
    if not value_cols:
        value_cols = [columns[1]] if len(columns) > 1 else [columns[0]]
    
    # Extract labels and values
    labels = [str(row.get(label_col, '')) for row in data[:20]]  # Limit to 20 for readability
    
    datasets = []
    for value_col in value_cols[:3]:  # Max 3 datasets
        values = [float(row.get(value_col, 0)) if row.get(value_col) is not None else 0 
                  for row in data[:20]]
        datasets.append({
            "label": value_col,
            "data": values
        })
    
    return {
        "type": "bar",
        "labels": labels,
        "datasets": datasets
    }


def create_line_chart(data, columns):
    """Create line chart data"""
    # Find date/label column
    label_col = None
    for col in columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
            label_col = col
            break
    
    if not label_col:
        label_col = columns[0]
    
    # Find numeric columns
    value_cols = []
    for col in columns:
        if col != label_col:
            first_value = data[0].get(col)
            if isinstance(first_value, (int, float)):
                value_cols.append(col)
    
    if not value_cols:
        value_cols = [columns[1]] if len(columns) > 1 else [columns[0]]
    
    # Extract labels and values
    labels = [str(row.get(label_col, '')) for row in data[:30]]
    
    datasets = []
    for value_col in value_cols[:3]:
        values = [float(row.get(value_col, 0)) if row.get(value_col) is not None else 0 
                  for row in data[:30]]
        datasets.append({
            "label": value_col,
            "data": values
        })
    
    return {
        "type": "line",
        "labels": labels,
        "datasets": datasets
    }


def create_pie_chart(data, columns):
    """Create pie chart data"""
    # Find label column
    label_col = columns[0]
    
    # Find value column (first numeric column)
    value_col = None
    for col in columns:
        if col != label_col:
            first_value = data[0].get(col)
            if isinstance(first_value, (int, float)):
                value_col = col
                break
    
    if not value_col:
        value_col = columns[1] if len(columns) > 1 else columns[0]
    
    # Extract labels and values (limit to 10 slices)
    labels = [str(row.get(label_col, '')) for row in data[:10]]
    values = [float(row.get(value_col, 0)) if row.get(value_col) is not None else 0 
              for row in data[:10]]
    
    return {
        "type": "pie",
        "labels": labels,
        "datasets": [{
            "label": value_col,
            "data": values
        }]
    }