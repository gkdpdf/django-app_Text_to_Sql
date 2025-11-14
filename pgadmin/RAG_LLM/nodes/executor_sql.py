import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Any, Dict, TypedDict, Optional, List

class GraphState(TypedDict, total=False):
    user_query: str
    catalog: Dict[str, Any]              
    table_columns: Dict[str, List[str]]  
    annotated_schema: str
    relationships: str
    resolved: Dict[str, Any]
    sql_result: Any                      
    validated_sql: str                  
    validation_status: str               
    validation_error: Optional[str]
    execution_result: Any                
    execution_status: str               
    execution_error: Optional[str]
    route_decision: str                
    final_output: str                    
    reasoning_trace: List[str]
    module_config: Optional[Dict[str, Any]]

def sql_executor_node(state: GraphState):
    """Execute SQL - Using connection from module config (DYNAMIC)"""
    validated_sql = state.get("validated_sql", "")
    module_config = state.get("module_config", {})
    
    print(f"🔍 DEBUG - Executing SQL: {validated_sql}")
    
    if not validated_sql:
        return {
            "execution_result": None,
            "execution_status": "failed",
            "execution_error": "No validated SQL query to execute"
        }
    
    try:
        # Get connection from main.py's pool
        from .main import get_db_connection, put_db_connection
        
        module_id = module_config.get("module_id")
        if not module_id:
            raise Exception("No module ID in config")
        
        conn = get_db_connection(module_id)
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            print(f"📋 Executing: {validated_sql}")
            cursor.execute(validated_sql)
            
            if cursor.description:
                results = cursor.fetchall()
                result_data = [dict(row) for row in results]
                print(f"✅ Success! Found {len(result_data)} rows")
                if result_data:
                    print(f"📊 Sample: {result_data[0]}")
            else:
                result_data = {"affected_rows": cursor.rowcount}
                print(f"✅ Success! Affected rows: {cursor.rowcount}")
            
            cursor.close()
            
            return {
                "execution_result": result_data,
                "execution_status": "success",
                "execution_error": None
            }
        finally:
            put_db_connection(conn, module_id)
        
    except Exception as e:
        print(f"❌ Execution error: {str(e)}")
        return {
            "execution_result": None,
            "execution_status": "failed",
            "execution_error": f"Execution error: {str(e)}"
        }