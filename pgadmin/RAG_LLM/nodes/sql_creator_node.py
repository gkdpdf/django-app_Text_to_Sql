from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


def sql_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: SQL Agent that uses EXACT entity matches from clarification.
    Pre-builds WHERE clause to prevent LLM from using fuzzy LIKE queries.
    """
    resolved = state.get("resolved", {})
    annotated_schema = state.get("annotated_schema", "")
    relationships = state.get("relationships", "")
    user_query = state.get("user_query", "")
    
    print("\n🔧 SQL Agent: Building query from resolved entities...")
    
    # CRITICAL: Prioritize session_entities (from clarifications)
    session_entities = state.get("session_entities", {})
    entities = session_entities if session_entities else resolved.get("entities", {})
    filters = resolved.get("filters", {})
    
    print(f"\n📤 Building SQL with validated entities...")
    print(f"   Session entities: {session_entities}")
    print(f"   Resolved entities: {resolved.get('entities', {})}")
    
    # ========== PRE-BUILD WHERE CLAUSE ==========
    where_conditions = []
    entity_info = []
    tables_needed = set()
    
    # Process each entity with EXACT matching
    for entity_type, entity_value in entities.items():
        # Handle both dict format and string format
        if isinstance(entity_value, dict):
            table = entity_value.get("table")
            column = entity_value.get("column")
            value = entity_value.get("value")
        else:
            # Direct string value - infer table/column
            value = entity_value
            column, table = _infer_column_table(entity_type)
        
        if value and column and table:
            # Get table alias
            alias = _get_table_alias(table)
            tables_needed.add(table)
            
            # CRITICAL: Use EXACT match with proper escaping
            escaped_value = value.replace("'", "''")  # Escape single quotes
            where_conditions.append(f"{alias}.{column} = '{escaped_value}'")
            
            entity_info.append(f"{entity_type}: {alias}.{column} = '{value}'")
            print(f"   ✅ Exact filter: {alias}.{column} = '{value}'")
    
    # Add time filters
    if filters and filters.get("time_range"):
        time_range = filters["time_range"]
        start_date, end_date = time_range[0], time_range[1]
        
        # Determine date column based on tables involved
        if "tbl_primary" in tables_needed or not tables_needed:
            date_column = "tp.sales_order_date"
            tables_needed.add("tbl_primary")
        else:
            date_column = "ts.invoice_date"
            tables_needed.add("tbl_shipment")
        
        where_conditions.append(f"{date_column} BETWEEN '{start_date}' AND '{end_date}'")
        entity_info.append(f"date: {start_date} to {end_date}")
        print(f"   📅 Date filter: {start_date} to {end_date}")
    
    # Build final WHERE clause
    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
    print(f"\n🔍 Pre-built WHERE clause:")
    print(f"   {where_clause}")
    
    # ========== DETERMINE MAIN TABLE AND JOINS ==========
    main_table = _determine_main_table(tables_needed, entities)
    joins_needed = _build_joins(tables_needed, main_table)
    
    print(f"\n📊 Query structure:")
    print(f"   Main table: {main_table}")
    print(f"   Tables needed: {tables_needed}")
    print(f"   Joins: {len(joins_needed)}")
    
    # ========== BUILD SQL PROMPT ==========
    system_prompt = f"""You are a PostgreSQL SQL query generator.

DATABASE SCHEMA:
{annotated_schema}

TABLE RELATIONSHIPS:
{relationships}

USER REQUEST: {user_query}

CRITICAL INSTRUCTIONS:
1. The WHERE clause has been PRE-BUILT for you (see below)
2. You MUST use this EXACT WHERE clause - DO NOT modify it
3. DO NOT add LIKE, ILIKE, or any fuzzy matching
4. Focus on building the SELECT clause and appropriate aggregations
5. Use the joins provided below

MAIN TABLE: {main_table}

REQUIRED JOINS:
{chr(10).join(joins_needed) if joins_needed else 'None needed'}

PRE-BUILT WHERE CLAUSE (USE EXACTLY AS IS):
{where_clause}

ENTITY FILTERS APPLIED:
{chr(10).join(f'- {info}' for info in entity_info)}

Your task:
1. Determine what columns to SELECT based on the user request
2. Add appropriate aggregations (SUM, COUNT, AVG, etc.)
3. Use the exact WHERE clause provided above
4. Add GROUP BY if needed for aggregations
5. Add ORDER BY if appropriate

Return ONLY the complete SQL query."""

    # Create agent
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    tools = []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    # Execute agent
    try:
        agent_input = f"""Generate SQL query for: {user_query}

MANDATORY: Use this EXACT WHERE clause (do not modify):
WHERE {where_clause}

Focus on determining the appropriate SELECT columns and aggregations."""
        
        result = agent_executor.invoke({"input": agent_input})
        sql_query = result.get("output", "")
        
        # Clean up SQL
        sql_query = sql_query.strip()
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
        
        # VALIDATION: Ensure WHERE clause wasn't modified
        if where_conditions and not _validate_where_clause(sql_query, where_conditions):
            print("⚠️ WARNING: LLM modified WHERE clause. Rebuilding query...")
            sql_query = _rebuild_sql_with_correct_where(
                sql_query, 
                where_clause, 
                main_table, 
                joins_needed
            )
        
        print(f"\n✅ Generated SQL:")
        print("=" * 70)
        print(sql_query)
        print("=" * 70)
        
        return {
            "sql_result": sql_query,
            "validation_status": "pending"
        }
        
    except Exception as e:
        print(f"\n❌ SQL Generation Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Build SQL manually
        print("\n⚠️ Building SQL manually...")
        
        # Determine aggregation based on query
        if any(keyword in user_query.lower() for keyword in ["sales", "total", "sum", "revenue"]):
            select_clause = "SELECT SUM(tp.invoiced_total_quantity) AS total_sales"
        elif "count" in user_query.lower():
            select_clause = "SELECT COUNT(*) AS count"
        else:
            select_clause = "SELECT *"
        
        # Build FROM clause with joins
        from_clause = f"FROM {main_table} {_get_table_alias(main_table)}"
        if joins_needed:
            from_clause += "\n" + "\n".join(joins_needed)
        
        # Combine all parts
        fallback_sql = f"{select_clause}\n{from_clause}\nWHERE {where_clause}"
        
        if "sum" in select_clause.lower() or "count" in select_clause.lower():
            # No GROUP BY needed for simple aggregations
            pass
        
        fallback_sql += ";"
        
        print(f"\n📝 Manual SQL:")
        print(fallback_sql)
        
        return {
            "sql_result": fallback_sql,
            "validation_status": "pending"
        }


# ========== HELPER FUNCTIONS ==========

def _infer_column_table(entity_type: str) -> tuple:
    """Infer column name and table from entity type"""
    mapping = {
        "product": ("product", "tbl_product_master"),
        "distributor": ("distributor_name", "tbl_primary"),
        "superstockist": ("superstockist_name", "tbl_primary"),
        "sold_to_party": ("sold_to_party_name", "tbl_shipment"),
    }
    return mapping.get(entity_type, (entity_type, "tbl_primary"))


def _get_table_alias(table_name: str) -> str:
    """Get standard table alias"""
    alias_map = {
        "tbl_primary": "tp",
        "tbl_product_master": "pm",
        "tbl_shipment": "ts",
        "tbl_distributor_master": "dm",
        "tbl_superstockist_master": "sm"
    }
    return alias_map.get(table_name, table_name[:2])


def _determine_main_table(tables_needed: set, entities: dict) -> str:
    """Determine which table should be the main table"""
    # Priority order
    if "tbl_primary" in tables_needed:
        return "tbl_primary"
    elif "tbl_shipment" in tables_needed:
        return "tbl_shipment"
    elif "tbl_product_master" in tables_needed:
        return "tbl_product_master"
    else:
        return "tbl_primary"  # Default


def _build_joins(tables_needed: set, main_table: str) -> list:
    """Build necessary JOIN clauses"""
    joins = []
    main_alias = _get_table_alias(main_table)
    
    # Remove main table from join candidates
    tables_to_join = tables_needed - {main_table}
    
    for table in tables_to_join:
        alias = _get_table_alias(table)
        
        if table == "tbl_product_master" and main_table == "tbl_primary":
            joins.append(f"JOIN tbl_product_master {alias} ON {main_alias}.product_id = {alias}.product_erp_id")
        
        elif table == "tbl_primary" and main_table == "tbl_product_master":
            joins.append(f"JOIN tbl_primary {alias} ON {alias}.product_id = pm.product_erp_id")
        
        elif table == "tbl_distributor_master" and main_table == "tbl_primary":
            joins.append(f"JOIN tbl_distributor_master {alias} ON {main_alias}.distributor_id = {alias}.distributor_erp_id")
        
        elif table == "tbl_superstockist_master" and main_table == "tbl_primary":
            joins.append(f"JOIN tbl_superstockist_master {alias} ON {main_alias}.super_stockist_id = {alias}.super_stockist_erp_id")
    
    return joins


def _validate_where_clause(sql: str, expected_conditions: list) -> bool:
    """Check if SQL contains all expected WHERE conditions"""
    sql_lower = sql.lower()
    
    # Check for fuzzy matching operators (these should NOT be present)
    if "like" in sql_lower or "ilike" in sql_lower:
        return False
    
    # Check that all expected conditions are present
    for condition in expected_conditions:
        # Extract the core condition (column = value part)
        if "=" in condition:
            column_part = condition.split("=")[0].strip().lower()
            if column_part not in sql_lower:
                return False
    
    return True


def _rebuild_sql_with_correct_where(sql: str, correct_where: str, main_table: str, joins: list) -> str:
    """Rebuild SQL with correct WHERE clause if LLM modified it"""
    # Extract SELECT part
    select_match = sql.upper().split("FROM")[0]
    
    # Rebuild query
    main_alias = _get_table_alias(main_table)
    from_clause = f"FROM {main_table} {main_alias}"
    
    if joins:
        from_clause += "\n" + "\n".join(joins)
    
    return f"{select_match}\n{from_clause}\nWHERE {correct_where};"


# For testing
if __name__ == "__main__":
    test_state = {
        "user_query": "sales of bhujia",
        "session_entities": {
            "product": "Bhujia MRP 5|17GM*8.568KG"
        },
        "resolved": {
            "intent": "query",
            "metrics": ["sales"],
            "entities": {},
            "filters": {},
        },
        "annotated_schema": "Sample schema...",
        "relationships": "Sample relationships..."
    }
    
    result = sql_agent_node(test_state)
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(result)