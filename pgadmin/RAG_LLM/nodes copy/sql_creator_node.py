from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
import os
from dotenv import load_dotenv

load_dotenv()


def sql_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: SQL Agent that properly uses validated entity names from resolver
    """
    resolved = state.get("resolved", {})
    annotated_schema = state.get("annotated_schema", "")
    relationships = state.get("relationships", "")
    user_query = state.get("user_query", "")
    
    print("\n🔧 SQL Agent: Building query from resolved entities...")
    
    # Extract validated entity values
    entities = resolved.get("entities", {})
    filters = resolved.get("filters", {})
    table = resolved.get("table", "")
    
    # Build entity constraints for SQL
    entity_constraints = []
    
    print(f"\n📤 Sending to SQL Agent...")
    print(f"   Main table: {table}")
    
    if entities:
        print(f"   Validated entities:")
        for entity_type, entity_data in entities.items():
            if isinstance(entity_data, dict):
                # Get the validated value
                value = entity_data.get("value")
                column = entity_data.get("column")
                entity_table = entity_data.get("table")
                
                if value and column:
                    print(f"      {entity_type}: {column} = '{value}' (from {entity_table})")
                    entity_constraints.append({
                        "type": entity_type,
                        "column": column,
                        "value": value,
                        "table": entity_table
                    })
            elif isinstance(entity_data, str):
                print(f"      {entity_type}: '{entity_data}' (raw value)")
                entity_constraints.append({
                    "type": entity_type,
                    "value": entity_data
                })
    
    # Build time filter info
    time_filter_info = ""
    if filters and filters.get("time_range"):
        time_range = filters["time_range"]
        print(f"   Time filter: {time_range[0]} to {time_range[1]}")
        time_filter_info = f"\n- Filter by date range: {time_range[0]} to {time_range[1]}"
    else:
        print(f"   Filters: no time filter")
    
    # Count joins needed
    unique_tables = set([table])
    for constraint in entity_constraints:
        if constraint.get("table"):
            unique_tables.add(constraint["table"])
    
    joins_needed = len(unique_tables) - 1
    print(f"   Joins needed: {joins_needed}")
    
    # Build entity filter instructions
    entity_filter_instructions = ""
    if entity_constraints:
        entity_filter_instructions = "\n\n**CRITICAL ENTITY FILTERS - USE EXACT VALUES:**\n"
        for constraint in entity_constraints:
            entity_type = constraint["type"]
            value = constraint["value"]
            column = constraint.get("column", entity_type)
            constraint_table = constraint.get("table", table)
            
            entity_filter_instructions += f"\n- Filter {constraint_table}.{column} = '{value}'"
            entity_filter_instructions += f"  (EXACT match required - this is the validated {entity_type} name)"
    
    # Create the SQL generation prompt
    system_prompt = f"""You are an expert SQL query generator for PostgreSQL databases.

**DATABASE SCHEMA:**
{annotated_schema}

**TABLE RELATIONSHIPS:**
{relationships}

**CRITICAL INSTRUCTIONS:**
1. Use ONLY the EXACT entity values provided below - these are pre-validated and must match exactly
2. DO NOT modify, shorten, or interpret entity names - use them EXACTLY as given
3. When an entity value is provided, you MUST use that exact string in your WHERE clause
4. Table and column names must match the schema exactly
5. Use proper JOINs when querying across tables
6. For date filters, use the date range provided
7. Return clean, executable PostgreSQL SQL only

**USER QUERY:**
{user_query}

**MAIN TABLE TO QUERY:**
{table}
{entity_filter_instructions}
{time_filter_info}

**IMPORTANT:** The entity values above are EXACT and VALIDATED. Use them exactly as written in your WHERE clauses.

Generate a PostgreSQL query that:
1. Uses the main table: {table}
2. Applies the exact entity filters listed above
3. Joins to other tables ONLY if needed for the filters
4. Returns the requested metrics (sales, quantity, etc.)
5. Groups and aggregates appropriately

Return ONLY the SQL query, no explanations."""

    # Create tools (empty for now, but required for agent)
    tools = []
    
    # Create agent
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
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
        # Build the input emphasizing the validated values
        agent_input = f"Generate SQL for: {user_query}\n\n"
        
        if entity_constraints:
            agent_input += "CRITICAL - Use these EXACT validated entity values:\n"
            for constraint in entity_constraints:
                agent_input += f"- {constraint.get('column', constraint['type'])} = '{constraint['value']}'\n"
        
        if filters and filters.get("time_range"):
            time_range = filters["time_range"]
            # Determine date column based on table
            date_column = "sales_order_date" if table == "tbl_primary" else "invoice_date"
            agent_input += f"\n- Date filter: {date_column} BETWEEN '{time_range[0]}' AND '{time_range[1]}'\n"
        
        result = agent_executor.invoke({"input": agent_input})
        
        sql_query = result.get("output", "")
        
        # Clean up the SQL
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
        
        print(f"\n✅ Generated SQL:")
        print("=" * 70)
        print(sql_query)
        print("=" * 70)
        
        # CRITICAL FIX: Return sql_result, not validated_sql
        return {
            "sql_result": sql_query,  # ← Changed from validated_sql
            "validation_status": "pending"
        }
        
    except Exception as e:
        print(f"\n❌ SQL Generation Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Build simple SQL manually
        print("\n⚠️ Falling back to manual SQL generation...")
        
        fallback_sql = f"SELECT * FROM {table}"
        
        where_clauses = []
        
        # Add entity filters
        for constraint in entity_constraints:
            column = constraint.get("column")
            value = constraint.get("value")
            if column and value:
                where_clauses.append(f"{column} = '{value}'")
        
        # Add time filter
        if filters and filters.get("time_range"):
            time_range = filters["time_range"]
            date_column = "sales_order_date" if table == "tbl_primary" else "invoice_date"
            where_clauses.append(f"{date_column} BETWEEN '{time_range[0]}' AND '{time_range[1]}'")
        
        if where_clauses:
            fallback_sql += "\nWHERE " + " AND ".join(where_clauses)
        
        fallback_sql += "\nLIMIT 100;"
        
        print(f"\n📝 Fallback SQL:")
        print(fallback_sql)
        
        return {
            "validated_sql": fallback_sql,
            "validation_status": "pending",
            "validation_error": str(e)
        }


# For testing
if __name__ == "__main__":
    # Test with sample state
    test_state = {
        "user_query": "sales of bhujia",
        "resolved": {
            "intent": "query",
            "metrics": ["sales"],
            "entities": {
                "product": {
                    "table": "tbl_product_master",
                    "column": "product",
                    "value": "Aloo Bhujia MRP 10|40 GM*12 KG",
                    "validated": True,
                    "confidence": 1.0
                }
            },
            "filters": {},
            "table": "tbl_primary",
            "columns": ["product_name"]
        },
        "annotated_schema": "Sample schema...",
        "relationships": "Sample relationships..."
    }
    
    result = sql_agent_node(test_state)
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(result)