from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
import os

load_dotenv()


class GraphState(TypedDict, total=False):
    user_query: str
    catalog: Dict[str, Any]
    table_columns: Dict[str, List[str]]
    annotated_schema: str
    relationships: str
    resolved: Dict[str, Any]
    sql_result: Any
    validated_sql: str
    module_config: Optional[Dict[str, Any]]


def build_where_clause(resolved_entities, table_columns):
    """Build WHERE clause from resolved entities"""
    conditions = []
    
    for entity_type, entity_data in resolved_entities.items():
        if isinstance(entity_data, dict) and 'value' in entity_data:
            table = entity_data['table']
            column = entity_data['column']
            value = entity_data['value']
            
            # Determine table alias
            alias = 't' + table.split('_')[-1][0] if '_' in table else 't'
            
            # Add condition
            conditions.append(f"{alias}.{column} = '{value}'")
    
    return " AND ".join(conditions) if conditions else ""


def sql_agent_node(state: GraphState):
    """Generate SQL using LLM with full module context"""
    
    print("\n🔧 SQL Agent: Building query from resolved entities...")
    
    user_query = state.get("user_query", "")
    resolved = state.get("resolved", {})
    table_columns = state.get("table_columns", {})
    annotated_schema = state.get("annotated_schema", "")
    relationships = state.get("relationships", "")
    module_config = state.get("module_config", {})
    
    resolved_entities = resolved.get("entities", {})
    
    print(f"\n📤 Building SQL with validated entities...")
    print(f"   Session entities: {state.get('session_entities', {})}")
    print(f"   Resolved entities: {resolved_entities}")
    
    # Build WHERE clause
    where_clause = build_where_clause(resolved_entities, table_columns)
    
    if where_clause:
        print(f"   ✅ Exact filter: {where_clause}")
    
    print(f"\n🔍 Pre-built WHERE clause:")
    print(f"   {where_clause if where_clause else 'No filters'}")
    
    # Determine tables needed
    tables_needed = set()
    for entity_data in resolved_entities.values():
        if isinstance(entity_data, dict) and 'table' in entity_data:
            tables_needed.add(entity_data['table'])
    
    if not tables_needed:
        tables_needed = set(module_config.get('tables', []))
    
    main_table = list(tables_needed)[0] if tables_needed else 'tbl_primary'
    
    print(f"\n📊 Query structure:")
    print(f"   Main table: {main_table}")
    print(f"   Tables needed: {tables_needed}")
    
    # Count necessary joins
    joins_needed = len(tables_needed) - 1 if len(tables_needed) > 1 else 0
    print(f"   Joins: {joins_needed}")
    
    # Build comprehensive context for LLM
    context_parts = []
    
    # 1. Schema with descriptions
    if annotated_schema:
        context_parts.append(f"SCHEMA WITH DESCRIPTIONS:\n{annotated_schema}")
    
    # 2. Relationships
    if relationships:
        context_parts.append(f"\n{relationships}")
    
    # 3. Metrics context
    metrics = module_config.get('metrics', {})
    if metrics:
        metrics_text = "AVAILABLE METRICS:\n"
        for metric_name, metric_desc in list(metrics.items())[:10]:
            metrics_text += f"  - {metric_name}: {metric_desc}\n"
        context_parts.append(metrics_text)
    
    # 4. RCA/Business context
    rca_context = module_config.get('rca_context', '')
    if rca_context:
        context_parts.append(f"\n{rca_context}")
    
    # 5. Extra suggestions
    extra_suggestions = module_config.get('extra_suggestions', '')
    if extra_suggestions:
        context_parts.append(f"\nADDITIONAL CONTEXT:\n{extra_suggestions}")
    
    full_context = "\n\n".join(context_parts)
    
    # Create SQL generation tool
    def generate_sql_query(query_intent: str) -> str:
        """Generate SQL query based on user intent and context"""
        return f"SQL generation based on: {query_intent}"
    
    sql_tool = Tool(
        name="GenerateSQL",
        func=generate_sql_query,
        description="Generates SQL query based on user request, resolved entities, and database schema"
    )
    
    # Enhanced prompt with full module context
    sql_prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "agent_scratchpad", "input"],
        template="""You are an expert SQL query generator for PostgreSQL.

{context}

RESOLVED ENTITIES (PRE-VALIDATED):
{resolved_entities_text}

PRE-BUILT WHERE CLAUSE (USE THIS EXACTLY):
{where_clause}

USER QUERY: {input}

IMPORTANT RULES:
1. **ALWAYS use the pre-built WHERE clause exactly as provided** - these entities are already validated
2. Use the schema descriptions to select appropriate columns
3. Use the relationships to construct proper JOINs when multiple tables are needed
4. Consider the metrics definitions for aggregations
5. Apply business context from RCA when relevant
6. Use table aliases for clarity (e.g., tp for tbl_primary)
7. For sales/revenue queries, use SUM aggregations
8. For counts, use COUNT
9. For rankings, use ORDER BY with LIMIT
10. Always include descriptive column aliases

GENERATE ONLY THE SQL QUERY. No explanations.

You have access to these tools:
{tools}

Tool names: {tool_names}

{agent_scratchpad}

SQL Query:"""
    )
    
    # Format resolved entities for prompt
    resolved_entities_text = ""
    for entity_type, entity_data in resolved_entities.items():
        if isinstance(entity_data, dict):
            table = entity_data.get('table', 'unknown')
            column = entity_data.get('column', 'unknown')
            value = entity_data.get('value', 'unknown')
            resolved_entities_text += f"  - {entity_type}: {value} (in {table}.{column})\n"
    
    # Create LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create agent
    agent = create_react_agent(llm, [sql_tool], sql_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[sql_tool],
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True
    )
    
    # Execute agent
    try:
        result = agent_executor.invoke({
            "input": user_query,
            "context": full_context,
            "resolved_entities_text": resolved_entities_text,
            "where_clause": where_clause if where_clause else "No pre-built filters"
        })
        
        sql_query = result.get("output", "").strip()
        
        # Clean SQL
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
        
        print(f"\n✅ Generated SQL:")
        print("=" * 70)
        print(sql_query)
        print("=" * 70)
        
        return {"sql_result": sql_query}
        
    except Exception as e:
        print(f"\n❌ SQL generation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: simple query
        fallback_sql = f"SELECT * FROM {main_table}"
        if where_clause:
            fallback_sql += f"\nWHERE {where_clause}"
        fallback_sql += "\nLIMIT 100;"
        
        return {"sql_result": fallback_sql}