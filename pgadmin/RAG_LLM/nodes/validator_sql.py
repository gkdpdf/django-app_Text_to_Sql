from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import re

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
    validation_status: str               
    validation_error: Optional[str]
    execution_result: Any                
    execution_status: str               
    execution_error: Optional[str]
    route_decision: str                
    final_output: str                    
    reasoning_trace: List[str]
    module_config: Optional[Dict[str, Any]]


def validator_agent(state: GraphState):
    """SQL validator with proper schema from table_columns"""
    original_sql = state.get("sql_result", "")
    table_columns = state.get("table_columns", {})
    module_config = state.get("module_config", {})
    
    print("\n" + "="*70)
    print("🔍 SQL VALIDATION STARTED")
    print("="*70)
    print(f"Original SQL:\n{original_sql}\n")
    
    # CRITICAL FIX: Build schema from table_columns
    print("📋 Building schema from table_columns...")
    schema_dict = {}
    
    for table_name, columns in table_columns.items():
        schema_dict[table_name] = {
            "columns": columns,
            "description": f"Table: {table_name}"
        }
    
    print(f"✅ Schema built with tables: {list(schema_dict.keys())}")
    
    # Validation loop
    max_attempts = 10
    attempt = 0
    current_sql = original_sql
    last_sql = None
    stuck_counter = 0
    
    while attempt < max_attempts:
        attempt += 1
        print(f"\n{'─'*70}")
        print(f"🔄 Validation Attempt {attempt}/{max_attempts}")
        print(f"{'─'*70}")
        
        # Check if stuck
        if current_sql == last_sql:
            stuck_counter += 1
            if stuck_counter >= 3:
                print("⚠️ Validator stuck. Applying rule-based fix...")
                current_sql = apply_rule_based_fixes(current_sql, schema_dict)
                stuck_counter = 0
        else:
            stuck_counter = 0
        
        last_sql = current_sql
        
        # Structure validation
        structure_errors = validate_structure(current_sql, schema_dict)
        
        if not structure_errors:
            print("✅ Structure validation passed!")
            
            # Syntax validation
            syntax_errors = validate_syntax(current_sql)
            
            if not syntax_errors:
                print("✅ Syntax validation passed!")
                print(f"\n{'='*70}")
                print("✅ VALIDATION COMPLETE")
                print(f"{'='*70}\n")
                
                return {
                    "validated_sql": current_sql,
                    "validation_status": "success",
                    "validation_error": None
                }
            else:
                print(f"❌ Syntax Error: {syntax_errors[0]}")
                current_sql = auto_fix_sql_llm(current_sql, syntax_errors, schema_dict)
        else:
            print(f"❌ Structure Error: {structure_errors[0]}")
            current_sql = auto_fix_sql_llm(current_sql, structure_errors, schema_dict)
        
        if current_sql:
            print(f"💡 Auto-corrected SQL:\n{current_sql}\n")
    
    print(f"\n⚠️ Maximum attempts ({max_attempts}) reached")
    
    return {
        "validated_sql": original_sql,
        "validation_status": "failed",
        "validation_error": "Could not validate SQL"
    }


def validate_structure(sql: str, schema_dict: dict) -> list:
    """Validate SQL structure"""
    errors = []
    
    # Extract table names
    from_pattern = r'\bFROM\s+(\w+)'
    join_pattern = r'\bJOIN\s+(\w+)'
    
    tables_in_sql = set()
    tables_in_sql.update(re.findall(from_pattern, sql, re.IGNORECASE))
    tables_in_sql.update(re.findall(join_pattern, sql, re.IGNORECASE))
    
    available_tables = list(schema_dict.keys())
    
    for table in tables_in_sql:
        if table not in available_tables:
            errors.append(f"Table '{table}' not in schema. Available: {available_tables}")
            return errors
    
    return errors


def validate_syntax(sql: str) -> list:
    """Basic syntax validation"""
    errors = []
    
    if sql.count('(') != sql.count(')'):
        errors.append("Unbalanced parentheses")
        return errors
    
    sql_upper = sql.upper()
    
    if 'SELECT' not in sql_upper:
        errors.append("Missing SELECT")
        return errors
    
    if 'FROM' not in sql_upper:
        errors.append("Missing FROM")
        return errors
    
    return errors


def auto_fix_sql_llm(sql: str, errors: list, schema_dict: dict) -> str:
    """Auto-fix SQL using LLM"""
    try:
        available_tables = list(schema_dict.keys())
        
        # Build schema context
        schema_context = "Available tables:\n"
        for table, info in schema_dict.items():
            cols = info.get("columns", [])
            schema_context += f"- {table}: {', '.join(cols[:10])}\n"
        
        error_msg = errors[0] if errors else "Unknown error"
        
        prompt = f"""Fix this SQL query to work with the schema.

{schema_context}

Current SQL:
{sql}

Error:
{error_msg}

Instructions:
- Use ONLY the tables listed above
- Use ONLY the columns from those tables
- Keep the query logic similar
- Return ONLY the corrected SQL, no explanation

Corrected SQL:"""
        
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        result = llm.invoke(prompt)
        fixed_sql = result.content.strip()
        
        # Clean up
        if "```sql" in fixed_sql:
            fixed_sql = fixed_sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in fixed_sql:
            fixed_sql = fixed_sql.split("```")[1].split("```")[0].strip()
        
        return fixed_sql
        
    except Exception as e:
        print(f"⚠️ LLM fix error: {e}")
        return apply_rule_based_fixes(sql, schema_dict)


def apply_rule_based_fixes(sql: str, schema_dict: dict) -> str:
    """Apply rule-based fixes"""
    available_tables = list(schema_dict.keys())
    
    if not available_tables:
        return sql
    
    primary_table = available_tables[0]
    
    # Replace incorrect table references
    sql_fixed = re.sub(
        r'\bFROM\s+\w+',
        f'FROM {primary_table}',
        sql,
        flags=re.IGNORECASE
    )
    
    sql_fixed = re.sub(
        r'\bJOIN\s+\w+',
        f'JOIN {primary_table}',
        sql_fixed,
        flags=re.IGNORECASE
    )
    
    # Replace common wrong names
    wrong_names = ['products', 'product_master', 'product_master_table', 'products_table', 'orders', 'distributors']
    for wrong in wrong_names:
        sql_fixed = re.sub(
            r'\b' + wrong + r'\b',
            primary_table,
            sql_fixed,
            flags=re.IGNORECASE
        )
    
    return sql_fixed