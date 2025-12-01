"""
LangGraph Workflow - FINAL PRODUCTION VERSION
Complete with proper context preservation and custom value search
"""
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from openai import OpenAI
from decimal import Decimal
import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_db_connection():
    """Get database connection"""
    from pgadmin.RAG_LLM.django_loader import get_db_credentials
    db_creds = get_db_credentials()
    conn = psycopg2.connect(**db_creds)
    conn.autocommit = False
    return conn


def load_catalog_values(catalog):
    """Load unique values"""
    logger.info("📊 Loading catalog values...")
    conn = get_db_connection()
    
    skip_columns = {'TABLE_INFO', 'tableoid', 'xmin', 'cmin', 'xmax', 'cmax', 'ctid'}
    
    try:
        for table, columns in catalog.items():
            for col_name, col_data in columns.items():
                if col_name in skip_columns:
                    continue
                
                try:
                    cursor = conn.cursor()
                    query = f'SELECT DISTINCT "{col_name}" FROM {table} WHERE "{col_name}" IS NOT NULL LIMIT 100'
                    cursor.execute(query)
                    
                    values = [row[0] for row in cursor.fetchall() if row[0]]
                    catalog[table][col_name]['values'] = values
                    
                    cursor.close()
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
        
        conn.close()
    except Exception as e:
        conn.close()
    
    return catalog


def find_fuzzy_matches(search_term, values):
    """Find matches"""
    if not search_term or not values:
        return []
    
    search_lower = search_term.lower().strip()
    exact_matches = []
    partial_matches = []
    
    for value in values:
        if value is None:
            continue
        
        value_str = str(value).lower().strip()
        
        if search_lower == value_str:
            exact_matches.append(value)
        elif search_lower in value_str:
            partial_matches.append(value)
    
    return exact_matches + partial_matches


def entity_resolver(user_query, catalog, session_data, feedback=None):
    """Resolve entities"""
    
    # CUSTOM VALUE - user typed a custom value
    if feedback and feedback.get('type') == 'custom_value':
        custom_value = feedback.get('custom_value')
        entity_type = feedback.get('entity_type')
        context = feedback.get('clarification_context', {})
        
        print(f"\n📝 CUSTOM VALUE ENTERED:")
        print(f"   Entity Type: {entity_type}")
        print(f"   Custom Value: {custom_value}")
        print(f"   Table: {context.get('table')}")
        print(f"   Column: {context.get('column')}\n")
        
        # Search for this custom value in the column
        table = context.get('table')
        column = context.get('column')
        
        if table and column and table in catalog and column in catalog[table]:
            values = catalog[table][column].get('values', [])
            matches = find_fuzzy_matches(custom_value, values)
            
            if len(matches) > 1:
                print(f"   Found {len(matches)} matches for custom search")
                return {
                    "needs_clarification": True,
                    "message": f"I found {len(matches)} matches for '{custom_value}'. Please select one:",
                    "options": [str(m) for m in matches[:20]],
                    "subtype": "value_selection",
                    "entity": custom_value,
                    "entity_type": entity_type,
                    "clarification_context": context
                }
            elif len(matches) == 1:
                print(f"   Found exact match: {matches[0]}")
                return {
                    "needs_clarification": False,
                    "entities": {
                        entity_type: {
                            "table": table,
                            "column": column,
                            "value": matches[0]
                        }
                    },
                    "intent": session_data.get('intent', 'total')
                }
            else:
                print(f"   No matches found for '{custom_value}'")
                return {
                    "needs_clarification": True,
                    "message": f"No matches found for '{custom_value}'. Please try again or select from the list.",
                    "options": [str(v) for v in values[:20]],
                    "subtype": "value_selection",
                    "entity": custom_value,
                    "entity_type": entity_type,
                    "clarification_context": context
                }
    
    # VALUE selection - user selected a specific value
    if feedback and feedback.get('type') == 'value_selection':
        selected_value = feedback.get('selected_option')
        entity_type = feedback.get('entity_type')
        context = feedback.get('clarification_context', {})
        
        table = context.get('table')
        column = context.get('column')
        
        print(f"\n📌 VALUE SELECTED:")
        print(f"   Entity Type: {entity_type}")
        print(f"   Selected: {selected_value}")
        print(f"   Context received: {context}")
        print(f"   Table: {table}")
        print(f"   Column: {column}\n")
        
        # If table/column missing, it's a frontend bug - but we can recover
        if not table or not column:
            print(f"   ⚠️ Table/Column missing in context! Searching session_data...")
            
            # Try to get from session_data
            if session_data and 'last_clarification' in session_data:
                last_ctx = session_data['last_clarification']
                table = last_ctx.get('table')
                column = last_ctx.get('column')
                print(f"   ✅ Recovered from session: {table}.{column}\n")
        
        if not table or not column:
            return {
                "type": "error",
                "message": "Error: Missing table/column information. Please start over.",
                "session_data": session_data
            }
        
        return {
            "needs_clarification": False,
            "entities": {
                entity_type: {
                    "table": table,
                    "column": column,
                    "value": selected_value
                }
            },
            "intent": session_data.get('intent', 'total')
        }
    
    # COLUMN selection - user selected which column to use
    if feedback and feedback.get('type') == 'column_selection':
        selected_option = feedback.get('selected_option', '')
        entity_type = feedback.get('entity_type')
        entity_value = feedback.get('entity')
        context = feedback.get('clarification_context', {})
        matches_by_column = context.get('matches_by_column', {})
        
        column_key = selected_option.split(' (')[0]
        
        print(f"\n📌 COLUMN SELECTED:")
        print(f"   Column Key: {column_key}")
        print(f"   Entity: {entity_value}")
        print(f"   Entity Type: {entity_type}")
        print(f"   Matches by column: {list(matches_by_column.keys())}\n")
        
        # Fallback if matches_by_column is missing
        if not matches_by_column or column_key not in matches_by_column:
            print(f"   ⚠️ matches_by_column missing, searching catalog...\n")
            
            for table, columns in catalog.items():
                for col_name, col_data in columns.items():
                    col_full = f"{table}.{col_name}"
                    if col_full == column_key:
                        matches = find_fuzzy_matches(entity_value, col_data.get('values', []))
                        
                        if matches:
                            print(f"   ✅ Found {len(matches)} matches in {col_full}")
                            
                            if len(matches) > 1:
                                # Store context in session for recovery
                                session_data['last_clarification'] = {
                                    'table': table,
                                    'column': col_name
                                }
                                
                                return {
                                    "needs_clarification": True,
                                    "message": f"I found {len(matches)} options. Please select one:",
                                    "options": [str(m) for m in matches[:20]],
                                    "subtype": "value_selection",
                                    "entity": entity_value,
                                    "entity_type": entity_type,
                                    "clarification_context": {
                                        "table": table,
                                        "column": col_name
                                    },
                                    "allow_custom": True  # Allow custom search
                                }
                            else:
                                return {
                                    "needs_clarification": False,
                                    "entities": {
                                        entity_type: {
                                            "table": table,
                                            "column": col_name,
                                            "value": matches[0]
                                        }
                                    },
                                    "intent": session_data.get('intent', 'total')
                                }
        
        # Original logic
        if column_key in matches_by_column:
            col_data = matches_by_column[column_key]
            values = col_data['values']
            
            print(f"   Found {len(values)} values")
            
            if len(values) > 1:
                print(f"   → Asking user to select from {len(values)} values\n")
                
                # Store context in session for recovery
                session_data['last_clarification'] = {
                    'table': col_data['table'],
                    'column': col_data['column']
                }
                
                return {
                    "needs_clarification": True,
                    "message": f"I found {len(values)} options. Please select one:",
                    "options": [str(v) for v in values[:20]],
                    "subtype": "value_selection",
                    "entity": entity_value,
                    "entity_type": entity_type,
                    "clarification_context": {
                        "table": col_data['table'],
                        "column": col_data['column']
                    },
                    "allow_custom": True
                }
            else:
                print(f"   → Only 1 value\n")
                return {
                    "needs_clarification": False,
                    "entities": {
                        entity_type: {
                            "table": col_data['table'],
                            "column": col_data['column'],
                            "value": values[0]
                        }
                    },
                    "intent": session_data.get('intent', 'total')
                }
    
    # Initial query
    try:
        catalog_context = {}
        for table, columns in catalog.items():
            catalog_context[table] = {}
            for col, col_data in columns.items():
                catalog_context[table][col] = {
                    'type': col_data.get('type'),
                    'samples': col_data.get('values', [])[:10]
                }
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """Extract entities and intent. Return JSON:
{"entities": [{"type": "product|customer|etc", "value": "value", "search_terms": ["value"]}], "intent": "total|count|list|average"}"""},
                {"role": "user", "content": f"Query: {user_query}\n\nDatabase: {json.dumps(catalog_context, default=str)}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        llm_result = json.loads(response.choices[0].message.content)
        extracted_entities = llm_result.get('entities', [])
        intent = llm_result.get('intent', 'total')
        session_data['intent'] = intent
        
        resolved_entities = {}
        
        for entity in extracted_entities:
            all_matches = []
            
            for table, columns in catalog.items():
                for col_name, col_data in columns.items():
                    for search_term in entity.get('search_terms', [entity.get('value', '')]):
                        matches = find_fuzzy_matches(search_term, col_data.get('values', []))
                        if matches:
                            all_matches.append({'table': table, 'column': col_name, 'matches': matches})
            
            if not all_matches:
                continue
            
            matches_by_column = {}
            for match in all_matches:
                col_key = f"{match['table']}.{match['column']}"
                if col_key not in matches_by_column:
                    matches_by_column[col_key] = {
                        'table': match['table'],
                        'column': match['column'],
                        'values': match['matches'],
                        'count': len(match['matches'])
                    }
            
            if len(matches_by_column) > 1:
                options = []
                for col_key, col_data in matches_by_column.items():
                    sample = ', '.join([str(v)[:40] for v in col_data['values'][:3]])
                    options.append(f"{col_key} ({col_data['count']} matches: {sample}...)")
                
                return {
                    "needs_clarification": True,
                    "message": f"I found '{entity.get('value')}' in multiple columns. Which one?",
                    "options": options,
                    "subtype": "column_selection",
                    "entity": entity.get('value'),
                    "entity_type": entity.get('type'),
                    "clarification_context": {"matches_by_column": matches_by_column}
                }
            
            col_key = list(matches_by_column.keys())[0]
            col_data = matches_by_column[col_key]
            matches = col_data['values']
            
            if len(matches) > 1:
                # Store context in session
                session_data['last_clarification'] = {
                    'table': col_data['table'],
                    'column': col_data['column']
                }
                
                return {
                    "needs_clarification": True,
                    "message": f"I found {len(matches)} options. Please select one:",
                    "options": [str(m) for m in matches[:20]],
                    "subtype": "value_selection",
                    "entity": entity.get('value'),
                    "entity_type": entity.get('type'),
                    "clarification_context": {
                        "table": col_data['table'],
                        "column": col_data['column']
                    },
                    "allow_custom": True
                }
            
            if len(matches) == 1:
                resolved_entities[entity.get('type')] = {
                    "table": col_data['table'],
                    "column": col_data['column'],
                    "value": matches[0]
                }
        
        return {"needs_clarification": False, "entities": resolved_entities, "intent": intent}
        
    except Exception as e:
        logger.error(f"❌ {e}")
        return {"needs_clarification": False, "entities": {}, "intent": "total"}


def generate_sql_with_llm(user_query, entities, config, intent="total"):
    """Generate SQL with LLM"""
    
    print("\n" + "="*80)
    print("🔧 SQL GENERATION")
    print("="*80)
    
    table_columns = config.get('table_columns', {})
    kg_data = config.get('knowledge_graph_data', {})
    relationships = config.get('relationships', [])
    
    schema_info = []
    for table, columns in table_columns.items():
        schema_info.append(f"\nTable: {table}")
        for col in columns:
            col_name = col['name']
            col_type = col['type']
            
            description = ''
            if table in kg_data and col_name in kg_data[table]:
                description = kg_data[table][col_name].get('desc', '')
            
            if description:
                schema_info.append(f"  - {col_name} ({col_type}): {description}")
            else:
                schema_info.append(f"  - {col_name} ({col_type})")
    
    schema_text = "\n".join(schema_info)
    
    relationships_text = ""
    if relationships:
        relationships_text = "\nRelationships:\n"
        for rel in relationships:
            left = f"{rel['left_table']}.{rel['left_column']}"
            right = f"{rel['right_table']}.{rel['right_column']}"
            relationships_text += f"  - {left} = {right}\n"
    
    where_conditions = []
    tables_used = set()
    
    for entity_type, entity_data in entities.items():
        table = entity_data.get('table')
        column = entity_data.get('column')
        value = entity_data.get('value')
        
        if table and column and value:
            tables_used.add(table)
            if isinstance(value, str):
                escaped_value = value.replace("'", "''")
                where_conditions.append(f'{table}."{column}" = \'{escaped_value}\'')
            else:
                where_conditions.append(f'{table}."{column}" = {value}')
    
    where_clause = " AND ".join(where_conditions) if where_conditions else ""
    
    print(f"User Query: {user_query}")
    print(f"Intent: {intent}")
    print(f"Resolved Entities:")
    for entity_type, entity_data in entities.items():
        print(f"  - {entity_type}: {entity_data['table']}.{entity_data['column']} = {entity_data['value']}")
    print(f"\nTables: {', '.join(tables_used)}")
    print(f"WHERE: {where_clause}")
    
    prompt = f"""Generate PostgreSQL query.

User Query: {user_query}
Intent: {intent}

Schema:
{schema_text}

{relationships_text}

WHERE: {where_clause}
Tables: {', '.join(tables_used)}

RULES:
1. Intent "total/sales" → SUM() on numeric amount/sales/revenue columns
2. Intent "count" → COUNT(*)
3. Intent "average" → AVG() on numeric columns
4. Intent "list" → SELECT *
5. ALWAYS include WHERE: {where_clause}
6. Use double quotes for columns: "column_name"
7. NEVER aggregate date/text columns
8. Add LIMIT 100

Generate SQL only."""
    
    print("\n📝 Generating SQL...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "PostgreSQL expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        sql = response.choices[0].message.content.strip()
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        if 'LIMIT' not in sql.upper():
            sql = sql.rstrip(';') + '\nLIMIT 100;'
        
        print("\n✅ GENERATED SQL:")
        print("-" * 80)
        print(sql)
        print("-" * 80)
        print("="*80 + "\n")
        
        return sql
        
    except Exception as e:
        print(f"\n❌ LLM failed: {e}")
        
        main_table = list(tables_used)[0] if tables_used else list(table_columns.keys())[0]
        sql = f'SELECT * FROM {main_table}'
        if where_clause:
            sql += f'\nWHERE {where_clause}'
        sql += '\nLIMIT 100;'
        
        return sql


def execute_sql(sql):
    """Execute SQL"""
    print(f"\n🔧 EXECUTING SQL...")
    print("-" * 80)
    print(sql)
    print("-" * 80)
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sql)
        results = cursor.fetchall()
        result_data = []
        
        for row in results:
            cleaned_row = {}
            for key, value in dict(row).items():
                if isinstance(value, Decimal):
                    cleaned_row[key] = float(value)
                elif isinstance(value, (datetime.date, datetime.datetime)):
                    cleaned_row[key] = value.isoformat()
                else:
                    cleaned_row[key] = value
            result_data.append(cleaned_row)
        
        cursor.close()
        conn.close()
        
        print(f"✅ Success! {len(result_data)} rows\n")
        
        return {"success": True, "data": result_data}
        
    except Exception as e:
        print(f"❌ Failed: {e}\n")
        conn.close()
        return {"success": False, "error": str(e)}


def format_results(data, intent):
    """Format results"""
    if not data:
        return "No results found."
    
    first_row = data[0]
    
    if 'sum' in first_row or 'total' in str(first_row.keys()).lower():
        for key, value in first_row.items():
            if 'sum' in key.lower() or 'total' in key.lower():
                if value is not None:
                    return f"Total: {value:,.2f}" if isinstance(value, float) else f"Total: {value:,}"
    
    if 'count' in first_row:
        return f"Count: {first_row['count']:,}"
    
    if 'avg' in first_row or 'average' in str(first_row.keys()).lower():
        for key, value in first_row.items():
            if 'avg' in key.lower() or 'average' in key.lower():
                if value is not None:
                    return f"Average: {value:,.2f}" if isinstance(value, float) else f"Average: {value}"
    
    return f"Found {len(data)} results"


def invoke_graph(user_query, module_id, session_data=None, feedback=None):
    """Main entry point"""
    try:
        from pgadmin.RAG_LLM.django_loader import load_module_config
        
        config = load_module_config(module_id)
        
        catalog = {}
        for table, columns in config['table_columns'].items():
            catalog[table] = {}
            for col in columns:
                catalog[table][col['name']] = {
                    'type': col['type'],
                    'values': []
                }
        
        catalog = load_catalog_values(catalog)
        
        if session_data is None:
            session_data = {"entities": {}, "history": [], "intent": "total"}
        
        resolution = entity_resolver(user_query, catalog, session_data, feedback)
        
        if resolution.get("needs_clarification"):
            return {
                "type": "clarification",
                "message": resolution.get("message"),
                "options": resolution.get("options"),
                "subtype": resolution.get("subtype"),
                "entity": resolution.get("entity"),
                "entity_type": resolution.get("entity_type"),
                "clarification_context": resolution.get("clarification_context", {}),
                "allow_custom": resolution.get("allow_custom", False),
                "session_data": session_data
            }
        
        entities = resolution.get("entities", {})
        intent = resolution.get("intent", "total")
        session_data["intent"] = intent
        
        if not entities:
            return {
                "type": "error",
                "message": "No data found.",
                "session_data": session_data
            }
        
        sql = generate_sql_with_llm(user_query, entities, config, intent)
        result = execute_sql(sql)
        
        if not result.get("success"):
            return {
                "type": "error",
                "message": f"Query failed: {result.get('error')}",
                "sql": sql,
                "session_data": session_data
            }
        
        data = result.get("data", [])
        
        chart = None
        if data and len(data) <= 20:
            chart = {
                "type": "table",
                "columns": list(data[0].keys()),
                "data": data
            }
        
        return {
            "type": "response",
            "message": format_results(data, intent),
            "sql": sql,
            "data": data,
            "chart": chart,
            "session_data": session_data,
            "metadata": {
                "row_count": len(data),
                "intent": intent
            }
        }
        
    except Exception as e:
        logger.error(f"❌ {e}")
        import traceback
        traceback.print_exc()
        return {
            "type": "error",
            "message": f"Error: {str(e)}",
            "session_data": session_data or {}
        }