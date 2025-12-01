"""
Main LangGraph workflow for SQL generation and execution
Simplified version with all nodes in one file
"""
import logging
from queue import Queue
from threading import Lock
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional
import os
from openai import OpenAI
import re
from decimal import Decimal
import datetime

logger = logging.getLogger(__name__)

# Connection pool
_connection_pools = {}
_pool_locks = {}

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_db_connection(module_id):
    """Get database connection from pool"""
    from pgadmin.RAG_LLM.django_loader import get_db_credentials
    
    if module_id not in _connection_pools:
        _connection_pools[module_id] = Queue(maxsize=10)
        _pool_locks[module_id] = Lock()
    
    pool = _connection_pools[module_id]
    
    # Try to get existing connection
    if not pool.empty():
        conn = pool.get()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return conn
        except:
            conn.close()
    
    # Create new connection
    db_creds = get_db_credentials()
    conn = psycopg2.connect(
        host=db_creds['host'],
        database=db_creds['database'],
        user=db_creds['user'],
        password=db_creds['password'],
        port=db_creds['port']
    )
    conn.autocommit = False
    return conn


def put_db_connection(conn, module_id):
    """Return connection to pool"""
    if module_id in _connection_pools:
        pool = _connection_pools[module_id]
        if not pool.full():
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                conn.commit()
                pool.put(conn)
                return
            except:
                pass
    try:
        conn.close()
    except:
        pass


def build_catalog(module_config):
    """Build catalog from module configuration"""
    logger.info("📚 Building catalog from module configuration...")
    
    catalog = {}
    table_columns = module_config.get('table_columns', {})
    
    for table, columns in table_columns.items():
        catalog[table] = {}
        
        for col_info in columns:
            col_name = col_info['name']
            
            # Skip invalid columns
            if ' ' in col_name:
                logger.warning(f"⚠️ Skipping column '{table}.{col_name}' - contains space")
                continue
            
            if col_name.startswith('_'):
                continue
            
            catalog[table][col_name] = {
                'type': col_info.get('type', 'unknown'),
                'values': []
            }
    
    logger.info(f"✅ Catalog built with {len(catalog)} tables")
    return catalog


def load_catalog_values(catalog, module_id):
    """Load unique values for catalog columns"""
    logger.info("📊 Loading catalog values from database...")
    
    conn = get_db_connection(module_id)
    
    try:
        for table, columns in catalog.items():
            for col_name in columns.keys():
                cursor = None
                try:
                    cursor = conn.cursor()
                    query = f'SELECT DISTINCT "{col_name}" FROM {table} WHERE "{col_name}" IS NOT NULL LIMIT 100'
                    cursor.execute(query)
                    
                    values = [row[0] for row in cursor.fetchall() if row[0]]
                    catalog[table][col_name]['values'] = values
                    
                    conn.commit()
                    logger.info(f"   ✅ Loaded {len(values)} values for {table}.{col_name}")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Skipping {table}.{col_name}: {str(e)}")
                    conn.rollback()
                finally:
                    if cursor:
                        try:
                            cursor.close()
                        except:
                            pass
        
    except Exception as e:
        logger.error(f"❌ Error loading catalog values: {e}")
        conn.rollback()
    finally:
        put_db_connection(conn, module_id)
    
    logger.info("✅ Catalog values loaded")
    return catalog


def question_validator(user_query):
    """Validate if question is valid for SQL generation"""
    logger.info(f"🔍 Validating question: {user_query}")
    
    # Simple validation - accept all queries for now
    return {"valid": True}


def entity_resolver(user_query, catalog, session_entities, feedback=None):
    """Resolve entities from user query"""
    logger.info(f"🔍 Resolving entities for: {user_query}")
    
    # If feedback provided, handle clarification response
    if feedback:
        entity_type = feedback.get('entity_type')
        
        if feedback.get('type') == 'value_selection':
            selected_value = feedback.get('selected_option')
            table = feedback.get('clarification_context', {}).get('table')
            column = feedback.get('clarification_context', {}).get('column')
            
            return {
                "needs_clarification": False,
                "entities": {
                    entity_type: {
                        "table": table,
                        "column": column,
                        "value": selected_value
                    }
                }
            }
    
    # Simple entity resolution using LLM
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Extract entities from the query. Return JSON with entity types as keys and entity values."
                },
                {
                    "role": "user",
                    "content": f"Query: {user_query}\n\nExtract entities like product names, dates, locations, etc. Return as JSON."
                }
            ],
            temperature=0
        )
        
        # For now, return simple resolution
        # In production, you'd parse LLM response and check against catalog
        
    except Exception as e:
        logger.error(f"Entity resolution error: {e}")
    
    # Simple resolution - look for product mentions
    entities = {}
    for table, columns in catalog.items():
        for col_name, col_data in columns.items():
            if 'product' in col_name.lower():
                for value in col_data.get('values', []):
                    if value and str(value).lower() in user_query.lower():
                        entities['product'] = {
                            "table": table,
                            "column": col_name,
                            "value": value
                        }
                        break
    
    return {
        "needs_clarification": False,
        "entities": entities
    }


def generate_sql(user_query, entities, table_columns, annotated_schema, relationships):
    """Generate SQL query"""
    logger.info(f"🔧 Generating SQL for: {user_query}")
    
    # Build WHERE clause
    where_conditions = []
    tables_needed = set()
    
    for entity_type, entity_data in entities.items():
        if isinstance(entity_data, dict):
            table = entity_data.get('table')
            column = entity_data.get('column')
            value = entity_data.get('value')
            
            if table and column and value:
                tables_needed.add(table)
                # Generate alias
                alias = ''.join([p[0] for p in table.replace('tbl_', '').split('_')])
                where_conditions.append(f"{alias}.{column} = '{value}'")
    
    # Default to first table if no entities
    if not tables_needed and table_columns:
        tables_needed.add(list(table_columns.keys())[0])
    
    main_table = list(tables_needed)[0] if tables_needed else 'tbl_primary'
    alias = ''.join([p[0] for p in main_table.replace('tbl_', '').split('_')])
    
    # Build SQL
    if 'sales' in user_query.lower() or 'revenue' in user_query.lower():
        sql = f"""
SELECT 
    {alias}.product_name,
    SUM({alias}.invoiced_total_quantity) AS total_sales
FROM {main_table} {alias}
"""
    else:
        sql = f"SELECT * FROM {main_table} {alias}"
    
    # Add WHERE clause
    if where_conditions:
        sql += "\nWHERE " + " AND ".join(where_conditions)
    
    # Add GROUP BY for aggregations
    if 'SUM(' in sql:
        sql += f"\nGROUP BY {alias}.product_name"
    
    sql += "\nLIMIT 100;"
    
    logger.info(f"✅ Generated SQL: {sql}")
    return sql


def validate_sql(sql, table_columns):
    """Validate SQL syntax"""
    logger.info("🔍 Validating SQL...")
    
    # Basic validation
    if not sql or len(sql.strip()) == 0:
        return {"valid": False, "error": "Empty SQL"}
    
    if not sql.strip().upper().startswith('SELECT'):
        return {"valid": False, "error": "Only SELECT queries allowed"}
    
    return {"valid": True, "sql": sql}


def execute_sql(sql, module_id):
    """Execute SQL query"""
    logger.info(f"🔧 Executing SQL: {sql}")
    
    conn = get_db_connection(module_id)
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sql)
        
        if cursor.description:
            results = cursor.fetchall()
            # Convert all values to JSON serializable types
            result_data = []
            for row in results:
                cleaned_row = {}
                for key, value in dict(row).items():
                    cleaned_row[key] = convert_to_json_serializable(value)
                result_data.append(cleaned_row)
            
            logger.info(f"✅ Query returned {len(result_data)} rows")
        else:
            result_data = []
        
        cursor.close()
        conn.commit()
        
        return {"success": True, "data": result_data}
        
    except Exception as e:
        logger.error(f"❌ SQL execution error: {e}")
        conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        put_db_connection(conn, module_id)

from decimal import Decimal
import datetime

def convert_to_json_serializable(obj):
    """Convert database objects to JSON serializable types"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj


def create_chart(data, user_query):
    """Create chart from data"""
    if not data or len(data) == 0:
        return None
    
    columns = list(data[0].keys())
    
    # Determine chart type
    if len(data) <= 10:
        chart_type = "bar"
    else:
        chart_type = "table"
    
    if chart_type == "table":
        return {
            "type": "table",
            "columns": columns,
            "data": data
        }
    
    # For bar chart
    label_col = columns[0]
    value_col = columns[1] if len(columns) > 1 else columns[0]
    
    labels = [str(row[label_col]) for row in data[:20]]
    values = [float(row[value_col]) if row[value_col] is not None else 0 for row in data[:20]]
    
    return {
        "type": "bar",
        "labels": labels,
        "datasets": [{
            "label": value_col,
            "data": values
        }]
    }


def invoke_graph(user_query, module_id, session_data=None, feedback=None):
    """Main entry point for graph execution"""
    logger.info("🚀 NEW REQUEST")
    logger.info(f"Query: {user_query}, Module: {module_id}")
    
    try:
        # Load module configuration
        from pgadmin.RAG_LLM.django_loader import load_module_config
        
        config = load_module_config(module_id)
        logger.info(f"✅ Module loaded: {config['module_name']}")
        
        # Build catalog
        catalog = build_catalog(config)
        catalog = load_catalog_values(catalog, module_id)
        
        # Build table_columns dict
        table_columns = {}
        for table, cols in catalog.items():
            table_columns[table] = list(cols.keys())
        
        # Initialize session data
        if session_data is None:
            session_data = {"entities": {}, "history": []}
        
        # Step 1: Validate question
        validation = question_validator(user_query)
        if not validation.get("valid"):
            return {
                "type": "error",
                "message": "Invalid question",
                "session_data": session_data
            }
        
        # Step 2: Resolve entities
        resolution = entity_resolver(
            user_query,
            catalog,
            session_data.get("entities", {}),
            feedback
        )
        
        if resolution.get("needs_clarification"):
            return {
                "type": "clarification",
                "message": resolution.get("message", "Please clarify"),
                "options": resolution.get("options", []),
                "subtype": resolution.get("subtype"),
                "entity": resolution.get("entity"),
                "entity_type": resolution.get("entity_type"),
                "table": resolution.get("table"),
                "column": resolution.get("column"),
                "session_data": session_data
            }
        
        entities = resolution.get("entities", {})
        session_data["entities"].update(entities)
        
        # Step 3: Generate SQL
        sql = generate_sql(
            user_query,
            entities,
            table_columns,
            config.get('annotated_schema', ''),
            config.get('relationships_text', '')
        )
        
        # Step 4: Validate SQL
        validation = validate_sql(sql, table_columns)
        if not validation.get("valid"):
            return {
                "type": "error",
                "message": f"SQL validation failed: {validation.get('error')}",
                "session_data": session_data
            }
        
        # Step 5: Execute SQL
        execution = execute_sql(sql, module_id)
        if not execution.get("success"):
            return {
                "type": "error",
                "message": f"Query failed: {execution.get('error')}",
                "session_data": session_data
            }
        
        data = execution.get("data", [])
        
        # Step 6: Create chart
        chart_data = create_chart(data, user_query)
        
        # Build response
        result_summary = f"Found {len(data)} results."
        
        # Update session history
        session_data["history"].append({
            "query": user_query,
            "sql": sql,
            "result_count": len(data)
        })
        
        logger.info("✅ Graph completed")
        
        return {
            "type": "response",
            "message": result_summary,
            "sql": sql,
            "data": data,
            "chart": chart_data,
            "session_data": session_data,
            "metadata": {
                "chart": chart_data,
                "sql": sql,
                "row_count": len(data)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Graph execution error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "type": "error",
            "message": f"An error occurred: {str(e)}",
            "session_data": session_data or {"entities": {}, "history": []}
        }