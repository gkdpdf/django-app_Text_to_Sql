from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

# Module-specific connection pools
MODULE_POOLS = {}

def create_connection_pool(module_config):
    """Create database connection pool using Django settings"""
    from .django_loader import get_db_credentials
    
    creds = get_db_credentials()
    
    logger.info(f"🔌 Creating connection pool...")
    
    try:
        return psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=creds['host'],
            dbname=creds['database'],
            user=creds['user'],
            password=creds['password'],
            port=creds['port']
        )
    except psycopg2.Error as e:
        logger.error(f"❌ Connection pool creation failed: {e}")
        raise

def get_db_connection(module_id):
    """Get connection from module-specific pool"""
    if module_id not in MODULE_POOLS:
        raise ValueError(f"No connection pool for module {module_id}")
    return MODULE_POOLS[module_id].getconn()

def put_db_connection(conn, module_id):
    """Return connection to pool"""
    if module_id in MODULE_POOLS:
        MODULE_POOLS[module_id].putconn(conn)

# Import nodes
from nodes.entity_clarity_node import (
    load_table_columns_pg, build_catalog, llm_understand, 
    resolve_entity_non_interactive, resolve_in_specific_column
)
from nodes.sql_creator_node import sql_agent_node
from nodes.validator_sql import validator_agent
from nodes.question_clean import question_validator
from nodes.summarized_result import summarized_results_node
from nodes.executor_sql import sql_executor_node
from nodes.chart_creation_node import chart_creation_node

# Graph State
class GraphState(TypedDict, total=False):
    user_query: str
    catalog: Dict[str, Any]
    table_columns: Dict[str, List[str]]
    annotated_schema: str
    relationships: str
    resolved: Dict[str, Any]
    session_entities: Dict[str, Any]
    query_history: List[str]
    clarification_needed: Optional[Dict[str, Any]]
    final_output: Optional[str]
    human_feedback: Optional[Dict[str, Any]]
    error: Optional[str]
    chart_data: Optional[Dict[str, Any]]
    sql_result: Optional[str]
    validated_sql: Optional[str]
    execution_result: Optional[List[Dict[str, Any]]]
    execution_status: Optional[str]
    module_config: Optional[Dict[str, Any]]

# Entity Resolver
def entity_resolver_with_memory(state: GraphState):
    """Enhanced entity resolver with module awareness"""
    user_query = state["user_query"]
    session_entities = state.get("session_entities", {})
    human_feedback = state.get("human_feedback")
    catalog = state.get("catalog", {})
    table_columns = state.get("table_columns", {})
    query_history = state.get("query_history", [])
    module_config = state.get("module_config", {})

    resolved = {"entities": {}, "filters": {}}
    clarification_needed = None

    logger.info(f"\n{'='*70}")
    logger.info(f"🔍 ENTITY RESOLVER")
    logger.info(f"Module: {module_config.get('module_name', 'Unknown')}")
    logger.info(f"Query: {user_query}")

    # Handle feedback
    if human_feedback:
        feedback_type = human_feedback.get("type")
        entity_type = human_feedback.get("entity_type")
        
        if feedback_type == "column_selection":
            entity_value = human_feedback.get("entity")
            selected_table = human_feedback.get("selected_table")
            selected_column = human_feedback.get("selected_column")
            
            module_id = module_config.get("module_id")
            conn = None
            try:
                conn = get_db_connection(module_id)
                result = resolve_in_specific_column(
                    entity_value, selected_table, selected_column, conn, max_options=15
                )
                
                if result.get("resolved"):
                    resolved["entities"][entity_type] = {
                        "table": selected_table,
                        "column": selected_column,
                        "value": result["value"]
                    }
                    session_entities[entity_type] = resolved["entities"][entity_type]
                    logger.info(f"✅ Resolved: {entity_type} = {result['value']}")
                elif result.get("clarification"):
                    clarification_needed = result["clarification"]
                    clarification_needed["entity_type"] = entity_type
            except Exception as e:
                logger.error(f"Error in column selection: {e}")
            finally:
                if conn:
                    put_db_connection(conn, module_id)
        
        elif feedback_type == "value_selection":
            selected_value = human_feedback.get("selected_option")
            
            if entity_type and selected_value:
                # CRITICAL FIX: Always use proper dict format
                if "clarification_context" in human_feedback:
                    table = human_feedback["clarification_context"].get("table")
                    column = human_feedback["clarification_context"].get("column")
                else:
                    column, table = _infer_column_table_dynamic(entity_type, module_config)
                
                resolved["entities"][entity_type] = {
                    "table": table,
                    "column": column,
                    "value": selected_value
                }
                session_entities[entity_type] = {
                    "table": table,
                    "column": column,
                    "value": selected_value
                }
                
                logger.info(f"✅ Selected: {entity_type} = {selected_value}")

    # Parse query with LLM
    if not human_feedback or human_feedback.get("type") == "column_selection":
        try:
            parsed = llm_understand(user_query, module_config)
            entities = parsed.get("entities", {})
            
            if not isinstance(entities, dict):
                entities = {}
            
            # Check for new question
            is_new_question = True
            if query_history:
                last_query = query_history[-1] if query_history else ""
                if user_query.lower() == last_query.lower():
                    is_new_question = False
            
            if is_new_question and query_history:
                entities_to_keep = {k: v for k, v in session_entities.items() if k in entities}
                session_entities = entities_to_keep
                logger.info(f"🔄 New question - kept: {list(entities_to_keep.keys())}")
            
            logger.info(f"📝 LLM parsed entities: {list(entities.keys())}")
            
        except Exception as e:
            logger.error(f"LLM parsing error: {e}", exc_info=True)
            entities = {}
        
        # Validate entities
        for e_type, e_value in entities.items():
            if e_type in session_entities and not is_new_question:
                resolved["entities"][e_type] = session_entities[e_type]
                logger.info(f"♻️ Using cached: {e_type}")
                continue
            
            if isinstance(e_value, list):
                e_value = e_value[0] if e_value else ""
            
            if not e_value:
                continue
            
            logger.info(f"🔍 Resolving: {e_type} = {e_value}")
            
            validation = resolve_entity_non_interactive(
                e_value, catalog, table_columns, max_options=15
            )
            
            if validation.get("resolved"):
                resolved["entities"][e_type] = {
                    "table": validation["table"],
                    "column": validation["column"],
                    "value": validation["value"]
                }
                session_entities[e_type] = resolved["entities"][e_type]
                logger.info(f"✅ Auto-resolved: {e_type} = {validation['value']}")
                
            elif validation.get("clarification") and not clarification_needed:
                clarification_needed = validation["clarification"]
                clarification_needed["entity_type"] = e_type
                logger.info(f"❓ Clarification needed for: {e_type}")
                break
    
    # Return state
    if clarification_needed:
        logger.info("⏸️ Returning clarification")
        return {
            "clarification_needed": clarification_needed,
            "session_entities": session_entities,
        }
    
    resolved["entities"].update(session_entities)
    
    logger.info(f"✅ All entities resolved: {list(resolved['entities'].keys())}")
    
    return {
        "resolved": resolved,
        "session_entities": resolved["entities"],
        "clarification_needed": None,
    }


def _infer_column_table_dynamic(entity_type: str, module_config: dict) -> tuple:
    """Infer column and table from entity type using module config"""
    entity_map = module_config.get("entity_inference_map", {})
    
    if entity_type.lower() in entity_map:
        return entity_map[entity_type.lower()]
    
    tables = module_config.get("tables", [])
    if tables:
        return (entity_type, tables[0])
    
    return (entity_type, "unknown_table")


def error_handler(state: GraphState):
    """Handle errors"""
    error = state.get("error", "Unknown error")
    logger.error(f"❌ Error handler: {error}")
    return {
        "final_output": f"Error: {error}. Please rephrase your query.",
        "error": error
    }


def create_graph_with_memory():
    """Build the LangGraph workflow"""
    graph = StateGraph(GraphState)
    
    graph.add_node("question_validator", question_validator)
    graph.add_node("entity_resolver", entity_resolver_with_memory)
    graph.add_node("sql_generator", sql_agent_node)
    graph.add_node("validator_sql", validator_agent)
    graph.add_node("executor_sql", sql_executor_node)
    graph.add_node("summarized_results", summarized_results_node)
    graph.add_node("chart_creation_node", chart_creation_node)
    graph.add_node("error_handler", error_handler)
    
    graph.set_entry_point("question_validator")
    graph.add_edge("question_validator", "entity_resolver")
    
    def route_after_resolver(state):
        if state.get("clarification_needed"):
            return END
        elif state.get("error"):
            return "error_handler"
        return "sql_generator"
    
    graph.add_conditional_edges(
        "entity_resolver",
        route_after_resolver,
        {
            "sql_generator": "sql_generator",
            "error_handler": "error_handler",
            END: END
        }
    )
    
    graph.add_edge("sql_generator", "validator_sql")
    graph.add_edge("validator_sql", "executor_sql")
    graph.add_edge("executor_sql", "summarized_results")
    graph.add_edge("summarized_results", "chart_creation_node")
    graph.add_edge("chart_creation_node", END)
    graph.add_edge("error_handler", END)
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def invoke_graph(user_query: str, session_data: dict, human_feedback: dict = None):
    """Main entry point - COMPLETE VERSION"""
    conn = None
    module_id = session_data.get("module_id")
    
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 NEW REQUEST")
        logger.info(f"Query: {user_query}, Module: {module_id}")
        
        if not module_id:
            logger.error("❌ No module ID")
            return {"type": "error", "message": "No module selected"}
        
        # Load module configuration
        from .django_loader import load_module_config
        
        config = load_module_config(module_id)
        logger.info(f"✅ Module loaded: {config['module_name']}")
        
        # Create connection pool if needed
        if module_id not in MODULE_POOLS:
            MODULE_POOLS[module_id] = create_connection_pool(config)
        
        conn = get_db_connection(module_id)
        
        # Load schema
        tables = config['tables']
        table_columns = load_table_columns_pg(conn, tables)
        catalog = build_catalog(conn, table_columns, max_values=100)
        
        # Create graph
        graph = create_graph_with_memory()
        
        # Prepare payload
        payload = {
            "user_query": user_query,
            "catalog": catalog,
            "table_columns": table_columns,
            "annotated_schema": config['annotated_schema'],
            "relationships": config['relationships'],
            "session_entities": session_data.get("entities", {}),
            "query_history": session_data.get("history", []),
            "module_config": config,
        }
        
        if human_feedback:
            payload["human_feedback"] = human_feedback
            
            if "clarification_context" not in human_feedback:
                pending = session_data.get("pending_clarification")
                if pending and pending.get("table") and pending.get("column"):
                    human_feedback["clarification_context"] = {
                        "table": pending["table"],
                        "column": pending["column"]
                    }
        
        thread_id = session_data.get("thread_id", f"module_{module_id}_user")
        
        result = graph.invoke(payload, config={"configurable": {"thread_id": thread_id}})
        logger.info(f"✅ Graph completed")
        
        # Handle clarification
        if result.get("clarification_needed"):
            clarification = result["clarification_needed"]
            
            if clarification.get("type") == "column":
                return {
                    "type": "clarification",
                    "subtype": "column",
                    "message": clarification["message"],
                    "options": clarification["options"],
                    "entity": clarification["entity"],
                    "entity_type": clarification.get("entity_type", "unknown"),
                }
            else:
                return {
                    "type": "clarification",
                    "subtype": "value",
                    "message": clarification["message"],
                    "options": clarification["options"],
                    "has_more": clarification.get("has_more", False),
                    "total_count": clarification.get("total_count", len(clarification["options"])),
                    "entity": clarification["entity"],
                    "entity_type": clarification.get("entity_type", "unknown"),
                    "table": clarification.get("table"),
                    "column": clarification.get("column"),
                }
        
        # Handle errors
        if result.get("error"):
            return {"type": "error", "message": result.get("final_output", "An error occurred.")}
        
        # Normal response
        response = {
            "type": "response",
            "message": result.get("final_output", "✅ Query processed."),
            "entities": result.get("session_entities", {}),
        }
        
        if result.get("chart_data"):
            response["chart_data"] = result["chart_data"]
        
        return response
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR", exc_info=True)
        return {"type": "error", "message": f"System error: {str(e)}"}
        
    finally:
        if conn and module_id:
            try:
                put_db_connection(conn, module_id)
            except:
                pass


# Cleanup
import atexit

def cleanup():
    """Cleanup connection pools"""
    for module_id, pool in MODULE_POOLS.items():
        try:
            pool.closeall()
        except:
            pass

atexit.register(cleanup)