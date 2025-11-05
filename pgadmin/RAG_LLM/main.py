from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== LOAD SCHEMA FILES ====================
def load_schema_files():
    """Load schema files with error handling"""
    annotated_schema = "### Schema file not found."
    relationships = "### Relationship file not found."
    
    try:
        with open(os.path.join(BASE_DIR, "annotated_schema.md"), "r", encoding="utf-8") as f:
            annotated_schema = f.read()
    except FileNotFoundError:
        logger.warning("annotated_schema.md not found")
    
    try:
        with open(os.path.join(BASE_DIR, "relationship.txt"), "r", encoding="utf-8") as f:
            relationships = f.read()
    except FileNotFoundError:
        logger.warning("relationship.txt not found")
    
    return annotated_schema, relationships

load_dotenv()

# ==================== DATABASE CONNECTION POOL ====================
try:
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        1, 20,
        host=os.getenv("PG_HOST", "localhost"),
        dbname=os.getenv("PG_DBNAME", "haldiram"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "12345678")
    )
except psycopg2.Error as e:
    logger.error(f"Failed to create connection pool: {e}")
    raise

def get_db_connection():
    return connection_pool.getconn()

def put_db_connection(conn):
    connection_pool.putconn(conn)

# ==================== IMPORT NODES ====================
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

# ==================== GRAPH STATE ====================
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

# ==================== ENTITY RESOLVER WITH PROPER FORMATTING ====================
def entity_resolver_with_memory(state: GraphState):
    """
    Enhanced resolver that:
    1. Handles clarifications properly
    2. Formats entities correctly for SQL generator
    3. Only keeps relevant entities from session
    """
    user_query = state["user_query"]
    session_entities = state.get("session_entities", {})
    human_feedback = state.get("human_feedback")
    catalog = state.get("catalog", {})
    table_columns = state.get("table_columns", {})
    query_history = state.get("query_history", [])

    resolved = {"entities": {}, "filters": {}, "table": "tbl_primary"}
    clarification_needed = None

    logger.info(f"\n{'='*70}")
    logger.info(f"🔍 ENTITY RESOLVER")
    logger.info(f"Query: {user_query}")
    logger.info(f"Session entities: {session_entities}")
    logger.info(f"Feedback: {human_feedback}")

    # ===== HANDLE FEEDBACK =====
    if human_feedback:
        feedback_type = human_feedback.get("type")
        entity_type = human_feedback.get("entity_type")
        
        if feedback_type == "column_selection":
            # User selected a column - search in that specific column
            entity_value = human_feedback.get("entity")
            selected_table = human_feedback.get("selected_table")
            selected_column = human_feedback.get("selected_column")
            
            conn = None
            try:
                conn = get_db_connection()
                result = resolve_in_specific_column(
                    entity_value, 
                    selected_table, 
                    selected_column, 
                    conn, 
                    max_options=15
                )
                
                if result.get("resolved"):
                    # CRITICAL: Store in proper format
                    resolved["entities"][entity_type] = {
                        "table": selected_table,
                        "column": selected_column,
                        "value": result["value"]
                    }
                    session_entities[entity_type] = {
                        "table": selected_table,
                        "column": selected_column,
                        "value": result["value"]
                    }
                    logger.info(f"✅ Resolved from column selection: {result['value']}")
                    
                elif result.get("clarification"):
                    clarification_needed = result["clarification"]
                    clarification_needed["entity_type"] = entity_type
                    
            except Exception as e:
                logger.error(f"Error in column selection: {e}")
            finally:
                if conn:
                    put_db_connection(conn)
        
        elif feedback_type == "value_selection":
            # User selected a value from options
            selected_value = human_feedback.get("selected_option")
            if entity_type and selected_value:
                # CRITICAL: Store with table/column info if available from clarification context
                if "clarification_context" in human_feedback:
                    table = human_feedback["clarification_context"].get("table")
                    column = human_feedback["clarification_context"].get("column")
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
                else:
                    # Fallback: infer table/column
                    column, table = _infer_column_table(entity_type)
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
                
                logger.info(f"✅ User selected: {entity_type} = {selected_value}")
        
        elif feedback_type == "custom_input":
            # User typed custom value
            custom_value = human_feedback.get("custom_value")
            if entity_type and custom_value:
                column, table = _infer_column_table(entity_type)
                resolved["entities"][entity_type] = {
                    "table": table,
                    "column": column,
                    "value": custom_value
                }
                session_entities[entity_type] = {
                    "table": table,
                    "column": column,
                    "value": custom_value
                }
                logger.info(f"✅ Custom input: {entity_type} = {custom_value}")
    
    # ===== PARSE QUERY WITH LLM =====
    if not human_feedback or human_feedback.get("type") == "column_selection":
        try:
            parsed = llm_understand(user_query)
            entities = parsed.get("entities", {})
            if not isinstance(entities, dict):
                entities = {}
            
            # CRITICAL: Check if this is a NEW question (different from last)
            is_new_question = True
            if query_history:
                last_query = query_history[-1] if query_history else ""
                # Simple check: if queries are significantly different
                if user_query.lower() != last_query.lower():
                    # Clear entities that aren't mentioned in new query
                    entities_to_keep = {}
                    for entity_type in session_entities.keys():
                        # Keep entity only if it's still relevant
                        if entity_type in entities:
                            entities_to_keep[entity_type] = session_entities[entity_type]
                    session_entities = entities_to_keep
                    logger.info(f"🔄 New question detected - cleared old entities")
                    logger.info(f"   Kept entities: {list(entities_to_keep.keys())}")
                else:
                    is_new_question = False
            
            logger.info(f"📝 LLM parsed entities: {list(entities.keys())}")
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            entities = {}
        
        # ===== VALIDATE ENTITIES =====
        for e_type, e_value in entities.items():
            # Skip if already in session AND same query
            if e_type in session_entities and not is_new_question:
                resolved["entities"][e_type] = session_entities[e_type]
                logger.info(f"♻️ Using cached: {e_type} = {session_entities[e_type].get('value')}")
                continue
            
            if isinstance(e_value, list):
                e_value = e_value[0] if e_value else ""
            
            if not e_value:
                continue
            
            logger.info(f"🔍 Resolving new entity: {e_type} = {e_value}")
            
            # Resolve entity
            validation = resolve_entity_non_interactive(
                e_value, 
                catalog, 
                table_columns, 
                max_options=15
            )
            
            if validation.get("resolved"):
                resolved["entities"][e_type] = {
                    "table": validation["table"],
                    "column": validation["column"],
                    "value": validation["value"]
                }
                session_entities[e_type] = {
                    "table": validation["table"],
                    "column": validation["column"],
                    "value": validation["value"]
                }
                logger.info(f"✅ Auto-resolved: {e_type} = {validation['value']}")
                
            elif validation.get("clarification") and not clarification_needed:
                clarification_needed = validation["clarification"]
                clarification_needed["entity_type"] = e_type
                logger.info(f"❓ Clarification needed for: {e_type}")
                break
    
    # ===== RETURN STATE =====
    if clarification_needed and not (human_feedback and human_feedback.get("type") == "value_selection"):
        logger.info(f"⏸️ Returning clarification request")
        return {
            "clarification_needed": clarification_needed,
            "session_entities": session_entities,
        }
    
    # All entities resolved - prepare for SQL generation
    resolved["entities"].update(session_entities)
    
    logger.info(f"\n✅ All entities resolved:")
    for e_type, e_data in resolved["entities"].items():
        if isinstance(e_data, dict):
            logger.info(f"   {e_type}: {e_data.get('value')} (from {e_data.get('table')}.{e_data.get('column')})")
    
    return {
        "resolved": resolved,
        "session_entities": resolved["entities"],
        "clarification_needed": None,
    }

def _infer_column_table(entity_type: str) -> tuple:
    """Infer column and table from entity type"""
    mapping = {
        "product": ("product", "tbl_product_master"),
        "distributor": ("distributor_name", "tbl_primary"),
        "superstockist": ("superstockist_name", "tbl_primary"),
        "sold_to_party": ("sold_to_party_name", "tbl_shipment"),
    }
    return mapping.get(entity_type, (entity_type, "tbl_primary"))

# ==================== ERROR HANDLER ====================
def error_handler(state: GraphState):
    error = state.get("error", "Unknown error")
    logger.error(f"Error: {error}")
    return {
        "final_output": f"Error: {error}. Please rephrase your query.",
        "error": error
    }

# ==================== BUILD GRAPH ====================
def create_graph_with_memory():
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("question_validator", question_validator)
    graph.add_node("entity_resolver", entity_resolver_with_memory)
    graph.add_node("sql_generator", sql_agent_node)
    graph.add_node("validator_sql", validator_agent)
    graph.add_node("executor_sql", sql_executor_node)
    graph.add_node("summarized_results", summarized_results_node)
    graph.add_node("chart_creation_node", chart_creation_node)
    graph.add_node("error_handler", error_handler)
    
    # Set entry point
    graph.set_entry_point("question_validator")
    graph.add_edge("question_validator", "entity_resolver")
    
    # Conditional routing
    def route_after_entity_resolver(state):
        if state.get("clarification_needed"):
            return END
        elif state.get("error"):
            return "error_handler"
        else:
            return "sql_generator"
    
    graph.add_conditional_edges(
        "entity_resolver",
        route_after_entity_resolver,
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

# ==================== DJANGO ENTRY POINT ====================
def invoke_graph(user_query: str, session_data: dict, human_feedback: dict = None):
    """
    Main entry point for Django views.
    """
    conn = None
    try:
        annotated_schema, relationships = load_schema_files()
        
        conn = get_db_connection()
        table_columns = load_table_columns_pg(
            conn, 
            ["tbl_shipment", "tbl_primary", "tbl_product_master"]
        )
        catalog = build_catalog(conn, table_columns, max_values=100)
        
        graph = create_graph_with_memory()
        
        payload = {
            "user_query": user_query,
            "catalog": catalog,
            "table_columns": table_columns,
            "annotated_schema": annotated_schema,
            "relationships": relationships,
            "session_entities": session_data.get("entities", {}),
            "query_history": session_data.get("history", []),
        }
        
        # Add feedback if provided
        if human_feedback:
            payload["human_feedback"] = human_feedback
            # CRITICAL: Add clarification context to feedback
            if "clarification_context" not in human_feedback and session_data.get("pending_clarification"):
                clarification = session_data["pending_clarification"]
                if clarification.get("table") and clarification.get("column"):
                    human_feedback["clarification_context"] = {
                        "table": clarification["table"],
                        "column": clarification["column"]
                    }
            logger.info(f"Processing with feedback: {human_feedback.get('type')}")
        
        thread_id = session_data.get("thread_id", "default_user")
        
        logger.info(f"Invoking graph for thread: {thread_id}")
        result = graph.invoke(payload, config={"thread_id": thread_id})
        
        logger.info(f"🔍 DEBUG - Graph result keys: {result.keys()}")
        logger.info(f"🔍 DEBUG - Has chart_data: {bool(result.get('chart_data'))}")
        
        # ===== HANDLE CLARIFICATION =====
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
        
        # ===== HANDLE ERRORS =====
        if result.get("error"):
            return {
                "type": "error",
                "message": result.get("final_output", "An error occurred."),
            }
        
        # ===== NORMAL RESPONSE =====
        response = {
            "type": "response",
            "message": result.get("final_output", "✅ Query processed."),
            "entities": result.get("session_entities", {}),
        }
        
        # CRITICAL FIX: Use chart_data (not chart)
        if result.get("chart_data"):
            response["chart_data"] = result["chart_data"]
            logger.info(f"✅ Chart data included in response")
            logger.info(f"📊 Chart info: {result['chart_data'].get('chart_info', {})}")
        
        logger.info(f"✅ Response ready: {response['type']}")
        return response
        
    except Exception as e:
        logger.error(f"Error in invoke_graph: {e}", exc_info=True)
        return {
            "type": "error",
            "message": "Error processing request. Please try again.",
        }
    finally:
        if conn:
            put_db_connection(conn)

# ==================== CLEANUP ====================
import atexit

def cleanup():
    if 'connection_pool' in globals():
        connection_pool.closeall()
        logger.info("Database connections closed")

atexit.register(cleanup)