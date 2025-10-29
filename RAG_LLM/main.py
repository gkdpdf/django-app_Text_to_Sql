from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()

with open("annotated_schema.md", "r", encoding="utf-8") as f:
    annotated_schema = f.read()

with open("relationship.txt", "r", encoding="utf-8") as f:
    relationships = f.read()

conn = psycopg2.connect(
    host="localhost",
    dbname="haldiram",
    user="postgres",
    password="12345678"
)

# Import after connection setup
from nodes.entity_clarity_node import load_table_columns_pg, build_catalog
from nodes.sql_creator_node import sql_agent_node
from nodes.validator_sql import validator_agent
from nodes.question_clean import question_validator
from nodes.summarized_result import summarized_results_node
from nodes.executor_sql import sql_executor_node
from nodes.chart_creation_node import chart_creation_node

# ---------- Enhanced Graph State with Memory ----------
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
    # Enhanced memory fields
    session_entities: Dict[str, Any]
    last_product: Optional[str]
    last_distributor: Optional[str]
    last_superstockist: Optional[str]
    last_sold_to_party: Optional[str]
    last_table: Optional[str]
    last_columns: List[str]
    query_history: List[str]


# ---------- Enhanced Entity Validator with Fuzzy Matching ----------
def validate_and_match_entity(entity_value: str, entity_type: str, catalog: Dict[str, Any], conn) -> Dict[str, Any]:
    """
    Validate entity with database lookup, fuzzy matching, and user confirmation
    Returns dict with: value, original, validated, confidence, source, table, column
    """
    from difflib import get_close_matches, SequenceMatcher
    
    print(f"\n🔍 Validating '{entity_value}' as {entity_type}")
    
    # Map entity types to database tables/columns
    table_column_map = {
        "sold_to_party": ("tbl_shipment", "sold_to_party_name"),
        "distributor": ("tbl_primary", "distributor_name"),
        "superstockist": ("tbl_primary", "superstockist_name"),
        "product": ("tbl_product_master", "product")
    }
    
    if entity_type not in table_column_map:
        print(f"⚠️ Unknown entity type: {entity_type}")
        return {
            "value": entity_value,
            "original": entity_value,
            "validated": False,
            "confidence": 0.0,
            "source": "unknown_type"
        }
    
    table, column = table_column_map[entity_type]
    
    try:
        cursor = conn.cursor()
        
        # Step 1: Try exact match
        query = f"SELECT DISTINCT {column} FROM {table} WHERE LOWER({column}) = LOWER(%s) LIMIT 1"
        cursor.execute(query, (entity_value,))
        result = cursor.fetchone()
        
        if result:
            matched_value = result[0]
            print(f"✅ Exact match in DB: {matched_value}")
            cursor.close()
            return {
                "value": matched_value,
                "original": entity_value,
                "validated": True,
                "confidence": 1.0,
                "source": f"database_{table}",
                "table": table,
                "column": column
            }
        
        # Step 2: Fuzzy search
        print(f"🔎 Searching database: {table}.{column}")
        words = entity_value.lower().split()
        all_candidates = []
        
        # Try full phrase
        pattern = f"%{entity_value}%"
        query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} ILIKE %s LIMIT 20"
        cursor.execute(query, (pattern,))
        results = cursor.fetchall()
        if results:
            print(f"   ✓ Found {len(results)} matches with full phrase")
            all_candidates.extend([r[0] for r in results])
        
        # Try individual words
        if len(words) > 1:
            for word in words:
                if len(word) > 2:
                    pattern = f"%{word}%"
                    cursor.execute(query, (pattern,))
                    results = cursor.fetchall()
                    if results:
                        new_candidates = [r[0] for r in results if r[0] not in all_candidates]
                        all_candidates.extend(new_candidates)
        
        # If no matches, get samples
        if not all_candidates:
            query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 10"
            cursor.execute(query)
            results = cursor.fetchall()
            if results:
                all_candidates = [r[0] for r in results]
        
        cursor.close()
        
        if all_candidates:
            # Remove duplicates
            seen = set()
            unique_candidates = []
            for candidate in all_candidates:
                if candidate.lower() not in seen:
                    seen.add(candidate.lower())
                    unique_candidates.append(candidate)
            
            all_candidates = unique_candidates[:10]
            
            # Score candidates
            scored_matches = []
            for candidate in all_candidates:
                score = SequenceMatcher(None, entity_value.lower(), candidate.lower()).ratio()
                scored_matches.append((candidate, score))
            
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Show options
            print("\n   Available options:")
            for i, (candidate, score) in enumerate(scored_matches, 1):
                bar = "█" * int(score * 10)
                print(f"   {i}. {candidate} {bar} ({score:.2f})")
            
            best_match, best_score = scored_matches[0]
            
            print(f"\n❓ Which one did you mean?")
            print(f"   Enter number (1-{len(scored_matches)}) or type exact name:")
            if best_score >= 0.6:
                print(f"   Press Enter for: {best_match}")
            
            choice = input("   > ").strip()
            
            if not choice and best_score >= 0.6:
                print(f"✅ Selected: {best_match}")
                return {
                    "value": best_match,
                    "original": entity_value,
                    "validated": True,
                    "confidence": best_score,
                    "source": f"database_{table}_confirmed",
                    "table": table,
                    "column": column
                }
            elif choice.isdigit() and 1 <= int(choice) <= len(scored_matches):
                selected = scored_matches[int(choice) - 1][0]
                print(f"✅ Selected: {selected}")
                return {
                    "value": selected,
                    "original": entity_value,
                    "validated": True,
                    "confidence": 1.0,
                    "source": f"database_{table}_user_selected",
                    "table": table,
                    "column": column
                }
            elif choice:
                return validate_and_match_entity(choice, entity_type, catalog, conn)
        
        # No matches found
        print(f"❌ No matches found for '{entity_value}'")
        return {
            "value": entity_value,
            "original": entity_value,
            "validated": False,
            "confidence": 0.0,
            "source": "not_found",
            "table": table,
            "column": column
        }
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return {
            "value": entity_value,
            "original": entity_value,
            "validated": False,
            "confidence": 0.0,
            "source": "error"
        }


def detect_entity_override(user_query: str) -> Dict[str, bool]:
    """
    Detect if user is explicitly specifying NEW entities that should OVERRIDE memory
    Returns dict with entity types that should be overridden
    """
    query_lower = user_query.lower()
    
    override_signals = {
        "product": False,
        "distributor": False,
        "superstockist": False,
        "sold_to_party": False
    }
    
    # Strong indicators for NEW product
    product_patterns = [
        r'(?:for|of)\s+([a-z\s]+?)\s+(?:sales|total|quantity)',
        r'sales\s+(?:of|for)\s+([a-z\s]+)',
        r'(?:product|item)\s+([a-z\s]+)',
    ]
    
    # Strong indicators for NEW distributor
    distributor_keywords = ['distributor', 'dist', 'dealer']
    if any(kw in query_lower for kw in distributor_keywords):
        override_signals["distributor"] = True
    
    # Check for product override
    import re
    for pattern in product_patterns:
        if re.search(pattern, query_lower):
            # Check if it's NOT a continuation phrase
            continuation_words = ['last', 'previous', 'same', 'that', 'this', 'it']
            if not any(word in query_lower.split()[:3] for word in continuation_words):
                override_signals["product"] = True
                break
    
    return override_signals


# ---------- Enhanced Entity Resolver with FIXED Memory ----------
def entity_resolver_with_memory(state: GraphState):
    """
    FIXED: Enhanced entity resolver with smart context reuse
    - Automatically carries forward last entities UNLESS new ones are specified
    - No need for "last" or "same" keywords
    """
    from nodes.entity_clarity_node import llm_understand
    
    user_query = state["user_query"]
    catalog = state.get("catalog", {})
    table_columns = state.get("table_columns", {})
    session_entities = state.get("session_entities", {})
    query_history = state.get("query_history", [])
    
    print("\n🧠 Analyzing query with session context...")
    print(f"📝 Query: {user_query}")
    
    if session_entities:
        print(f"💾 Session memory contains:")
        for key, val in session_entities.items():
            if key.startswith("last_") and val:
                if isinstance(val, dict):
                    print(f"   - {key}: {val.get('value', val)}")
                else:
                    print(f"   - {key}: {val}")
    
    query_lower = user_query.lower()
    
    # CRITICAL FIX: Detect if user is specifying NEW entities
    override_signals = detect_entity_override(user_query)
    
    # Check if query is asking about metrics/aggregations WITHOUT specifying entities
    metric_keywords = ['sales', 'total', 'quantity', 'revenue', 'count', 'sum', 'how much', 'how many']
    has_metrics = any(kw in query_lower for kw in metric_keywords)
    
    # Detect if this is a pure metric query (no new entities specified)
    is_metric_continuation = (
        has_metrics and
        session_entities and
        not any(override_signals.values()) and
        len(query_history) > 0
    )
    
    print(f"\n🔍 Query Analysis:")
    print(f"   Has metrics: {has_metrics}")
    print(f"   Override signals: {override_signals}")
    print(f"   Is metric continuation: {is_metric_continuation}")
    
    # DECISION: Use memory or resolve new entities?
    if is_metric_continuation:
        print("✅ USING MEMORY - Carrying forward last entities")
        
        # Parse query for NEW information (metrics, filters)
        parsed = llm_understand(user_query)
        
        from nodes.entity_clarity_node import detect_time_filters
        filters = detect_time_filters(user_query)
        
        # Build resolved state with MEMORY entities
        resolved = {
            "intent": parsed.get("intent", "query"),
            "metrics": parsed.get("metrics", []),
            "entities": {},
            "filters": filters,
            "table": session_entities.get("last_table"),
            "columns": session_entities.get("last_columns", [])
        }
        
        # CARRY FORWARD entities from memory
        for entity_type in ["product", "distributor", "superstockist", "sold_to_party"]:
            session_key = f"last_{entity_type}"
            if session_key in session_entities and session_entities[session_key]:
                resolved["entities"][entity_type] = session_entities[session_key]
                entity_val = session_entities[session_key].get('value') if isinstance(session_entities[session_key], dict) else session_entities[session_key]
                print(f"   💾 Reusing {entity_type}: {entity_val}")
        
        updated_session_entities = session_entities.copy()
        
    else:
        # NEW QUERY or OVERRIDE - Full resolution
        print("🆕 RESOLVING NEW ENTITIES")
        
        from nodes.entity_clarity_node import resolve_with_human_in_loop_pg
        
        try:
            resolved = resolve_with_human_in_loop_pg(user_query, catalog, table_columns)
            
            # Remove auto-generated time filters
            if resolved.get("filters") and resolved["filters"].get("time_range"):
                explicit_time_keywords = [
                    'january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november', 'december',
                    'last month', 'this month', 'yesterday', 'today', 'last week'
                ]
                has_explicit_time = any(kw in query_lower for kw in explicit_time_keywords)
                
                if not has_explicit_time:
                    print("⚠️ REMOVING auto-generated time filter")
                    resolved["filters"] = {}
        
        except Exception as e:
            print(f"⚠️ Resolution error: {e}")
            resolved = {"intent": "query", "entities": {}, "filters": {}, "table": None, "columns": []}
        
        if not isinstance(resolved.get("entities"), dict):
            resolved["entities"] = {}
        
        # CRITICAL FIX: Extract entity from query if not resolved
        if not resolved.get("entities"):
            parsed_llm = llm_understand(user_query)
            detected_entities = parsed_llm.get("entities", {})
            
            print(f"\n🔍 LLM detected entities: {detected_entities}")
            
            if detected_entities:
                import re
                patterns = [
                    r'(?:sales|total|revenue|quantity)\s+(?:of|for)\s+(.+?)(?:\s+in|\s+for|\s+from|\s+by|$)',
                    r'(?:for|of)\s+(.+?)(?:\s+sales|\s+total|\s+distributor|$)',
                    r'^(.+?)\s+(?:sales|total|revenue)',
                ]
                
                entity_name = None
                for pattern in patterns:
                    match = re.search(pattern, query_lower, re.IGNORECASE)
                    if match:
                        entity_name = match.group(1).strip()
                        # Clean up
                        stop_words = ['the', 'a', 'an', 'in', 'on', 'at', 'by', 'from']
                        words = entity_name.split()
                        entity_name = ' '.join([w for w in words if w not in stop_words])
                        break
                
                if entity_name:
                    print(f"🎯 Extracted entity: '{entity_name}'")
                    
                    # Determine entity type
                    table = resolved.get("table", "")
                    if table == "tbl_shipment":
                        entity_type = "sold_to_party"
                    elif "distributor" in query_lower:
                        entity_type = "distributor"
                    elif "superstockist" in query_lower or "super stockist" in query_lower:
                        entity_type = "superstockist"
                    elif "product" in detected_entities or table == "tbl_primary":
                        entity_type = "product"
                    else:
                        entity_type = "product"  # default
                    
                    validation = validate_and_match_entity(entity_name, entity_type, catalog, conn)
                    if validation["validated"]:
                        resolved["entities"][entity_type] = validation
                        print(f"✅ Added {entity_type}: {validation['value']}")
        
        # Validate all entities
        validated_entities = {}
        for entity_type, entity_data in resolved.get("entities", {}).items():
            if not entity_data:
                continue
            
            # Handle if override NOT signaled for this entity type - check memory
            if not override_signals.get(entity_type) and session_entities.get(f"last_{entity_type}"):
                print(f"   💾 No override for {entity_type}, keeping memory")
                validated_entities[entity_type] = session_entities[f"last_{entity_type}"]
                continue
            
            if isinstance(entity_data, dict):
                entity_value = entity_data.get("value", entity_data.get("name", ""))
                original_table = entity_data.get("table")
                original_column = entity_data.get("column")
            else:
                entity_value = str(entity_data)
                original_table = None
                original_column = None
            
            if not entity_value:
                continue
            
            validation_result = validate_and_match_entity(entity_value, entity_type, catalog, conn)
            
            if validation_result["validated"]:
                print(f"  ✅ Validated {entity_type}: '{entity_value}' → '{validation_result['value']}'")
                validated_entities[entity_type] = {
                    "value": validation_result["value"],
                    "original": validation_result.get("original", entity_value),
                    "validated": True,
                    "confidence": validation_result["confidence"],
                    "source": validation_result.get("source"),
                    "table": validation_result.get("table", original_table),
                    "column": validation_result.get("column", original_column)
                }
            else:
                print(f"  ⚠️ Could not validate {entity_type}: '{entity_value}'")
                validated_entities[entity_type] = validation_result
        
        resolved["entities"] = validated_entities
        
        # Update session memory
        updated_session_entities = session_entities.copy()
        
        for entity_type, entity_data in validated_entities.items():
            session_key = f"last_{entity_type}"
            updated_session_entities[session_key] = entity_data
            entity_val = entity_data.get('value') if isinstance(entity_data, dict) else entity_data
            print(f"💾 Stored {entity_type}: {entity_val}")
        
        if resolved.get("table"):
            updated_session_entities["last_table"] = resolved["table"]
        
        if resolved.get("columns"):
            updated_session_entities["last_columns"] = resolved["columns"]
    
    # Load schema
    try:
        with open("annotated_schema.md", "r", encoding="utf-8") as f:
            annotated_schema = f.read()
    except FileNotFoundError:
        annotated_schema = "Schema not found"
    
    # Update query history
    updated_history = query_history + [user_query]
    if len(updated_history) > 10:
        updated_history = updated_history[-10:]
    
    print(f"\n✅ Resolution complete:")
    print(f"   Intent: {resolved.get('intent')}")
    print(f"   Entities: {list(resolved.get('entities', {}).keys())}")
    for entity_type, entity_data in resolved.get('entities', {}).items():
        if isinstance(entity_data, dict):
            val = entity_data.get('value', 'N/A')
            table = entity_data.get('table', 'N/A')
            column = entity_data.get('column', 'N/A')
            print(f"      ✅ {entity_type}: {val} ({table}.{column})")
    print(f"   Filters: {list(resolved.get('filters', {}).keys())}")
    print(f"   Table: {resolved.get('table')}")
    
    return {
        "resolved": resolved,
        "annotated_schema": annotated_schema,
        "session_entities": updated_session_entities,
        "query_history": updated_history,
        "last_product": resolved.get("entities", {}).get("product", {}).get("value") if isinstance(resolved.get("entities", {}).get("product"), dict) else resolved.get("entities", {}).get("product"),
        "last_distributor": resolved.get("entities", {}).get("distributor", {}).get("value") if isinstance(resolved.get("entities", {}).get("distributor"), dict) else resolved.get("entities", {}).get("distributor"),
        "last_superstockist": resolved.get("entities", {}).get("superstockist", {}).get("value") if isinstance(resolved.get("entities", {}).get("superstockist"), dict) else resolved.get("entities", {}).get("superstockist"),
        "last_sold_to_party": resolved.get("entities", {}).get("sold_to_party", {}).get("value") if isinstance(resolved.get("entities", {}).get("sold_to_party"), dict) else resolved.get("entities", {}).get("sold_to_party"),
        "last_table": resolved.get("table"),
        "last_columns": resolved.get("columns", [])
    }


# ---------- Build Graph with Memory ----------
def create_graph_with_memory():
    """Create the LangGraph with MemorySaver for session persistence"""
    
    graph = StateGraph(GraphState)
    
    # Add all nodes
    graph.add_node("question_validator", question_validator)
    graph.add_node("entity_resolver", entity_resolver_with_memory)
    graph.add_node("sql_generator", sql_agent_node)
    graph.add_node("validator_sql", validator_agent)
    graph.add_node('executor_sql', sql_executor_node)
    graph.add_node("summarized_results", summarized_results_node)
    graph.add_node("chart_creation_node", chart_creation_node)
    
    # Set entry point
    graph.set_entry_point("question_validator")
    
    # Routing logic
    def route_question(state):
        if state.get("route_decision") == "entity_resolver":
            return "entity_resolver"
        else:
            return "summarized_results"
    
    graph.add_conditional_edges(
        "question_validator",
        route_question,
        {
            "entity_resolver": "entity_resolver",
            "summarized_results": "summarized_results"
        }
    )
    
    # Add edges
    graph.add_edge("entity_resolver", "sql_generator")
    graph.add_edge("sql_generator", "validator_sql")
    graph.add_edge("validator_sql", "executor_sql")
    graph.add_edge("executor_sql", "summarized_results")
    graph.add_edge("summarized_results", "chart_creation_node")
    graph.add_edge("chart_creation_node", END)
    
    # Compile with MemorySaver
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    
    return compiled


# ---------- Enhanced Session Management ----------
class SessionManager:
    """Manage conversation sessions with persistent memory"""
    
    def __init__(self, graph, catalog, table_columns, annotated_schema, relationships):
        self.graph = graph
        self.catalog = catalog
        self.table_columns = table_columns
        self.annotated_schema = annotated_schema
        self.relationships = relationships
        self.sessions = {}
    
    def create_session(self, session_id: str):
        """Create a new session"""
        self.sessions[session_id] = {
            "created_at": __import__('datetime').datetime.now(),
            "query_count": 0,
            "entities": {}
        }
        print(f"\n🆕 Created new session: {session_id}")
    
    def invoke(self, user_query: str, session_id: str = "default"):
        """Invoke graph with session memory"""
        
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        self.sessions[session_id]["query_count"] += 1
        
        print(f"\n{'='*70}")
        print(f"📊 Session: {session_id} | Query #{self.sessions[session_id]['query_count']}")
        print(f"{'='*70}")
        
        config = {"configurable": {"thread_id": session_id}}
        
        result = self.graph.invoke(
            {
                "user_query": user_query,
                "catalog": self.catalog,
                "table_columns": self.table_columns,
                "annotated_schema": self.annotated_schema,
                "relationships": self.relationships
            },
            config=config
        )
        
        # Store entities in session
        if result.get("session_entities"):
            self.sessions[session_id]["entities"] = result["session_entities"]
        
        return result
    
    def show_session_memory(self, session_id: str):
        """Display current session memory"""
        context = self.sessions.get(session_id)
        if context:
            print(f"\n📝 Session {session_id} Memory:")
            print(f"   Queries: {context['query_count']}")
            if context.get("entities"):
                print("   Stored Entities:")
                for key, val in context["entities"].items():
                    if key.startswith("last_") and val:
                        if isinstance(val, dict) and 'value' in val:
                            print(f"      - {key}: {val['value']}")
                        elif val:
                            print(f"      - {key}: {val}")
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"🗑️ Cleared session: {session_id}")


# ---------- Interactive Loop ----------
if __name__ == "__main__":
    # Load tables and catalog
    table_columns = load_table_columns_pg(
        conn, 
        ["tbl_shipment", "tbl_primary", "tbl_product_master"]
    )
    catalog = build_catalog(conn, table_columns)
    
    print("\n📚 Catalog loaded")
    
    # Create graph with memory
    graph = create_graph_with_memory()
    
    # Create session manager
    session_manager = SessionManager(
        graph=graph,
        catalog=catalog,
        table_columns=table_columns,
        annotated_schema=annotated_schema,
        relationships=relationships
    )
    
    # Interactive conversation loop
    print("\n" + "="*70)
    print("Text-to-SQL Interactive System with Memory")
    print("="*70)
    print("Commands:")
    print("  - Type your question to query the database")
    print("  - 'new session' - Start a new conversation session")
    print("  - 'show memory' - Display current session context")
    print("  - 'clear' - Clear current session memory")
    print("  - 'exit' - Quit the program")
    print("="*70)
    print("\n💡 TIP: After asking about an entity, just ask about metrics!")
    print("   Example: 'sales of V H Trading' then 'what about last month?'\n")
    
    session_id = "default"
    query_number = 0
    
    while True:
        print(f"\nSession: {session_id} | Query #{query_number + 1}")
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'new session':
            session_id = input("Enter new session ID (or press Enter for auto-generated): ").strip()
            if not session_id:
                import uuid
                session_id = str(uuid.uuid4())[:8]
            query_number = 0
            print(f"Started new session: {session_id}")
            continue
        
        if user_input.lower() == 'show memory':
            session_manager.show_session_memory(session_id)
            continue
        
        if user_input.lower() == 'clear':
            session_manager.clear_session(session_id)
            session_id = "default"
            query_number = 0
            print("Memory cleared, starting fresh!")
            continue
        
        query_number += 1
        
        try:
            result = session_manager.invoke(
                user_query=user_input,
                session_id=session_id
            )
            
            print("\n" + "-"*70)
            print("RESULT:")
            print("-"*70)
            print(result.get('final_output', 'No output generated'))
            print("-"*70)
            
        except Exception as e:
            print(f"\nError processing query: {e}")
            import traceback
            traceback.print_exc()