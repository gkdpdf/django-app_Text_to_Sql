from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
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
    validation_status: str               
    validation_error: Optional[str]
    execution_result: Any                
    execution_status: str               
    execution_error: Optional[str]
    route_decision: str                
    final_output: str                    
    reasoning_trace: List[str]
    module_config: Optional[Dict[str, Any]]


def question_validator(state: GraphState):
    """
    Enhanced question validator - FULLY DYNAMIC from knowledge graph
    """
    user_query = state['user_query']
    module_config = state.get('module_config', {})
    
    print(f"\n{'='*70}")
    print(f"🔍 QUESTION VALIDATOR")
    print(f"{'='*70}")
    print(f"Query: {user_query}")
    
    # Quick validation for empty queries
    if not user_query or not user_query.strip():
        return {
            "route_decision": "summarized_results",
            "final_output": "Please provide a query.",
            "validation_status": "invalid_query"
        }
    
    query_lower = user_query.lower().strip()
    
    # ========== BUILD DYNAMIC KEYWORDS FROM MODULE CONFIG ==========
    database_keywords = []
    
    # From metrics
    metrics = module_config.get('metrics', {})
    if metrics:
        for metric_name in metrics.keys():
            database_keywords.extend(metric_name.lower().split())
    
    # From POS tagging (entity types)
    pos_tagging = module_config.get('pos_tagging', [])
    if pos_tagging:
        for pos in pos_tagging:
            if pos.get('name'):
                database_keywords.append(pos['name'].lower())
            if pos.get('reference'):
                database_keywords.extend(pos['reference'].lower().split(','))
    
    # From table names
    tables = module_config.get('tables', [])
    for table in tables:
        database_keywords.append(table.lower())
    
    # Generic data keywords (minimal hardcoding)
    generic_keywords = [
        'show', 'get', 'fetch', 'retrieve', 'find', 'list', 'display',
        'give', 'tell', 'what is', 'what are', 'how many', 'how much',
        'total', 'sum', 'count', 'average', 'avg', 'max', 'min',
        'top', 'bottom', 'best', 'worst', 'highest', 'lowest',
        'compare', 'comparison', 'trend', 'analysis', 'report'
    ]
    
    database_keywords.extend(generic_keywords)
    
    # Remove duplicates
    database_keywords = list(set(database_keywords))
    
    print(f"📋 Dynamic keywords loaded: {len(database_keywords)} keywords")
    
    # ========== FAST-PATH: Database Query Keywords ==========
    has_database_keyword = any(keyword in query_lower for keyword in database_keywords)
    
    if has_database_keyword:
        print("✅ Fast-path: Database keyword detected")
        print("   Routing to: entity_resolver")
        return {
            "route_decision": "entity_resolver",
            "validation_status": "valid_query"
        }
    
    # ========== FAST-PATH: Non-Database Queries ==========
    greeting_patterns = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'greetings'
    ]
    
    if any(query_lower.startswith(pattern) for pattern in greeting_patterns):
        print("👋 Greeting detected")
        
        # Build dynamic greeting response
        module_name = module_config.get('module_name', 'database')
        greeting_msg = f"Hello! I'm your {module_name} assistant. I can help you query data"
        
        if metrics:
            metric_examples = list(metrics.keys())[:3]
            greeting_msg += f" about {', '.join(metric_examples)}"
        
        greeting_msg += ". What would you like to know?"
        
        return {
            "route_decision": "summarized_results",
            "final_output": greeting_msg,
            "validation_status": "greeting"
        }
    
    # Help requests
    help_patterns = ['help', 'what can you do', 'how does this work', 'capabilities']
    
    if any(pattern in query_lower for pattern in help_patterns):
        print("❓ Help request detected")
        
        # Build dynamic help message
        help_msg = f"I can help you query the {module_config.get('module_name', 'database')}. "
        
        if metrics:
            help_msg += f"\n\n📊 **Available Metrics:**\n"
            for metric_name, metric_desc in list(metrics.items())[:5]:
                help_msg += f"   • {metric_name}: {metric_desc}\n"
        
        if pos_tagging:
            help_msg += f"\n\n🏷️ **Entity Types:**\n"
            for pos in pos_tagging[:5]:
                if pos.get('name') and pos.get('reference'):
                    help_msg += f"   • {pos['name']}: {pos['reference']}\n"
        
        help_msg += "\n\nJust ask naturally - I'll understand your queries!"
        
        return {
            "route_decision": "summarized_results",
            "final_output": help_msg,
            "validation_status": "help_request"
        }
    
    # ========== LLM VALIDATION (For Ambiguous Cases) ==========
    print("🤔 Using LLM for ambiguous query classification...")
    
    # Build dynamic domain description
    domain_desc = f"Database: {module_config.get('module_name', 'Unknown')}\n"
    
    if metrics:
        domain_desc += f"Metrics: {', '.join(list(metrics.keys())[:10])}\n"
    
    if pos_tagging:
        entity_types = [pos['name'] for pos in pos_tagging if pos.get('name')]
        domain_desc += f"Entities: {', '.join(entity_types[:10])}\n"
    
    if tables:
        domain_desc += f"Tables: {', '.join(tables[:5])}\n"
    
    validation_prompt = f"""You are validating queries for a database system.

{domain_desc}

User Query: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Be VERY INCLUSIVE - if there's ANY chance this relates to data, mark as VALID
2. Even abbreviated or partial entity names should be VALID
3. Even simple data requests should be VALID
4. Only mark INVALID if it's clearly unrelated (weather, jokes, math, cooking, etc.)

Respond with ONLY ONE WORD:
- "VALID" if it's a database/data query
- "INVALID" if it's completely unrelated to the database

Your response:"""
    
    try:
        validation_llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4o-mini",
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        result = validation_llm.invoke(validation_prompt)
        validation_result = result.content.strip().upper()
        
        print(f"📋 LLM Response: {validation_result}")
        
        if "VALID" in validation_result:
            print("✅ Routing to: entity_resolver")
            return {
                "route_decision": "entity_resolver",
                "validation_status": "valid_query"
            }
        else:
            print("❌ Not a database query")
            return {
                "route_decision": "summarized_results", 
                "final_output": "I can only help with database queries. Please ask about retrieving or analyzing data.",
                "validation_status": "invalid_query"
            }
    
    except Exception as e:
        print(f"⚠️ LLM validation error: {e}")
        print("⚠️ Defaulting to VALID (safe routing to entity_resolver)")
        
        return {
            "route_decision": "entity_resolver",
            "validation_status": "valid_query_default"
        }