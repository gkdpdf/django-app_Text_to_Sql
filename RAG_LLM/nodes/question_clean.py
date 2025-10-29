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


def question_validator(state: GraphState):
    """
    Enhanced question validator with fast-path routing for database queries
    """
    user_query = state['user_query']
    
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
    
    # ========== FAST-PATH: Database Query Keywords ==========
    # These indicate clear database queries - route immediately
    database_keywords = [
        # Metrics & Aggregations
        'sales', 'revenue', 'total', 'sum', 'count', 'average', 'avg',
        'maximum', 'max', 'minimum', 'min', 'quantity', 'amount', 'value',
        
        # Actions
        'show', 'get', 'fetch', 'retrieve', 'find', 'list', 'display',
        'give', 'tell', 'what is', 'what are', 'how many', 'how much',
        
        # Entities (Haldiram specific)
        'product', 'distributor', 'superstockist', 'customer', 'party',
        'sold to', 'shipment', 'invoice', 'order', 'bhujia', 'namkeen',
        'sev', 'mixture', 'chips', 'papad',
        
        # Comparisons & Rankings
        'top', 'bottom', 'best', 'worst', 'highest', 'lowest', 'rank',
        'compare', 'comparison', 'versus', 'vs',
        
        # Time references
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'month', 'year', 'week', 'day', 'today', 'yesterday', 'last',
        'this', 'current', 'previous', 'recent', 'latest',
        
        # Analysis
        'trend', 'analysis', 'performance', 'growth', 'report', 'summary'
    ]
    
    has_database_keyword = any(keyword in query_lower for keyword in database_keywords)
    
    if has_database_keyword:
        print("✅ Fast-path: Database keyword detected")
        print("   Routing to: entity_resolver")
        return {
            "route_decision": "entity_resolver",
            "validation_status": "valid_query"
        }
    
    # ========== FAST-PATH: Non-Database Queries ==========
    # Greetings
    greeting_patterns = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'greetings'
    ]
    
    if any(query_lower.startswith(pattern) for pattern in greeting_patterns):
        print("👋 Greeting detected")
        return {
            "route_decision": "summarized_results",
            "final_output": "Hello! I'm your Haldiram database assistant. I can help you query sales data, products, distributors, superstockists, and more. What would you like to know?",
            "validation_status": "greeting"
        }
    
    # Help requests
    help_patterns = ['help', 'what can you do', 'how does this work', 'capabilities']
    
    if any(pattern in query_lower for pattern in help_patterns):
        print("❓ Help request detected")
        return {
            "route_decision": "summarized_results",
            "final_output": """I can help you query the Haldiram database. Here are some examples:

📊 **Sales Queries:**
   • "What are the total sales of Bhujia?"
   • "Show me sales for XYZ distributor in May"
   • "Top 10 products by revenue"
   • "Sales of sb marke in March"

📅 **Time-based Queries:**
   • "Sales in last 3 months"
   • "Monthly trend for ABC product"
   • "Compare this month vs last month"

🏢 **Entity Queries:**
   • "List all products"
   • "Show distributors in Delhi"
   • "Superstockist performance"

Just ask naturally - I'll understand abbreviated names and partial matches!""",
            "validation_status": "help_request"
        }
    
    # ========== LLM VALIDATION (For Ambiguous Cases) ==========
    print("🤔 Using LLM for ambiguous query classification...")
    
    validation_prompt = f"""You are validating queries for a Haldiram sales database system.

Database Domain:
- Sales data (products, quantities, revenue)
- Products (Bhujia, Namkeen, Sev, Chips, etc.)
- Distributors and Superstockists (companies, parties)
- Shipments and invoices
- Time-based sales records

User Query: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Be VERY INCLUSIVE - if there's ANY chance this relates to data, mark as VALID
2. Even abbreviated or partial entity names should be VALID (e.g., "sb marke", "xyz company")
3. Even simple data requests should be VALID (e.g., "show X", "get Y", "total Z")
4. Only mark INVALID if it's clearly unrelated (weather, jokes, math, cooking, etc.)

Examples of VALID queries:
✅ "sales of sb marke" - asking for sales data
✅ "total bhujia" - abbreviated but clearly data request
✅ "xyz company revenue" - entity + metric
✅ "show products" - data retrieval
✅ "march sales" - time-based data
✅ "top 5" - ranking query
✅ "how much did we sell" - sales question
✅ "distributor performance" - analytics

Examples of INVALID queries:
❌ "what's the weather" - not database related
❌ "tell me a joke" - entertainment
❌ "how to cook pasta" - cooking instructions
❌ "what's 2+2" - math calculation
❌ "who is the president" - general knowledge

Respond with ONLY ONE WORD:
- "VALID" if it's a database/data query
- "INVALID" if it's completely unrelated to the database

Your response:"""
    
    try:
        validation_llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4o-mini",  # Changed from gpt-3.5-turbo for better accuracy
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
        
        # In case of error, default to entity_resolver (safer than rejecting)
        return {
            "route_decision": "entity_resolver",
            "validation_status": "valid_query_default"
        }


# ========== Testing Function ==========
if __name__ == "__main__":
    """Test the question validator with various queries"""
    
    test_cases = [
        # Should be VALID
        ("sales of sb marke", "VALID"),
        ("total sales of bhujia", "VALID"),
        ("show me xyz company revenue", "VALID"),
        ("march sales", "VALID"),
        ("top 10 products", "VALID"),
        ("distributor performance", "VALID"),
        ("how much did we sell", "VALID"),
        
        # Should be INVALID
        ("what's the weather", "INVALID"),
        ("tell me a joke", "INVALID"),
        ("how to cook pasta", "INVALID"),
        
        # Edge cases
        ("hi", "GREETING"),
        ("help", "HELP"),
    ]
    
    print("\n" + "="*70)
    print("TESTING QUESTION VALIDATOR")
    print("="*70)
    
    for query, expected in test_cases:
        state = {"user_query": query}
        result = question_validator(state)
        
        actual = result.get("validation_status", "unknown")
        route = result.get("route_decision")
        
        status = "✅" if (
            (expected == "VALID" and route == "entity_resolver") or
            (expected == "INVALID" and route == "summarized_results") or
            (expected in actual)
        ) else "❌"
        
        print(f"\n{status} Query: '{query}'")
        print(f"   Expected: {expected}, Got: {actual}, Route: {route}")