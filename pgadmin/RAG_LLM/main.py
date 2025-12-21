"""
LangGraph Workflow - COMPLETE WITH NON-DATA QUERY HANDLING
- Handles greetings and non-data queries gracefully
- LLM-based entity extraction
- Column selection FIRST with KG descriptions
- Temporal filtering (last 6 months, daily, weekly, monthly)
- Full context usage (KG, Relationships, RCA, POS, Metrics)
"""
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from openai import OpenAI
from decimal import Decimal
import datetime
import json
from datetime import datetime as dt, timedelta

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# 🔒 QUERY CLASSIFIER - Enhanced
# ============================================================

def classify_query(user_query):
    """
    Classify the user query into categories:
    - greeting: hi, hello, hey, etc.
    - non_data: weather, news, jokes, general knowledge
    - data: actual data/analytics questions
    """
    if not user_query or len(user_query.strip()) < 2:
        return "invalid"
    
    q = user_query.lower().strip()
    
    # Greetings
    greetings = ['hi', 'hello', 'hey', 'hii', 'hiii', 'good morning', 'good afternoon', 
                 'good evening', 'howdy', 'greetings', 'sup', 'yo', 'hola']
    if q in greetings or any(q.startswith(g + ' ') for g in greetings) or any(q.startswith(g + ',') for g in greetings):
        return "greeting"
    
    # Non-data queries (general knowledge, chitchat)
    non_data_keywords = [
        'weather', 'news', 'movie', 'song', 'cricket', 'football', 'sports',
        'joke', 'story', 'poem', 'recipe', 'travel', 'holiday',
        'who is', 'what is the capital', 'how to cook', 'tell me about',
        'define', 'meaning of', 'translate', 'calculate', 'math',
        'thank you', 'thanks', 'bye', 'goodbye', 'see you'
    ]
    
    if any(kw in q for kw in non_data_keywords):
        return "non_data"
    
    # Check if it looks like a data query
    data_keywords = [
        'sales', 'revenue', 'quantity', 'total', 'sum', 'average', 'count',
        'product', 'distributor', 'region', 'zone', 'state', 'customer',
        'order', 'invoice', 'monthly', 'daily', 'weekly', 'yearly',
        'last', 'this', 'previous', 'trend', 'compare', 'top', 'bottom',
        'highest', 'lowest', 'best', 'worst', 'mrp', 'price', 'value'
    ]
    
    if any(kw in q for kw in data_keywords):
        return "data"
    
    # Default: assume data query if not clearly non-data
    # This allows queries like "bhujia" or "north zone" to work
    return "data"


def get_non_data_response(query_type, user_query):
    """Generate appropriate response for non-data queries"""
    
    if query_type == "greeting":
        return {
            "type": "response",
            "message": "👋 Hello! I'm your data assistant. I can help you query and analyze your sales data.\n\nTry asking questions like:\n• \"What are the total sales of Bhujia?\"\n• \"Show me monthly sales for North zone\"\n• \"Top 10 distributors by revenue\"",
            "sql": None,
            "data": None
        }
    
    if query_type == "non_data":
        return {
            "type": "response", 
            "message": "🤖 I'm specialized in analyzing your business data. I can't help with general questions, but I'd be happy to help you with:\n\n• Sales analysis\n• Product performance\n• Distributor insights\n• Regional comparisons\n• Trend analysis\n\nPlease ask a question related to your data!",
            "sql": None,
            "data": None
        }
    
    if query_type == "invalid":
        return {
            "type": "response",
            "message": "❓ I didn't quite understand that. Please ask a question about your data, like:\n\n• \"Total sales of [product name]\"\n• \"Sales in [region/zone] last month\"\n• \"Compare distributor performance\"",
            "sql": None,
            "data": None
        }
    
    return None


# ============================================================
# 🔌 DATABASE CONNECTION
# ============================================================

def get_db_connection():
    """Get database connection"""
    from pgadmin.RAG_LLM.django_loader import get_db_credentials
    db_creds = get_db_credentials()
    conn = psycopg2.connect(**db_creds)
    conn.autocommit = False
    return conn


# ============================================================
# 📊 LOAD CATALOG VALUES
# ============================================================

def load_catalog_values(catalog):
    """Load unique values for each column"""
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


# ============================================================
# 🔍 FUZZY MATCHING
# ============================================================

def find_fuzzy_matches(search_term, values):
    """Find matches with fuzzy logic"""
    if not search_term or not values:
        return []
    
    search_lower = search_term.lower().strip()
    exact_matches = []
    partial_matches = []
    word_matches = []
    
    for value in values:
        if value is None:
            continue
        
        value_str = str(value).lower().strip()
        
        if search_lower == value_str:
            exact_matches.append(value)
        elif search_lower in value_str:
            partial_matches.append(value)
        else:
            search_words = search_lower.split()
            value_words = value_str.split()
            if any(sw in vw for sw in search_words for vw in value_words):
                word_matches.append(value)
    
    return exact_matches + partial_matches + word_matches


# ============================================================
# ⏰ TEMPORAL EXTRACTION
# ============================================================

def extract_temporal_info(user_query):
    """Extract temporal requirements from query"""
    import re
    query_lower = user_query.lower()
    
    temporal_info = {
        'grouping': None,
        'period': None,
        'has_temporal': False,
        'interval_value': None,
        'interval_unit': None,
        'needs_date_column': False
    }
    
    # Detect grouping
    if any(word in query_lower for word in ['daily', 'day-wise', 'per day', 'each day']):
        temporal_info['grouping'] = 'daily'
        temporal_info['has_temporal'] = True
        temporal_info['needs_date_column'] = True
    elif any(word in query_lower for word in ['weekly', 'week-wise', 'per week', 'each week']):
        temporal_info['grouping'] = 'weekly'
        temporal_info['has_temporal'] = True
        temporal_info['needs_date_column'] = True
    elif any(word in query_lower for word in ['monthly', 'month-wise', 'per month', 'each month']):
        temporal_info['grouping'] = 'monthly'
        temporal_info['has_temporal'] = True
        temporal_info['needs_date_column'] = True
    elif any(word in query_lower for word in ['yearly', 'year-wise', 'per year', 'each year']):
        temporal_info['grouping'] = 'yearly'
        temporal_info['has_temporal'] = True
        temporal_info['needs_date_column'] = True
    
    # Detect period
    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
        'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'twelve': 12
    }
    
    patterns = [
        r'(?:last|past|previous)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(month|week|day|year)s?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            num_str = match.group(1)
            unit = match.group(2)
            
            num_value = number_words.get(num_str, int(num_str) if num_str.isdigit() else 1)
            
            temporal_info['interval_value'] = num_value
            temporal_info['interval_unit'] = unit + 's'
            temporal_info['has_temporal'] = True
            temporal_info['needs_date_column'] = True
            
            if unit == 'month':
                temporal_info['period'] = f'last_{num_value}_months'
            elif unit == 'week':
                temporal_info['period'] = f'last_{num_value}_weeks'
            elif unit == 'day':
                temporal_info['period'] = f'last_{num_value}_days'
            elif unit == 'year':
                temporal_info['period'] = f'last_{num_value}_years'
            
            break
    
    # Detect specific months
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    for month in months:
        if month in query_lower:
            temporal_info['has_temporal'] = True
            temporal_info['needs_date_column'] = True
            temporal_info['specific_month'] = month
            break
    
    return temporal_info


def extract_aggregation_info(user_query):
    """Extract aggregation requirements"""
    query_lower = user_query.lower()
    
    agg_info = {'type': 'total', 'unit': None}
    
    if any(word in query_lower for word in ['average', 'avg', 'mean']):
        agg_info['type'] = 'average'
    elif any(word in query_lower for word in ['count', 'number of', 'how many']):
        agg_info['type'] = 'count'
    
    return agg_info


# ============================================================
# 📅 DATE COLUMN FINDER
# ============================================================

def find_date_columns(catalog, config):
    """Find all potential date columns in the schema"""
    date_columns = []
    
    kg_data = config.get('knowledge_graph_data', {})
    
    for table, columns in catalog.items():
        for col_name, col_data in columns.items():
            col_type = col_data.get('type', '').lower()
            col_name_lower = col_name.lower()
            
            if any(dt in col_type for dt in ['date', 'time', 'timestamp']):
                desc = ""
                if table in kg_data and col_name in kg_data[table]:
                    desc = kg_data[table][col_name].get('desc', '')
                
                date_columns.append({
                    'table': table,
                    'column': col_name,
                    'type': col_type,
                    'description': desc,
                    'full_name': f"{table}.{col_name}"
                })
            elif any(kw in col_name_lower for kw in ['date', 'time', 'day', 'month', 'year']):
                desc = ""
                if table in kg_data and col_name in kg_data[table]:
                    desc = kg_data[table][col_name].get('desc', '')
                
                date_columns.append({
                    'table': table,
                    'column': col_name,
                    'type': col_type,
                    'description': desc,
                    'full_name': f"{table}.{col_name}"
                })
    
    return date_columns


# ============================================================
# 🔍 ENTITY RESOLVER - Main Logic
# ============================================================

def entity_resolver(user_query, catalog, session_data, feedback=None, module_id=None):
    """Resolve entities with clarification flow"""
    
    print("\n" + "="*80)
    print("🔍 ENTITY_RESOLVER START")
    print("="*80)
    print(f"Query: {user_query[:60]}..." if len(user_query) > 60 else f"Query: {user_query}")
    print(f"Feedback: {feedback.get('type') if feedback else 'None'}")
    print("="*80 + "\n")
    
    # Store original query in session
    if user_query and not feedback:
        session_data['original_user_query'] = user_query
    
    # Extract temporal/agg info
    temporal_info = extract_temporal_info(user_query if user_query else session_data.get('original_user_query', ''))
    agg_info = extract_aggregation_info(user_query if user_query else session_data.get('original_user_query', ''))
    session_data['temporal_info'] = temporal_info
    session_data['agg_info'] = agg_info
    
    # Load config
    config = {}
    if module_id:
        try:
            from pgadmin.RAG_LLM.django_loader import load_module_config
            config = load_module_config(module_id)
            session_data['config'] = config
        except Exception as e:
            print(f"⚠️ Config load error: {e}")
    
    # === DATE COLUMN SELECTION ===
    if feedback and feedback.get('type') == 'date_column_selection':
        selected_option = feedback.get('selected_option', '')
        column_full_name = selected_option.split('\n')[0].split(' - ')[0].strip()
        table, column = column_full_name.split('.')
        
        session_data['selected_date_column'] = {
            'table': table,
            'column': column,
            'full_name': column_full_name
        }
        
        original_query = session_data.get('original_user_query', user_query)
        return entity_resolver(original_query, catalog, session_data, None, module_id)
    
    # === VALUE SELECTION ===
    if feedback and feedback.get('type') == 'value_selection':
        selected_value = feedback.get('selected_option')
        entity_type = feedback.get('entity_type')
        context = feedback.get('clarification_context', {})
        multiple_values = feedback.get('multiple_values')
        
        table = context.get('table')
        column = context.get('column')
        
        if not table or not column:
            if 'last_clarification' in session_data:
                table = session_data['last_clarification'].get('table')
                column = session_data['last_clarification'].get('column')
        
        if 'resolved_entities' not in session_data:
            session_data['resolved_entities'] = {}
        
        if multiple_values and len(multiple_values) > 1:
            session_data['resolved_entities'][entity_type] = {
                "table": table,
                "column": column,
                "values": multiple_values,
                "is_multiple": True
            }
        else:
            session_data['resolved_entities'][entity_type] = {
                "table": table,
                "column": column,
                "value": selected_value
            }
        
        return {
            "needs_clarification": False,
            "entities": session_data['resolved_entities'],
            "intent": session_data.get('intent', 'total')
        }
    
    # === COLUMN SELECTION ===
    if feedback and feedback.get('type') == 'column_selection':
        selected_option = feedback.get('selected_option', '')
        entity_type = feedback.get('entity_type')
        entity_value = feedback.get('entity')
        context = feedback.get('clarification_context', {})
        
        matches_by_column = context.get('matches_by_column', {})
        if not matches_by_column and 'last_matches_by_column' in session_data:
            matches_by_column = session_data['last_matches_by_column']
        
        column_key = selected_option.split('\n')[0].split(' (')[0].strip()
        
        if column_key not in matches_by_column:
            # Re-find matches
            all_matches = []
            for table, columns in catalog.items():
                for col_name, col_data in columns.items():
                    matches = find_fuzzy_matches(entity_value, col_data.get('values', []))
                    if matches:
                        all_matches.append({'table': table, 'column': col_name, 'matches': matches})
            
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
        
        if column_key not in matches_by_column:
            return {"type": "error", "message": "Column not found"}
        
        col_data = matches_by_column[column_key]
        table = col_data['table']
        column = col_data['column']
        values = col_data['values']
        
        kg_data = config.get('knowledge_graph_data', {})
        col_desc = ""
        if table in kg_data and column in kg_data[table]:
            col_desc = kg_data[table][column].get('desc', '')
        
        session_data['last_clarification'] = {'table': table, 'column': column}
        
        if len(values) > 1:
            message = f"Found {len(values)} values in {column}"
            if col_desc:
                message += f" ({col_desc})"
            message += ". Please select one or more:"
            
            return {
                "needs_clarification": True,
                "message": message,
                "options": [str(v) for v in values[:20]],
                "subtype": "value_selection",
                "entity": entity_value,
                "entity_type": entity_type,
                "clarification_context": {"table": table, "column": column},
                "allow_custom": True
            }
        elif len(values) == 1:
            if 'resolved_entities' not in session_data:
                session_data['resolved_entities'] = {}
            
            session_data['resolved_entities'][entity_type] = {
                "table": table,
                "column": column,
                "value": values[0]
            }
            
            return {
                "needs_clarification": False,
                "entities": session_data['resolved_entities'],
                "intent": session_data.get('intent', 'total')
            }
    
    # === CUSTOM VALUE ===
    if feedback and feedback.get('type') == 'custom_value':
        custom_value = feedback.get('custom_value')
        entity_type = feedback.get('entity_type')
        context = feedback.get('clarification_context', {})
        
        table = context.get('table')
        column = context.get('column')
        
        if table and column and table in catalog and column in catalog[table]:
            values = catalog[table][column].get('values', [])
            matches = find_fuzzy_matches(custom_value, values)
            
            if len(matches) > 1:
                return {
                    "needs_clarification": True,
                    "message": f"Found {len(matches)} matches for '{custom_value}'. Select one:",
                    "options": [str(m) for m in matches[:20]],
                    "subtype": "value_selection",
                    "entity": custom_value,
                    "entity_type": entity_type,
                    "clarification_context": context,
                    "allow_custom": True
                }
            elif len(matches) == 1:
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
    
    # === INITIAL QUERY - LLM Entity Extraction ===
    query_for_extraction = user_query if user_query else session_data.get('original_user_query', '')
    
    # Apply POS tagging to expand query with aliases
    pos_tagging = config.get('pos_tagging', [])
    expanded_query = query_for_extraction
    if pos_tagging:
        for pos in pos_tagging:
            name = pos.get('name', '').lower()
            reference = pos.get('reference', '')
            if name and reference and name in query_for_extraction.lower():
                # Add the reference as an alias
                expanded_query = f"{query_for_extraction} (also known as: {reference})"
                print(f"🏷️ POS: '{name}' → '{reference}'")
    
    # Check for date column selection first
    if temporal_info.get('needs_date_column') and 'selected_date_column' not in session_data:
        date_columns = find_date_columns(catalog, config)
        
        if len(date_columns) == 1:
            date_col = date_columns[0]
            session_data['selected_date_column'] = {
                'table': date_col['table'],
                'column': date_col['column'],
                'full_name': date_col['full_name']
            }
        elif len(date_columns) > 1:
            options = []
            for dc in date_columns:
                opt = f"{dc['full_name']} - {dc['type']}"
                if dc['description']:
                    opt += f"\n    📝 {dc['description']}"
                options.append(opt)
            
            return {
                "needs_clarification": True,
                "message": "Which date column should I use for this query?",
                "options": options,
                "subtype": "date_column_selection",
                "clarification_context": {"date_columns": date_columns},
                "allow_custom": False
            }
    
    # LLM Entity Extraction
    try:
        catalog_context = {}
        for table, columns in catalog.items():
            catalog_context[table] = {}
            for col, col_data in columns.items():
                catalog_context[table][col] = {
                    'type': col_data.get('type'),
                    'samples': col_data.get('values', [])[:10]
                }
        
        system_prompt = """Extract entities and intent from query.

Return JSON:
{
    "entities": [
        {"type": "product|customer|region|distributor|plant|category", "value": "...", "search_terms": ["..."]},
        ...
    ],
    "intent": "total|count|list|average|trend"
}

Extract ALL entities. Add synonyms to search_terms. Do NOT extract temporal terms like 'may', 'monthly', 'last 6 months' as entities."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {expanded_query}\n\nDatabase: {json.dumps(catalog_context, default=str)[:2000]}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        llm_result = json.loads(response.choices[0].message.content)
        extracted_entities = llm_result.get('entities', [])
        intent = llm_result.get('intent', 'total')
        
        session_data['intent'] = intent
        
        print(f"\n🔍 Extracted {len(extracted_entities)} entities:")
        for ent in extracted_entities:
            print(f"   - {ent['type']}: {ent['value']}")
        
        # Filter out temporal entities
        extracted_entities = [
            e for e in extracted_entities 
            if e.get('type') not in ['time', 'date', 'month', 'period']
            and e.get('value', '').lower() not in ['sales', 'may', 'june', 'monthly', 'daily', 'weekly']
        ]
        
        if not extracted_entities:
            return {"needs_clarification": False, "entities": {}, "intent": intent}
        
        resolved_entities = {}
        
        # Process first entity
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
            
            if len(matches_by_column) >= 1:
                options = []
                kg_data = config.get('knowledge_graph_data', {})
                
                for col_key, col_data in matches_by_column.items():
                    table = col_data['table']
                    column = col_data['column']
                    
                    col_desc = ""
                    col_datatype = ""
                    if table in kg_data and column in kg_data[table]:
                        col_desc = kg_data[table][column].get('desc', '')
                        col_datatype = kg_data[table][column].get('datatype', '')
                    
                    sample_values = ', '.join([str(v)[:25] for v in col_data['values'][:2]])
                    option_text = f"{col_key} ({col_data['count']} matches)"
                    
                    if col_desc:
                        option_text += f"\n    📝 {col_desc}"
                    if col_datatype:
                        option_text += f"\n    🏷️ {col_datatype}"
                    option_text += f"\n    📊 {sample_values}..."
                    
                    options.append(option_text)
                
                session_data['pending_entities'] = extracted_entities[extracted_entities.index(entity)+1:]
                session_data['resolved_entities'] = resolved_entities
                session_data['last_matches_by_column'] = matches_by_column
                
                message = f"I found '{entity.get('value')}' in {len(matches_by_column)} column{'s' if len(matches_by_column) > 1 else ''}. "
                message += "Which column?" if len(matches_by_column) > 1 else "Is this correct?"
                
                return {
                    "needs_clarification": True,
                    "message": message,
                    "options": options,
                    "subtype": "column_selection",
                    "entity": entity.get('value'),
                    "entity_type": entity.get('type'),
                    "clarification_context": {"matches_by_column": matches_by_column},
                    "allow_custom": False
                }
        
        return {"needs_clarification": False, "entities": resolved_entities, "intent": intent}
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"needs_clarification": False, "entities": {}, "intent": "total"}


# ============================================================
# 🧠 SQL GENERATION
# ============================================================

def generate_sql_with_llm(user_query, entities, config, intent="total", temporal_info=None, agg_info=None, selected_date_column=None, context_settings=None):
    """Generate SQL with full context including RCA, POS, Metrics, Extra Suggestions"""
    
    print("\n" + "="*80)
    print("🔧 SQL GENERATION")
    print("="*80)
    
    # Default context settings
    if context_settings is None:
        context_settings = {}
    
    use_kg = context_settings.get('use_kg', True)
    use_relationships = context_settings.get('use_relationships', True)
    use_rca = context_settings.get('use_rca', True)
    use_pos = context_settings.get('use_pos', True)
    use_metrics = context_settings.get('use_metrics', True)
    use_extra_suggestions = context_settings.get('use_extra_suggestions', True)
    
    table_columns = config.get('table_columns', {})
    kg_data = config.get('knowledge_graph_data', {}) if use_kg else {}
    relationships = config.get('relationships', []) if use_relationships else []
    
    # Load RCA context
    rca_list = config.get('rca_list', []) if use_rca else []
    rca_context = ""
    if rca_list:
        rca_context = "\n\nBUSINESS RULES (RCA):\n"
        for rca in rca_list:
            title = rca.get('title', '')
            content = rca.get('content', '')
            if title or content:
                rca_context += f"• {title}: {content}\n"
        rca_context += "(Use these rules to guide calculations, but keep the query simple and executable)\n"
        print(f"📋 RCA Rules loaded: {len(rca_list)} rules")
    
    # Load POS tagging context
    pos_tagging = config.get('pos_tagging', []) if use_pos else []
    pos_context = ""
    if pos_tagging:
        pos_context = "\n\nPOS TAGGING (Entity Aliases):\n"
        for pos in pos_tagging:
            name = pos.get('name', '')
            reference = pos.get('reference', '')
            if name and reference:
                pos_context += f"• '{name}' refers to '{reference}'\n"
        print(f"🏷️ POS Tags loaded: {len(pos_tagging)} tags")
    
    # Load Metrics context
    metrics_data = config.get('metrics_data', {}) if use_metrics else {}
    metrics_context = ""
    if metrics_data:
        metrics_context = "\n\nMETRICS DEFINITIONS:\n"
        for metric_name, metric_desc in metrics_data.items():
            if metric_name and metric_desc:
                metrics_context += f"• {metric_name}: {metric_desc}\n"
        print(f"📊 Metrics loaded: {len(metrics_data)} metrics")
    
    # Load Extra Suggestions
    extra_suggestions = config.get('extra_suggestions', '') if use_extra_suggestions else ''
    extra_context = ""
    if extra_suggestions:
        extra_context = f"\n\nADDITIONAL INSTRUCTIONS:\n{extra_suggestions}\n"
        print(f"💡 Extra suggestions loaded")
    
    # Build schema
    schema_info = []
    for table, columns in table_columns.items():
        schema_info.append(f"\nTABLE: {table}")
        for col in columns:
            col_name = col['name']
            col_type = col['type']
            description = ''
            if table in kg_data and col_name in kg_data[table]:
                description = kg_data[table][col_name].get('desc', '')
            if description:
                schema_info.append(f"  • {col_name} ({col_type}) - {description}")
            else:
                schema_info.append(f"  • {col_name} ({col_type})")
    
    schema_text = "\n".join(schema_info)
    
    # Build WHERE
    where_conditions = []
    tables_used = set()
    
    for entity_type, entity_data in entities.items():
        table = entity_data.get('table')
        column = entity_data.get('column')
        
        if entity_data.get('is_multiple'):
            values = entity_data.get('values', [])
            if values and table and column:
                tables_used.add(table)
                escaped_values = []
                for v in values:
                    if isinstance(v, str):
                        escaped = v.replace("'", "''")
                        escaped_values.append(f"'{escaped}'")
                    else:
                        escaped_values.append(str(v))
                in_clause = f'{table}."{column}" IN ({", ".join(escaped_values)})'
                where_conditions.append(in_clause)
        else:
            value = entity_data.get('value')
            if table and column and value:
                tables_used.add(table)
                if isinstance(value, str):
                    escaped = value.replace("'", "''")
                    where_conditions.append(f'{table}."{column}" = \'{escaped}\'')
                else:
                    where_conditions.append(f'{table}."{column}" = {value}')
    
    # Temporal filter
    grouping_clause = ""
    if temporal_info and temporal_info.get('has_temporal') and selected_date_column:
        date_table = selected_date_column['table']
        date_col = selected_date_column['column']
        tables_used.add(date_table)
        
        if temporal_info.get('interval_value') and temporal_info.get('interval_unit'):
            val = temporal_info['interval_value']
            unit = temporal_info['interval_unit']
            where_conditions.append(f'{date_table}."{date_col}" >= CURRENT_DATE - INTERVAL \'{val} {unit}\'')
        
        if temporal_info.get('specific_month'):
            month_name = temporal_info['specific_month'].capitalize()
            month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
            month_num = month_map.get(month_name, 5)
            where_conditions.append(f'EXTRACT(MONTH FROM {date_table}."{date_col}") = {month_num}')
        
        if temporal_info.get('grouping'):
            grouping = temporal_info['grouping']
            if grouping == 'daily':
                grouping_clause = f'DATE({date_table}."{date_col}")'
            elif grouping == 'weekly':
                grouping_clause = f'DATE_TRUNC(\'week\', {date_table}."{date_col}")'
            elif grouping == 'monthly':
                grouping_clause = f'DATE_TRUNC(\'month\', {date_table}."{date_col}")'
            elif grouping == 'yearly':
                grouping_clause = f'DATE_TRUNC(\'year\', {date_table}."{date_col}")'
    
    where_clause = " AND ".join(where_conditions) if where_conditions else ""
    
    # Build comprehensive prompt with all context
    prompt = f"""Generate a SINGLE, SIMPLE PostgreSQL query for this request.

USER QUERY: {user_query}
INTENT: {intent}

SCHEMA:
{schema_text}

WHERE (MANDATORY): {where_clause}
TABLES: {', '.join(tables_used) if tables_used else 'All'}
GROUP BY: {grouping_clause if grouping_clause else 'None'}

RELATIONSHIPS:
{json.dumps(relationships, indent=2) if relationships else 'None'}
{rca_context}{pos_context}{metrics_context}{extra_context}
RULES:
1. Generate ONE simple, executable query
2. Use exact WHERE clause if provided
3. Double quotes for column names
4. If grouping needed: SELECT period_column as period, SUM(...) GROUP BY period_column
5. Add LIMIT 100 at the end
6. Use JOINs based on relationships when needed
7. For simple sales queries: just SUM the quantity or calculate quantity * price
8. Avoid overly complex CTEs unless absolutely necessary
9. Keep it simple - basic SELECT, JOIN, WHERE, GROUP BY, ORDER BY

Generate ONLY the SQL:
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "PostgreSQL expert. Generate a SINGLE executable query. Keep it simple - avoid complex CTEs with window functions unless explicitly requested. For sales queries, just use SUM with GROUP BY."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        sql = response.choices[0].message.content.strip()
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        # Validate SQL completeness
        sql_upper = sql.upper()
        
        # Check for incomplete CTE (WITH ... AS ( without closing )
        if sql_upper.startswith('WITH'):
            # Count opening and closing parentheses
            open_parens = sql.count('(')
            close_parens = sql.count(')')
            
            if open_parens != close_parens:
                print(f"⚠️ Incomplete CTE detected (parens: {open_parens} open, {close_parens} close)")
                print("🔄 Regenerating simpler query...")
                
                # Generate a simpler fallback query
                simple_prompt = f"""Generate a SIMPLE SQL query. NO CTEs, NO window functions.

USER QUERY: {user_query}
WHERE: {where_clause}
TABLES: {', '.join(tables_used) if tables_used else list(table_columns.keys())[0]}

Just use: SELECT SUM/COUNT/AVG, FROM, JOIN if needed, WHERE, GROUP BY, ORDER BY, LIMIT 100

SQL:"""
                
                response2 = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Generate simple SQL. No CTEs."},
                        {"role": "user", "content": simple_prompt}
                    ],
                    temperature=0
                )
                sql = response2.choices[0].message.content.strip()
                sql = sql.replace('```sql', '').replace('```', '').strip()
        
        # Handle case where LLM returned multiple separate queries
        if ';\nSELECT' in sql or ';\nWITH' in sql or '; SELECT' in sql or '; WITH' in sql:
            first_semicolon = sql.find(';')
            if first_semicolon > 0:
                rest = sql[first_semicolon+1:].strip()
                if rest.upper().startswith('SELECT') or rest.upper().startswith('WITH'):
                    sql = sql[:first_semicolon+1]
                    print("⚠️ Multiple queries detected - using first query only")
        
        # Add LIMIT if not present
        if 'LIMIT' not in sql_upper:
            sql = sql.rstrip(';').rstrip() + '\nLIMIT 100;'
        elif not sql.rstrip().endswith(';'):
            sql = sql.rstrip() + ';'
        
        print(f"\n✅ SQL:\n{sql}\n")
        return sql
        
    except Exception as e:
        print(f"❌ LLM error: {e}")
        main_table = list(tables_used)[0] if tables_used else list(table_columns.keys())[0]
        sql = f'SELECT * FROM {main_table}'
        if where_clause:
            sql += f'\nWHERE {where_clause}'
        sql += '\nLIMIT 100;'
        return sql


# ============================================================
# ▶️ EXECUTE SQL
# ============================================================

def execute_sql(sql):
    """Execute SQL and return results"""
    print(f"🔧 Executing SQL...")
    
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


# ============================================================
# 📝 FORMAT RESULTS
# ============================================================

def format_results(data, intent, temporal_info=None):
    """Format results for display"""
    if not data:
        return "No results found."
    
    first_row = data[0]
    
    # Temporal queries with multiple rows
    if temporal_info and temporal_info.get('has_temporal') and len(data) > 1:
        grouping = temporal_info.get('grouping', 'period')
        message = f"Found {len(data)} {grouping} records.\n\n"
        
        preview_count = min(5, len(data))
        for i, row in enumerate(data[:preview_count]):
            for key, value in row.items():
                if key.lower() != 'period' and value is not None:
                    period_val = row.get('period', f'Row {i+1}')
                    if isinstance(value, (int, float)):
                        message += f"• {period_val}: ₹{value:,.2f}\n"
                    else:
                        message += f"• {period_val}: {value}\n"
                    break
        
        if len(data) > preview_count:
            message += f"\n... and {len(data) - preview_count} more records"
        
        return message.strip()
    
    # Aggregations
    for key, value in first_row.items():
        key_lower = key.lower()
        
        if 'sum' in key_lower or 'total' in key_lower:
            if value is not None:
                if isinstance(value, (int, float)):
                    return f"Total: ₹{value:,.2f}"
                return f"Total: {value}"
        
        if 'avg' in key_lower or 'average' in key_lower:
            if value is not None:
                if isinstance(value, (int, float)):
                    return f"Average: ₹{value:,.2f}"
                return f"Average: {value}"
        
        if 'count' in key_lower:
            if value is not None:
                return f"Count: {value:,}"
    
    # Single row single value
    if len(data) == 1 and len(first_row) == 1:
        key = list(first_row.keys())[0]
        value = first_row[key]
        if value is not None:
            if isinstance(value, (int, float)):
                return f"{key}: ₹{value:,.2f}"
            return f"{key}: {value}"
    
    # Single row multiple values
    if len(data) == 1:
        message = ""
        for key, value in first_row.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    message += f"{key}: ₹{value:,.2f}\n"
                else:
                    message += f"{key}: {value}\n"
        return message.strip() if message else "Found 1 result"
    
    return f"Found {len(data)} results"


# ============================================================
# 🚀 MAIN ENTRY POINT
# ============================================================

def invoke_graph(user_query, module_id, session_data=None, feedback=None, stream_callback=None, context_settings=None):
    """Main entry point for the RAG engine"""
    
    print("\n" + "🔵"*40)
    print("🚀 INVOKE_GRAPH")
    print(f"   Query: {user_query}")
    print(f"   Feedback: {feedback.get('type') if feedback else 'None'}")
    print("🔵"*40 + "\n")
    
    # Initialize session
    if session_data is None:
        session_data = {}
    
    # === RESET SESSION FOR NEW QUERIES ===
    # If this is a new query (not feedback), reset the session state
    if not feedback and user_query:
        # Clear previous resolution state for fresh start
        keys_to_clear = [
            'resolved_entities', 
            'pending_entities', 
            'last_clarification',
            'last_matches_by_column',
            'original_user_query',
            'selected_date_column',
            'temporal_info',
            'agg_info',
            'intent',
            'config'
        ]
        for key in keys_to_clear:
            session_data.pop(key, None)
        print("🔄 Session reset for new query")
    
    # === NON-DATA QUERY HANDLING ===
    # Only classify if this is a fresh query (not feedback)
    if not feedback:
        query_type = classify_query(user_query)
        print(f"📋 Query Type: {query_type}")
        
        non_data_response = get_non_data_response(query_type, user_query)
        if non_data_response:
            non_data_response["session_data"] = session_data
            return non_data_response
    
    # === DATA QUERY PROCESSING ===
    try:
        from pgadmin.RAG_LLM.django_loader import load_module_config
        
        config = load_module_config(module_id)
        
        # Apply context settings
        if context_settings:
            if not context_settings.get('use_kg', True):
                config['knowledge_graph_data'] = {}
            if not context_settings.get('use_relationships', True):
                config['relationships'] = []
        
        # Build catalog
        catalog = {}
        for table, columns in config['table_columns'].items():
            catalog[table] = {}
            for col in columns:
                catalog[table][col['name']] = {
                    'type': col['type'],
                    'values': []
                }
        
        catalog = load_catalog_values(catalog)
        
        # Entity Resolution
        resolution = entity_resolver(user_query, catalog, session_data, feedback, module_id)
        
        # Return clarification if needed
        if resolution.get("needs_clarification"):
            return {
                "type": "clarification",
                "message": resolution.get("message"),
                "options": resolution.get("options"),
                "subtype": resolution.get("subtype"),
                "entity": resolution.get("entity"),
                "entity_type": resolution.get("entity_type"),
                "clarification_context": resolution.get("clarification_context", {}),
                "allow_custom": bool(resolution.get("allow_custom", False)),
                "session_data": session_data
            }
        
        # Generate and execute SQL
        entities = resolution.get("entities", {})
        intent = resolution.get("intent", "total")
        temporal_info = session_data.get('temporal_info')
        agg_info = session_data.get('agg_info')
        selected_date_column = session_data.get('selected_date_column')
        
        query_for_sql = session_data.get('original_user_query', user_query) if not user_query else user_query
        
        sql = generate_sql_with_llm(
            query_for_sql,
            entities, 
            config, 
            intent, 
            temporal_info, 
            agg_info,
            selected_date_column,
            context_settings  # Pass context settings for RCA, POS, etc.
        )
        
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
        if data and len(data) <= 100:
            chart = {
                "type": "table",
                "columns": list(data[0].keys()) if data else [],
                "data": data
            }
        
        return {
            "type": "response",
            "message": format_results(data, intent, temporal_info),
            "sql": sql,
            "data": data,
            "chart": chart,
            "session_data": session_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "type": "error",
            "message": f"Error: {str(e)}",
            "session_data": session_data or {}
        }