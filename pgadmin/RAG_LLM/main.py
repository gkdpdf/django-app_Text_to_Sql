"""
LangGraph Workflow - FINAL WITH TEMPORAL DATE COLUMN SELECTION
- Asks for date column when temporal query detected
- Column selection FIRST with KG descriptions
- Proper temporal filtering (last 6 months, daily, weekly, monthly)
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


def extract_temporal_info(user_query):
    """Extract temporal requirements - ENHANCED"""
    import re
    query_lower = user_query.lower()
    
    temporal_info = {
        'grouping': None,        # daily, weekly, monthly, yearly
        'period': None,          # last_6_months, last_year, etc
        'has_temporal': False,
        'interval_value': None,  # 6, 3, 2, etc
        'interval_unit': None,   # months, weeks, days, years
        'needs_date_column': False  # NEW: Flag if we need to ask for date column
    }
    
    # Detect grouping
    if any(word in query_lower for word in ['daily', 'day-wise', 'per day', 'each day', 'day by day']):
        temporal_info['grouping'] = 'daily'
        temporal_info['has_temporal'] = True
        temporal_info['needs_date_column'] = True
    elif any(word in query_lower for word in ['weekly', 'week-wise', 'per week', 'each week', 'week by week']):
        temporal_info['grouping'] = 'weekly'
        temporal_info['has_temporal'] = True
        temporal_info['needs_date_column'] = True
    elif any(word in query_lower for word in ['monthly', 'month-wise', 'per month', 'each month', 'month by month']):
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
        r'(?:in|for|over|during)\s+(?:the\s+)?(?:last|past|previous)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(month|week|day|year)s?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            num_str = match.group(1)
            unit = match.group(2)
            
            num_value = number_words.get(num_str, int(num_str))
            
            temporal_info['interval_value'] = num_value
            temporal_info['interval_unit'] = unit + 's'
            temporal_info['has_temporal'] = True
            temporal_info['needs_date_column'] = True
            
            if unit == 'month':
                temporal_info['period'] = f'last_{num_value}_months' if num_value > 1 else 'last_month'
            elif unit == 'week':
                temporal_info['period'] = f'last_{num_value}_weeks'
            elif unit == 'day':
                temporal_info['period'] = f'last_{num_value}_days'
            elif unit == 'year':
                temporal_info['period'] = 'last_year' if num_value == 1 else f'last_{num_value}_years'
            
            break
    
    # Detect specific months (may, june, etc.)
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    for month in months:
        if month in query_lower:
            temporal_info['has_temporal'] = True
            temporal_info['needs_date_column'] = True
            temporal_info['specific_month'] = month
            break
    
    # Detect trends
    if any(word in query_lower for word in ['trend', 'pattern', 'over time', 'progression']):
        temporal_info['is_trend'] = True
        temporal_info['has_temporal'] = True
        temporal_info['needs_date_column'] = True
        if not temporal_info['grouping']:
            temporal_info['grouping'] = 'monthly'
    
    return temporal_info


def extract_aggregation_info(user_query):
    """Extract aggregation requirements"""
    query_lower = user_query.lower()
    
    agg_info = {
        'type': 'total',
        'unit': None
    }
    
    if any(word in query_lower for word in ['average', 'avg', 'mean']):
        agg_info['type'] = 'average'
    elif any(word in query_lower for word in ['count', 'number of', 'how many']):
        agg_info['type'] = 'count'
    elif 'per gram' in query_lower or 'per gm' in query_lower:
        agg_info['type'] = 'per_unit'
        agg_info['unit'] = 'gram'
    elif 'per kg' in query_lower or 'per kilogram' in query_lower:
        agg_info['type'] = 'per_unit'
        agg_info['unit'] = 'kg'
    
    return agg_info


def find_date_columns(catalog, config):
    """Find all potential date columns in the schema"""
    date_columns = []
    
    kg_data = config.get('knowledge_graph_data', {})
    
    for table, columns in catalog.items():
        for col_name, col_data in columns.items():
            col_type = col_data.get('type', '').lower()
            col_name_lower = col_name.lower()
            
            # Check by type
            if any(dt in col_type for dt in ['date', 'time', 'timestamp']):
                # Get description
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
            # Check by name
            elif any(kw in col_name_lower for kw in ['date', 'time', 'day', 'month', 'year', 'created', 'updated', 'modified']):
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


def entity_resolver(user_query, catalog, session_data, feedback=None, module_id=None):
    """Resolve entities with date column selection for temporal queries"""
    
    print("\n" + "="*80)
    print("🔍 ENTITY_RESOLVER START")
    print("="*80)
    print(f"Query: {user_query[:60]}..." if len(user_query) > 60 else f"Query: {user_query}")
    print(f"Feedback: {feedback.get('type') if feedback else 'None'}")
    print("="*80 + "\n")
    
    # Store original query in session (for date column selection callback)
    if user_query and not feedback:
        session_data['original_user_query'] = user_query
        print(f"📝 Stored original query in session\n")
    
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
    
    # === DATE COLUMN SELECTION (for temporal queries) ===
    if feedback and feedback.get('type') == 'date_column_selection':
        selected_option = feedback.get('selected_option', '')
        
        # Extract column info
        column_full_name = selected_option.split('\n')[0].split(' - ')[0].strip()
        table, column = column_full_name.split('.')
        
        print(f"📅 Date column selected: {column_full_name}")
        
        # Store in session
        session_data['selected_date_column'] = {
            'table': table,
            'column': column,
            'full_name': column_full_name
        }
        
        # Get original query from session
        original_query = session_data.get('original_user_query', user_query)
        
        print(f"   ✅ Stored date column in session")
        print(f"   🔄 Continuing with original query: {original_query}\n")
        
        # Now continue with entity resolution using ORIGINAL query
        return entity_resolver(original_query, catalog, session_data, None, module_id)
    
    # === CUSTOM VALUE ===
    if feedback and feedback.get('type') == 'custom_value':
        custom_value = feedback.get('custom_value')
        entity_type = feedback.get('entity_type')
        context = feedback.get('clarification_context', {})
        
        print(f"📝 Custom value: '{custom_value}'")
        
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
            else:
                return {
                    "needs_clarification": True,
                    "message": f"No matches for '{custom_value}'. Try again or select from list.",
                    "options": [str(v) for v in values[:20]],
                    "subtype": "value_selection",
                    "entity": custom_value,
                    "entity_type": entity_type,
                    "clarification_context": context,
                    "allow_custom": True
                }
    
    # === VALUE SELECTION ===
    if feedback and feedback.get('type') == 'value_selection':
        selected_value = feedback.get('selected_option')
        entity_type = feedback.get('entity_type')
        context = feedback.get('clarification_context', {})
        
        # NEW: Multi-select support
        multiple_values = feedback.get('multiple_values')
        
        table = context.get('table')
        column = context.get('column')
        
        if not table or not column:
            if 'last_clarification' in session_data:
                table = session_data['last_clarification'].get('table')
                column = session_data['last_clarification'].get('column')
        
        if not table or not column:
            return {
                "type": "error",
                "message": "Missing table/column info",
                "session_data": session_data
            }
        
        if 'resolved_entities' not in session_data:
            session_data['resolved_entities'] = {}
        
        # Handle multiple values (OR condition)
        if multiple_values and len(multiple_values) > 1:
            print(f"📌 Multiple values selected: {len(multiple_values)} items")
            print(f"   Values: {multiple_values[:3]}{'...' if len(multiple_values) > 3 else ''}")
            
            session_data['resolved_entities'][entity_type] = {
                "table": table,
                "column": column,
                "values": multiple_values,  # Store as array
                "is_multiple": True
            }
            
            print(f"✅ Multiple entities resolved: {entity_type} = {len(multiple_values)} values\n")
        else:
            # Single value (existing behavior)
            print(f"📌 Value selected: {selected_value}")
            
            session_data['resolved_entities'][entity_type] = {
                "table": table,
                "column": column,
                "value": selected_value
            }
            
            print(f"✅ Entity resolved: {entity_type} = {selected_value}\n")
        
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
        
        # NEW: Check for multiple columns selection
        multiple_values = feedback.get('multiple_values')
        
        # Recover matches
        matches_by_column = context.get('matches_by_column', {})
        if not matches_by_column and 'last_matches_by_column' in session_data:
            matches_by_column = session_data['last_matches_by_column']
            print(f"   📦 Recovered matches from session")
        
        # Handle multiple column selection
        if multiple_values and len(multiple_values) > 1:
            print(f"📌 Multiple columns selected: {len(multiple_values)} columns")
            
            all_values = []
            primary_table = None
            primary_column = None
            
            for selected in multiple_values:
                column_key = selected.split('\n')[0].split(' (')[0].strip()
                
                if column_key not in matches_by_column:
                    continue
                
                col_data = matches_by_column[column_key]
                table = col_data['table']
                column = col_data['column']
                values = col_data['values']
                
                print(f"   → {table}.{column}: {len(values)} values")
                
                # Use first column as primary
                if primary_table is None:
                    primary_table = table
                    primary_column = column
                
                # Collect all values
                all_values.extend(values)
            
            # Remove duplicates while preserving order
            unique_values = list(dict.fromkeys(all_values))
            
            print(f"   Combined: {len(unique_values)} unique values across {len(multiple_values)} columns")
            
            # Store in session
            session_data['last_clarification'] = {
                'table': primary_table, 
                'column': primary_column,
                'is_multi_column': True,
                'all_columns': multiple_values
            }
            
            # Ask for value selection from combined list
            kg_data = config.get('knowledge_graph_data', {})
            col_desc = ""
            if primary_table in kg_data and primary_column in kg_data[primary_table]:
                col_desc = kg_data[primary_table][primary_column].get('desc', '')
            
            message = f"Found {len(unique_values)} values across {len(multiple_values)} columns"
            if col_desc:
                message += f" ({col_desc})"
            message += ". Please select one or more:"
            
            print(f"   ✅ Returning {len(unique_values)} value options\n")
            
            return {
                "needs_clarification": True,
                "message": message,
                "options": [str(v) for v in unique_values[:50]],  # Limit to 50 for performance
                "subtype": "value_selection",
                "entity": entity_value,
                "entity_type": entity_type,
                "clarification_context": {"table": primary_table, "column": primary_column},
                "allow_custom": True
            }
        
        # Single column selection (existing logic)
        column_key = selected_option.split('\n')[0].split(' (')[0].strip()
        
        print(f"📌 Column selected: {column_key}")
        print(f"   Entity: {entity_value}")
        
        if column_key not in matches_by_column:
            print(f"   🔄 Re-finding matches...")
            
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
                return {
                    "type": "error",
                    "message": "Column not found",
                    "session_data": session_data
                }
        
        col_data = matches_by_column[column_key]
        table = col_data['table']
        column = col_data['column']
        values = col_data['values']
        
        print(f"   Table: {table}, Column: {column}, Values: {len(values)}")
        
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
            
            print(f"   ✅ Returning {len(values)} value options\n")
            
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
            print(f"   ✅ Auto-resolve: {values[0]}\n")
            
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
    
    # === INITIAL QUERY ===
    
    # Get the query to use for entity extraction
    query_for_extraction = user_query if user_query else session_data.get('original_user_query', '')
    
    print(f"🔍 Initial entity extraction...")
    print(f"   Using query: {query_for_extraction}\n")
    
    # Check if we need date column selection
    if temporal_info.get('needs_date_column') and 'selected_date_column' not in session_data:
        print(f"📅 Temporal query detected - need date column selection")
        
        date_columns = find_date_columns(catalog, config)
        
        if not date_columns:
            print(f"   ⚠️ No date columns found!")
        elif len(date_columns) == 1:
            # Auto-select single date column
            date_col = date_columns[0]
            session_data['selected_date_column'] = {
                'table': date_col['table'],
                'column': date_col['column'],
                'full_name': date_col['full_name']
            }
            print(f"   ✅ Auto-selected date column: {date_col['full_name']}\n")
        else:
            # Ask user to select date column
            options = []
            for dc in date_columns:
                opt = f"{dc['full_name']} - {dc['type']}"
                if dc['description']:
                    opt += f"\n    📝 {dc['description']}"
                options.append(opt)
            
            message = f"This query needs a date column. Which date should I use?"
            if temporal_info.get('grouping'):
                message += f"\n(for {temporal_info['grouping']} grouping)"
            if temporal_info.get('period'):
                message += f"\n(for {temporal_info['period']} filter)"
            
            print(f"   ✅ Asking for date column selection ({len(date_columns)} options)\n")
            
            return {
                "needs_clarification": True,
                "message": message,
                "options": options,
                "subtype": "date_column_selection",
                "clarification_context": {"date_columns": date_columns},
                "allow_custom": False
            }
    
    # Continue with entity extraction
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
        {"type": "product|customer|region|distributor|plant|super_stockist|category", "value": "...", "search_terms": ["..."]},
        ...
    ],
    "intent": "total|count|list|average|trend"
}

Extract ALL entities. Add synonyms to search_terms. Do NOT extract temporal terms like 'may', 'monthly', 'last 6 months' as entities."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query_for_extraction}\n\nDatabase: {json.dumps(catalog_context, default=str)[:2000]}"}
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
        print(f"Intent: {intent}\n")
        
        # Filter out temporal entities
        extracted_entities = [
            e for e in extracted_entities 
            if e.get('type') not in ['time', 'date', 'month', 'period']
            and e.get('value').lower() not in ['sales', 'may', 'june', 'monthly', 'daily', 'weekly']
        ]
        
        if not extracted_entities:
            print("   ℹ️ No entities to resolve (temporal-only query)\n")
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
                print(f"   ⚠️ No matches for {entity['type']}: {entity['value']}")
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
                
                current_index = extracted_entities.index(entity)
                session_data['pending_entities'] = extracted_entities[current_index+1:]
                session_data['resolved_entities'] = resolved_entities
                session_data['last_matches_by_column'] = matches_by_column
                
                message = f"I found '{entity.get('value')}' in {len(matches_by_column)} column{'s' if len(matches_by_column) > 1 else ''}. "
                message += "Which column?" if len(matches_by_column) > 1 else "Is this correct?"
                
                print(f"✅ Asking for column selection ({len(matches_by_column)} options)")
                print(f"   📦 Storing matches in session\n")
                
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


def generate_sql_with_llm(user_query, entities, config, intent="total", temporal_info=None, agg_info=None, selected_date_column=None, stream_callback=None):
    """Generate SQL with full context and streaming support"""
    
    print("\n" + "="*80)
    print("🔧 SQL GENERATION")
    print("="*80)
    
    table_columns = config.get('table_columns', {})
    kg_data = config.get('knowledge_graph_data', {})
    relationships = config.get('relationships', [])
    
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
        
        # NEW: Handle multiple values (OR condition with IN clause)
        if entity_data.get('is_multiple'):
            values = entity_data.get('values', [])
            if values and table and column:
                tables_used.add(table)
                # Create IN clause for multiple values
                escaped_values = []
                for v in values:
                    if isinstance(v, str):
                        escaped = v.replace("'", "''")
                        escaped_values.append(f"'{escaped}'")
                    else:
                        escaped_values.append(str(v))
                
                in_clause = f'{table}."{column}" IN ({", ".join(escaped_values)})'
                where_conditions.append(in_clause)
                print(f"   Using IN clause with {len(values)} values")
        else:
            # Single value (existing behavior)
            value = entity_data.get('value')
            
            if table and column and value:
                tables_used.add(table)
                if isinstance(value, str):
                    escaped = value.replace("'", "''")
                    where_conditions.append(f'{table}."{column}" = \'{escaped}\'')
                else:
                    where_conditions.append(f'{table}."{column}" = {value}')
    
    # Add temporal filter
    temporal_filter = ""
    grouping_clause = ""
    
    if temporal_info and temporal_info.get('has_temporal') and selected_date_column:
        date_col_full = selected_date_column['full_name']
        date_table = selected_date_column['table']
        date_col = selected_date_column['column']
        tables_used.add(date_table)
        
        print(f"📅 Using date column: {date_col_full}")
        
        # Add period filter
        if temporal_info.get('interval_value') and temporal_info.get('interval_unit'):
            val = temporal_info['interval_value']
            unit = temporal_info['interval_unit']
            temporal_filter = f'{date_table}."{date_col}" >= CURRENT_DATE - INTERVAL \'{val} {unit}\''
            where_conditions.append(temporal_filter)
            print(f"   Period filter: last {val} {unit}")
        
        # Handle specific month
        if temporal_info.get('specific_month'):
            month_name = temporal_info['specific_month'].capitalize()
            month_num = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }.get(month_name, 5)
            
            temporal_filter = f'EXTRACT(MONTH FROM {date_table}."{date_col}") = {month_num}'
            where_conditions.append(temporal_filter)
            print(f"   Month filter: {month_name}")
        
        # Add grouping
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
            
            print(f"   Grouping: {grouping}")
    
    where_clause = " AND ".join(where_conditions) if where_conditions else ""
    
    print(f"Query: {user_query}")
    print(f"WHERE: {where_clause}")
    print(f"GROUP BY: {grouping_clause if grouping_clause else 'None'}")
    
    # Build prompt
    prompt = f"""Generate PostgreSQL query for this request.

USER QUERY: {user_query}
INTENT: {intent}

SCHEMA:
{schema_text}

WHERE (MANDATORY): {where_clause}
TABLES: {', '.join(tables_used)}

TEMPORAL:
- Grouping: {temporal_info.get('grouping') if temporal_info else 'None'}
- Period: {temporal_info.get('period') if temporal_info else 'None'}
- Date Column: {selected_date_column['full_name'] if selected_date_column else 'None'}
- GROUP BY clause: {grouping_clause if grouping_clause else 'None'}

AGGREGATION:
- Type: {agg_info.get('type') if agg_info else 'total'}
- Unit: {agg_info.get('unit') if agg_info else 'None'}

RULES:
1. Use exact WHERE clause: {where_clause}
2. Double quotes for columns
3. If grouping: SELECT {grouping_clause} as period, SUM(...), etc GROUP BY {grouping_clause}
4. Add LIMIT 100
5. Use appropriate aggregation (SUM, AVG, COUNT)

Generate ONLY SQL:
"""
    
    try:
        if stream_callback:
            # Stream the response
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "PostgreSQL expert. Generate queries with proper temporal grouping."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                stream=True  # Enable streaming
            )
            
            sql = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    sql += token
                    stream_callback(token)
            
        else:
            # Non-streaming
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "PostgreSQL expert. Generate queries with proper temporal grouping."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            sql = response.choices[0].message.content.strip()
        
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        if 'LIMIT' not in sql.upper():
            sql = sql.rstrip(';') + '\nLIMIT 100;'
        
        print(f"\n✅ SQL:\n{sql}\n")
        print("="*80 + "\n")
        
        return sql
        
    except Exception as e:
        print(f"❌ LLM error: {e}")
        main_table = list(tables_used)[0] if tables_used else list(table_columns.keys())[0]
        sql = f'SELECT * FROM {main_table}'
        if where_clause:
            sql += f'\nWHERE {where_clause}'
        sql += '\nLIMIT 100;'
        return sql


def execute_sql(sql):
    """Execute SQL"""
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


def format_results(data, intent, temporal_info=None):
    """Format results with preview for temporal queries"""
    if not data:
        return "No results found."
    
    if len(data) == 0:
        return "No results found."
    
    first_row = data[0]
    
    # For temporal queries with multiple rows - show preview
    if temporal_info and temporal_info.get('has_temporal') and len(data) > 1:
        grouping = temporal_info.get('grouping', 'period')
        
        # Build preview message
        message = f"Found {len(data)} {grouping} records.\n\n"
        
        # Show first few rows as preview
        preview_count = min(5, len(data))
        for i, row in enumerate(data[:preview_count]):
            # Find the value column (not period)
            for key, value in row.items():
                if key.lower() != 'period' and value is not None:
                    period_val = row.get('period', f'Row {i+1}')
                    if isinstance(value, (int, float)):
                        message += f"• {period_val}: {value:,.2f}\n"
                    else:
                        message += f"• {period_val}: {value}\n"
                    break
        
        if len(data) > preview_count:
            message += f"\n... and {len(data) - preview_count} more records"
        
        return message.strip()
    
    # Check for aggregations in first row
    for key, value in first_row.items():
        key_lower = key.lower()
        
        # Handle SUM aggregations
        if 'sum' in key_lower or 'total' in key_lower:
            if value is not None:
                if isinstance(value, (int, float)):
                    return f"Total: {value:,.2f}"
                else:
                    return f"Total: {value}"
        
        # Handle AVG aggregations
        if 'avg' in key_lower or 'average' in key_lower:
            if value is not None:
                if isinstance(value, (int, float)):
                    return f"Average: {value:,.2f}"
                else:
                    return f"Average: {value}"
        
        # Handle COUNT aggregations
        if 'count' in key_lower:
            if value is not None:
                return f"Count: {value:,}"
    
    # If single row with single value, show it
    if len(data) == 1 and len(first_row) == 1:
        key = list(first_row.keys())[0]
        value = first_row[key]
        if value is not None:
            if isinstance(value, (int, float)):
                return f"{key}: {value:,.2f}"
            else:
                return f"{key}: {value}"
    
    # If single row with multiple values, show all
    if len(data) == 1:
        message = ""
        for key, value in first_row.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    message += f"{key}: {value:,.2f}\n"
                else:
                    message += f"{key}: {value}\n"
        return message.strip() if message else "Found 1 result"
    
    # Default: show row count
    return f"Found {len(data)} results"


def invoke_graph(user_query, module_id, session_data=None, feedback=None, stream_callback=None, context_settings=None):
    """Main entry point
    
    Args:
        user_query: User's question
        module_id: Module ID
        session_data: Session state
        feedback: User feedback for clarifications
        stream_callback: Optional streaming callback
        context_settings: Dict with context toggle flags (use_kg, use_relationships, use_rca, use_extra_suggestions, use_pos)
    """
    
    print("\n" + "🔵"*40)
    print("🚀 INVOKE_GRAPH")
    print(f"   Query: {user_query}")
    print(f"   Feedback: {feedback.get('type') if feedback else 'None'}")
    
    # NEW: Log context settings
    if context_settings:
        print(f"📊 Context Settings:")
        print(f"   KG: {'✅' if context_settings.get('use_kg', True) else '❌'}")
        print(f"   Relationships: {'✅' if context_settings.get('use_relationships', True) else '❌'}")
        print(f"   RCA: {'✅' if context_settings.get('use_rca', True) else '❌'}")
        print(f"   Extra Suggestions: {'✅' if context_settings.get('use_extra_suggestions', True) else '❌'}")
        print(f"   POS: {'✅' if context_settings.get('use_pos', True) else '❌'}")
    else:
        print(f"📊 Context Settings: All enabled (default)")
    
    print("🔵"*40 + "\n")
    
    try:
        from pgadmin.RAG_LLM.django_loader import load_module_config
        
        # Load config normally
        config = load_module_config(module_id)
        
        # NEW: Filter context based on settings
        if context_settings:
            # Default all to True if not specified
            use_kg = context_settings.get('use_kg', True)
            use_relationships = context_settings.get('use_relationships', True)
            use_rca = context_settings.get('use_rca', True)
            use_extra_suggestions = context_settings.get('use_extra_suggestions', True)
            use_pos = context_settings.get('use_pos', True)
            
            # Filter Knowledge Graph
            if not use_kg:
                config['kg_data'] = {}
                print("   ⚠️ Knowledge Graph disabled")
            
            # Filter Relationships
            if not use_relationships:
                config['relationships'] = []
                print("   ⚠️ Relationships disabled")
            
            # Filter RCA
            if not use_rca:
                config['rca'] = []
                print("   ⚠️ RCA disabled")
            
            # Filter Extra Suggestions
            if not use_extra_suggestions:
                config['extra_suggestions'] = ''
                print("   ⚠️ Extra Suggestions disabled")
            
            # Filter POS Tagging
            if not use_pos:
                config['pos_tagging'] = []
                print("   ⚠️ POS Tagging disabled")
        
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
            session_data = {}
        
        resolution = entity_resolver(user_query, catalog, session_data, feedback, module_id)
        
        if resolution.get("needs_clarification"):
            print("✅ Returning clarification\n")
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
        temporal_info = session_data.get('temporal_info')
        agg_info = session_data.get('agg_info')
        selected_date_column = session_data.get('selected_date_column')
        
        # Use original query for SQL generation
        query_for_sql = session_data.get('original_user_query', user_query) if not user_query else user_query
        
        sql = generate_sql_with_llm(
            query_for_sql,  # Use original query
            entities, 
            config, 
            intent, 
            temporal_info, 
            agg_info,
            selected_date_column
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
        
        # DEBUG: Show actual result data
        print("📊 RESULT DATA:")
        if data:
            print(f"   Rows: {len(data)}")
            print(f"   First row: {data[0]}")
            for key, value in data[0].items():
                print(f"      {key}: {value}")
        else:
            print("   No data returned")
        print()
        
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


def generate_knowledge_graph(module_id, selected_tables, selected_columns):
    """Generate knowledge graph with AI"""
    from pgadmin.RAG_LLM.django_loader import get_db_credentials
    import psycopg2
    
    db_creds = get_db_credentials()
    conn = psycopg2.connect(**db_creds)
    
    kg_data = {}
    
    for table in selected_tables:
        kg_data[table] = {}
        cols_to_process = selected_columns.get(table, [])
        
        for col_name in cols_to_process:
            try:
                cursor = conn.cursor()
                cursor.execute(f'SELECT DISTINCT "{col_name}" FROM {table} LIMIT 10')
                samples = [str(row[0]) for row in cursor.fetchall() if row[0]]
                cursor.close()
                
                prompt = f"""Describe this column:
Table: {table}
Column: {col_name}
Samples: {', '.join(samples[:5])}

Return JSON: {{"desc": "...", "datatype": "identifier|monetary|quantity|categorical|temporal|text"}}"""
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Database expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                kg_info = json.loads(response.choices[0].message.content)
                kg_data[table][col_name] = kg_info
                
            except Exception as e:
                print(f"Error: {table}.{col_name}: {e}")
    
    conn.close()
    return kg_data