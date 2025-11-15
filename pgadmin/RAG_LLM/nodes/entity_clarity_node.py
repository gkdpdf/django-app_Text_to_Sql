import os
import re
from openai import OpenAI
from rapidfuzz import process, fuzz
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==================== HELPER FUNCTIONS ====================
def normalize(t):
    """Normalize text for fuzzy matching"""
    return re.sub(r'[^a-zA-Z0-9 ]+', '', str(t)).lower().strip()

def detect_time_filters(user_query: str):
    """Detect time filters using word boundary matching"""
    query = user_query.lower()
    today = datetime.today().date()
    
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    for month_name, month_num in months.items():
        pattern = r'\b' + month_name + r'\b'
        if re.search(pattern, query):
            year = today.year
            return {
                "time_range": [f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-31"],
                "month": month_num
            }
    
    if "last 2 months" in query:
        return {"time_range": [str(today - timedelta(days=60)), str(today)]}
    if "last 3 months" in query:
        return {"time_range": [str(today - timedelta(days=90)), str(today)]}
    if "last month" in query:
        return {"time_range": [str(today - timedelta(days=30)), str(today)]}
    if "last week" in query:
        return {"time_range": [str(today - timedelta(days=7)), str(today)]}
    if "this month" in query:
        month = today.month
        year = today.year
        return {"time_range": [f"{year}-{month:02d}-01", f"{year}-{month:02d}-31"], "month": month}
    if "this year" in query:
        year = today.year
        return {"time_range": [f"{year}-01-01", f"{year}-12-31"]}
    if re.search(r'\btoday\b', query):
        return {"time_range": [str(today), str(today)]}
    if re.search(r'\byesterday\b', query):
        yesterday = today - timedelta(days=1)
        return {"time_range": [str(yesterday), str(yesterday)]}
    
    return {}

def shortlist_candidates_with_scores(text, options, k=15, score_cutoff=60):
    """Fuzzy match with scoring"""
    if not options:
        return []
    
    text_norm = normalize(text)
    normalized_options = [normalize(o) for o in options]
    norm_to_original = dict(zip(normalized_options, options))

    exact_matches = []
    for norm_option, orig_option in norm_to_original.items():
        if text_norm in norm_option:
            score = 90 + (len(text_norm) / len(norm_option)) * 10
            exact_matches.append((orig_option, score))

    fuzzy_matches = process.extract(
        text_norm, normalized_options, 
        scorer=fuzz.token_set_ratio,
        limit=k, 
        score_cutoff=score_cutoff
    )

    all_matches = {}
    for match, score in exact_matches:
        all_matches[match] = score
    for match_result in fuzzy_matches:
        match, score = match_result[0], match_result[1]
        orig_match = norm_to_original[match]
        if orig_match not in all_matches:
            all_matches[orig_match] = score

    final_matches = list(all_matches.items())
    final_matches.sort(key=lambda x: x[1], reverse=True)
    return final_matches[:k]

# ==================== LLM PARSING ====================
def llm_understand(user_query, module_config=None):
    """Extract intent, metrics, and entities using LLM with module context - FULLY DYNAMIC"""
    
    # Build context from module config - NO HARDCODING
    context_lines = []
    
    if module_config:
        module_name = module_config.get('module_name', 'Database')
        context_lines.append(f"Module: {module_name}")
        
        # Add entity types from POS tagging
        pos_tagging = module_config.get('pos_tagging', [])
        if pos_tagging:
            entity_types = [pos['name'] for pos in pos_tagging if pos.get('name')]
            if entity_types:
                context_lines.append(f"\nAvailable entity types: {', '.join(entity_types)}")
                context_lines.append("Entity references:")
                for pos in pos_tagging[:10]:  # Limit to 10 for prompt size
                    if pos.get('name') and pos.get('reference'):
                        context_lines.append(f"  - {pos['name']}: {pos['reference']}")
        
        # Add metrics from module config
        metrics = module_config.get('metrics', {})
        if metrics:
            context_lines.append(f"\nAvailable metrics:")
            for metric_name, metric_desc in list(metrics.items())[:10]:
                context_lines.append(f"  - {metric_name}: {metric_desc}")
        
        # Add RCA context
        rca_list = module_config.get('rca_list', [])
        if rca_list:
            context_lines.append(f"\nBusiness context (RCA):")
            for rca in rca_list[:3]:  # Limit to 3 for prompt size
                if rca.get('title') and rca.get('content'):
                    context_lines.append(f"  - {rca['title']}: {rca['content'][:200]}...")
        
        # Add extra suggestions
        extra_suggestions = module_config.get('extra_suggestions', '')
        if extra_suggestions and len(extra_suggestions.strip()) > 0:
            context_lines.append(f"\nAdditional context:\n{extra_suggestions[:300]}...")
    
    context = "\n".join(context_lines) if context_lines else "No additional context available."
    
    prompt = f"""
You are a business query analyzer for a database system.

{context}

1. If irrelevant to this database, return {{"intent": "irrelevant"}}.

2. If relevant, extract:
   - intent (query|aggregation|ranking|comparison)
   - metrics (from available metrics above, or infer from query)
   - entities: COMPLETE entity names as dictionary with entity type as key

Examples:
- "XYZ company" → {{"intent": "query", "entities": {{"company": ["XYZ company"]}}, "metrics": []}}
- "product sales" → {{"intent": "query", "entities": {{"product": ["product"]}}, "metrics": ["sales"]}}
- "sales in may" → {{"intent": "query", "entities": {{}}, "metrics": ["sales"]}}

User query: ```{user_query}```

Return ONLY valid JSON with intent, metrics, entities keys.
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        content = resp.choices[0].message.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = eval(content)
        
        if "intent" not in result:
            result["intent"] = "query"
        if "metrics" not in result:
            result["metrics"] = []
        if "entities" not in result:
            result["entities"] = {}
        
        if not isinstance(result["entities"], dict):
            result["entities"] = {}
        
        if not isinstance(result["metrics"], list):
            result["metrics"] = [result["metrics"]] if result["metrics"] else []
        
        return result
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        return {"intent": "query", "metrics": [], "entities": {}}

# ==================== ENTITY RESOLUTION ====================
def resolve_entity_non_interactive(entity_value, catalog, table_columns, max_options=15):
    """
    Non-interactive entity resolution for web.
    Returns clarification dict if multiple matches found.
    """
    all_matches = {}
    
    for table, cols in catalog.items():
        for col, values in cols.items():
            candidates = shortlist_candidates_with_scores(entity_value, values, k=max_options)
            if candidates:
                all_matches[(table, col)] = candidates
    
    if not all_matches:
        return {"resolved": False, "not_found": True, "entity": entity_value}
    
    # Multiple columns - need column disambiguation
    if len(all_matches) > 1:
        column_options = []
        for (table, col), candidates in all_matches.items():
            column_options.append({
                "table": table,
                "column": col,
                "best_match": candidates[0][0],
                "count": len(candidates),
                "score": candidates[0][1]
            })
        
        column_options.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "resolved": False,
            "clarification": {
                "type": "column",
                "entity": entity_value,
                "options": column_options,
                "message": f"'{entity_value}' found in multiple columns. Which column do you mean?"
            }
        }
    
    # Single column
    selected_key = next(iter(all_matches.keys()))
    table, column = selected_key
    candidates = all_matches[selected_key]
    
    if len(candidates) == 1:
        return {
            "resolved": True,
            "table": table,
            "column": column,
            "value": candidates[0][0]
        }
    
    # Multiple values - need value disambiguation
    options = [cand[0] for cand in candidates[:max_options]]
    has_more = len(candidates) > max_options
    
    return {
        "resolved": False,
        "clarification": {
            "type": "value",
            "entity": entity_value,
            "table": table,
            "column": column,
            "options": options,
            "has_more": has_more,
            "total_count": len(candidates),
            "message": f"I found {len(candidates)} matches for '{entity_value}'. Which one did you mean?"
        }
    }

# ==================== SPECIFIC COLUMN RESOLUTION ====================
def resolve_in_specific_column(entity_value, table, column, conn, max_options=15):
    """Search in specific table/column after column selection"""
    cursor = None
    try:
        cursor = conn.cursor()
        
        query = f"SELECT DISTINCT {column} FROM {table} WHERE LOWER({column}) LIKE LOWER(%s) LIMIT {max_options * 2}"
        cursor.execute(query, (f"%{entity_value}%",))
        results = cursor.fetchall()
        
        if not results:
            return {"resolved": False, "not_found": True}
        
        values = [r[0] for r in results]
        candidates = shortlist_candidates_with_scores(entity_value, values, k=max_options)
        
        if len(candidates) == 1:
            return {
                "resolved": True,
                "table": table,
                "column": column,
                "value": candidates[0][0]
            }
        
        options = [cand[0] for cand in candidates]
        has_more = len(results) > max_options
        
        return {
            "resolved": False,
            "clarification": {
                "type": "value",
                "entity": entity_value,
                "table": table,
                "column": column,
                "options": options,
                "has_more": has_more,
                "total_count": len(results),
                "message": f"Which {column} did you mean?"
            }
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"resolved": False, "error": str(e)}
    finally:
        if cursor:
            cursor.close()

# ==================== DATABASE UTILITIES ====================
def build_catalog(conn, table_columns, max_values=100):
    """Build catalog from database"""
    catalog = {}
    cur = conn.cursor()

    for table, cols in table_columns.items():
        catalog[table] = {}
        for col in cols:
            try:
                q = f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT {max_values};"
                cur.execute(q)
                values = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
                catalog[table][col] = values
            except Exception as e:
                print(f"⚠️ Skipping {table}.{col}: {e}")
    cur.close()
    return catalog

def load_table_columns_pg(conn, tables):
    """Load column names from PostgreSQL"""
    table_columns = {}
    with conn.cursor() as cur:
        for table in tables:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table,))
            cols = [row[0] for row in cur.fetchall()]
            table_columns[table] = cols
    return table_columns