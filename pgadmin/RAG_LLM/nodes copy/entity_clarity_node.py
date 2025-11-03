import os
import re
from openai import OpenAI
from rapidfuzz import process, fuzz
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg2

load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Helpers ----------
def normalize(t):
    return re.sub(r'[^a-zA-Z0-9 ]+', '', str(t)).lower().strip()

def detect_time_filters(user_query: str):
    """
    FIXED: Only returns time filter if explicitly mentioned in query
    Uses word boundary matching to avoid false positives like "markeplus" containing "mar"
    """
    query = user_query.lower()
    today = datetime.today().date()
    
    # Month detection - MUST be whole words or with "in" prefix
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # Check for explicit month mentions using word boundaries
    for month_name, month_num in months.items():
        # Create regex pattern for whole word match
        # Matches: "in march", "march sales", "for march", but NOT "markeplus"
        pattern = r'\b' + month_name + r'\b'
        
        if re.search(pattern, query):
            year = today.year
            print(f"📅 Detected month: {month_name} {year}")
            return {
                "time_range": [f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-31"],
                "month": month_num
            }
    
    # Check for relative time periods (these are safe as they're full phrases)
    if "last 2 months" in query:
        print("📅 Detected: last 2 months")
        return {"time_range": [str(today - timedelta(days=60)), str(today)]}
    if "last 3 months" in query:
        print("📅 Detected: last 3 months")
        return {"time_range": [str(today - timedelta(days=90)), str(today)]}
    if "last month" in query:
        print("📅 Detected: last month")
        return {"time_range": [str(today - timedelta(days=30)), str(today)]}
    if "last week" in query:
        print("📅 Detected: last week")
        return {"time_range": [str(today - timedelta(days=7)), str(today)]}
    if "this month" in query:
        print("📅 Detected: this month")
        month = today.month
        year = today.year
        return {
            "time_range": [f"{year}-{month:02d}-01", f"{year}-{month:02d}-31"],
            "month": month
        }
    if "this year" in query:
        print("📅 Detected: this year")
        year = today.year
        return {"time_range": [f"{year}-01-01", f"{year}-12-31"]}
    if re.search(r'\btoday\b', query):
        print("📅 Detected: today")
        return {"time_range": [str(today), str(today)]}
    if re.search(r'\byesterday\b', query):
        print("📅 Detected: yesterday")
        yesterday = today - timedelta(days=1)
        return {"time_range": [str(yesterday), str(yesterday)]}
    
    # CRITICAL FIX: No time mentioned - explicitly state it
    print("ℹ️  No time period mentioned - will query all available data")
    return {}

def shortlist_candidates_with_scores(text, options, k=15, score_cutoff=60):
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
        text_norm, normalized_options, scorer=fuzz.token_set_ratio,
        limit=k, score_cutoff=score_cutoff
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

# ---------- LLM Step 1 ----------
def llm_understand(user_query):
    prompt = f"""
You are a business query analyzer.

1. If the query is irrelevant to sales, products, distributors, or superstockists,
   return JSON with "intent": "irrelevant".

2. If relevant, extract:
   - intent (query|aggregation|ranking|comparison)
   - metrics (sales, revenue, quantity, etc.)
   - entities: Extract COMPLETE entity names as a dictionary, don't split them.

Examples:
- "VH trading" → {{"intent": "query", "entities": {{"distributor": ["VH trading"]}}, "metrics": []}}
- "takatak" → {{"intent": "query", "entities": {{"product": ["takatak"]}}, "metrics": []}}
- "sales in may" → {{"intent": "query", "entities": {{}}, "metrics": ["sales"]}}

CRITICAL: Do NOT infer or add time information. Only extract what's explicitly mentioned.

IMPORTANT: ALWAYS include "intent", "metrics", and "entities" keys in your response.

User query:
```{user_query}```

Return ONLY valid JSON with all required keys: intent, metrics, entities.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Return only valid JSON with intent, metrics, and entities keys."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        content = resp.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = eval(content)
        
        # Ensure all required keys exist with defaults
        if "intent" not in result:
            result["intent"] = "query"
        if "metrics" not in result:
            result["metrics"] = []
        if "entities" not in result:
            result["entities"] = {}
        
        # Ensure entities is always a dict
        if not isinstance(result["entities"], dict):
            print(f"⚠️ Converting entities from {type(result['entities'])} to dict")
            result["entities"] = {}
        
        # Ensure metrics is always a list
        if not isinstance(result["metrics"], list):
            result["metrics"] = [result["metrics"]] if result["metrics"] else []
        
        print(f"✅ LLM parsed: intent={result['intent']}, metrics={result['metrics']}, entities={list(result['entities'].keys())}")
        
        return result
    except Exception as e:
        print(f"⚠️ LLM parsing error: {e}, using defaults")
        return {"intent": "query", "metrics": [], "entities": {}}

# ---------- Entity Resolution ----------
def resolve_entity_with_disambiguation(entity_value, catalog, table_columns):
    """
    Resolve entity by:
    1. Checking all columns
    2. Asking user to pick column if multiple matches
    3. Asking user to pick value if multiple candidates
    """
    all_matches = {}
    for table, cols in catalog.items():
        for col, values in cols.items():
            candidates = shortlist_candidates_with_scores(entity_value, values)
            if candidates:
                all_matches[(table, col)] = candidates

    if not all_matches:
        return {"status": "not found", "value": entity_value}

    # Step 1: Let user pick column if entity appears in multiple columns
    if len(all_matches) > 1:
        print(f"\n🤔 '{entity_value}' found in multiple columns:")
        keys = list(all_matches.keys())
        for i, (table, col) in enumerate(keys, 1):
            best = all_matches[(table, col)][0][0]
            print(f"  {i}. {col} (in {table}) → best match: '{best}'")
        while True:
            try:
                choice = int(input(f"Which column do you mean? (1-{len(keys)}): "))
                if 1 <= choice <= len(keys):
                    selected_key = keys[choice - 1]
                    break
            except ValueError:
                pass
    else:
        selected_key = next(iter(all_matches.keys()))

    table, column = selected_key
    candidates = all_matches[selected_key]

    # Step 2: Let user pick entity if multiple values match in same column
    if len(candidates) > 1:
        print(f"\n🎯 Multiple matches found for '{entity_value}' in {column}:")
        for i, (cand, score) in enumerate(candidates, 1):
            print(f"  {i}. {cand} (similarity {score}%)")
        while True:
            try:
                choice = int(input(f"Which one do you mean? (1-{len(candidates)}): "))
                if 1 <= choice <= len(candidates):
                    final_value = candidates[choice - 1][0]
                    break
            except ValueError:
                pass
    else:
        final_value = candidates[0][0]

    return {"table": table, "column": column, "value": final_value}

# ---------- Main Resolver ----------
def resolve_with_human_in_loop_pg(user_query, catalog, table_columns):
    parsed = llm_understand(user_query)
    
    # Safe access to intent with default
    intent = parsed.get("intent", "query")

    if intent == "irrelevant":
        print("🙅 This question doesn't relate to products, distributors, or sales in the DB.")
        return {
            "intent": "irrelevant",
            "metrics": [],
            "entities": {},
            "filters": {},
            "table": None,
            "columns": []
        }

    metrics = parsed.get("metrics", [])
    entities = parsed.get("entities", {})
    
    # CRITICAL FIX: Ensure entities is always a dict
    if not isinstance(entities, dict):
        print(f"⚠️ Warning: entities was {type(entities)}, converting to dict")
        entities = {}
    
    # CRITICAL FIX: Only detect time filters if explicitly mentioned
    filters = detect_time_filters(user_query)
    
    print(f"📝 Query intent: {intent}, metrics: {metrics}, filters: {filters if filters else 'None'}")

    resolved_entities = {}
    
    # Only try to resolve if entities exist
    if entities:
        for entity_type, values in entities.items():
            if not values:
                continue
            # Handle both list and single values
            entity_value = values[0] if isinstance(values, list) else values
            print(f"\n🔍 Resolving entity: '{entity_value}'")
            result = resolve_entity_with_disambiguation(entity_value, catalog, table_columns)
            if result.get("status") != "not found":
                resolved_entities[entity_type] = result
    else:
        print("ℹ️ No entities detected in query")

    # Determine table
    table = None
    if "sales" in metrics and not any(tbl in user_query.lower() for tbl in ["primary", "shipment"]):
        print("\n❓ 'Sales' found. Do you mean:")
        print("  1. tbl_primary")
        print("  2. tbl_shipment")
        while True:
            try:
                choice = int(input("Select table (1-2): "))
                if choice in [1, 2]:
                    table = "tbl_primary" if choice == 1 else "tbl_shipment"
                    break
            except ValueError:
                print("Invalid input. Please enter 1 or 2.")
    elif resolved_entities:
        table = next(iter(resolved_entities.values())).get("table")
    else:
        # Default table selection
        print("\n📊 Which table do you want to query?")
        available_tables = list(table_columns.keys())
        for i, t in enumerate(available_tables, 1):
            print(f"  {i}. {t}")
        while True:
            try:
                choice = int(input(f"Select table (1-{len(available_tables)}): "))
                if 1 <= choice <= len(available_tables):
                    table = available_tables[choice-1]
                    break
            except ValueError:
                print(f"Invalid input. Please enter a number between 1 and {len(available_tables)}.")

    # Determine columns
    candidate_cols = [ent["column"] for ent in resolved_entities.values() if "column" in ent]
    candidate_cols = list(dict.fromkeys(candidate_cols)) if candidate_cols else list(table_columns.get(table, []))

    if candidate_cols:
        print(f"\n📊 Candidate columns in {table}:")
        for i, col in enumerate(candidate_cols, 1):
            print(f"  {i}. {col}")
        cols_input = input("Select columns by number (comma separated, or Enter for auto): ")
        if cols_input.strip():
            col_indices = [int(x.strip()) for x in cols_input.split(",") if x.strip().isdigit()]
            selected_cols = [candidate_cols[i-1] for i in col_indices if 1 <= i <= len(candidate_cols)]
        else:
            selected_cols = candidate_cols
    else:
        selected_cols = []

    return {
        "intent": intent,
        "metrics": metrics,
        "entities": resolved_entities,
        "filters": filters,
        "table": table,
        "columns": selected_cols
    }


def build_catalog(conn, table_columns, max_values=50):
    """
    Build catalog = {table: {column: [distinct values...]}} 
    from Postgres DB.
    """
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
                print(f"⚠️ Skipping {table}.{col} → {e}")
    cur.close()
    return catalog


def load_table_columns_pg(conn, tables):
    """
    Load column names for given tables from PostgreSQL.
    Returns a dict {table_name: [col1, col2, ...]}
    """
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


def entity_resolver_node(state: GraphState):
    """
    Resolves entities (products, distributors, etc.) FIRST,
    then provides annotated schema.
    """
    user_query = state["user_query"]
    catalog = state.get("catalog", {})
    table_columns = state.get("table_columns", {})

    reasoning = []

    # --- Try to resolve entities ---
    resolved = resolve_with_human_in_loop_pg(user_query, catalog, table_columns)

    # Safety: If resolver failed, fallback to raw query text
    if not resolved.get("entities"):
        reasoning.append(f"⚠️ No entity found in schema for '{user_query}'. Passing raw text downstream.")
        resolved = {
            "intent": "fallback",
            "entities": {"raw_text": user_query},
            "filters": {},
            "message": "Entity not found, using raw query."
        }

    # --- Annotated Schema (from file) ---
    try:
        with open("annotated_schema.md", "r", encoding="utf-8") as f:
            annotated_schema = f.read()
    except FileNotFoundError:
        annotated_schema = """
        ### tbl_primary
        - product_id → references tbl_product_master.product_erp_id
        - distributor_name : name of distributor
        - sales_order_date : date of order
        - invoiced_total_quantity : actual billed sales

        ### tbl_product_master
        - product_erp_id : unique product key
        - product : product description/name
        """

    resolved["message"] = "Entities resolved or fallback applied."
    resolved["thinking"] = reasoning

    return {
        "resolved": resolved,
        "annotated_schema": annotated_schema
    }