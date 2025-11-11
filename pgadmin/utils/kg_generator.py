import os
from django.db import connection
import json
import re

def get_table_schema(table_name):
    """Get detailed schema information for a table"""
    with connection.cursor() as cursor:
        # Get column information with data types
        cursor.execute(f"""
            SELECT 
                column_name,
                data_type,
                character_maximum_length,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position
        """, [table_name])
        
        columns = cursor.fetchall()
        
        # Get primary keys
        cursor.execute(f"""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary
        """, [table_name])
        
        primary_keys = [row[0] for row in cursor.fetchall()]
        
        # Get foreign keys
        cursor.execute(f"""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = %s
        """, [table_name])
        
        foreign_keys = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        
        # Get sample data (safely)
        try:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 5')
            sample_data = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
        except:
            sample_data = []
            column_names = []
        
        return {
            'columns': columns,
            'primary_keys': primary_keys,
            'foreign_keys': foreign_keys,
            'sample_data': sample_data,
            'column_names': column_names
        }


def generate_knowledge_graph_with_llm(tables):
    """Generate knowledge graph descriptions using OpenAI API with enhanced prompts"""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found, using basic generation")
        return generate_basic_knowledge_graph(tables)
    
    # Import OpenAI only if key exists
    try:
        import openai
        openai.api_key = api_key
    except ImportError:
        print("⚠️  OpenAI library not installed, using basic generation")
        return generate_basic_knowledge_graph(tables)
    
    knowledge_graph = {}
    
    for table in tables:
        try:
            schema_info = get_table_schema(table)
            
            # Prepare detailed column information
            column_details = []
            for col in schema_info['columns']:
                col_name = col[0]
                sql_type = col[1]
                max_length = col[2]
                is_nullable = col[3]
                default_val = col[4]
                
                detail = f"  - **{col_name}** ({sql_type}"
                if max_length:
                    detail += f"({max_length})"
                detail += ")"
                
                # Add constraints
                if col_name in schema_info['primary_keys']:
                    detail += " [PRIMARY KEY]"
                if col_name in schema_info['foreign_keys']:
                    fk_table, fk_col = schema_info['foreign_keys'][col_name]
                    detail += f" [FK → {fk_table}.{fk_col}]"
                if is_nullable == 'NO':
                    detail += " [NOT NULL]"
                if default_val:
                    detail += f" [DEFAULT: {default_val}]"
                
                column_details.append(detail)
            
            column_details_str = "\n".join(column_details)
            
            # Prepare sample data display
            sample_str = ""
            if schema_info['sample_data'] and schema_info['column_names']:
                sample_str = f"\n\n**Sample Data (first {min(3, len(schema_info['sample_data']))} rows):**\n"
                for i, row in enumerate(schema_info['sample_data'][:3], 1):
                    sample_str += f"\nRow {i}:\n"
                    for col_name, value in zip(schema_info['column_names'], row):
                        # Truncate long values
                        val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                        sample_str += f"  {col_name}: {val_str}\n"
            
            # Enhanced prompt with strict data type requirements
            prompt = f"""You are a database documentation expert. Analyze the table "{table}" and generate clear, business-friendly descriptions for each column.

**Table: {table}**

**Column Details:**
{column_details_str}
{sample_str}

**Task:**
For EACH column, provide:
1. **desc**: A clear, concise business description (1-2 sentences) explaining what this column represents in business terms. Focus on the PURPOSE and MEANING of the data, not technical details.
2. **datatype**: Choose EXACTLY ONE from this list based on the SQL type and business usage:
   - "Text" - for any text/string data (VARCHAR, CHAR, TEXT)
   - "Decimal" - for decimal numbers, currency, percentages (NUMERIC, DECIMAL, REAL, DOUBLE PRECISION, MONEY)
   - "Whole Number" - for integers, counts, IDs (INTEGER, BIGINT, SMALLINT)
   - "Boolean" - for true/false, yes/no values (BOOLEAN)
   - "Datetime" - for timestamps with time (TIMESTAMP, TIMESTAMP WITH TIME ZONE)
   - "Date" - for dates without time (DATE)

**Important Guidelines:**
- Make descriptions business-friendly and meaningful
- For ID columns, explain what entity they identify (e.g., "Unique identifier for each customer order")
- For foreign keys, explain the relationship (e.g., "Links to the customer who placed this order")
- For timestamps, explain what event they track (e.g., "When the order was created in the system")
- For status/category columns, mention what they represent (e.g., "Current status of the order: pending, completed, or cancelled")
- Be specific about what the column measures or tracks

**Response Format:**
Respond ONLY with valid JSON in this EXACT format (no markdown, no code blocks, no backticks):

{{
    "column_name_1": {{
        "desc": "Business-friendly description here",
        "datatype": "Text"
    }},
    "column_name_2": {{
        "desc": "Business-friendly description here",
        "datatype": "Whole Number"
    }}
}}

CRITICAL: Use ONLY the 6 datatype values listed above. Do not use any other values."""

            print(f"🤖 Generating knowledge graph for table: {table}...")
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a database documentation expert who generates accurate, business-friendly documentation. You ALWAYS respond with valid JSON and use ONLY the specified datatype values."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean markdown code blocks if present
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            response_text = response_text.strip()
            
            # Parse JSON
            column_descriptions = json.loads(response_text)
            
            # Validate and normalize datatypes
            valid_datatypes = ["Text", "Decimal", "Whole Number", "Boolean", "Datetime", "Date"]
            for col_name, col_data in column_descriptions.items():
                if col_data.get("datatype") not in valid_datatypes:
                    # Fallback to basic mapping
                    col_data["datatype"] = map_sql_type_to_semantic(
                        next((c[1] for c in schema_info['columns'] if c[0] == col_name), "text")
                    )
            
            knowledge_graph[table] = column_descriptions
            
            print(f"✅ Generated knowledge graph for table: {table}")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error for {table}: {e}")
            print(f"Response preview: {response_text[:300]}...")
            # Fallback to basic generation
            schema_info = get_table_schema(table)
            knowledge_graph[table] = generate_basic_columns(table, schema_info)
            
        except Exception as e:
            print(f"❌ Error generating KG for {table}: {e}")
            # Fallback to basic generation
            schema_info = get_table_schema(table)
            knowledge_graph[table] = generate_basic_columns(table, schema_info)
    
    return knowledge_graph


def generate_basic_columns(table, schema_info):
    """Generate basic column descriptions without LLM"""
    columns_data = {}
    
    for col in schema_info['columns']:
        col_name = col[0]
        sql_type = col[1]
        
        # Generate basic description
        desc_parts = []
        
        if col_name in schema_info['primary_keys']:
            desc_parts.append(f"Primary key - unique identifier for each {table.replace('tbl_', '').replace('_', ' ')} record")
        elif col_name in schema_info['foreign_keys']:
            fk_table, fk_col = schema_info['foreign_keys'][col_name]
            desc_parts.append(f"References {fk_table.replace('tbl_', '').replace('_', ' ')} - links to related data")
        else:
            desc_parts.append(f"Stores {col_name.replace('_', ' ')} information")
        
        columns_data[col_name] = {
            "desc": " ".join(desc_parts),
            "datatype": map_sql_type_to_semantic(sql_type)
        }
    
    return columns_data


def generate_basic_knowledge_graph(tables):
    """Fallback: Generate basic knowledge graph without LLM"""
    knowledge_graph = {}
    
    for table in tables:
        schema_info = get_table_schema(table)
        knowledge_graph[table] = generate_basic_columns(table, schema_info)
    
    return knowledge_graph


def map_sql_type_to_semantic(sql_type):
    """Map SQL data types to semantic types (restricted list)"""
    sql_type = sql_type.lower()
    
    # Whole Number types
    if any(t in sql_type for t in ['integer', 'bigint', 'smallint', 'int']):
        return 'Whole Number'
    
    # Decimal types
    if any(t in sql_type for t in ['numeric', 'decimal', 'real', 'double precision', 'money', 'float']):
        return 'Decimal'
    
    # Date types
    if sql_type == 'date':
        return 'Date'
    
    # Datetime types
    if any(t in sql_type for t in ['timestamp', 'datetime', 'time']):
        return 'Datetime'
    
    # Boolean types
    if any(t in sql_type for t in ['boolean', 'bool']):
        return 'Boolean'
    
    # Everything else is Text
    return 'Text'