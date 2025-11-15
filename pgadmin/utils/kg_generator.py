"""
Knowledge Graph Generator
"""
import os
import logging

logger = logging.getLogger(__name__)


def generate_knowledge_graph_with_llm(tables, selected_columns):
    """
    Generate knowledge graph using OpenAI API
    
    Args:
        tables: List of table names
        selected_columns: Dict of {table_name: [column_names]}
    
    Returns:
        Dict of {table_name: {column_name: {desc: str, datatype: str}}}
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("No OpenAI API key found - using basic generation")
        return generate_basic_knowledge_graph(tables, selected_columns)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        kg_data = {}
        
        for table in tables:
            columns = selected_columns.get(table, [])
            if not columns:
                continue
            
            kg_data[table] = {}
            
            # Create prompt
            prompt = f"""Generate descriptions and data types for these database columns:

Table: {table}
Columns: {', '.join(columns)}

For each column, provide:
1. A clear description of what the column represents
2. The data type/category (e.g., identifier, text, number, date, boolean, etc.)

Format as JSON:
{{
  "column_name": {{"desc": "description", "datatype": "type"}}
}}
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a database documentation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            for column in columns:
                if column in result:
                    kg_data[table][column] = result[column]
                else:
                    kg_data[table][column] = {
                        "desc": f"{column} column",
                        "datatype": "unknown"
                    }
        
        return kg_data
        
    except Exception as e:
        logger.error(f"Error generating KG with AI: {e}")
        return generate_basic_knowledge_graph(tables, selected_columns)


def generate_basic_knowledge_graph(tables, selected_columns):
    """
    Generate basic knowledge graph without AI
    """
    kg_data = {}
    
    for table in tables:
        columns = selected_columns.get(table, [])
        if not columns:
            continue
        
        kg_data[table] = {}
        
        for column in columns:
            # Generate basic description based on column name
            desc = column.replace('_', ' ').title()
            
            # Guess datatype based on common patterns
            datatype = "text"
            if any(x in column.lower() for x in ['id', 'code', 'number']):
                datatype = "identifier"
            elif any(x in column.lower() for x in ['date', 'time', 'created', 'updated']):
                datatype = "date/time"
            elif any(x in column.lower() for x in ['amount', 'price', 'quantity', 'total']):
                datatype = "number"
            elif any(x in column.lower() for x in ['is_', 'has_', 'active', 'enabled']):
                datatype = "boolean"
            
            kg_data[table][column] = {
                "desc": f"{desc} field",
                "datatype": datatype
            }
    
    return kg_data