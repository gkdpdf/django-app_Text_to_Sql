"""
Knowledge Graph Generator using OpenAI
"""
import os
from openai import OpenAI
import json
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_knowledge_graph(table_columns):
    """
    Generate knowledge graph descriptions for database tables and columns
    
    Args:
        table_columns: Dict of {table_name: [{"name": col_name, "type": col_type}, ...]}
    
    Returns:
        Dict of {table_name: {column_name: {"desc": description, "datatype": category}, ...}}
    """
    logger.info(f"🤖 Starting KG generation for {len(table_columns)} tables")
    
    # Build prompt
    prompt = """You are a database documentation expert. Analyze the following database schema and provide comprehensive descriptions.

DATABASE SCHEMA:
"""
    
    for table, columns in table_columns.items():
        prompt += f"\n\nTable: {table}\nColumns:\n"
        for col in columns:
            prompt += f"  - {col['name']} ({col['type']})\n"
    
    prompt += """

TASK: For each table and column, provide:
1. Table description (what the table represents, its purpose)
2. Column descriptions (what each column stores, its business meaning)
3. Column categories (identifier, text, number, date, boolean, etc.)

Return ONLY a valid JSON object in this exact format:
{
  "table_name": {
    "TABLE_INFO": {
      "desc": "Description of what this table represents",
      "datatype": "meta"
    },
    "column_name": {
      "desc": "Detailed description of this column",
      "datatype": "identifier|text|number|date|boolean|etc"
    }
  }
}

IMPORTANT RULES:
- Use "TABLE_INFO" key for table-level description
- Be specific and business-focused
- Use proper categories: identifier, text, number, date, datetime, boolean, json, array
- DO NOT include markdown formatting or code blocks
- Return ONLY the JSON object, nothing else
"""
    
    try:
        logger.info("📤 Sending request to OpenAI...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a database documentation expert. You provide clear, concise, and accurate descriptions of database schemas. You always return valid JSON without any markdown formatting."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content.strip()
        logger.info(f"📥 Received response from OpenAI ({len(content)} chars)")
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        # Parse JSON
        kg_data = json.loads(content)
        
        logger.info(f"✅ Successfully generated KG for {len(kg_data)} tables")
        
        # Validate structure
        for table, columns in kg_data.items():
            if not isinstance(columns, dict):
                logger.warning(f"⚠️ Invalid structure for table {table}")
                continue
            
            for col_name, col_data in columns.items():
                if not isinstance(col_data, dict):
                    logger.warning(f"⚠️ Invalid column data for {table}.{col_name}")
                    continue
                
                if "desc" not in col_data:
                    col_data["desc"] = ""
                if "datatype" not in col_data:
                    col_data["datatype"] = "unknown"
        
        return kg_data
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse JSON response: {e}")
        logger.error(f"Response content: {content[:500]}...")
        raise Exception(f"Failed to parse AI response as JSON: {str(e)}")
    
    except Exception as e:
        logger.error(f"❌ Error generating knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Knowledge graph generation failed: {str(e)}")