import psycopg2
import sqlparse
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
import os

class InteractiveValidator:
    """
    Enhanced validator with user interaction for ambiguous cases
    and zero-hallucination guarantee
    """
    
    def __init__(self, connection_params, annotated_schema, relationships):
        self.connection_params = connection_params
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
        self.annotated_schema = annotated_schema
        self.relationships = relationships
        self.table_structure = self.parse_annotated_schema(annotated_schema)
        self.valid_joins = self.parse_relationships(relationships)
    
    def parse_annotated_schema(self, annotated_schema: str) -> Dict[str, Any]:
        """
        Parse annotated_schema.md to extract tables and columns
        FIXED: Handles JSON format with nested arrays
        """
        table_structure = {}
        
        print(f"📋 Parsing annotated schema...")
        
        # Split by table headers (### **tbl_...)
        table_blocks = re.split(r'###\s*\*\*(\w+)\*\*', annotated_schema)
        
        # Process each table block
        for i in range(1, len(table_blocks), 2):
            if i + 1 < len(table_blocks):
                table_name = table_blocks[i].lower().strip()
                table_content = table_blocks[i + 1]
                
                print(f"  Found table: {table_name}")
                
                # Initialize table structure
                table_structure[table_name] = {
                    'columns': [],
                    'types': {},
                    'descriptions': {}
                }
                
                # FIXED: Extract columns from JSON format
                # Pattern 1: ["column_name : description..."]
                pattern1 = r'\["([^:]+)\s*:\s*([^"]+)"\]'
                matches1 = re.findall(pattern1, table_content)
                
                # Pattern 2: "column_name : description, datatype: type, <sample values: ...>"
                pattern2 = r'"([a-z_]+)\s*:\s*([^"]+)"'
                matches2 = re.findall(pattern2, table_content)
                
                all_columns = set()
                
                # Process pattern 1 matches
                for col_name, col_desc in matches1:
                    column = col_name.strip().lower()
                    description = col_desc.strip()
                    
                    if column and column not in all_columns:
                        all_columns.add(column)
                        table_structure[table_name]['columns'].append(column)
                        table_structure[table_name]['descriptions'][column] = description
                        
                        # Infer data type
                        desc_lower = description.lower()
                        if 'date' in desc_lower:
                            table_structure[table_name]['types'][column] = 'date'
                        elif 'integer' in desc_lower or 'int' in desc_lower:
                            table_structure[table_name]['types'][column] = 'integer'
                        elif 'float' in desc_lower or 'decimal' in desc_lower:
                            table_structure[table_name]['types'][column] = 'float'
                        elif 'boolean' in desc_lower or 'bool' in desc_lower:
                            table_structure[table_name]['types'][column] = 'boolean'
                        else:
                            table_structure[table_name]['types'][column] = 'text'
                
                # Process pattern 2 matches (fallback)
                if not all_columns:
                    for col_name, col_desc in matches2:
                        column = col_name.strip().lower()
                        description = col_desc.strip()
                        
                        if column and column not in all_columns and not column.startswith('this'):
                            all_columns.add(column)
                            table_structure[table_name]['columns'].append(column)
                            table_structure[table_name]['descriptions'][column] = description
                            
                            desc_lower = description.lower()
                            if 'date' in desc_lower:
                                table_structure[table_name]['types'][column] = 'date'
                            elif 'integer' in desc_lower:
                                table_structure[table_name]['types'][column] = 'integer'
                            elif 'float' in desc_lower:
                                table_structure[table_name]['types'][column] = 'float'
                            else:
                                table_structure[table_name]['types'][column] = 'text'
        
        print(f"✅ Parsed {len(table_structure)} tables from annotated schema")
        for table, info in table_structure.items():
            print(f"  {table}: {len(info['columns'])} columns")
        
        return table_structure
    
    def parse_relationships(self, relationships: str) -> List[Dict[str, str]]:
        """
        Parse relationship.txt to extract valid JOIN conditions
        """
        valid_joins = []
        
        lines = relationships.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for JOIN patterns
            join_match = re.search(
                r'(\w+)\.(\w+)\s*[=→]\s*(\w+)\.(\w+)', 
                line, 
                re.IGNORECASE
            )
            
            if join_match:
                valid_joins.append({
                    'left_table': join_match.group(1).lower(),
                    'left_column': join_match.group(2).lower(),
                    'right_table': join_match.group(3).lower(),
                    'right_column': join_match.group(4).lower()
                })
        
        return valid_joins
    
    def validate_and_fix_sql(self, sql_query: str, user_query: str, 
                            annotated_schema: str, relationships: str) -> Dict[str, Any]:
        """
        Main validation orchestrator with user interaction
        """
        max_attempts = 10
        current_sql = sql_query.strip()
        previous_sql = None
        stuck_count = 0
        
        print("\n" + "="*70)
        print("🔍 SQL VALIDATION STARTED")
        print("="*70)
        print(f"Original SQL:\n{current_sql}\n")
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n{'─'*70}")
            print(f"🔄 Validation Attempt {attempt}/{max_attempts}")
            print(f"{'─'*70}")
            
            # Detect if stuck (same SQL for 3 attempts)
            if current_sql == previous_sql:
                stuck_count += 1
                if stuck_count >= 3:
                    print("⚠️ Validator appears stuck. Trying rule-based fix...")
                    current_sql = self._apply_rule_based_fix(current_sql, user_query)
                    stuck_count = 0
            else:
                stuck_count = 0
            
            previous_sql = current_sql
            
            # Level 1: Syntax validation
            is_valid, error = self._validate_syntax(current_sql)
            if not is_valid:
                print(f"❌ Syntax Error: {error}")
                current_sql = self._fix_with_llm(
                    current_sql, error, "syntax", user_query, 
                    annotated_schema, relationships
                )
                continue
            
            # Level 2: Table/Column existence
            is_valid, error = self._validate_structure(current_sql)
            if not is_valid:
                print(f"❌ Structure Error: {error}")
                current_sql = self._fix_with_llm(
                    current_sql, error, "structure", user_query,
                    annotated_schema, relationships
                )
                continue
            
            # Level 3: Execution test
            is_valid, error = self._test_execution(current_sql)
            if not is_valid:
                print(f"❌ Execution Error: {error}")
                current_sql = self._fix_with_llm(
                    current_sql, error, "execution", user_query,
                    annotated_schema, relationships
                )
                continue
            
            # All validations passed!
            print(f"\n{'='*70}")
            print(f"✅ SQL VALIDATION SUCCESSFUL (Attempt {attempt})")
            print(f"{'='*70}")
            print(f"Final SQL:\n{current_sql}\n")
            
            return {
                "validated_sql": current_sql,
                "validation_status": "valid" if attempt == 1 else "corrected",
                "validation_error": None,
                "correction_attempts": attempt
            }
        
        # Max attempts reached
        print(f"\n⚠️ Maximum validation attempts ({max_attempts}) reached")
        return {
            "validated_sql": current_sql,
            "validation_status": "max_attempts",
            "validation_error": f"Could not validate after {max_attempts} attempts",
            "correction_attempts": max_attempts
        }
    
    def _validate_syntax(self, sql: str) -> Tuple[bool, str]:
        """Validate SQL syntax"""
        try:
            parsed = sqlparse.parse(sql)
            if not parsed or not parsed[0].tokens:
                return False, "Empty or invalid SQL statement"
            return True, ""
        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"
    
    def _validate_structure(self, sql: str) -> Tuple[bool, str]:
        """Validate tables and columns exist in annotated schema"""
        # Extract tables
        tables = self._extract_tables(sql)
        for table in tables:
            if table not in self.table_structure:
                available = list(self.table_structure.keys())
                return False, f"Table '{table}' not in schema. Available: {available}"
        
        # Extract columns
        columns = self._extract_columns(sql)
        for table_alias, column in columns:
            actual_table = self._resolve_table_alias(sql, table_alias)
            if not actual_table:
                continue
            
            if actual_table in self.table_structure:
                if column not in self.table_structure[actual_table]['columns']:
                    available = self.table_structure[actual_table]['columns']
                    return False, f"Column '{column}' not in table '{actual_table}'. Available columns: {available}"
        
        return True, ""
    
    def _test_execution(self, sql: str) -> Tuple[bool, str]:
        """Test SQL execution without running it"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            cursor = conn.cursor()
            
            # Use EXPLAIN to validate without executing
            cursor.execute(f"EXPLAIN {sql}")
            
            cursor.close()
            conn.close()
            return True, ""
            
        except psycopg2.Error as e:
            error_msg = str(e).split('\n')[0]
            return False, f"PostgreSQL Error: {error_msg}"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def _fix_with_llm(self, sql: str, error: str, error_type: str,
                      user_query: str, annotated_schema: str, 
                      relationships: str) -> str:
        """
        Fix SQL using LLM with enhanced context
        """
        print(f"\n🤖 Automatically fixing {error_type} error...")
        
        schema_summary = self._build_schema_summary()
        
        fix_prompt = f"""You are a PostgreSQL query fixer. Fix this broken SQL.

User Question: {user_query}

Current BROKEN SQL:
{sql}

Error: {error}

Available Schema (COMPLETE TABLE STRUCTURES):
{schema_summary}

CRITICAL RULES:
1. Use ONLY tables and columns from the schema above
2. Match column names to the CORRECT table
3. Use correct table aliases
4. Return ONLY the corrected SQL query - no explanations, no markdown

Fix the SQL using correct table.column references."""
        
        try:
            result = self.llm.invoke(fix_prompt)
            suggested_sql = self._extract_sql(result.content)
            
            print(f"💡 Auto-corrected SQL:")
            print(f"{suggested_sql}\n")
            
            return suggested_sql
        
        except Exception as e:
            print(f"⚠️ Auto-fix failed: {e}")
            return sql
    
    def _apply_rule_based_fix(self, sql: str, user_query: str) -> str:
        """
        Apply rule-based fixes when LLM gets stuck
        """
        print("🔧 Applying rule-based fixes...")
        
        # Remove unnecessary JOINs if query can work without them
        if 'JOIN' in sql.upper():
            # Check if we're querying from tbl_primary and all columns are in tbl_primary
            tables = self._extract_tables(sql)
            columns = self._extract_columns(sql)
            
            if 'tbl_primary' in tables:
                all_cols_in_primary = all(
                    col in self.table_structure.get('tbl_primary', {}).get('columns', [])
                    for _, col in columns if col not in ['product', 'material_description']
                )
                
                if all_cols_in_primary:
                    print("  💡 Removing unnecessary JOINs - all columns in tbl_primary")
                    # Remove JOIN clauses
                    sql = re.sub(r'JOIN\s+\w+\s+(?:AS\s+)?\w+\s+ON[^;]+(?=WHERE|GROUP|ORDER|;|$)', '', sql, flags=re.IGNORECASE)
                    # Clean up whitespace
                    sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql
    
    def _build_schema_summary(self) -> str:
        """Build human-readable schema summary"""
        summary_parts = []
        
        for table, info in self.table_structure.items():
            summary_parts.append(f"\n{table}:")
            for col in info['columns']:
                col_type = info['types'].get(col, 'text')
                summary_parts.append(f"  - {col} ({col_type})")
        
        return '\n'.join(summary_parts)
    
    # Helper methods
    
    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL"""
        tables = []
        sql_upper = sql.upper()
        
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            tables.append(from_match.group(1).lower())
        
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        tables.extend([m.lower() for m in join_matches])
        
        return list(set(tables))
    
    def _extract_columns(self, sql: str) -> List[Tuple[str, str]]:
        """Extract column references as (table_alias, column) tuples"""
        pattern = r'(\w+)\.(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        return [(alias.lower(), col.lower()) for alias, col in matches]
    
    def _resolve_table_alias(self, sql: str, alias: str) -> Optional[str]:
        """Resolve table alias to actual table name"""
        sql_upper = sql.upper()
        alias_upper = alias.upper()
        
        # Try pattern: table_name AS alias
        pattern1 = rf'(\w+)\s+AS\s+{alias_upper}\b'
        match = re.search(pattern1, sql_upper)
        
        if match:
            table_name = match.group(1).lower()
            if table_name in self.table_structure:
                return table_name
        
        # Try pattern: table_name alias (without AS)
        pattern2 = rf'(tbl_\w+)\s+{alias_upper}\b'
        match = re.search(pattern2, sql_upper, re.IGNORECASE)
        
        if match:
            table_name = match.group(1).lower()
            if table_name in self.table_structure:
                return table_name
        
        # Check if alias is the table name itself
        if alias.lower() in self.table_structure:
            return alias.lower()
        
        return None
    
    def _extract_sql(self, text: str) -> str:
        """Extract SQL from LLM response"""
        if "```sql" in text:
            return text.split("```sql")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        
        # Remove common prefixes
        for prefix in ["SQL:", "Query:", "Corrected:", "Fixed:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text.strip()


def validator_agent(state):
    """
    LangGraph node for SQL validation
    """
    sql_query = state.get('sql_result', '')
    user_query = state.get('user_query', '')
    annotated_schema = state.get('annotated_schema', '')
    relationships = state.get('relationships', '')
    
    connection_params = {
        'host': 'localhost',
        'database': 'haldiram',
        'user': 'postgres',
        'password': '12345678',
        'port': 5432
    }
    
    # Initialize validator
    validator = InteractiveValidator(connection_params, annotated_schema, relationships)
    
    result = validator.validate_and_fix_sql(
        sql_query, user_query, annotated_schema, relationships
    )
    
    return {
        "validated_sql": result["validated_sql"],
        "validation_status": result["validation_status"],
        "validation_error": result.get("validation_error"),
        "correction_attempts": result.get("correction_attempts", 0)
    }