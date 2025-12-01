"""
Django Model Loader for LangGraph - PRODUCTION READY
Auto-discovers tables from database with proper filtering
VERSION: Final - No TABLE_INFO errors
"""
import os
import sys
import django

# Setup Django
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

if not django.apps.apps.ready:
    django.setup()

from django.conf import settings
import psycopg2


def get_db_credentials():
    """Get database credentials from Django settings"""
    db_config = settings.DATABASES['default']
    
    return {
        'host': db_config.get('HOST', 'localhost'),
        'database': db_config['NAME'],
        'user': db_config['USER'],
        'password': db_config['PASSWORD'],
        'port': db_config.get('PORT', 5432)
    }


def get_table_columns_from_database(table_name):
    """Get columns for a table - filters out system columns"""
    db_creds = get_db_credentials()
    conn = psycopg2.connect(**db_creds)
    
    # Columns to skip
    skip_columns = {
        'TABLE_INFO', 'tableoid', 'xmin', 'cmin', 
        'xmax', 'cmax', 'ctid'
    }
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        
        columns = []
        for row in cursor.fetchall():
            col_name = row[0]
            
            # Skip system columns
            if col_name in skip_columns or col_name.startswith('_') or ' ' in col_name:
                continue
            
            columns.append({
                'name': col_name,
                'type': row[1],
                'description': ''
            })
        
        cursor.close()
        conn.close()
        return columns
        
    except Exception as e:
        if conn:
            conn.close()
        return []


def build_table_columns(knowledge_graph_data=None, module_tables=None):
    """Build table_columns structure with proper filtering"""
    table_columns = {}
    skip_columns = {'TABLE_INFO', 'tableoid', 'xmin', 'cmin', 'xmax', 'cmax', 'ctid'}
    
    # Method 1: Use knowledge_graph_data
    if knowledge_graph_data:
        for table, columns in knowledge_graph_data.items():
            table_columns[table] = []
            for column, info in columns.items():
                if column in skip_columns or column.startswith('_') or ' ' in column:
                    continue
                
                table_columns[table].append({
                    'name': column,
                    'type': info.get('datatype', 'unknown'),
                    'description': info.get('desc', '')
                })
        
        if table_columns:
            return table_columns
    
    # Method 2: Use module.tables
    if module_tables:
        for table in module_tables:
            columns = get_table_columns_from_database(table)
            if columns:
                table_columns[table] = columns
        if table_columns:
            return table_columns
    
    # Method 3: Introspect database
    db_creds = get_db_credentials()
    conn = psycopg2.connect(**db_creds)
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        for row in cursor.fetchall():
            table = row[0]
            if table.startswith(('django_', 'auth_', 'pg_')):
                continue
            
            columns = get_table_columns_from_database(table)
            if columns:
                table_columns[table] = columns
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        if conn:
            conn.close()
    
    return table_columns


def load_module_config(module_id):
    """Load complete module configuration"""
    from pgadmin.models import Module
    
    try:
        module = Module.objects.get(id=module_id)
    except Module.DoesNotExist:
        raise ValueError(f"Module with ID {module_id} not found")
    
    kg_data = module.knowledge_graph_data or {}
    module_tables = module.tables or []
    
    table_columns = build_table_columns(kg_data, module_tables)
    
    if not table_columns:
        raise ValueError("No tables found!")
    
    annotated_schema = ""
    for table, columns in table_columns.items():
        annotated_schema += f"\nTable: {table}\n"
        for col in columns:
            annotated_schema += f"  - {col['name']} ({col['type']})\n"
    
    relationships_text = "No relationships defined."
    if module.relationships:
        lines = []
        for rel in module.relationships:
            left = f"{rel['left_table']}.{rel['left_column']}"
            right = f"{rel['right_table']}.{rel['right_column']}"
            lines.append(f"  - {left} → {right}")
        if lines:
            relationships_text = "Relationships:\n" + "\n".join(lines)
    
    return {
        'module_id': module.id,
        'module_name': module.name,
        'tables': list(table_columns.keys()),
        'table_columns': table_columns,
        'knowledge_graph_data': kg_data,
        'relationships': module.relationships or [],
        'metrics': module.metrics_data or {},
        'rca_list': module.rca_list or [],
        'pos_tagging': module.pos_tagging or [],
        'extra_suggestions': module.extra_suggestions or '',
        'annotated_schema': annotated_schema,
        'relationships_text': relationships_text,
        'entity_inference_map': {},
        'rca_context': '',
        'created_at': module.created_at.isoformat() if hasattr(module, 'created_at') else '',
        'kg_auto_generated': getattr(module, 'kg_auto_generated', False),
    }