"""
Django Model Loader for LangGraph
Loads module configuration from Django models
"""
import os
import sys
import django

# Setup Django
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from django.conf import settings


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


def load_module_config(module_id):
    """
    Load complete module configuration including:
    - Tables
    - Knowledge Graph (column descriptions)
    - Relationships (table joins)
    - Metrics
    - RCA (business context)
    - POS Tagging (entity types)
    - Extra Suggestions
    """
    from pgadmin.models import Module
    
    try:
        module = Module.objects.get(id=module_id)
    except Module.DoesNotExist:
        raise ValueError(f"Module {module_id} not found")
    
    # Build annotated schema from knowledge graph
    annotated_schema = build_annotated_schema(module.knowledge_graph_data)
    
    # Build relationships string for SQL context
    relationships_text = build_relationships_text(module.relationships)
    
    # Build entity inference map from POS tagging
    entity_inference_map = build_entity_map(module.pos_tagging)
    
    # Build RCA context
    rca_context = build_rca_context(module.rca_list)
    
    config = {
        'module_id': module.id,
        'module_name': module.name,
        'tables': module.tables or [],
        'knowledge_graph_data': module.knowledge_graph_data or {},
        'relationships': module.relationships or [],
        'metrics': module.metrics_data or {},
        'rca_list': module.rca_list or [],
        'pos_tagging': module.pos_tagging or [],
        'extra_suggestions': module.extra_suggestions or '',
        
        # Processed/formatted data for LLM
        'annotated_schema': annotated_schema,
        'relationships_text': relationships_text,
        'entity_inference_map': entity_inference_map,
        'rca_context': rca_context,
        
        # Metadata
        'created_at': module.created_at.isoformat(),
        'kg_auto_generated': module.kg_auto_generated,
    }
    
    return config


def build_annotated_schema(knowledge_graph_data):
    """
    Build annotated schema text from knowledge graph
    
    Format:
    Table: customers
      - customer_id (identifier): Unique customer identifier
      - customer_name (text): Full name of the customer
      - email (text): Customer email address
    """
    if not knowledge_graph_data:
        return "No schema annotations available."
    
    lines = []
    for table, columns in knowledge_graph_data.items():
        lines.append(f"\nTable: {table}")
        for column, info in columns.items():
            desc = info.get('desc', 'No description')
            datatype = info.get('datatype', 'unknown')
            lines.append(f"  - {column} ({datatype}): {desc}")
    
    return "\n".join(lines)


def build_relationships_text(relationships):
    """
    Build relationships text for SQL JOIN context
    
    Format:
    - customers.customer_id (1:N) orders.customer_id
    - orders.product_id (N:1) products.product_id
    """
    if not relationships:
        return "No relationships defined."
    
    lines = []
    for rel in relationships:
        left = f"{rel['left_table']}.{rel['left_column']}"
        right = f"{rel['right_table']}.{rel['right_column']}"
        rel_type = rel.get('type', 'one-to-many')
        
        # Format relationship type
        type_map = {
            'one-to-one': '1:1',
            'one-to-many': '1:N',
            'many-to-one': 'N:1',
            'many-to-many': 'N:N'
        }
        type_symbol = type_map.get(rel_type, '1:N')
        
        lines.append(f"  - {left} ({type_symbol}) {right}")
    
    return "Relationships:\n" + "\n".join(lines) if lines else "No relationships defined."


def build_entity_map(pos_tagging):
    """
    Build entity inference map from POS tagging
    
    Returns dict: {entity_type: (column, table)}
    """
    entity_map = {}
    
    for pos in pos_tagging:
        entity_name = pos.get('name', '').lower()
        reference = pos.get('reference', '')
        
        if entity_name and reference:
            # Try to extract column/table from reference
            # Format could be: "product_name column" or "tbl_products.product_name"
            if '.' in reference:
                parts = reference.split('.')
                if len(parts) == 2:
                    entity_map[entity_name] = (parts[1], parts[0])
            else:
                # Just column name, will infer table later
                entity_map[entity_name] = (entity_name, None)
    
    return entity_map


def build_rca_context(rca_list):
    """
    Build RCA context text for business understanding
    
    Format:
    Business Context:
    1. [Title]: [Content]
    2. [Title]: [Content]
    """
    if not rca_list:
        return ""
    
    lines = ["Business Context:"]
    for i, rca in enumerate(rca_list, 1):
        title = rca.get('title', 'Untitled')
        content = rca.get('content', '')
        if title and content:
            lines.append(f"{i}. {title}: {content}")
    
    return "\n".join(lines) if len(lines) > 1 else ""