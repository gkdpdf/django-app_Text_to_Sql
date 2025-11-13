"""
Dynamic Module Configuration Loader - NO HARDCODING
"""
from django.apps import apps
from django.db import connection
import logging
import os

logger = logging.getLogger(__name__)

def load_module_config(module_id: int):
    """Load complete module configuration"""
    try:
        Module = apps.get_model('pgadmin', 'Module')
        module = Module.objects.get(id=module_id)
        
        tables = module.tables or []
        if not tables:
            raise ValueError("No tables selected")
        
        logger.info(f"📦 Loading: {module.name}")
        
        # Build schema
        annotated_schema = build_annotated_schema(
            module.knowledge_graph_data or {}, 
            tables,
            module.name
        )
        
        # Build relationships
        relationships = build_relationships(tables)
        
        # Build entity map
        entity_map = build_entity_inference_map(
            module.knowledge_graph_data or {},
            tables,
            module.pos_tagging or []
        )
        
        # Get metrics
        metrics = module.metrics_data or {}
        
        # Get RCAs
        rca_context = ""
        if module.rca_list:
            rca_parts = []
            for rca in module.rca_list:
                title = rca.get('title', '')
                content = rca.get('content', '')
                if title and content:
                    rca_parts.append(f"**{title}**\n{content}")
            rca_context = "\n\n".join(rca_parts)
        
        logger.info(f"✅ Loaded: {len(entity_map)} entity types")
        
        return {
            "module_id": module_id,
            "module_name": module.name,
            "tables": tables,
            "annotated_schema": annotated_schema,
            "relationships": relationships,
            "entity_inference_map": entity_map,
            "metrics": metrics,
            "rca_context": rca_context,
            "extra_context": module.extra_suggestions or "",
            "knowledge_graph": module.knowledge_graph_data or {}
        }
        
    except Exception as e:
        logger.error(f"❌ Error loading module: {e}", exc_info=True)
        raise


def build_annotated_schema(kg: dict, tables: list, name: str) -> str:
    """Build schema documentation"""
    if not kg:
        return f"# {name}\n\nNo schema available."
    
    lines = [f"# {name} - Schema", f"\n{len(tables)} tables.\n"]
    
    for table in tables:
        if table not in kg:
            continue
        
        lines.append(f"\n## {table}\n")
        
        for col, info in kg.get(table, {}).items():
            desc = info.get('desc', 'No description')
            dtype = info.get('datatype', 'Text')
            lines.append(f"- **{col}** ({dtype}): {desc}")
    
    return "\n".join(lines)


def build_relationships(tables: list) -> str:
    """Build relationships"""
    lines = ["# Relationships\n"]
    
    with connection.cursor() as cursor:
        for table in tables:
            try:
                cursor.execute("""
                    SELECT kcu.column_name, ccu.table_name, ccu.column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_name = %s
                """, [table])
                
                fks = cursor.fetchall()
                if fks:
                    lines.append(f"\n## {table}")
                    for fk in fks:
                        lines.append(f"- `{fk[0]}` → `{fk[1]}.{fk[2]}`")
            except:
                pass
    
    return "\n".join(lines)


def build_entity_inference_map(kg: dict, tables: list, pos_tagging: list) -> dict:
    """Build entity type mapping"""
    entity_map = {}
    
    # Use POS tagging
    for pos in pos_tagging:
        entity_type = pos.get('name', '').strip().lower()
        reference = pos.get('reference', '').strip()
        
        if not entity_type or not reference:
            continue
        
        if '.' in reference:
            parts = reference.split('.', 1)
            table, column = parts[0], parts[1]
            if table in tables:
                entity_map[entity_type] = (column, table)
        else:
            for table in tables:
                if table in kg and reference in kg[table]:
                    entity_map[entity_type] = (reference, table)
                    break
    
    # Infer from descriptions
    if kg:
        for table in tables:
            if table not in kg:
                continue
            
            for col, info in kg[table].items():
                desc = info.get('desc', '').lower()
                
                if ('product' in desc or 'item' in desc) and 'product' not in entity_map:
                    entity_map['product'] = (col, table)
                
                if any(w in desc for w in ['distributor', 'dealer']) and 'distributor' not in entity_map:
                    entity_map['distributor'] = (col, table)
                
                if 'customer' in desc and 'customer' not in entity_map:
                    entity_map['customer'] = (col, table)
    
    return entity_map


def get_db_credentials():
    """Get DB credentials from Django"""
    from django.conf import settings
    
    db = settings.DATABASES.get('default', {})
    
    return {
        'host': db.get('HOST') or os.getenv('PG_HOST', 'localhost'),
        'database': db.get('NAME') or os.getenv('PG_DBNAME', 'postgres'),
        'user': db.get('USER') or os.getenv('PG_USER', 'postgres'),
        'password': db.get('PASSWORD') or os.getenv('PG_PASSWORD', ''),
        'port': db.get('PORT', 5432)
    }