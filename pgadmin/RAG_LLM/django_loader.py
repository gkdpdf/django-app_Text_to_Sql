"""
Dynamic Module Configuration Loader - COMPLETE VERSION
"""
from django.apps import apps
from django.db import connection
import logging
import os

logger = logging.getLogger(__name__)

def load_module_config(module_id: int):
    """Load complete module configuration from database"""
    try:
        Module = apps.get_model('pgadmin', 'Module')
        module = Module.objects.get(id=module_id)
        
        tables = module.tables or []
        if not tables:
            raise ValueError("No tables selected for this module")
        
        logger.info(f"📦 Loading module: {module.name}")
        logger.info(f"   Tables: {tables}")
        
        # Build annotated schema from knowledge graph
        annotated_schema = build_annotated_schema(
            module.knowledge_graph_data or {}, 
            tables,
            module.name
        )
        
        # Build relationships
        relationships = build_relationships(tables)
        
        # Build entity inference map
        entity_map = build_entity_inference_map(
            module.knowledge_graph_data or {},
            tables,
            module.pos_tagging or []
        )
        
        # Get metrics
        metrics = module.metrics_data or {}
        
        # Get RCAs for business context
        rca_context = ""
        if module.rca_list:
            rca_parts = []
            for rca in module.rca_list:
                title = rca.get('title', 'Context')
                content = rca.get('content', '')
                if title and content:
                    rca_parts.append(f"**{title}**\n{content}")
            rca_context = "\n\n".join(rca_parts)
        
        logger.info(f"✅ Module loaded successfully")
        logger.info(f"   Entity types: {list(entity_map.keys())}")
        logger.info(f"   Metrics: {list(metrics.keys())}")
        
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
        logger.error(f"❌ Error loading module {module_id}: {e}", exc_info=True)
        raise


def build_annotated_schema(kg: dict, tables: list, name: str) -> str:
    """Build detailed schema documentation"""
    if not kg:
        logger.warning("Empty knowledge graph!")
        return f"# {name} - Database Schema\n\nNo schema documentation available."
    
    lines = [
        f"# {name} - Database Schema",
        f"\nThis module contains {len(tables)} tables.\n"
    ]
    
    for table in tables:
        if table not in kg:
            lines.append(f"\n## {table}\n(No documentation)")
            continue
        
        lines.append(f"\n## Table: {table}\n")
        lines.append("**Columns:**\n")
        
        columns = kg.get(table, {})
        for col_name, col_info in columns.items():
            desc = col_info.get('desc', 'No description available')
            dtype = col_info.get('datatype', 'Text')
            lines.append(f"- **{col_name}** ({dtype}): {desc}")
    
    schema = "\n".join(lines)
    logger.info(f"📄 Schema built: {len(schema)} characters")
    return schema


def build_relationships(tables: list) -> str:
    """Build relationships from database introspection"""
    lines = ["# Table Relationships\n"]
    
    found_any = False
    with connection.cursor() as cursor:
        for table in tables:
            try:
                cursor.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table,
                        ccu.column_name AS foreign_column
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
                    found_any = True
                    lines.append(f"\n## {table}")
                    for fk in fks:
                        lines.append(f"- `{fk[0]}` references `{fk[1]}.{fk[2]}`")
            except Exception as e:
                logger.warning(f"Could not get FKs for {table}: {e}")
    
    if not found_any:
        lines.append("\nNo foreign key relationships found.")
    
    return "\n".join(lines)


def build_entity_inference_map(kg: dict, tables: list, pos_tagging: list) -> dict:
    """
    Build entity type to (column, table) mapping
    This is critical for entity resolution
    """
    entity_map = {}
    
    logger.info("🔍 Building entity inference map...")
    
    # First: Use POS tagging (explicit mapping)
    for pos in pos_tagging:
        entity_type = pos.get('name', '').strip().lower()
        reference = pos.get('reference', '').strip()
        
        if not entity_type or not reference:
            continue
        
        # Handle "table.column" format
        if '.' in reference:
            parts = reference.split('.', 1)
            table = parts[0]
            column = parts[1]
            if table in tables:
                entity_map[entity_type] = (column, table)
                logger.info(f"   POS: {entity_type} → {table}.{column}")
        else:
            # Find column across tables
            for table in tables:
                if table in kg and reference in kg[table]:
                    entity_map[entity_type] = (reference, table)
                    logger.info(f"   POS: {entity_type} → {table}.{reference}")
                    break
    
    # Second: Infer from knowledge graph descriptions
    if kg:
        for table in tables:
            if table not in kg:
                continue
            
            for col_name, col_info in kg[table].items():
                desc = col_info.get('desc', '').lower()
                
                # Product patterns
                if ('product' in desc or 'item' in desc) and 'product' not in entity_map:
                    entity_map['product'] = (col_name, table)
                    logger.info(f"   Inferred: product → {table}.{col_name}")
                
                # Distributor patterns
                if any(w in desc for w in ['distributor', 'dealer', 'vendor', 'reseller']) and 'distributor' not in entity_map:
                    entity_map['distributor'] = (col_name, table)
                    logger.info(f"   Inferred: distributor → {table}.{col_name}")
                
                # Customer patterns
                if 'customer' in desc and 'customer' not in entity_map:
                    entity_map['customer'] = (col_name, table)
                    logger.info(f"   Inferred: customer → {table}.{col_name}")
                
                # Superstockist patterns
                if 'superstockist' in desc and 'superstockist' not in entity_map:
                    entity_map['superstockist'] = (col_name, table)
                    logger.info(f"   Inferred: superstockist → {table}.{col_name}")
                
                # Party/Sold to patterns
                if any(w in desc for w in ['party', 'sold to', 'ship to']) and 'party' not in entity_map:
                    entity_map['party'] = (col_name, table)
                    logger.info(f"   Inferred: party → {table}.{col_name}")
    
    logger.info(f"✅ Entity map built: {len(entity_map)} types")
    return entity_map


def get_db_credentials():
    """Get database credentials from Django settings"""
    from django.conf import settings
    
    db = settings.DATABASES.get('default', {})
    
    # Also check environment variables as fallback
    return {
        'host': db.get('HOST') or os.getenv('PG_HOST', 'localhost'),
        'database': db.get('NAME') or os.getenv('PG_DBNAME', 'postgres'),
        'user': db.get('USER') or os.getenv('PG_USER', 'postgres'),
        'password': db.get('PASSWORD') or os.getenv('PG_PASSWORD', ''),
        'port': db.get('PORT', 5432)
    }