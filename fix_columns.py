import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from django.db import connection

with connection.cursor() as cursor:
    # Add missing columns to all tables
    tables_to_fix = [
        'pgadmin_knowledgegraph',
        'pgadmin_metrics',
        'pgadmin_rca',
        'pgadmin_extra_suggestion'
    ]
    
    for table in tables_to_fix:
        try:
            cursor.execute(f"""
                ALTER TABLE {table} 
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
            """)
            cursor.execute(f"""
                ALTER TABLE {table} 
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
            """)
            print(f"✅ Fixed {table}")
        except Exception as e:
            print(f"❌ Error fixing {table}: {e}")
    
    # Also add tables column to module if missing
    try:
        cursor.execute("""
            ALTER TABLE pgadmin_module 
            ADD COLUMN IF NOT EXISTS tables varchar(200)[] DEFAULT '{}';
        """)
        print("✅ Fixed pgadmin_module")
    except Exception as e:
        print(f"❌ Error fixing pgadmin_module: {e}")