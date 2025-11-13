import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from django.db import connection

with connection.cursor() as cursor:
    try:
        # Add tables column as array
        cursor.execute("""
            ALTER TABLE pgadmin_module 
            ADD COLUMN IF NOT EXISTS tables varchar(200)[] DEFAULT '{}';
        """)
        print("✅ Added 'tables' column successfully!")
        
        # Add other missing columns just in case
        cursor.execute("""
            ALTER TABLE pgadmin_module 
            ADD COLUMN IF NOT EXISTS knowledge_graph_data JSONB DEFAULT '{}';
        """)
        cursor.execute("""
            ALTER TABLE pgadmin_module 
            ADD COLUMN IF NOT EXISTS rca_list JSONB DEFAULT '[]';
        """)
        cursor.execute("""
            ALTER TABLE pgadmin_module 
            ADD COLUMN IF NOT EXISTS pos_tagging JSONB DEFAULT '[]';
        """)
        cursor.execute("""
            ALTER TABLE pgadmin_module 
            ADD COLUMN IF NOT EXISTS metrics_data JSONB DEFAULT '{}';
        """)
        cursor.execute("""
            ALTER TABLE pgadmin_module 
            ADD COLUMN IF NOT EXISTS extra_suggestions TEXT DEFAULT '';
        """)
        cursor.execute("""
            ALTER TABLE pgadmin_module 
            ADD COLUMN IF NOT EXISTS kg_auto_generated BOOLEAN DEFAULT FALSE;
        """)
        cursor.execute("""
            ALTER TABLE pgadmin_module 
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        """)
        print("✅ All columns added successfully!")
        
    except Exception as e:
        print(f"Error: {e}")