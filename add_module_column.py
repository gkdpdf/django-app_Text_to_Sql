import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from django.db import connection

with connection.cursor() as cursor:
    try:
        # Add module_id column to conversation table
        cursor.execute("""
            ALTER TABLE pgadmin_conversation 
            ADD COLUMN IF NOT EXISTS module_id INTEGER;
        """)
        
        # Add foreign key constraint
        cursor.execute("""
            ALTER TABLE pgadmin_conversation 
            ADD CONSTRAINT fk_module
            FOREIGN KEY (module_id) 
            REFERENCES pgadmin_module(id)
            ON DELETE CASCADE;
        """)
        
        print("✅ Added module_id column successfully!")
        
    except Exception as e:
        print(f"Error: {e}")