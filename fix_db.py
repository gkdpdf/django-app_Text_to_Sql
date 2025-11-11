import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from django.db import connection

with connection.cursor() as cursor:
    try:
        cursor.execute("ALTER TABLE pgadmin_module DROP COLUMN IF EXISTS tables;")
        print("✅ Column 'tables' dropped successfully!")
    except Exception as e:
        print(f"Error: {e}")