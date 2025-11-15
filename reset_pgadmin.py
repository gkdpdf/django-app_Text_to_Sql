#!/usr/bin/env python
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from django.db import connection
from django.core.management import call_command

def reset_pgadmin():
    print("=" * 60)
    print("RESETTING PGADMIN APP")
    print("=" * 60)
    
    # Drop all pgadmin tables
    with connection.cursor() as cursor:
        tables = [
            'pgadmin_message',
            'pgadmin_conversation',
            'pgadmin_module',
            'pgadmin_knowledgegraph',
            'pgadmin_metrics',
            'pgadmin_rca',
            'pgadmin_extra_suggestion'
        ]
        
        print("\n1. Dropping tables...")
        for table in tables:
            try:
                cursor.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
                print(f"   ✅ Dropped {table}")
            except Exception as e:
                print(f"   ⚠️  {table}: {e}")
    
    # Delete migration files
    print("\n2. Deleting migration files...")
    migrations_dir = 'pgadmin/migrations'
    if os.path.exists(migrations_dir):
        for file in os.listdir(migrations_dir):
            if file.startswith('0') and file.endswith('.py'):
                filepath = os.path.join(migrations_dir, file)
                os.remove(filepath)
                print(f"   ✅ Deleted {file}")
    
    # Create new migrations
    print("\n3. Creating fresh migrations...")
    call_command('makemigrations', 'pgadmin')
    
    # Run migrations
    print("\n4. Running migrations...")
    call_command('migrate', 'pgadmin')
    
    print("\n" + "=" * 60)
    print("✅ RESET COMPLETE!")
    print("=" * 60)
    print("\nYou can now run: python manage.py runserver")

if __name__ == '__main__':
    reset_pgadmin()