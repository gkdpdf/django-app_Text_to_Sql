#!/usr/bin/env python
"""
Complete fix for pgadmin app - Run this once to reset everything
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from django.db import connection
from django.core.management import call_command
import glob

def main():
    print("\n" + "="*70)
    print("COMPLETE PGADMIN FIX - STARTING")
    print("="*70)
    
    # Step 1: Drop all pgadmin tables
    print("\n[1/5] Dropping all pgadmin tables...")
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
        
        for table in tables:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"   ✅ Dropped {table}")
            except Exception as e:
                print(f"   ⚠️  {table}: {e}")
    
    # Step 2: Delete migration files
    print("\n[2/5] Deleting old migration files...")
    migrations_dir = 'pgadmin/migrations'
    
    if os.path.exists(migrations_dir):
        # Delete all migration files except __init__.py
        for file in glob.glob(os.path.join(migrations_dir, '0*.py')):
            try:
                os.remove(file)
                print(f"   ✅ Deleted {os.path.basename(file)}")
            except Exception as e:
                print(f"   ⚠️  Error deleting {file}: {e}")
        
        # Also delete .pyc files
        for file in glob.glob(os.path.join(migrations_dir, '__pycache__', '0*.pyc')):
            try:
                os.remove(file)
            except:
                pass
    
    # Ensure __init__.py exists
    init_file = os.path.join(migrations_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('')
        print(f"   ✅ Created __init__.py")
    
    # Step 3: Delete migration records from database
    print("\n[3/5] Cleaning migration records from database...")
    with connection.cursor() as cursor:
        try:
            cursor.execute("""
                DELETE FROM django_migrations 
                WHERE app = 'pgadmin'
            """)
            print("   ✅ Cleaned migration records")
        except Exception as e:
            print(f"   ⚠️  {e}")
    
    # Step 4: Create fresh migrations
    print("\n[4/5] Creating fresh migrations...")
    try:
        call_command('makemigrations', 'pgadmin', verbosity=2)
        print("   ✅ Migrations created")
    except Exception as e:
        print(f"   ❌ Error creating migrations: {e}")
        return False
    
    # Step 5: Apply migrations
    print("\n[5/5] Applying migrations...")
    try:
        call_command('migrate', 'pgadmin', verbosity=2)
        print("   ✅ Migrations applied")
    except Exception as e:
        print(f"   ❌ Error applying migrations: {e}")
        return False
    
    # Verify tables exist
    print("\n[VERIFY] Checking if tables were created...")
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename LIKE 'pgadmin_%'
            ORDER BY tablename
        """)
        tables = cursor.fetchall()
        
        if tables:
            print("   ✅ Tables created successfully:")
            for table in tables:
                print(f"      - {table[0]}")
        else:
            print("   ❌ No tables found!")
            return False
    
    print("\n" + "="*70)
    print("✅ COMPLETE FIX SUCCESSFUL!")
    print("="*70)
    print("\nYou can now run: python manage.py runserver")
    print()
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)