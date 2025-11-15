from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Fix database by dropping and recreating pgadmin tables'

    def handle(self, *args, **options):
        with connection.cursor() as cursor:
            self.stdout.write('Dropping pgadmin tables...')
            
            # Drop tables in correct order (respect foreign keys)
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
                    cursor.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
                    self.stdout.write(f'✅ Dropped {table}')
                except Exception as e:
                    self.stdout.write(f'⚠️  Could not drop {table}: {e}')
            
            self.stdout.write(self.style.SUCCESS('\n✅ All tables dropped!'))
            self.stdout.write('\nNow run: python manage.py migrate')