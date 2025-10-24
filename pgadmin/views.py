from django.shortcuts import render
from django.db import connection

def login_view(request):
    error_message = ""
    tables = []
    user_name = None

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        # Debug print (optional): to check data in terminal
        print(f"Username: {username}, Password: {password}")

        # Simple demo authentication
        if username.lower() == "gaurav" and password == "12345678":
            user_name = username

            # Fetch all non-Django tables from PostgreSQL
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                """)
                all_tables = [row[0] for row in cursor.fetchall()]

            # Exclude Django system tables
            exclude_tables = [
                'auth_group', 'auth_group_permissions', 'auth_permission',
                'auth_user', 'auth_user_groups', 'auth_user_user_permissions',
                'django_admin_log', 'django_content_type',
                'django_migrations', 'django_session'
            ]
            tables = [t for t in all_tables if t not in exclude_tables]

        else:
            error_message = "Incorrect username or password."

    return render(request, "login.html", {
        "error_message": error_message,
        "tables": tables,
        "user_name": user_name
    })
