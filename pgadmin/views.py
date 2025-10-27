from django.shortcuts import render, redirect, get_object_or_404
from django.db import connection
from django.contrib.auth.models import User
from .models import Module

# ---------- LOGIN ----------
def login_view(request):
    error_message = ""

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        # Demo static credentials for simplicity
        if username.lower() == "gaurav" and password == "12345678":
            request.session["user_name"] = username
            return redirect("dashboard")
        else:
            error_message = "Incorrect username or password."

    return render(request, "login.html", {"error_message": error_message})


# ---------- DASHBOARD ----------
def dashboard_view(request):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    # Fetch all modules created by this user
    modules = Module.objects.filter(user_name=user_name)
    return render(request, "dashboard.html", {"modules": modules, "user_name": user_name})


# ---------- CREATE NEW MODULE ----------
def new_module_view(request):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    if request.method == "POST":
        module_name = request.POST.get("module_name")
        selected_tables = request.POST.getlist("selected_tables")

        # Save to DB
        Module.objects.create(
            user_name=user_name,
            name=module_name,
            tables=selected_tables,
        )
        return redirect("dashboard")

    # Fetch PostgreSQL user-defined tables dynamically
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        all_tables = [row[0] for row in cursor.fetchall()]

    exclude_tables = [
        'auth_group', 'auth_group_permissions', 'auth_permission',
        'auth_user', 'auth_user_groups', 'auth_user_user_permissions',
        'django_admin_log', 'django_content_type',
        'django_migrations', 'django_session'
    ]
    tables = [t for t in all_tables if t not in exclude_tables]

    return render(request, "new_module.html", {"tables": tables, "user_name": user_name})


# ---------- EDIT MODULE / UPLOAD DOCS ----------
def edit_module_view(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    module = get_object_or_404(Module, id=module_id, user_name=user_name)

    if request.method == "POST":
        if "knowledge_graph" in request.FILES:
            module.knowledge_graph = request.FILES["knowledge_graph"]
        if "metrics" in request.FILES:
            module.metrics = request.FILES["metrics"]
        module.save()
        return redirect("dashboard")

    return render(request, "upload_docs.html", {
        "module": module,
        "user_name": user_name
    })
