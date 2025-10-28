from django.shortcuts import render, redirect, get_object_or_404
from django.db import connection
from .models import Module

# ---------- LOGIN ----------
def login_view(request):
    error_message = ""

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

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

    modules = Module.objects.filter(user_name=user_name).order_by("-created_at")
    return render(request, "dashboard.html", {"modules": modules, "user_name": user_name})


# ---------- CREATE NEW MODULE ----------
def new_module_view(request):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    if request.method == "POST":
        module_name = request.POST.get("module_name")
        selected_tables = request.POST.getlist("selected_tables")

        Module.objects.create(
            user_name=user_name,
            name=module_name,
            tables=selected_tables,
        )
        return redirect("dashboard")

    with connection.cursor() as cursor:
        cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        all_tables = [row[0] for row in cursor.fetchall()]

    exclude_tables = [
        'auth_group', 'auth_group_permissions', 'auth_permission',
        'auth_user', 'auth_user_groups', 'auth_user_user_permissions',
        'django_admin_log', 'django_content_type', 'django_migrations', 'django_session'
    ]
    tables = [t for t in all_tables if t not in exclude_tables]

    return render(request, "new_module.html", {"tables": tables, "user_name": user_name})


# ---------- UPLOAD FILES (EDIT MODULE) ----------
# ---------- UPLOAD FILES & RESELECT TABLES (EDIT MODULE) ----------
def edit_module_view(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    module = get_object_or_404(Module, id=module_id, user_name=user_name)

    # Fetch all available tables from database
    with connection.cursor() as cursor:
        cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        all_tables = [row[0] for row in cursor.fetchall()]

    exclude_tables = [
        'auth_group', 'auth_group_permissions', 'auth_permission',
        'auth_user', 'auth_user_groups', 'auth_user_user_permissions',
        'django_admin_log', 'django_content_type', 'django_migrations', 'django_session'
    ]
    tables = [t for t in all_tables if t not in exclude_tables]

    if request.method == "POST":
        # Update file uploads
        if "knowledge_graph" in request.FILES:
            module.knowledge_graph = request.FILES["knowledge_graph"]
        if "metrics" in request.FILES:
            module.metrics = request.FILES["metrics"]

        # Update table selections
        selected_tables = request.POST.getlist("selected_tables")
        if selected_tables:
            module.tables = selected_tables

        module.save()
        return redirect("dashboard")

    # Convert current tables list for preselection
    selected_tables = module.tables if isinstance(module.tables, list) else []

    return render(request, "upload_docs.html", {
        "module": module,
        "tables": tables,
        "selected_tables": selected_tables,
        "user_name": user_name,
    })


# ---------- CHATBOT VIEW ----------
def module_chat_view(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    tables = module.tables

    return render(request, "module_chat.html", {
        "module_name": module.name,
        "tables": tables,
        "user_name": user_name,
    })
from django.shortcuts import render, redirect
from django.db import connection
from .models import KnowledgeGraph

def knowledge_graph_view(request):
    # fetch or create the single record
    instance, _ = KnowledgeGraph.objects.get_or_create(id=1)
    existing_data = instance.data or {}

    tables = connection.introspection.table_names()
    data = {}

    # collect all table-column data
    for table in tables:
        columns = [col.name for col in connection.introspection.get_table_description(connection.cursor(), table)]
        data[table] = {}
        for col in columns:
            data[table][col] = existing_data.get(table, {}).get(col, "")

    if request.method == 'POST':
        # extract descriptions from form
        updated_data = {}
        for key, value in request.POST.items():
            if key.startswith("desc_"):
                _, table, column = key.split("__")
                if table not in updated_data:
                    updated_data[table] = {}
                updated_data[table][column] = value
        instance.data = updated_data
        instance.save()
        return redirect('knowledge_graph')

    return render(request, 'knowledge-graph.html', {'data': data})
