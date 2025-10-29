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
from django.http import HttpResponse
from .models import KnowledgeGraph, Metrics, RCA, Extra_suggestion
import csv
import io
import json


def knowledge_graph_view(request):
    kg_instance, _ = KnowledgeGraph.objects.get_or_create(id=1)
    metrics_instance, _ = Metrics.objects.get_or_create(id=1)
    rca_instance, _ = RCA.objects.get_or_create(id=1)
    extra_instance, _ = Extra_suggestion.objects.get_or_create(id=1)

    try:
        existing_kg_data = json.loads(kg_instance.data) if isinstance(kg_instance.data, str) else kg_instance.data or {}
    except:
        existing_kg_data = {}

    try:
        existing_metrics_data = json.loads(metrics_instance.data) if isinstance(metrics_instance.data, str) else metrics_instance.data or {}
    except:
        existing_metrics_data = {}

    existing_rca_data = ""
    if rca_instance.data:
        if isinstance(rca_instance.data, dict):
            existing_rca_data = rca_instance.data.get("text", "")
        else:
            try:
                existing_rca_data = json.loads(rca_instance.data).get("text", "")
            except:
                existing_rca_data = ""

    existing_extra_data = ""
    if extra_instance.data:
        if isinstance(extra_instance.data, dict):
            existing_extra_data = extra_instance.data.get("text", "")
        else:
            try:
                existing_extra_data = json.loads(extra_instance.data).get("text", "")
            except:
                existing_extra_data = ""

    # Get tables starting with tbl_
    tables = [t for t in connection.introspection.table_names() if t.startswith("tbl_")]
    knowledge_data = {}

    for table in tables:
        columns = [col.name for col in connection.introspection.get_table_description(connection.cursor(), table)]
        knowledge_data[table] = {}
        for col in columns:
            existing_info = existing_kg_data.get(table, {}).get(col, {})
            knowledge_data[table][col] = {
                "desc": existing_info.get("desc", ""),
                "datatype": existing_info.get("datatype", "")
            }

    # Handle CSV upload
    if request.method == "POST" and "upload_csv" in request.FILES:
        csv_file = request.FILES["upload_csv"]
        decoded_file = csv_file.read().decode("utf-8")
        io_string = io.StringIO(decoded_file)
        reader = csv.DictReader(io_string)
        uploaded_data = {}

        for row in reader:
            table = row["table"]
            column = row["column"]
            desc = row.get("desc", "")
            datatype = row.get("datatype", "")
            uploaded_data.setdefault(table, {}).setdefault(column, {})["desc"] = desc
            uploaded_data.setdefault(table, {}).setdefault(column, {})["datatype"] = datatype

        kg_instance.data = uploaded_data
        kg_instance.save()
        return redirect("knowledge_graph")

    # Handle Save All
    if request.method == "POST" and "save_all" in request.POST:
        updated_kg_data = {}
        for key, value in request.POST.items():
            if key.startswith("desc__"):
                _, table, column = key.split("__")
                updated_kg_data.setdefault(table, {}).setdefault(column, {})["desc"] = value
            elif key.startswith("datatype__"):
                _, table, column = key.split("__")
                updated_kg_data.setdefault(table, {}).setdefault(column, {})["datatype"] = value

        kg_instance.data = updated_kg_data
        kg_instance.save()

        updated_metrics_data = {}
        names = request.POST.getlist("kpi_name")
        descs = request.POST.getlist("kpi_desc")
        for i in range(len(names)):
            if names[i].strip():
                updated_metrics_data[names[i].strip()] = descs[i].strip()
        metrics_instance.data = updated_metrics_data
        metrics_instance.save()

        rca_instance.data = {"text": request.POST.get("rca_text", "").strip()}
        rca_instance.save()

        extra_instance.data = {"text": request.POST.get("extra_text", "").strip()}
        extra_instance.save()

        return redirect("knowledge_graph")

    return render(request, "knowledge-graph.html", {
        "data": knowledge_data,
        "metrics_data": existing_metrics_data,
        "rca_text": existing_rca_data,
        "extra_text": existing_extra_data
    })


from django.http import HttpResponse
import csv
import json

def download_knowledge_graph_csv(request):
    kg_instance = KnowledgeGraph.objects.first()

    # Safely parse stored JSON data
    try:
        data = json.loads(kg_instance.data) if isinstance(kg_instance.data, str) else kg_instance.data or {}
    except:
        data = {}

    # Create CSV response
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="knowledge_graph.csv"'

    writer = csv.writer(response)
    writer.writerow(["table", "column", "desc", "datatype"])  # CSV headers

    for table, columns in data.items():
        for column, info in columns.items():
            writer.writerow([
                table,
                column,
                info.get("desc", ""),
                info.get("datatype", "")
            ])

    return response

import csv
from django.http import HttpResponse
import json

def upload_knowledge_graph(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        decoded_file = csv_file.read().decode('utf-8').splitlines()
        reader = csv.DictReader(decoded_file)

        kg_instance, _ = KnowledgeGraph.objects.get_or_create(id=1)
        data = {}

        for row in reader:
            table = row.get('table', '').strip()
            column = row.get('column', '').strip()
            desc = row.get('desc', '').strip()
            datatype = row.get('datatype', '').strip()

            if table and column:
                data.setdefault(table, {})[column] = {
                    'desc': desc,
                    'datatype': datatype
                }

        kg_instance.data = data
        kg_instance.save()

        return redirect('knowledge_graph')
    return redirect('knowledge_graph')


