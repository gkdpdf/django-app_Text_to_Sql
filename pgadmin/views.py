# pgadmin/views.py
import csv
import io
import json
import logging
import traceback
from datetime import datetime
from decimal import Decimal

from django.shortcuts import render, redirect, get_object_or_404
from django.db import connection as django_connection
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone

from .models import (
    Module, KnowledgeGraph, Metrics, RCA,
    Extra_suggestion, Conversation, Message
)

# Try to import Dashboard model, create if not exists
try:
    from .models import Dashboard
except ImportError:
    Dashboard = None

logger = logging.getLogger(__name__)
SESSION_STORE = {}


# --------------------
# Helper Functions
# --------------------
def _load_public_tables(prefix=None):
    """Helper: return list of public table names; optional prefix filter"""
    with django_connection.cursor() as cursor:
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        all_tables = [r[0] for r in cursor.fetchall()]
    exclude = [
        'auth_group', 'auth_group_permissions', 'auth_permission',
        'auth_user', 'auth_user_groups', 'auth_user_user_permissions',
        'django_admin_log', 'django_content_type', 'django_migrations', 'django_session',
    ]
    tables = [t for t in all_tables if t not in exclude]
    if prefix:
        tables = [t for t in tables if t.startswith(prefix)]
    return tables


def _get_table_columns(table_name):
    """Helper: fetch columns for a given table from the database"""
    with django_connection.cursor() as cursor:
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, [table_name])
        return [{"name": r[0], "type": r[1]} for r in cursor.fetchall()]


def api_table_columns(request):
    """API endpoint to get columns for a table"""
    table_name = request.GET.get("table", "")
    
    if not table_name:
        return JsonResponse({"error": "table parameter required", "columns": []}, status=400)
    
    try:
        columns = _get_table_columns(table_name)
        return JsonResponse({"columns": columns})
    except Exception as e:
        logger.exception(f"Error getting columns for table {table_name}")
        return JsonResponse({"error": str(e), "columns": []}, status=500)


def _get_sample_data(table_name, limit=3):
    """Helper: fetch sample data from a table for AI context"""
    try:
        with django_connection.cursor() as cursor:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT %s', [limit])
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            serializable_rows = []
            for row in rows:
                serializable_row = []
                for val in row:
                    if val is None:
                        serializable_row.append(None)
                    elif isinstance(val, (datetime,)):
                        serializable_row.append(str(val))
                    else:
                        serializable_row.append(val)
                serializable_rows.append(serializable_row)
            return {"columns": columns, "rows": serializable_rows}
    except Exception as e:
        logger.warning(f"Could not fetch sample data for {table_name}: {e}")
        return {"columns": [], "rows": []}


def _build_initial_kg_from_selection(selected_tables, selected_columns):
    """Build initial knowledge graph structure from selected tables and columns."""
    kg_data = {}
    
    for table in selected_tables:
        kg_data[table] = {}
        kg_data[table]["TABLE_INFO"] = {"desc": "", "datatype": "meta"}
        
        columns_in_selection = selected_columns.get(table, [])
        
        if columns_in_selection:
            db_columns = _get_table_columns(table)
            db_col_map = {c["name"]: c["type"] for c in db_columns}
            
            for col_name in columns_in_selection:
                kg_data[table][col_name] = {
                    "desc": "",
                    "datatype": db_col_map.get(col_name, "")
                }
    
    return kg_data


def _merge_kg_with_selection(existing_kg, selected_tables, selected_columns):
    """Merge existing KG data with new selection, preserving existing descriptions."""
    merged_kg = {}
    
    for table in selected_tables:
        merged_kg[table] = {}
        existing_table_data = existing_kg.get(table, {}) if isinstance(existing_kg, dict) else {}
        
        if "TABLE_INFO" in existing_table_data:
            merged_kg[table]["TABLE_INFO"] = existing_table_data["TABLE_INFO"]
        else:
            merged_kg[table]["TABLE_INFO"] = {"desc": "", "datatype": "meta"}
        
        columns_in_selection = selected_columns.get(table, [])
        
        if columns_in_selection:
            db_columns = _get_table_columns(table)
            db_col_map = {c["name"]: c["type"] for c in db_columns}
            
            for col_name in columns_in_selection:
                if col_name in existing_table_data:
                    merged_kg[table][col_name] = existing_table_data[col_name]
                    if not merged_kg[table][col_name].get("datatype"):
                        merged_kg[table][col_name]["datatype"] = db_col_map.get(col_name, "")
                else:
                    merged_kg[table][col_name] = {
                        "desc": "",
                        "datatype": db_col_map.get(col_name, "")
                    }
    
    return merged_kg


# --------------------
# AI Description Generation Functions
# --------------------
def _generate_kg_descriptions_with_ai(kg_data, selected_columns):
    """Use AI to generate descriptions for tables and columns with empty descriptions."""
    import os
    
    items_to_describe = []
    
    for table, cols in kg_data.items():
        table_info = cols.get("TABLE_INFO", {})
        if not table_info.get("desc", "").strip():
            sample = _get_sample_data(table, limit=3)
            column_list = selected_columns.get(table, [])
            items_to_describe.append({
                "type": "table",
                "table": table,
                "columns": column_list,
                "sample_data": sample
            })
        
        for col_name, col_info in cols.items():
            if col_name == "TABLE_INFO":
                continue
            if not col_info.get("desc", "").strip():
                items_to_describe.append({
                    "type": "column",
                    "table": table,
                    "column": col_name,
                    "datatype": col_info.get("datatype", "")
                })
    
    if not items_to_describe:
        return kg_data
    
    prompt = _build_kg_generation_prompt(items_to_describe)
    
    try:
        ai_response = _call_llm_for_kg(prompt)
        if ai_response:
            kg_data = _parse_and_apply_ai_descriptions(kg_data, ai_response)
    except Exception as e:
        logger.exception("Error generating AI descriptions: %s", e)
    
    return kg_data


def _build_kg_generation_prompt(items_to_describe):
    """Build a prompt for the LLM to generate KG descriptions"""
    prompt = """You are a database documentation expert. Generate clear, concise descriptions for the following database tables and columns.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
    "tables": {"table_name": "description"},
    "columns": {"table_name.column_name": "description"}
}

Items to describe:
"""
    
    for item in items_to_describe:
        if item["type"] == "table":
            prompt += f"\nTABLE: {item['table']} (Columns: {', '.join(item['columns'][:10])})"
        else:
            prompt += f"\nCOLUMN: {item['table']}.{item['column']} ({item['datatype']})"
    
    return prompt


def _call_llm_for_kg(prompt):
    """Call the LLM API to generate descriptions"""
    import os
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if anthropic_key:
        return _call_anthropic(prompt, anthropic_key)
    elif openai_key:
        return _call_openai(prompt, openai_key)
    return None


def _call_anthropic(prompt, api_key):
    """Call Anthropic Claude API"""
    import requests
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("content", [{}])[0].get("text", "")
    except Exception as e:
        logger.exception("Anthropic API error: %s", e)
        return None


def _call_openai(prompt, api_key):
    """Call OpenAI API"""
    import requests
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Database documentation expert. Respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.exception("OpenAI API error: %s", e)
        return None


def _parse_and_apply_ai_descriptions(kg_data, ai_response):
    """Parse AI response and apply descriptions to kg_data"""
    import re
    
    try:
        json_match = re.search(r'\{[\s\S]*\}', ai_response)
        if not json_match:
            return kg_data
        
        ai_data = json.loads(json_match.group())
        
        tables_desc = ai_data.get("tables", {})
        for table_name, desc in tables_desc.items():
            if table_name in kg_data:
                if "TABLE_INFO" not in kg_data[table_name]:
                    kg_data[table_name]["TABLE_INFO"] = {"desc": "", "datatype": "meta"}
                if not kg_data[table_name]["TABLE_INFO"].get("desc", "").strip():
                    kg_data[table_name]["TABLE_INFO"]["desc"] = desc.strip()
        
        columns_desc = ai_data.get("columns", {})
        for full_col_name, desc in columns_desc.items():
            parts = full_col_name.split(".", 1)
            if len(parts) == 2:
                table_name, col_name = parts
                if table_name in kg_data and col_name in kg_data[table_name]:
                    if not kg_data[table_name][col_name].get("desc", "").strip():
                        kg_data[table_name][col_name]["desc"] = desc.strip()
        
        return kg_data
        
    except Exception as e:
        logger.exception("Error applying AI descriptions: %s", e)
        return kg_data


# --------------------
# Auth
# --------------------
def home_view(request):
    if request.session.get("user_name"):
        return redirect("dashboard")
    return redirect("login")


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


# --------------------
# Dashboard
# --------------------
def dashboard_view(request):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")
    try:
        modules = Module.objects.filter(user_name=user_name).order_by("-created_at")
        for m in modules:
            if not isinstance(m.tables, list):
                m.tables = []
                m.save()
        return render(request, "dashboard.html", {"modules": modules, "user_name": user_name})
    except Exception as e:
        logger.error("Error in dashboard_view: %s", e, exc_info=True)
        return render(request, "dashboard.html", {"modules": [], "user_name": user_name, "error": str(e)})


# --------------------
# New Module
# --------------------
def new_module_view(request):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    if request.method == "POST":
        module_name = request.POST.get("module_name", "").strip()
        selected_tables = request.POST.getlist("selected_tables")
        selected_columns_json = request.POST.get("selected_columns", "{}")

        try:
            selected_columns = json.loads(selected_columns_json or "{}")
            if not isinstance(selected_columns, dict):
                selected_columns = {}
        except Exception as e:
            selected_columns = {}

        if not module_name:
            tables = _load_public_tables(prefix="tbl_")
            return render(request, "new_module.html", {
                "tables": tables,
                "user_name": user_name,
                "error": "Please provide module name"
            })

        if not selected_tables:
            tables = _load_public_tables(prefix="tbl_")
            return render(request, "new_module.html", {
                "tables": tables,
                "user_name": user_name,
                "error": "Please select at least one table"
            })

        try:
            initial_kg = _build_initial_kg_from_selection(selected_tables, selected_columns)
            
            module = Module.objects.create(
                user_name=user_name,
                name=module_name,
                tables=selected_tables,
                selected_columns=selected_columns,
                knowledge_graph_data=initial_kg
            )
            
            return redirect("edit_module", module_id=module.id)
            
        except Exception as e:
            logger.exception("Failed to create module")
            tables = _load_public_tables(prefix="tbl_")
            return render(request, "new_module.html", {
                "tables": tables,
                "user_name": user_name,
                "error": f"Failed to create module: {str(e)}"
            })

    tables = _load_public_tables(prefix="tbl_")
    return render(request, "new_module.html", {"tables": tables, "user_name": user_name})


# --------------------
# Edit Module
# --------------------
def edit_module_view(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    module = get_object_or_404(Module, id=module_id, user_name=user_name)

    with django_connection.cursor() as cursor:
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            AND table_name NOT LIKE 'auth_%'
            AND table_name NOT LIKE 'django_%'
            AND table_name NOT LIKE 'pgadmin_%'
            ORDER BY table_name
        """)
        tables = [r[0] for r in cursor.fetchall()]

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "save_all":
            selected_tables = request.POST.getlist("selected_tables")
            selected_columns_json = request.POST.get("selected_columns", "{}")
            run_ai = request.POST.get("run_ai", "false") == "true"
            
            try:
                selected_columns = json.loads(selected_columns_json or "{}")
                if not isinstance(selected_columns, dict):
                    selected_columns = {}
            except:
                selected_columns = {}

            module.tables = selected_tables or []
            module.selected_columns = selected_columns or {}

            posted_kg = {}
            for key, val in request.POST.items():
                if key.startswith("table_info__"):
                    table = key.replace("table_info__", "")
                    if table and table in selected_tables:
                        posted_kg.setdefault(table, {})
                        posted_kg[table]["TABLE_INFO"] = {"desc": val.strip(), "datatype": "meta"}
                        
                elif key.startswith("kg_desc__"):
                    parts = key.split("__", 2)
                    if len(parts) == 3:
                        _, table, column = parts
                        if table in selected_tables:
                            cols_for_table = selected_columns.get(table, [])
                            if column in cols_for_table or column == "TABLE_INFO":
                                posted_kg.setdefault(table, {})
                                posted_kg[table].setdefault(column, {})
                                posted_kg[table][column]["desc"] = val.strip()
                                
                elif key.startswith("kg_datatype__"):
                    parts = key.split("__", 2)
                    if len(parts) == 3:
                        _, table, column = parts
                        if table in selected_tables:
                            cols_for_table = selected_columns.get(table, [])
                            if column in cols_for_table or column == "TABLE_INFO":
                                posted_kg.setdefault(table, {})
                                posted_kg[table].setdefault(column, {})
                                posted_kg[table][column]["datatype"] = val.strip()

            final_kg = _merge_kg_with_selection(posted_kg, selected_tables, selected_columns)
            
            for table in posted_kg:
                if table in final_kg:
                    for col, data in posted_kg[table].items():
                        if col in final_kg[table]:
                            if data.get("desc"):
                                final_kg[table][col]["desc"] = data["desc"]
                            if data.get("datatype"):
                                final_kg[table][col]["datatype"] = data["datatype"]

            if run_ai:
                final_kg = _generate_kg_descriptions_with_ai(final_kg, selected_columns)

            module.knowledge_graph_data = final_kg

            # Relationships
            relationships = []
            rel_left_tables = request.POST.getlist("rel_left_table")
            rel_left_columns = request.POST.getlist("rel_left_column")
            rel_types = request.POST.getlist("rel_type")
            rel_right_tables = request.POST.getlist("rel_right_table")
            rel_right_columns = request.POST.getlist("rel_right_column")
            
            for i in range(len(rel_left_tables)):
                if rel_left_tables[i] and rel_right_tables[i]:
                    relationships.append({
                        "left_table": rel_left_tables[i],
                        "left_column": rel_left_columns[i] if i < len(rel_left_columns) else "",
                        "type": rel_types[i] if i < len(rel_types) else "one-to-many",
                        "right_table": rel_right_tables[i],
                        "right_column": rel_right_columns[i] if i < len(rel_right_columns) else ""
                    })
            module.relationships = relationships

            # RCAs
            rca_list = []
            rca_titles = request.POST.getlist("rca_title")
            rca_contents = request.POST.getlist("rca_content")
            for i in range(len(rca_titles)):
                if rca_titles[i].strip():
                    rca_list.append({
                        "title": rca_titles[i].strip(),
                        "content": rca_contents[i].strip() if i < len(rca_contents) else ""
                    })
            module.rca_list = rca_list

            # POS tagging
            pos_list = []
            pos_names = request.POST.getlist("pos_name")
            pos_refs = request.POST.getlist("pos_reference")
            for i in range(len(pos_names)):
                if pos_names[i].strip():
                    pos_list.append({
                        "name": pos_names[i].strip(),
                        "reference": pos_refs[i].strip() if i < len(pos_refs) else ""
                    })
            module.pos_tagging = pos_list

            # Metrics
            metrics_data = {}
            metric_names = request.POST.getlist("metric_name")
            metric_descs = request.POST.getlist("metric_desc")
            for i in range(len(metric_names)):
                if metric_names[i].strip():
                    metrics_data[metric_names[i].strip()] = metric_descs[i].strip() if i < len(metric_descs) else ""
            module.metrics_data = metrics_data

            module.extra_suggestions = request.POST.get("extra_suggestions", "").strip()
            module.save()
            
            return redirect("edit_module", module_id=module.id)

    # GET
    selected_tables = module.tables or []
    selected_columns = module.selected_columns if isinstance(module.selected_columns, dict) else {}

    try:
        selected_columns_json = json.dumps(selected_columns)
    except:
        selected_columns_json = "{}"

    knowledge_data = {}
    table_info_map = {}
    
    kg_data = module.knowledge_graph_data or {}
    
    if selected_tables:
        kg_data = _merge_kg_with_selection(kg_data, selected_tables, selected_columns)
        if kg_data != module.knowledge_graph_data:
            module.knowledge_graph_data = kg_data
            module.save()
    
    for table, cols in kg_data.items():
        knowledge_data[table] = {}
        if isinstance(cols, dict):
            for col_name, col_data in cols.items():
                if col_name == "TABLE_INFO":
                    table_info_map[table] = col_data.get("desc", "") if isinstance(col_data, dict) else ""
                else:
                    knowledge_data[table][col_name] = col_data if isinstance(col_data, dict) else {"desc": str(col_data), "datatype": ""}

    return render(request, "edit_module.html", {
        "module": module,
        "tables": tables,
        "selected_tables": selected_tables,
        "selected_columns": selected_columns,
        "selected_columns_json": selected_columns_json,
        "knowledge_data": knowledge_data,
        "table_info_map": table_info_map,
        "relationships": module.relationships or [],
        "rca_list": module.rca_list or [],
        "pos_tagging": module.pos_tagging or [],
        "metrics_data": module.metrics_data or {},
        "extra_suggestions": module.extra_suggestions or "",
        "user_name": user_name,
    })


# --------------------
# Generate KG API
# --------------------
@csrf_exempt
@require_http_methods(["POST"])
def generate_kg_api(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    
    try:
        module = get_object_or_404(Module, id=module_id, user_name=user_name)
        
        selected_tables = module.tables or []
        selected_columns = module.selected_columns or {}
        kg_data = module.knowledge_graph_data or {}
        
        kg_data = _merge_kg_with_selection(kg_data, selected_tables, selected_columns)
        kg_data = _generate_kg_descriptions_with_ai(kg_data, selected_columns)
        
        module.knowledge_graph_data = kg_data
        module.save()
        
        return JsonResponse({
            "success": True,
            "message": "AI descriptions generated successfully",
            "kg_data": kg_data
        })
        
    except Exception as e:
        logger.exception("Error in generate_kg_api")
        return JsonResponse({"error": str(e)}, status=500)


# --------------------
# Delete module
# --------------------
@csrf_exempt
@require_http_methods(["DELETE", "POST"])
def delete_module_view(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    try:
        module = Module.objects.get(id=module_id, user_name=user_name)
        name = module.name
        module.delete()
        return JsonResponse({"success": True, "message": f"Deleted {name}"})
    except Module.DoesNotExist:
        return JsonResponse({"error": "Module not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# --------------------
# Utilities
# --------------------
@csrf_exempt
def get_table_columns_view(request):
    table_name = request.GET.get("table")
    if not table_name:
        return JsonResponse({"error": "No table specified"}, status=400)
    try:
        columns = _get_table_columns(table_name)
        return JsonResponse({"columns": columns})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# --------------------
# Download / Upload KG CSV
# --------------------
def download_module_kg_csv(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{module.name}_knowledge_graph.csv"'
    writer = csv.writer(response)
    writer.writerow(["table", "table_info", "column", "desc", "datatype"])
    kg = module.knowledge_graph_data or {}
    for table, cols in kg.items():
        table_info = cols.get("TABLE_INFO", {}).get("desc", "") if isinstance(cols, dict) else ""
        first = True
        for col, info in (cols or {}).items():
            if col == "TABLE_INFO":
                continue
            writer.writerow([table, table_info if first else "", col, info.get("desc", ""), info.get("datatype", "")])
            first = False
    return response


def upload_module_kg_csv(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")
    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]
        decoded = csv_file.read().decode("utf-8").splitlines()
        reader = csv.DictReader(decoded)
        data = {}
        for row in reader:
            table = row.get("table", "").strip()
            table_info = row.get("table_info", "").strip()
            column = row.get("column", "").strip()
            desc = row.get("desc", "").strip()
            datatype = row.get("datatype", "").strip()
            if table:
                data.setdefault(table, {})
                if table_info and "TABLE_INFO" not in data[table]:
                    data[table]["TABLE_INFO"] = {"desc": table_info, "datatype": "meta"}
                if column:
                    data[table][column] = {"desc": desc, "datatype": datatype}
        module.knowledge_graph_data = data
        module.save()
    return redirect("edit_module", module_id=module_id)


# --------------------
# Global KG views (legacy)
# --------------------
def knowledge_graph_view(request):
    kg_inst, _ = KnowledgeGraph.objects.get_or_create(id=1)
    metrics_inst, _ = Metrics.objects.get_or_create(id=1)
    rca_inst, _ = RCA.objects.get_or_create(id=1)
    extra_inst, _ = Extra_suggestion.objects.get_or_create(id=1)

    data = {}
    if isinstance(kg_inst.data, dict):
        data = kg_inst.data
    elif isinstance(kg_inst.data, str):
        try:
            data = json.loads(kg_inst.data)
        except:
            data = {}

    metrics_data = metrics_inst.data if isinstance(metrics_inst.data, dict) else {}
    rca_text = ""
    extra_text = ""
    if isinstance(rca_inst.data, dict):
        rca_text = rca_inst.data.get("text", "")
    if isinstance(extra_inst.data, dict):
        extra_text = extra_inst.data.get("text", "")

    return render(request, "knowledge-graph.html", {
        "data": data,
        "metrics_data": metrics_data,
        "rca_text": rca_text,
        "extra_text": extra_text
    })


def download_knowledge_graph_csv(request):
    kg_instance = KnowledgeGraph.objects.first()
    data = {}
    if isinstance(kg_instance.data, dict):
        data = kg_instance.data
    elif isinstance(kg_instance.data, str):
        try:
            data = json.loads(kg_instance.data)
        except:
            data = {}
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="knowledge_graph.csv"'
    writer = csv.writer(response)
    writer.writerow(["table", "column", "desc", "datatype"])
    for table, cols in data.items():
        for column, info in cols.items():
            writer.writerow([table, column, info.get("desc", ""), info.get("datatype", "")])
    return response


def upload_knowledge_graph(request):
    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]
        decoded = csv_file.read().decode("utf-8").splitlines()
        reader = csv.DictReader(decoded)
        data = {}
        for row in reader:
            table = row.get("table", "").strip()
            column = row.get("column", "").strip()
            desc = row.get("desc", "").strip()
            datatype = row.get("datatype", "").strip()
            if table and column:
                data.setdefault(table, {})[column] = {"desc": desc, "datatype": datatype}
        kg, _ = KnowledgeGraph.objects.get_or_create(id=1)
        kg.data = data
        kg.save()
    return redirect("knowledge_graph")


# --------------------
# Chat View
# --------------------
def chat_view(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")
    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    return render(request, "chat.html", {"module": module, "module_id": module_id, "user_name": user_name})


# ==================================================
# 🚀 CHAT API - MAIN ENDPOINT
# ==================================================
@csrf_exempt
def chat_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body or "{}")

        user_message = (payload.get("message") or "").strip()
        conversation_id = payload.get("conversation_id")
        module_id = payload.get("module_id")
        feedback = payload.get("feedback")
        context_settings = payload.get("context_settings", {})

        if not module_id:
            return JsonResponse({"error": "module_id required"}, status=400)

        user_name = request.session.get("user_name")
        if not user_name:
            return JsonResponse({"error": "Not authenticated"}, status=401)

        module = get_object_or_404(Module, id=module_id, user_name=user_name)

        # Get or create conversation
        conversation = None
        if conversation_id:
            conversation = Conversation.objects.filter(id=conversation_id, module=module).first()

        if not conversation:
            conversation = Conversation.objects.create(
                module=module,
                session_id=f"conv_{Conversation.objects.filter(module=module).count() + 1}",
                title=user_message[:50] if user_message else "New Chat"
            )

        # Load session data from conversation context
        session_data = conversation.context or {"entities": {}, "history": []}

        # === PROTECTION: If user sends a new message, ignore stale feedback ===
        if user_message and feedback:
            # New query with feedback = user typed new question while clarification was pending
            # Treat as new query, ignore the feedback
            logger.info("New message with stale feedback - treating as new query")
            feedback = None

        # === PROTECTION: Validate feedback has required context ===
        if feedback and not user_message:
            feedback_type = feedback.get('type')
            
            # Check if we have the required session context for this feedback
            if feedback_type == 'value_selection':
                if 'last_clarification' not in session_data and not feedback.get('clarification_context', {}).get('table'):
                    logger.warning("Stale value_selection feedback - missing context")
                    return JsonResponse({
                        "conversation_id": conversation.id,
                        "type": "error",
                        "message": "Session expired. Please ask your question again.",
                        "session_data": {}
                    })
            
            if feedback_type == 'column_selection':
                if 'last_matches_by_column' not in session_data and not feedback.get('clarification_context', {}).get('matches_by_column'):
                    logger.warning("Stale column_selection feedback - missing context")
                    return JsonResponse({
                        "conversation_id": conversation.id,
                        "type": "error",
                        "message": "Session expired. Please ask your question again.",
                        "session_data": {}
                    })

        # Save user message
        if user_message:
            Message.objects.create(
                conversation=conversation,
                content=user_message,
                is_user=True
            )

        # Call RAG engine
        from pgadmin.RAG_LLM.main import invoke_graph

        result = invoke_graph(
            user_query=user_message,
            module_id=module_id,
            session_data=session_data,
            feedback=feedback,
            context_settings=context_settings
        )

        # Update session data
        updated_session_data = result.get("session_data", session_data)
        conversation.context = updated_session_data

        # Update conversation title
        if conversation.messages.count() == 1 and user_message:
            conversation.title = user_message[:50]

        conversation.save()

        # Debug logging
        logger.info("RAG RESULT TYPE: %s", result.get("type"))

        # Build response
        response = {
            "conversation_id": conversation.id,
            "session_data": updated_session_data
        }

        # ==================================================
        # ✅ CLARIFICATION RESPONSE
        # ==================================================
        if result.get("type") == "clarification":
            response.update({
                "type": "clarification",
                "message": result.get("message", "Please clarify"),
                "options": result.get("options", []),
                "subtype": result.get("subtype"),
                "entity": result.get("entity"),
                "entity_type": result.get("entity_type"),
                "clarification_context": result.get("clarification_context"),
                "allow_custom": bool(result.get("allow_custom", False))
            })
            return JsonResponse(response)

        # ==================================================
        # ✅ NORMAL RESPONSE
        # ==================================================
        if result.get("type") == "response":
            assistant_message = result.get("message", "OK")

            Message.objects.create(
                conversation=conversation,
                content=assistant_message,
                is_user=False,
                metadata={
                    "sql": result.get("sql"),
                    "data": result.get("data"),
                    "chart": result.get("chart")
                }
            )

            response.update({
                "type": "response",
                "message": assistant_message,
                "sql": result.get("sql"),
                "data": result.get("data"),
                "chart": result.get("chart")
            })
            return JsonResponse(response)

        # ==================================================
        # ❌ ERROR RESPONSE
        # ==================================================
        if result.get("type") == "error":
            error_msg = result.get("message", "Error")
            Message.objects.create(conversation=conversation, content=error_msg, is_user=False)
            return JsonResponse({
                "conversation_id": conversation.id,
                "type": "error",
                "message": error_msg,
                "session_data": updated_session_data
            })

        # ==================================================
        # 🚨 FALLBACK
        # ==================================================
        logger.error("Unhandled RAG response type: %s", result.get("type"))
        return JsonResponse({
            "conversation_id": conversation.id,
            "type": "error",
            "message": "Invalid response from AI engine",
            "session_data": updated_session_data
        })

    except Exception as e:
        logger.exception("chat_api failed")
        return JsonResponse({"error": str(e)}, status=500)


# --------------------
# Conversation Management
# --------------------
@csrf_exempt
def get_conversations(request):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    module_id = request.GET.get("module_id")
    if not module_id:
        return JsonResponse({"conversations": []})
    try:
        convs = Conversation.objects.filter(module_id=module_id).order_by("-updated_at")[:50]
        out = []
        for c in convs:
            out.append({
                "id": c.id, 
                "title": c.title, 
                "updated_at": c.updated_at.isoformat(), 
                "message_count": c.messages.count()
            })
        return JsonResponse({"conversations": out})
    except Exception as e:
        logger.exception("get_conversations error")
        return JsonResponse({"conversations": []})


@csrf_exempt
def load_conversation(request, conversation_id):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    try:
        conv = Conversation.objects.get(id=conversation_id)
        msgs = []
        for m in conv.messages.all():
            msgs.append({
                "content": m.content, 
                "is_user": m.is_user, 
                "timestamp": m.timestamp.isoformat(), 
                "metadata": getattr(m, "metadata", {})
            })
        return JsonResponse({
            "conversation": {"id": conv.id, "title": conv.title, "context": conv.context}, 
            "messages": msgs
        })
    except Conversation.DoesNotExist:
        return JsonResponse({"error": "Conversation not found"}, status=404)
    except Exception as e:
        logger.exception("load_conversation error")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def delete_conversation(request, conversation_id):
    if request.method != "DELETE":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    try:
        conv = Conversation.objects.get(id=conversation_id)
        Message.objects.filter(conversation=conv).delete()
        conv.delete()
        if conversation_id in SESSION_STORE:
            del SESSION_STORE[conversation_id]
        return JsonResponse({"success": True})
    except Conversation.DoesNotExist:
        return JsonResponse({"error": "Conversation not found"}, status=404)
    except Exception as e:
        logger.exception("delete_conversation error")
        return JsonResponse({"error": str(e)}, status=500)


# ==================================================
# 📊 DASHBOARD BUILDER
# ==================================================

def dashboard_builder_view(request, module_id):
    """Dashboard Builder page"""
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")
    
    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    return render(request, "dashboard_builder.html", {"module": module})


@csrf_exempt
def dashboard_generate_api(request):
    """Generate dashboard widgets based on prompt"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        from openai import OpenAI
        import os
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        payload = json.loads(request.body or "{}")
        prompt = payload.get("prompt", "").strip()
        module_id = payload.get("module_id")
        
        if not prompt or not module_id:
            return JsonResponse({"error": "prompt and module_id required"}, status=400)
        
        user_name = request.session.get("user_name")
        if not user_name:
            return JsonResponse({"error": "Not authenticated"}, status=401)
        
        module = get_object_or_404(Module, id=module_id, user_name=user_name)
        
        # Get module schema
        tables_data = module.tables or []
        schema_info = []
        
        for table in tables_data:
            # Handle both string and dict formats
            if isinstance(table, str):
                table_name = table
            elif isinstance(table, dict):
                table_name = table.get('name', '')
            else:
                continue
            
            if not table_name:
                continue
                
            columns = _get_table_columns(table_name)
            col_names = [c['name'] if isinstance(c, dict) else str(c) for c in columns]
            schema_info.append(f"Table: {table_name}\nColumns: {', '.join(col_names)}")
        
        schema_text = "\n\n".join(schema_info)
        
        if not schema_text:
            return JsonResponse({"error": "No tables found in module"}, status=400)
        
        # Generate dashboard specification using LLM
        system_prompt = """You are a dashboard designer. Based on the user's request and database schema, generate a dashboard specification.

Return ONLY valid JSON with this structure:
{
    "title": "Dashboard Title",
    "widgets": [
        {
            "type": "kpi",
            "title": "Key Metrics",
            "sql": "SELECT COUNT(*) as total_orders, SUM(amount) as total_sales FROM table",
            "kpis": [
                {"label": "Total Orders", "value_key": "total_orders", "format": "number"},
                {"label": "Total Sales", "value_key": "total_sales", "format": "currency"}
            ]
        },
        {
            "type": "chart",
            "title": "Monthly Trend",
            "chartType": "line|bar|pie|doughnut",
            "sql": "SELECT month, SUM(sales) as sales FROM table GROUP BY month",
            "labelKey": "month",
            "valueKey": "sales"
        },
        {
            "type": "table",
            "title": "Top Products",
            "sql": "SELECT product_name, SUM(quantity) as qty FROM table GROUP BY product_name ORDER BY qty DESC LIMIT 10"
        }
    ]
}

Rules:
1. Generate 3-5 relevant widgets based on the prompt
2. Use valid SQL for PostgreSQL
3. Include at least one KPI widget
4. Include at least one chart
5. Use appropriate chart types (line for trends, bar for comparisons, pie for distributions)
6. All SQL should use table/column names from the schema"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Database Schema:\n{schema_text}\n\nUser Request: {prompt}"}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        dashboard_spec = json.loads(response.choices[0].message.content)
        
        # Execute SQL for each widget and populate data
        widgets = []
        for widget_spec in dashboard_spec.get("widgets", []):
            widget = process_widget(widget_spec)
            if widget:
                widgets.append(widget)
        
        return JsonResponse({
            "success": True,
            "title": dashboard_spec.get("title", "Dashboard"),
            "widgets": widgets
        })
        
    except Exception as e:
        logger.exception("Dashboard generate error")
        return JsonResponse({"error": str(e)}, status=500)


def process_widget(widget_spec):
    """Execute SQL and process widget data"""
    try:
        sql = widget_spec.get("sql", "")
        if not sql:
            return None
        
        # Execute SQL
        with django_connection.cursor() as cursor:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
        
        # Convert to list of dicts
        data = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                if isinstance(value, Decimal):
                    value = float(value)
                elif hasattr(value, 'isoformat'):
                    value = value.isoformat()
                row_dict[col] = value
            data.append(row_dict)
        
        widget_type = widget_spec.get("type", "table")
        
        if widget_type == "kpi":
            # Process KPI widget
            kpis = []
            if data:
                row = data[0]
                for kpi_spec in widget_spec.get("kpis", []):
                    value_key = kpi_spec.get("value_key", "")
                    kpis.append({
                        "label": kpi_spec.get("label", value_key),
                        "value": row.get(value_key, 0),
                        "format": kpi_spec.get("format", "number"),
                        "change": kpi_spec.get("change")
                    })
            
            return {
                "type": "kpi",
                "title": widget_spec.get("title", "KPIs"),
                "kpis": kpis
            }
        
        elif widget_type == "chart":
            # Process chart widget
            labels = []
            values = []
            label_key = widget_spec.get("labelKey", columns[0] if columns else "label")
            value_key = widget_spec.get("valueKey", columns[1] if len(columns) > 1 else "value")
            
            for row in data[:20]:  # Limit to 20 data points
                labels.append(str(row.get(label_key, "")))
                values.append(row.get(value_key, 0))
            
            return {
                "type": "chart",
                "title": widget_spec.get("title", "Chart"),
                "chartType": widget_spec.get("chartType", "bar"),
                "labels": labels,
                "values": values
            }
        
        else:
            # Table widget
            return {
                "type": "table",
                "title": widget_spec.get("title", "Data"),
                "columns": columns,
                "data": data[:100]  # Limit to 100 rows
            }
    
    except Exception as e:
        logger.error(f"Widget processing error: {e}")
        return {
            "type": "table",
            "title": widget_spec.get("title", "Error"),
            "error": str(e),
            "data": []
        }


@csrf_exempt
def dashboard_save_api(request):
    """Save dashboard"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        payload = json.loads(request.body or "{}")
        dashboard_id = payload.get("dashboard_id")
        module_id = payload.get("module_id")
        title = payload.get("title", "Untitled Dashboard")
        widgets = payload.get("widgets", [])
        filters = payload.get("filters", [])
        prompt = payload.get("prompt", "")
        
        logger.info(f"Saving dashboard: module={module_id}, title={title}, widgets={len(widgets)}")
        
        user_name = request.session.get("user_name")
        if not user_name:
            return JsonResponse({"error": "Not authenticated"}, status=401)
        
        module = get_object_or_404(Module, id=module_id, user_name=user_name)
        
        # Context data to store
        context_data = {
            "type": "dashboard",
            "widgets": widgets,
            "filters": filters,
            "prompt": prompt
        }
        
        # Store in Conversation model with session_id starting with 'dashboard_'
        dashboard = None
        if dashboard_id:
            dashboard = Conversation.objects.filter(
                id=dashboard_id, 
                module=module,
                session_id__startswith='dashboard_'
            ).first()
        
        if dashboard:
            dashboard.title = title
            dashboard.context = context_data
            dashboard.updated_at = timezone.now()
            dashboard.save()
            logger.info(f"Dashboard updated: {dashboard.id} - {title}")
        else:
            dashboard = Conversation.objects.create(
                module=module,
                session_id=f"dashboard_{int(timezone.now().timestamp())}",
                title=title,
                context=context_data
            )
            logger.info(f"Dashboard created: {dashboard.id} - {title}")
        
        return JsonResponse({
            "success": True,
            "dashboard_id": dashboard.id,
            "title": dashboard.title
        })
        
    except Exception as e:
        logger.exception("Dashboard save error")
        return JsonResponse({"error": str(e)}, status=500)


def dashboard_list_api(request):
    """List all dashboards for a module"""
    try:
        module_id = request.GET.get("module_id")
        user_name = request.session.get("user_name")
        
        logger.info(f"Dashboard list request: module_id={module_id}, user={user_name}")
        
        if not user_name:
            return JsonResponse({"error": "Not authenticated"}, status=401)
        
        if not module_id:
            return JsonResponse({"error": "module_id required"}, status=400)
        
        module = get_object_or_404(Module, id=module_id, user_name=user_name)
        
        # Get dashboards - filter by session_id starting with 'dashboard_'
        dashboards = Conversation.objects.filter(
            module=module,
            session_id__startswith='dashboard_'
        ).order_by("-updated_at")
        
        logger.info(f"Found {dashboards.count()} dashboards for module {module_id}")
        
        dashboard_list = []
        for dash in dashboards:
            try:
                context = dash.context or {}
                # Handle if context is stored as string
                if isinstance(context, str):
                    try:
                        context = json.loads(context)
                    except:
                        context = {}
                
                widget_count = len(context.get("widgets", [])) if isinstance(context, dict) else 0
                dashboard_list.append({
                    "id": dash.id,
                    "title": dash.title or "Untitled",
                    "widget_count": widget_count,
                    "updated_at": dash.updated_at.isoformat() if dash.updated_at else None
                })
            except Exception as e:
                logger.error(f"Error processing dashboard {dash.id}: {e}")
                continue
        
        logger.info(f"Returning {len(dashboard_list)} dashboards")
        return JsonResponse({"dashboards": dashboard_list})
        
    except Exception as e:
        logger.exception("Dashboard list error")
        return JsonResponse({"error": str(e), "dashboards": []}, status=500)


def dashboard_get_api(request, dashboard_id):
    """Get a specific dashboard"""
    try:
        user_name = request.session.get("user_name")
        if not user_name:
            return JsonResponse({"error": "Not authenticated"}, status=401)
        
        dashboard = get_object_or_404(Conversation, id=dashboard_id)
        
        # Verify ownership
        if dashboard.module.user_name != user_name:
            return JsonResponse({"error": "Access denied"}, status=403)
        
        context = dashboard.context or {}
        
        # Ensure context is a dict
        if not isinstance(context, dict):
            try:
                context = json.loads(context) if isinstance(context, str) else {}
            except:
                context = {}
        
        return JsonResponse({
            "success": True,
            "id": dashboard.id,
            "title": dashboard.title,
            "widgets": context.get("widgets", []),
            "filters": context.get("filters", []),
            "prompt": context.get("prompt", "")
        })
        
    except Conversation.DoesNotExist:
        return JsonResponse({"error": "Dashboard not found"}, status=404)
    except Exception as e:
        logger.exception("Dashboard get error")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def dashboard_delete_api(request, dashboard_id):
    """Delete a dashboard"""
    if request.method != "DELETE":
        return JsonResponse({"error": "DELETE required"}, status=405)
    
    try:
        user_name = request.session.get("user_name")
        if not user_name:
            return JsonResponse({"error": "Not authenticated"}, status=401)
        
        dashboard = get_object_or_404(Conversation, id=dashboard_id)
        
        # Verify ownership
        if dashboard.module.user_name != user_name:
            return JsonResponse({"error": "Access denied"}, status=403)
        
        dashboard.delete()
        
        return JsonResponse({"success": True})
        
    except Exception as e:
        logger.exception("Dashboard delete error")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def dashboard_filter_values_api(request):
    """Get distinct values for a column to use as filter options"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        payload = json.loads(request.body or "{}")
        table = payload.get("table", "")
        column = payload.get("column", "")
        
        if not table or not column:
            return JsonResponse({"error": "table and column required"}, status=400)
        
        user_name = request.session.get("user_name")
        if not user_name:
            return JsonResponse({"error": "Not authenticated"}, status=401)
        
        # Get distinct values from the column
        with django_connection.cursor() as cursor:
            # Use parameterized query for table/column names
            # Note: Can't use params for identifiers, so we validate
            safe_table = table.replace('"', '').replace("'", "").replace(";", "")
            safe_column = column.replace('"', '').replace("'", "").replace(";", "")
            
            cursor.execute(f'''
                SELECT DISTINCT "{safe_column}" 
                FROM "{safe_table}" 
                WHERE "{safe_column}" IS NOT NULL 
                ORDER BY "{safe_column}" 
                LIMIT 100
            ''')
            
            values = [row[0] for row in cursor.fetchall()]
            
            # Convert to string for JSON
            values = [str(v) if v is not None else None for v in values]
        
        return JsonResponse({
            "success": True,
            "table": table,
            "column": column,
            "values": values
        })
        
    except Exception as e:
        logger.exception("Filter values error")
        return JsonResponse({"error": str(e), "values": []}, status=500)