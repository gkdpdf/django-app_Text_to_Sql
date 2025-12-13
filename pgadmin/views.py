# pgadmin/views.py
import csv
import io
import json
import logging
import traceback
from datetime import datetime

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


def _get_sample_data(table_name, limit=3):
    """Helper: fetch sample data from a table for AI context"""
    try:
        with django_connection.cursor() as cursor:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT %s', [limit])
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            # Convert to serializable format
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
    """
    Use AI (LLM) to generate descriptions for tables and columns with empty descriptions.
    Returns updated kg_data with AI-generated descriptions.
    """
    import os
    
    # Collect items that need descriptions
    items_to_describe = []
    
    for table, cols in kg_data.items():
        # Check table description
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
        
        # Check column descriptions
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
        logger.info("No empty descriptions to fill")
        return kg_data
    
    logger.info(f"Generating AI descriptions for {len(items_to_describe)} items")
    
    # Build prompt for AI
    prompt = _build_kg_generation_prompt(items_to_describe)
    
    # Call LLM
    try:
        ai_response = _call_llm_for_kg(prompt)
        if ai_response:
            kg_data = _parse_and_apply_ai_descriptions(kg_data, ai_response)
            logger.info("Successfully generated AI descriptions")
    except Exception as e:
        logger.exception("Error generating AI descriptions: %s", e)
    
    return kg_data


def _build_kg_generation_prompt(items_to_describe):
    """Build a prompt for the LLM to generate KG descriptions"""
    
    prompt = """You are a database documentation expert. Generate clear, concise descriptions for the following database tables and columns.

For each item, provide a brief but informative description that explains:
- For tables: What data the table stores and its purpose (1-2 sentences)
- For columns: What the column represents, its meaning, and any relevant details (1 sentence)

IMPORTANT: Respond ONLY with valid JSON in this exact format, no other text:
{
    "tables": {
        "table_name": "description of the table"
    },
    "columns": {
        "table_name.column_name": "description of the column"
    }
}

Here are the items that need descriptions:

"""
    
    tables_section = []
    columns_section = []
    
    for item in items_to_describe:
        if item["type"] == "table":
            table_info = f"TABLE: {item['table']}\n"
            table_info += f"  Columns: {', '.join(item['columns'][:15])}"  # Limit columns shown
            if len(item['columns']) > 15:
                table_info += f" (and {len(item['columns']) - 15} more)"
            table_info += "\n"
            if item.get("sample_data", {}).get("rows"):
                # Show limited sample data
                sample_preview = str(item['sample_data']['rows'][:2])[:200]
                table_info += f"  Sample data: {sample_preview}...\n"
            tables_section.append(table_info)
        else:
            col_info = f"COLUMN: {item['table']}.{item['column']} (datatype: {item['datatype']})"
            columns_section.append(col_info)
    
    if tables_section:
        prompt += "TABLES TO DESCRIBE:\n" + "\n".join(tables_section) + "\n\n"
    
    if columns_section:
        prompt += "COLUMNS TO DESCRIBE:\n" + "\n".join(columns_section) + "\n\n"
    
    prompt += "Generate descriptions for ALL items above. Respond with JSON only."
    
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
    else:
        logger.warning("No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
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
        "messages": [
            {"role": "user", "content": prompt}
        ]
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
            {"role": "system", "content": "You are a database documentation expert. Always respond with valid JSON only."},
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
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', ai_response)
        if not json_match:
            logger.warning("No JSON found in AI response")
            return kg_data
        
        ai_data = json.loads(json_match.group())
        
        # Apply table descriptions
        tables_desc = ai_data.get("tables", {})
        for table_name, desc in tables_desc.items():
            if table_name in kg_data:
                if "TABLE_INFO" not in kg_data[table_name]:
                    kg_data[table_name]["TABLE_INFO"] = {"desc": "", "datatype": "meta"}
                if not kg_data[table_name]["TABLE_INFO"].get("desc", "").strip():
                    kg_data[table_name]["TABLE_INFO"]["desc"] = desc.strip()
        
        # Apply column descriptions
        columns_desc = ai_data.get("columns", {})
        for full_col_name, desc in columns_desc.items():
            parts = full_col_name.split(".", 1)
            if len(parts) == 2:
                table_name, col_name = parts
                if table_name in kg_data and col_name in kg_data[table_name]:
                    if not kg_data[table_name][col_name].get("desc", "").strip():
                        kg_data[table_name][col_name]["desc"] = desc.strip()
        
        return kg_data
        
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse AI response as JSON: %s", e)
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
            logger.warning("Invalid selected_columns JSON: %s", e)
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
            
            logger.info("Created module id=%s name=%s", module.id, module.name)
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

        # ===== save_all with optional AI generation =====
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

            selected_tables = selected_tables or []
            selected_columns = selected_columns or {}

            module.tables = selected_tables
            module.selected_columns = selected_columns

            # Build KG from posted form fields
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

            # Merge with selection
            final_kg = _merge_kg_with_selection(posted_kg, selected_tables, selected_columns)
            
            # Preserve user-entered descriptions
            for table in posted_kg:
                if table in final_kg:
                    for col, data in posted_kg[table].items():
                        if col in final_kg[table]:
                            if data.get("desc"):
                                final_kg[table][col]["desc"] = data["desc"]
                            if data.get("datatype"):
                                final_kg[table][col]["datatype"] = data["datatype"]

            # Run AI to fill blank descriptions if requested
            if run_ai:
                logger.info("Running AI to fill blank descriptions...")
                final_kg = _generate_kg_descriptions_with_ai(final_kg, selected_columns)

            module.knowledge_graph_data = final_kg

            # Save relationships
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

            # Extra suggestions
            module.extra_suggestions = request.POST.get("extra_suggestions", "").strip()

            module.save()
            
            logger.info("Saved module %s (run_ai=%s)", module.id, run_ai)
            return redirect("edit_module", module_id=module.id)

    # GET: prepare template data
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
# Generate KG API (AJAX endpoint)
# --------------------
@csrf_exempt
@require_http_methods(["POST"])
def generate_kg_api(request, module_id):
    """AJAX endpoint to generate AI descriptions for empty KG fields."""
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
        logger.info("Deleted module %s id=%s", name, module_id)
        return JsonResponse({"success": True, "message": f"Deleted {name}"})
    except Module.DoesNotExist:
        return JsonResponse({"error": "Module not found"}, status=404)
    except Exception as e:
        logger.exception("Error deleting module")
        return JsonResponse({"error": str(e)}, status=500)


# --------------------
# Utilities: get_table_columns
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
        logger.exception("Error loading columns for %s", table_name)
        return JsonResponse({"error": str(e)}, status=500)


# --------------------
# Download / Upload per-module KG CSV
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
# Global (legacy) KG views
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
# Chat & Conversation helpers
# --------------------
def chat_view(request, module_id):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")
    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    return render(request, "chat.html", {"module": module, "module_id": module_id, "user_name": user_name})

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

        conversation = None
        if conversation_id:
            conversation = Conversation.objects.filter(id=conversation_id, module=module).first()

        if not conversation:
            conversation = Conversation.objects.create(
                module=module,
                session_id=f"conv_{Conversation.objects.filter(module=module).count() + 1}",
                title=user_message[:50] if user_message else "New Chat"
            )

        session_data = conversation.context or {"entities": {}, "history": []}

        if user_message:
            Message.objects.create(
                conversation=conversation,
                content=user_message,
                is_user=True
            )

        from pgadmin.RAG_LLM.main import invoke_graph

        result = invoke_graph(
            user_query=user_message,
            module_id=module_id,
            session_data=session_data,
            feedback=feedback,
            context_settings=context_settings
        )

        updated_session_data = result.get("session_data", session_data)
        conversation.context = updated_session_data

        if conversation.messages.count() == 1 and user_message:
            conversation.title = user_message[:50]

        conversation.save()

        # 🔍 DEBUG (TEMP – KEEP UNTIL STABLE)
        logger.warning("RAG RESULT KEYS: %s", list(result.keys()))
        logger.warning("RAG RESULT TYPE: %s", result.get("type"))
        logger.warning("RAG needs_clarification: %s", result.get("needs_clarification"))

        response = {
            "conversation_id": conversation.id,
            "session_data": updated_session_data
        }

        # ==================================================
        # ✅ CLARIFICATION — FINAL & SAFE
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
        # ❌ ERROR
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
        # 🚨 HARD FAIL (NO SILENT FALLBACK)
        # ==================================================
        logger.error("Unhandled RAG response: %s", result)

        return JsonResponse({
            "conversation_id": conversation.id,
            "type": "error",
            "message": "Invalid response from AI engine",
            "session_data": updated_session_data
        })

    except Exception as e:
        logger.exception("chat_api failed")
        return JsonResponse({"error": str(e)}, status=500)



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
            out.append({"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat(), "message_count": c.messages.count()})
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
            msgs.append({"content": m.content, "is_user": m.is_user, "timestamp": m.timestamp.isoformat(), "metadata": getattr(m, "metadata", {})})
        return JsonResponse({"conversation": {"id": conv.id, "title": conv.title, "context": conv.context}, "messages": msgs})
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