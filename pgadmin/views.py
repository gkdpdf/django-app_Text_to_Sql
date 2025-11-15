from django.shortcuts import render, redirect, get_object_or_404
from django.db import connection
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import Module, KnowledgeGraph, Metrics, RCA, Extra_suggestion, Conversation, Message
import csv
import io
import json
import logging
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)
SESSION_STORE = {}

# ---------- HOME ----------
def home_view(request):
    """Redirect to appropriate page based on login status"""
    if request.session.get("user_name"):
        return redirect("dashboard")
    return redirect("login")


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

    try:
        modules = Module.objects.filter(user_name=user_name).order_by("-created_at")
        
        # Force evaluation to catch any serialization errors
        modules_list = []
        for module in modules:
            # Ensure all JSON fields are valid
            if not isinstance(module.tables, list):
                module.tables = []
                module.save()
            modules_list.append(module)
        
        return render(request, "dashboard.html", {
            "modules": modules_list,
            "user_name": user_name
        })
        
    except Exception as e:
        logger.error(f"❌ Error in dashboard_view: {str(e)}", exc_info=True)
        return render(request, "dashboard.html", {
            "modules": [],
            "user_name": user_name,
            "error": "Error loading modules. Please try refreshing the page."
        })


# ---------- CREATE NEW MODULE ----------
def new_module_view(request):
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    if request.method == "POST":
        module_name = request.POST.get("module_name")
        selected_tables = request.POST.getlist("selected_tables")
        selected_columns_json = request.POST.get("selected_columns", "{}")
        
        try:
            selected_columns = json.loads(selected_columns_json)
        except:
            selected_columns = {}

        # -------------- FIX STARTS HERE --------------
        # Ensure selected_tables is always a Python list, not a JSON string
        if isinstance(selected_tables, str):
            try:
                selected_tables = json.loads(selected_tables)
            except:
                selected_tables = [selected_tables]
        # -------------- FIX ENDS HERE ----------------

        # Create module with selected tables
        module = Module.objects.create(
            user_name=user_name,
            name=module_name,
            tables=selected_tables,
        )
        
        # If user wants to auto-generate KG
        generate_kg = request.POST.get("generate_kg") == "true"
        if generate_kg and selected_tables:
            try:
                from .utils.kg_generator import generate_knowledge_graph_with_llm
                logger.info(f"Auto-generating KG for new module '{module_name}' with tables: {selected_tables}")
                module.knowledge_graph_data = generate_knowledge_graph_with_llm(selected_tables, selected_columns)
                module.kg_auto_generated = True
                module.save()
                logger.info(f"Successfully generated KG for module '{module_name}'")
            except Exception as e:
                logger.error(f"Error auto-generating KG for new module: {str(e)}")
                logger.error(traceback.format_exc())
        
        return redirect("edit_module", module_id=module.id)

    # fetch list of tables
    with connection.cursor() as cursor:
        cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        all_tables = [row[0] for row in cursor.fetchall()]

    exclude_tables = [
        'auth_group', 'auth_group_permissions', 'auth_permission',
        'auth_user', 'auth_user_groups', 'auth_user_user_permissions',
        'django_admin_log', 'django_content_type', 'django_migrations', 'django_session',
        'pgadmin_module', 'pgadmin_conversation', 'pgadmin_message',
        'pgadmin_knowledgegraph', 'pgadmin_metrics', 'pgadmin_rca', 'pgadmin_extra_suggestion'
    ]
    tables = [t for t in all_tables if t not in exclude_tables]

    return render(request, "new_module.html", {"tables": tables, "user_name": user_name})


# ---------- EDIT MODULE ----------
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
        'django_admin_log', 'django_content_type', 'django_migrations', 'django_session',
        'pgadmin_module', 'pgadmin_conversation', 'pgadmin_message',
        'pgadmin_knowledgegraph', 'pgadmin_metrics', 'pgadmin_rca', 'pgadmin_extra_suggestion'
    ]
    tables = [t for t in all_tables if t not in exclude_tables]

    if request.method == "POST":
        action = request.POST.get("action", "save_all")
        
        if action == "generate_kg":
            # Read from POST request
            selected_tables_from_form = request.POST.getlist("selected_tables")
            selected_columns_json = request.POST.get("selected_columns", "{}")
            
            try:
                selected_columns = json.loads(selected_columns_json)
            except json.JSONDecodeError:
                selected_columns = {}
            
            # Validate we have data
            if not selected_tables_from_form:
                return JsonResponse({
                    "success": False, 
                    "error": "No tables selected. Please select at least one table."
                }, status=400)
            
            # Validate columns
            tables_without_columns = [t for t in selected_tables_from_form if not selected_columns.get(t)]
            if tables_without_columns:
                return JsonResponse({
                    "success": False,
                    "error": f"Please select columns for: {', '.join(tables_without_columns)}"
                }, status=400)
            
            logger.info(f"=== AI GENERATION REQUEST ===")
            logger.info(f"Module: {module.name} (ID: {module_id})")
            logger.info(f"Tables: {selected_tables_from_form}")
            logger.info(f"Selected columns: {selected_columns}")
            logger.info(f"===============================")
            
            try:
                from .utils.kg_generator import generate_knowledge_graph_with_llm
                
                # Check API key
                import os
                if not os.environ.get("OPENAI_API_KEY"):
                    return JsonResponse({
                        "success": False,
                        "error": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
                    }, status=500)
                
                # ===== SMART GENERATION: Only generate for blank/missing fields =====
                existing_kg_data = module.knowledge_graph_data if isinstance(module.knowledge_graph_data, dict) else {}
                
                logger.info(f"Existing KG data has {len(existing_kg_data)} tables")
                
                # Identify which columns need generation (blank or missing)
                columns_needing_generation = {}
                
                for table in selected_tables_from_form:
                    table_columns = selected_columns.get(table, [])
                    columns_to_generate = []
                    
                    for column in table_columns:
                        needs_generation = False
                        
                        # Check if table doesn't exist
                        if table not in existing_kg_data:
                            needs_generation = True
                            logger.info(f"  - {table}.{column}: TABLE NEW, needs generation")
                        # Check if column doesn't exist in table
                        elif column not in existing_kg_data[table]:
                            needs_generation = True
                            logger.info(f"  - {table}.{column}: COLUMN NEW, needs generation")
                        else:
                            # Column exists - check if description or datatype is blank
                            existing_col_data = existing_kg_data[table][column]
                            desc = existing_col_data.get('desc', '').strip()
                            datatype = existing_col_data.get('datatype', '').strip()
                            
                            if not desc or not datatype:
                                needs_generation = True
                                logger.info(f"  - {table}.{column}: BLANK FIELDS (desc={bool(desc)}, type={bool(datatype)}), needs generation")
                            else:
                                logger.info(f"  - {table}.{column}: ALREADY FILLED, will preserve")
                        
                        if needs_generation:
                            columns_to_generate.append(column)
                    
                    if columns_to_generate:
                        columns_needing_generation[table] = columns_to_generate
                
                if not columns_needing_generation:
                    logger.info("✅ All fields already filled - nothing to generate")
                    return JsonResponse({
                        "success": True,
                        "message": "✅ All fields are already filled! No generation needed.",
                        "tables_processed": 0,
                        "columns_generated": 0
                    })
                
                # Log what we're generating
                total_columns = sum(len(cols) for cols in columns_needing_generation.values())
                logger.info(f"📝 Will generate {total_columns} column(s) across {len(columns_needing_generation)} table(s)")
                for table, cols in columns_needing_generation.items():
                    logger.info(f"   {table}: {cols}")
                
                # Generate KG only for needed columns
                new_kg_data = generate_knowledge_graph_with_llm(
                    list(columns_needing_generation.keys()), 
                    columns_needing_generation
                )
                
                if not new_kg_data:
                    raise Exception("Generated knowledge graph is empty")
                
                logger.info(f"✅ AI generated data for {len(new_kg_data)} tables")
                
                # ===== MERGE: Preserve all existing data, only add new fields =====
                merged_kg_data = existing_kg_data.copy()
                
                columns_generated = 0
                columns_preserved = 0
                
                # Process all selected tables
                for table in selected_tables_from_form:
                    if table not in merged_kg_data:
                        merged_kg_data[table] = {}
                    
                    table_columns = selected_columns.get(table, [])
                    
                    for column in table_columns:
                        if table in new_kg_data and column in new_kg_data[table]:
                            # We generated this column - add/update it
                            merged_kg_data[table][column] = new_kg_data[table][column]
                            columns_generated += 1
                            logger.info(f"  ✅ Generated: {table}.{column}")
                        elif column in merged_kg_data[table]:
                            # Column already exists with data - preserve it (DO NOTHING)
                            columns_preserved += 1
                            # Don't log to reduce noise
                        else:
                            # New column but wasn't in generation list (shouldn't happen)
                            merged_kg_data[table][column] = {
                                "desc": "",
                                "datatype": ""
                            }
                            logger.warning(f"  ⚠️ Column {table}.{column} not generated, creating blank")
                
                # Remove tables that are no longer selected
                tables_to_remove = [t for t in list(merged_kg_data.keys()) if t not in selected_tables_from_form]
                for table in tables_to_remove:
                    del merged_kg_data[table]
                    logger.info(f"  🗑️ Removed deselected table: {table}")
                
                # ===== CRITICAL: Don't clear table selections! =====
                # Only update tables list if it's different
                if set(module.tables or []) != set(selected_tables_from_form):
                    module.tables = selected_tables_from_form
                    logger.info(f"Updated table selections")
                
                # Update module with merged data
                module.knowledge_graph_data = merged_kg_data
                module.kg_auto_generated = True
                module.save()
                
                logger.info(f"💾 Saved: {columns_generated} generated, {columns_preserved} preserved")
                
                return JsonResponse({
                    "success": True, 
                    "message": f"✅ Generated {columns_generated} field(s)! {columns_preserved} existing field(s) preserved.",
                    "columns_generated": columns_generated,
                    "columns_preserved": columns_preserved
                })
                
            except ImportError as e:
                error_msg = f"Import error: {str(e)}. Please ensure kg_generator module is available."
                logger.error(error_msg)
                return JsonResponse({"success": False, "error": error_msg}, status=500)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Error generating KG: {error_msg}")
                logger.error(traceback.format_exc())
                return JsonResponse({"success": False, "error": error_msg}, status=500)
        
        elif action == "save_all":
            # Update table selections
            selected_tables = request.POST.getlist("selected_tables")
            module.tables = selected_tables if selected_tables else []
            
            # Get existing KG data
            existing_kg_data = module.knowledge_graph_data if isinstance(module.knowledge_graph_data, dict) else {}
            
            # Parse form data for knowledge graph
            kg_data_from_form = {}
            for key, value in request.POST.items():
                if key.startswith("kg_desc__"):
                    parts = key.split("__", 2)
                    if len(parts) == 3:
                        _, table, column = parts
                        if table not in kg_data_from_form:
                            kg_data_from_form[table] = {}
                        if column not in kg_data_from_form[table]:
                            kg_data_from_form[table][column] = {}
                        kg_data_from_form[table][column]["desc"] = value.strip()
                        
                elif key.startswith("kg_datatype__"):
                    parts = key.split("__", 2)
                    if len(parts) == 3:
                        _, table, column = parts
                        if table not in kg_data_from_form:
                            kg_data_from_form[table] = {}
                        if column not in kg_data_from_form[table]:
                            kg_data_from_form[table][column] = {}
                        kg_data_from_form[table][column]["datatype"] = value.strip()
            
            # Merge with existing data
            merged_kg_data = existing_kg_data.copy()
            for table in selected_tables:
                if table in kg_data_from_form:
                    if table not in merged_kg_data:
                        merged_kg_data[table] = {}
                    for column, data in kg_data_from_form[table].items():
                        merged_kg_data[table][column] = data
            
            # Remove deselected tables
            tables_to_remove = [t for t in list(merged_kg_data.keys()) if t not in selected_tables]
            for table in tables_to_remove:
                del merged_kg_data[table]
            
            module.knowledge_graph_data = merged_kg_data
            logger.info(f"💾 Saved manual edits for {len(merged_kg_data)} tables")
            
            # ========== Save Relationships (NEW) ==========
            relationships_list = []
            rel_left_tables = request.POST.getlist("rel_left_table")
            rel_left_columns = request.POST.getlist("rel_left_column")
            rel_types = request.POST.getlist("rel_type")
            rel_right_tables = request.POST.getlist("rel_right_table")
            rel_right_columns = request.POST.getlist("rel_right_column")
            
            for i in range(len(rel_left_tables)):
                if rel_left_tables[i] and rel_right_tables[i]:
                    relationships_list.append({
                        "left_table": rel_left_tables[i],
                        "left_column": rel_left_columns[i] if i < len(rel_left_columns) else "",
                        "type": rel_types[i] if i < len(rel_types) else "one-to-many",
                        "right_table": rel_right_tables[i],
                        "right_column": rel_right_columns[i] if i < len(rel_right_columns) else ""
                    })
            module.relationships = relationships_list
            logger.info(f"💾 Saved {len(relationships_list)} relationships")
            
            # Save RCAs
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
            
            # Save POS Tagging
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
            
            # Save Metrics
            metrics_data = {}
            metric_names = request.POST.getlist("metric_name")
            metric_descs = request.POST.getlist("metric_desc")
            for i in range(len(metric_names)):
                if metric_names[i].strip():
                    metrics_data[metric_names[i].strip()] = metric_descs[i].strip() if i < len(metric_descs) else ""
            module.metrics_data = metrics_data
            
            # Save Extra Suggestions
            module.extra_suggestions = request.POST.get("extra_suggestions", "").strip()
            
            module.save()
            logger.info(f"✅ Saved all module data for '{module.name}'")
            return redirect("dashboard")
    # ===== GET REQUEST: Prepare data for display =====
    knowledge_data = {}
    if module.tables:
        for table in module.tables:
            try:
                with connection.cursor() as cursor:
                    columns = [col.name for col in connection.introspection.get_table_description(cursor, table)]
                
                knowledge_data[table] = {}
                for col in columns:
                    existing_info = module.knowledge_graph_data.get(table, {}).get(col, {})
                    knowledge_data[table][col] = {
                        "desc": existing_info.get("desc", ""),
                        "datatype": existing_info.get("datatype", "")
                    }
            except Exception as e:
                logger.error(f"Error processing table {table}: {e}")
                continue

    selected_tables = module.tables if isinstance(module.tables, list) else []

    return render(request, "edit_module.html", {
        "module": module,
        "tables": tables,
        "selected_tables": selected_tables,
        "knowledge_data": knowledge_data,
        "relationships": module.relationships or [],  # ← NEW
        "rca_list": module.rca_list or [],
        "pos_tagging": module.pos_tagging or [],
        "metrics_data": module.metrics_data or {},
        "extra_suggestions": module.extra_suggestions or "",
        "kg_auto_generated": module.kg_auto_generated,
        "user_name": user_name,
    })

# ---------- DELETE MODULE ----------
@csrf_exempt
@require_http_methods(["DELETE", "POST"])
def delete_module_view(request, module_id):
    """Delete a module"""
    user_name = request.session.get("user_name")
    if not user_name:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    
    try:
        module = Module.objects.get(id=module_id, user_name=user_name)
        module_name = module.name
        
        # Delete all conversations for this module
        deleted_count = module.conversations.count()
        module.delete()
        
        logger.info(f"Module '{module_name}' (ID: {module_id}) and {deleted_count} conversations deleted")
        return JsonResponse({
            "success": True, 
            "message": f"Module '{module_name}' deleted successfully"
        })
    except Module.DoesNotExist:
        return JsonResponse({"error": "Module not found"}, status=404)
    except Exception as e:
        logger.error(f"Error deleting module {module_id}: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


# ---------- UTILITY FUNCTIONS ----------
@csrf_exempt
def get_table_columns(request):
    """Get columns for a specific table"""
    table_name = request.GET.get('table')
    if not table_name:
        return JsonResponse({"error": "No table specified"}, status=400)
    
    try:
        with connection.cursor() as cursor:
            columns = connection.introspection.get_table_description(cursor, table_name)
            column_list = [{"name": col.name, "type": str(col.type_code)} for col in columns]
        
        return JsonResponse({"columns": column_list})
    except Exception as e:
        logger.error(f"Error getting columns for table {table_name}: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def download_module_kg_csv(request, module_id):
    """Download knowledge graph as CSV for specific module"""
    user_name = request.session.get("user_name")
    if not user_name:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    
    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{module.name}_knowledge_graph.csv"'

    writer = csv.writer(response)
    writer.writerow(["table", "column", "desc", "datatype"])

    for table, columns in module.knowledge_graph_data.items():
        for column, info in columns.items():
            writer.writerow([
                table,
                column,
                info.get("desc", ""),
                info.get("datatype", "")
            ])

    return response


@csrf_exempt
def upload_module_kg_csv(request, module_id):
    """Upload knowledge graph CSV for specific module"""
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")
    
    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        decoded_file = csv_file.read().decode('utf-8').splitlines()
        reader = csv.DictReader(decoded_file)

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

        module.knowledge_graph_data = data
        module.save()

    return redirect('edit_module', module_id=module_id)


# ---------- GLOBAL KNOWLEDGE GRAPH (Legacy) - FIXED ----------
def knowledge_graph_view(request):
    kg_instance, _ = KnowledgeGraph.objects.get_or_create(id=1)
    metrics_instance, _ = Metrics.objects.get_or_create(id=1)
    rca_instance, _ = RCA.objects.get_or_create(id=1)
    extra_instance, _ = Extra_suggestion.objects.get_or_create(id=1)

    # FIX: Don't call json.loads on data that's already a dict/list
    if isinstance(kg_instance.data, str):
        try:
            existing_kg_data = json.loads(kg_instance.data)
        except:
            existing_kg_data = {}
    elif isinstance(kg_instance.data, dict):
        existing_kg_data = kg_instance.data
    else:
        existing_kg_data = {}

    if isinstance(metrics_instance.data, str):
        try:
            existing_metrics_data = json.loads(metrics_instance.data)
        except:
            existing_metrics_data = {}
    elif isinstance(metrics_instance.data, dict):
        existing_metrics_data = metrics_instance.data
    else:
        existing_metrics_data = {}

    # Handle RCA data
    existing_rca_data = ""
    if rca_instance.data:
        if isinstance(rca_instance.data, str):
            try:
                parsed_rca = json.loads(rca_instance.data)
                existing_rca_data = parsed_rca.get("text", "") if isinstance(parsed_rca, dict) else ""
            except:
                existing_rca_data = ""
        elif isinstance(rca_instance.data, dict):
            existing_rca_data = rca_instance.data.get("text", "")

    # Handle Extra suggestion data
    existing_extra_data = ""
    if extra_instance.data:
        if isinstance(extra_instance.data, str):
            try:
                parsed_extra = json.loads(extra_instance.data)
                existing_extra_data = parsed_extra.get("text", "") if isinstance(parsed_extra, dict) else ""
            except:
                existing_extra_data = ""
        elif isinstance(extra_instance.data, dict):
            existing_extra_data = extra_instance.data.get("text", "")

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


def download_knowledge_graph_csv(request):
    kg_instance = KnowledgeGraph.objects.first()

    # FIX: Handle both string and dict data
    if isinstance(kg_instance.data, str):
        try:
            data = json.loads(kg_instance.data)
        except:
            data = {}
    elif isinstance(kg_instance.data, dict):
        data = kg_instance.data
    else:
        data = {}

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="knowledge_graph.csv"'

    writer = csv.writer(response)
    writer.writerow(["table", "column", "desc", "datatype"])

    for table, columns in data.items():
        for column, info in columns.items():
            writer.writerow([
                table,
                column,
                info.get("desc", ""),
                info.get("datatype", "")
            ])

    return response


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


# ---------- CHAT VIEWS ----------

def chat_view(request, module_id):
    """Render the chat interface for a specific module"""
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")
    
    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    
    return render(request, "chat.html", {
        "module": module,
        "module_id": module_id,
        "user_name": user_name
    })


@csrf_exempt
def chat_api(request):
    """Handle chat messages with conversation persistence"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    user_query = data.get("message", "").strip()
    feedback = data.get("feedback", None)
    conversation_id = data.get("conversation_id", None)
    module_id = data.get("module_id", None)
    
    logger.info(f"📨 Chat API: conv_id={conversation_id}, module_id={module_id}, msg='{user_query[:50] if user_query else ''}', feedback={bool(feedback)}")
    
    if not user_query and not feedback:
        return JsonResponse({"error": "Empty message"}, status=400)
    
    if not module_id:
        return JsonResponse({"error": "Module ID required"}, status=400)
    
    # Ensure session exists
    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key
    
    # Get or create conversation
    conversation = None
    if conversation_id:
        try:
            conversation = Conversation.objects.get(id=conversation_id, session_id=session_id)
            logger.info(f"✅ Using existing conversation {conversation.id}")
        except Conversation.DoesNotExist:
            logger.warning(f"⚠️ Conversation {conversation_id} not found")
            conversation = None
    
    # Create new conversation if needed
    if not conversation and user_query and not feedback:
        try:
            module = Module.objects.get(id=module_id)
            conversation = Conversation.objects.create(
                module=module,
                session_id=session_id,
                title=user_query[:50] + ('...' if len(user_query) > 50 else '')
            )
            logger.info(f"✅ Created NEW conversation {conversation.id}: '{conversation.title}'")
        except Module.DoesNotExist:
            return JsonResponse({"error": "Module not found"}, status=404)
    
    # Error if feedback without conversation
    if not conversation and feedback:
        return JsonResponse({"error": "Invalid conversation state"}, status=400)
    
    # Session management
    if conversation.id not in SESSION_STORE:
        SESSION_STORE[conversation.id] = {
            "entities": conversation.entities or {},
            "history": list(conversation.messages.filter(is_user=True).values_list('content', flat=True)),
            "thread_id": f"conv_{conversation.id}",
            "last_query": None,
            "pending_clarification": conversation.context.get("pending_clarification") if conversation.context else None,
            "module_id": module_id
        }
    
    session_data = SESSION_STORE[conversation.id]
    
    if module_id:
        session_data["module_id"] = module_id
    
    # Process request
    try:
        if feedback:
            entity_type = feedback.get("entity_type")
            feedback_type = feedback.get("type")
            
            if session_data.get("pending_clarification"):
                feedback["clarification_context"] = {
                    "table": session_data["pending_clarification"].get("table"),
                    "column": session_data["pending_clarification"].get("column")
                }
            
            if feedback_type == "value_selection":
                selected_value = feedback.get("selected_option")
                if entity_type and selected_value:
                    session_data["entities"][entity_type] = selected_value
            
            elif feedback_type == "custom_input":
                custom_value = feedback.get("custom_value")
                if entity_type and custom_value:
                    session_data["entities"][entity_type] = custom_value
            
            if feedback_type in ["value_selection", "custom_input"]:
                selection_text = feedback.get("selected_option") or feedback.get("custom_value")
                Message.objects.create(
                    conversation=conversation,
                    content=f"Selected: {selection_text}",
                    is_user=True,
                    metadata={"type": "selection", "entity_type": entity_type}
                )
            
            original_query = session_data.get("last_query", "")
            from .RAG_LLM.main import invoke_graph
            result = invoke_graph(original_query, session_data, human_feedback=feedback)
            session_data["pending_clarification"] = None
            
        else:
            # New user message
            user_message = Message.objects.create(
                conversation=conversation,
                content=user_query,
                is_user=True
            )
            logger.info(f"💾 Saved user message to DB")
            
            session_data["last_query"] = user_query
            from .RAG_LLM.main import invoke_graph
            result = invoke_graph(user_query, session_data)
            
            if result.get("type") == "clarification":
                session_data["pending_clarification"] = {
                    "entity_type": result.get("entity_type"),
                    "table": result.get("table"),
                    "column": result.get("column"),
                    "entity": result.get("entity")
                }
            
            if result.get("type") != "clarification":
                session_data["history"].append(user_query)
        
        # Save assistant response
        if result.get("type") == "response":
            chart_data = result.get("chart_data")
            Message.objects.create(
                conversation=conversation,
                content=result.get("message", ""),
                is_user=False,
                metadata={
                    "entities": result.get("entities", {}),
                    "chart": chart_data
                }
            )
            logger.info(f"💾 Saved assistant message to DB")
        
        # Update session entities
        if result.get("entities"):
            for e_type, e_data in result["entities"].items():
                if isinstance(e_data, dict):
                    session_data["entities"][e_type] = e_data.get("value", e_data)
                else:
                    session_data["entities"][e_type] = e_data
        
        # Update conversation
        conversation.entities = session_data["entities"]
        if not conversation.context:
            conversation.context = {}
        conversation.context.update({
            "pending_clarification": session_data.get("pending_clarification"),
            "last_query": session_data.get("last_query"),
            "module_id": session_data.get("module_id")
        })
        
        # Update title if still "New Chat"
        if conversation.title == "New Chat" and conversation.messages.filter(is_user=True).count() > 0:
            first_message = conversation.get_first_message()
            conversation.title = first_message[:50] + ("..." if len(first_message) > 50 else "")
        
        conversation.save()
        logger.info(f"💾 Updated conversation {conversation.id}")
        
        # Add conversation_id to response
        result["conversation_id"] = conversation.id
        
        return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"❌ Error in chat_api: {str(e)}", exc_info=True)
        return JsonResponse({
            "type": "error",
            "message": "I encountered an error processing your request. Please try again.",
            "conversation_id": conversation.id if conversation else None
        }, status=500)


@csrf_exempt
def get_conversations(request):
    """Get list of conversations for sidebar"""
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    module_id = request.GET.get('module_id', None)
    
    if not module_id:
        return JsonResponse({"conversations": []})
    
    try:
        conversations = Conversation.objects.filter(
            module_id=module_id
        ).order_by('-updated_at')[:50]
        
        conv_list = []
        for conv in conversations:
            conv_list.append({
                "id": conv.id,
                "title": conv.title,
                "updated_at": conv.updated_at.isoformat(),
                "message_count": conv.messages.count()
            })
        
        logger.info(f"📋 Returning {len(conv_list)} conversations for module {module_id}")
        return JsonResponse({"conversations": conv_list})
        
    except Exception as e:
        logger.error(f"❌ Error loading conversations: {e}", exc_info=True)
        return JsonResponse({"conversations": []})


@csrf_exempt
def load_conversation(request, conversation_id):
    """Load a specific conversation with all messages"""
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key
    
    try:
        conversation = Conversation.objects.get(id=conversation_id, session_id=session_id)
    except Conversation.DoesNotExist:
        return JsonResponse({"error": "Conversation not found"}, status=404)
    
    messages = []
    for msg in conversation.messages.all():
        messages.append({
            "content": msg.content,
            "is_user": msg.is_user,
            "timestamp": msg.timestamp.isoformat(),
            "metadata": msg.metadata
        })
    
    return JsonResponse({
        "conversation": {
            "id": conversation.id,
            "title": conversation.title,
            "entities": conversation.entities,
            "context": conversation.context
        },
        "messages": messages
    })


@csrf_exempt
def delete_conversation(request, conversation_id):
    """Delete a conversation"""
    if request.method != "DELETE":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if not request.session.session_key:
        return JsonResponse({"error": "No session"}, status=400)
    session_id = request.session.session_key
    
    try:
        conversation = Conversation.objects.get(id=conversation_id, session_id=session_id)
        conversation.delete()
        
        if conversation_id in SESSION_STORE:
            del SESSION_STORE[conversation_id]
        
        logger.info(f"🗑️ Deleted conversation {conversation_id}")
        return JsonResponse({"success": True})
    except Conversation.DoesNotExist:
        return JsonResponse({"error": "Conversation not found"}, status=404)