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
import logging
import json
import csv
import logging
import json
import csv
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.db import connection as django_connection
from .models import Module, Conversation, Message

import logging
import json
import csv
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.db import connection as django_connection
from .models import Module, Conversation, Message

logger = logging.getLogger(__name__)
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

        logger.info(f"📝 Creating module '{module_name}'")
        logger.info(f"   Tables: {selected_tables}")
        logger.info(f"   Columns: {selected_columns}")

        # Create module with selected tables AND columns
        module = Module.objects.create(
            user_name=user_name,
            name=module_name,
            tables=selected_tables,
            selected_columns=selected_columns,  # ← STORE COLUMNS!
        )
        
        logger.info(f"✅ Created module ID {module.id}")
        
        # Redirect to edit page (user will click "Update" there to generate KG)
        return redirect("edit_module", module_id=module.id)

    # GET request - show form
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


# ---------- EDIT MODULE (FIXED) ----------
def edit_module_view(request, module_id):
    """Edit module - PRESERVES existing KG data, only generates for new entries"""
    user_name = request.session.get("user_name")
    if not user_name:
        return redirect("login")

    module = get_object_or_404(Module, id=module_id, user_name=user_name)

    # Get tables
    tables = []
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
        tables = [row[0] for row in cursor.fetchall()]

    if request.method == "POST":
        action = request.POST.get("action")
        
        if action == "generate_kg":
            selected_tables = request.POST.getlist("selected_tables")
            selected_columns_json = request.POST.get("selected_columns", "{}")
            
            logger.info(f"🔍 Generate KG request for module {module_id}")
            logger.info(f"   Selected tables: {selected_tables}")
            
            try:
                selected_columns = json.loads(selected_columns_json)
            except Exception as e:
                logger.error(f"❌ Failed to parse selected_columns: {e}")
                return JsonResponse({
                    "success": False,
                    "error": "Invalid column selection data"
                })
            
            if not selected_tables:
                return JsonResponse({
                    "success": False,
                    "error": "No tables selected"
                })
            module.tables = selected_tables
            module.selected_columns = selected_columns
            module.save()
            # ========== KEY FIX: Only generate for NEW tables/columns ==========
            
            # Get existing KG data
            existing_kg = module.knowledge_graph_data if isinstance(module.knowledge_graph_data, dict) else {}
            
            # Determine which tables/columns need AI generation
            tables_to_generate = {}
            
            for table in selected_tables:
                # Get columns for this table from database
                with django_connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                    """, [table])
                    
                    all_cols = cursor.fetchall()
                    
                    # Filter to only selected columns
                    table_cols = []
                    for col_name, col_type in all_cols:
                        if selected_columns.get(table) and col_name in selected_columns[table]:
                            # Check if this column already has KG data
                            if table not in existing_kg or col_name not in existing_kg.get(table, {}):
                                # NEW column - needs AI generation
                                table_cols.append({
                                    "name": col_name,
                                    "type": col_type
                                })
                            else:
                                logger.info(f"   ✓ Skipping {table}.{col_name} (already has description)")
                    
                    if table_cols:
                        tables_to_generate[table] = table_cols
                        logger.info(f"   → Will generate KG for {table}: {len(table_cols)} new columns")
            
            # Only call AI if there are new tables/columns
            if tables_to_generate:
                logger.info(f"🤖 Generating KG for {len(tables_to_generate)} tables with new columns...")
                
                try:
                    from pgadmin.utils.kg_generator import generate_knowledge_graph
                    
                    # Generate only for new entries
                    new_kg_data = generate_knowledge_graph(tables_to_generate)
                    
                    # Merge with existing data (PRESERVE existing descriptions!)
                    merged_kg = existing_kg.copy()
                    
                    for table, columns in new_kg_data.items():
                        if table not in merged_kg:
                            merged_kg[table] = {}
                        
                        for col, col_data in columns.items():
                            # Only add if not already present
                            if col not in merged_kg[table]:
                                merged_kg[table][col] = col_data
                                logger.info(f"   ✅ Added KG for {table}.{col}")
                    
                    module.knowledge_graph_data = merged_kg
                    module.kg_auto_generated = True
                    module.save()
                    
                    logger.info(f"💾 Merged KG data: {len(tables_to_generate)} tables updated")
                    
                    return JsonResponse({
                        "success": True,
                        "message": f"Generated descriptions for {len(tables_to_generate)} new tables/columns!"
                    })
                    
                except ImportError as e:
                    logger.error(f"❌ Cannot import kg_generator: {e}")
                    return JsonResponse({
                        "success": False,
                        "error": "Knowledge graph generator not available"
                    })
                except Exception as e:
                    logger.error(f"❌ KG generation error: {e}")
                    import traceback
                    traceback.print_exc()
                    return JsonResponse({
                        "success": False,
                        "error": f"Error generating knowledge graph: {str(e)}"
                    })
            else:
                # No new columns - all already have descriptions
                logger.info("✅ All selected columns already have descriptions")
                return JsonResponse({
                    "success": True,
                    "message": "All selected columns already have descriptions. No AI generation needed."
                })
        
        elif action == "save_all":
            # Update table selections
            selected_tables = request.POST.getlist("selected_tables")
            selected_columns_json = request.POST.get("selected_columns", "{}")
            
            try:
                selected_columns = json.loads(selected_columns_json)
            except:
                selected_columns = {}
            
            module.tables = selected_tables if selected_tables else []
            module.selected_columns = selected_columns  # ← SAVE SELECTED COLUMNS
            
            # Get existing KG data
            existing_kg_data = module.knowledge_graph_data if isinstance(module.knowledge_graph_data, dict) else {}
            
            # Parse form data for knowledge graph
            kg_data_from_form = {}
            
            # Process table info
            for key, value in request.POST.items():
                if key.startswith("table_info__"):
                    table = key.replace("table_info__", "")
                    if table not in kg_data_from_form:
                        kg_data_from_form[table] = {}
                    kg_data_from_form[table]["TABLE_INFO"] = {
                        "desc": value.strip(),
                        "datatype": "meta"
                    }
            
            # Process column descriptions
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
            
            # Save Relationships
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

    # GET request - prepare data for template
    selected_tables = module.tables or []
    selected_columns = module.selected_columns if hasattr(module, 'selected_columns') else {}
    
    # Separate table info from column data
    knowledge_data = {}
    table_info_map = {}
    
    for table, columns in (module.knowledge_graph_data or {}).items():
        knowledge_data[table] = {}
        for col_name, col_data in columns.items():
            if col_name == "TABLE_INFO":
                table_info_map[table] = col_data.get("desc", "")
            else:
                knowledge_data[table][col_name] = col_data
    
    return render(request, "edit_module.html", {
        "module": module,
        "tables": tables,
        "selected_tables": selected_tables,
        "selected_columns": selected_columns,  # ← PASS TO TEMPLATE
        "knowledge_data": knowledge_data,
        "table_info_map": table_info_map,
        "relationships": module.relationships or [],
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
def get_table_columns_view(request):
    """Get columns for a specific table"""
    table_name = request.GET.get('table')
    
    if not table_name:
        return JsonResponse({"error": "No table specified"}, status=400)
    
    try:
        with django_connection.cursor() as cursor:
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, [table_name])
            
            columns = [
                {"name": row[0], "type": row[1]}
                for row in cursor.fetchall()
            ]
        
        logger.info(f"✅ Loaded {len(columns)} columns for table '{table_name}'")
        return JsonResponse({"columns": columns})
        
    except Exception as e:
        logger.error(f"❌ Error getting columns for {table_name}: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def download_module_kg_csv(request, module_id):
    """Download knowledge graph as CSV for specific module - WITH TABLE INFO"""
    user_name = request.session.get("user_name")
    if not user_name:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    
    module = get_object_or_404(Module, id=module_id, user_name=user_name)
    
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{module.name}_knowledge_graph.csv"'

    writer = csv.writer(response)
    writer.writerow(["table", "table_info", "column", "desc", "datatype"])

    for table, columns in module.knowledge_graph_data.items():
        # Get table info from _table_info key
        table_info = ""
        if "_table_info" in columns:
            table_info = columns["_table_info"].get("desc", "")
        
        first_column = True
        for column, info in columns.items():
            if column == "_table_info":
                continue  # Skip the meta field
            
            writer.writerow([
                table,
                table_info if first_column else "",  # Only write table info once per table
                column,
                info.get("desc", ""),
                info.get("datatype", "")
            ])
            first_column = False

    return response


@csrf_exempt
@csrf_exempt
def upload_module_kg_csv(request, module_id):
    """Upload knowledge graph CSV for specific module - WITH TABLE INFO"""
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
            table_info = row.get('table_info', '').strip()
            column = row.get('column', '').strip()
            desc = row.get('desc', '').strip()
            datatype = row.get('datatype', '').strip()

            if table:
                if table not in data:
                    data[table] = {}
                
                # Store table info if present
                if table_info and "_table_info" not in data[table]:
                    data[table]["_table_info"] = {
                        "desc": table_info,
                        "datatype": "meta"
                    }
                
                # Store column data
                if column:
                    data[table][column] = {
                        'desc': desc,
                        'datatype': datatype
                    }
        
        module.knowledge_graph_data = data
        module.save()
        
        logger.info(f"✅ Uploaded KG CSV for module '{module.name}' with table info")

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
@csrf_exempt
def chat_api(request):
    """API endpoint for chat interactions"""
    if request.method != 'POST':
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        module_id = data.get('module_id')
        feedback = data.get('feedback')
        
        logger.info(f"📨 Chat API called: message='{user_message}', conv_id={conversation_id}, module_id={module_id}")
        
        if not module_id:
            return JsonResponse({"error": "module_id required"}, status=400)
        
        # Get or create conversation
        if conversation_id:
            try:
                conversation = Conversation.objects.get(id=conversation_id)
                logger.info(f"📂 Using existing conversation {conversation_id}")
            except Conversation.DoesNotExist:
                logger.warning(f"⚠️ Conversation {conversation_id} not found, creating new one")
                conversation = None
        else:
            conversation = None
        
        # Get module
        user_name = request.session.get("user_name")
        if not user_name:
            return JsonResponse({"error": "Not authenticated"}, status=401)
        
        module = get_object_or_404(Module, id=module_id, user_name=user_name)
        
        # Create conversation if needed
        if not conversation:
            conversation = Conversation.objects.create(
                module=module,
                session_id=f"conv_{Conversation.objects.filter(module=module).count() + 1}",
                title=user_message[:50] if user_message else "New Chat"
            )
            logger.info(f"✨ Created new conversation {conversation.id}")
        
        # Get session data from conversation
        session_data = conversation.context or {
            "entities": {},
            "history": []
        }
        
        # If feedback provided, it's a clarification response
        if feedback:
            logger.info(f"🔄 Processing feedback: {feedback}")
            # Don't save user message for feedback responses
        else:
            # Save user message
            if user_message:
                Message.objects.create(
                    conversation=conversation,
                    content=user_message,
                    is_user=True
                )
                logger.info("💾 Saved user message to DB")
        
        # Call LangGraph - CORRECT PARAMETER ORDER
        from pgadmin.RAG_LLM.main import invoke_graph
        
        result = invoke_graph(
            user_query=user_message,
            module_id=module_id,  # ← Pass module_id as integer
            session_data=session_data,
            feedback=feedback
        )
        
        # Update session data
        updated_session_data = result.get("session_data", session_data)
        conversation.context = updated_session_data
        
        # Update conversation title if it's the first message
        if conversation.messages.count() == 1 and user_message:
            conversation.title = user_message[:50]
        
        conversation.save()
        logger.info(f"💾 Updated conversation {conversation.id}")
        
        # Prepare response based on result type
        response_data = {
            "conversation_id": conversation.id,
            "type": result.get("type", "response")
        }
        
        if result["type"] == "clarification":
            # Clarification needed
            response_data.update({
                "message": result.get("message"),
                "options": result.get("options", []),
                "subtype": result.get("subtype"),
                "entity": result.get("entity"),
                "entity_type": result.get("entity_type"),
                "table": result.get("table"),
                "column": result.get("column")
            })
            
        elif result["type"] == "response":
            # Successful response
            assistant_message = result.get("message", "Query completed successfully.")
            
            # Save assistant message
            Message.objects.create(
                conversation=conversation,
                content=assistant_message,
                is_user=False,
                metadata=result.get("metadata", {})
            )
            logger.info("💾 Saved assistant message to DB")
            
            response_data.update({
                "message": assistant_message,
                "metadata": result.get("metadata", {})
            })
            
        elif result["type"] == "error":
            # Error response
            error_message = result.get("message", "An error occurred")
            
            # Save error message
            Message.objects.create(
                conversation=conversation,
                content=error_message,
                is_user=False
            )
            
            response_data.update({
                "message": error_message
            })
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        logger.error("❌ Invalid JSON in request")
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    except Exception as e:
        logger.error(f"❌ Error in chat_api: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            "type": "error",
            "message": f"Server error: {str(e)}"
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