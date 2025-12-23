from django.urls import path
from . import views

urlpatterns = [
    # Home
    path('', views.home_view, name='home'),
    
    # Authentication
    path('login/', views.login_view, name='login'),
    
    # Dashboard and Modules
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('new-module/', views.new_module_view, name='new_module'),
    path('edit-module/<int:module_id>/', views.edit_module_view, name='edit_module'),
    path('delete-module/<int:module_id>/', views.delete_module_view, name='delete_module'),
    
    # Utility
    #  path('get-table-columns/', views.get_table_columns, name='get_table_columns'),
    
    # Module-specific Knowledge Graph
    path('download-module-kg/<int:module_id>/', views.download_module_kg_csv, name='download_module_kg_csv'),
    path('upload-module-kg/<int:module_id>/', views.upload_module_kg_csv, name='upload_module_kg_csv'),
    
    # Global Knowledge Graph (Legacy)
    path('knowledge-graph/', views.knowledge_graph_view, name='knowledge_graph'),
    path('download-kg-csv/', views.download_knowledge_graph_csv, name='download_knowledge_graph'),
    path('upload-kg/', views.upload_knowledge_graph, name='upload_knowledge_graph'),
    
    # Chat Interface
    path('chat/<int:module_id>/', views.chat_view, name='chat_module'),
    path('chat/api/', views.chat_api, name='chat_api'),
    
    # Conversation Management
    path('conversations/', views.get_conversations, name='get_conversations'),
    path('conversation/<int:conversation_id>/', views.load_conversation, name='load_conversation'),
    path('conversation/<int:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
     path('download-module-kg-csv/<int:module_id>/', views.download_module_kg_csv, name='download_module_kg_csv'),
    path('upload-module-kg-csv/<int:module_id>/', views.upload_module_kg_csv, name='upload_module_kg_csv'),
    path('get-table-columns/', views.get_table_columns_view, name='get_table_columns'),
    #path('conversation/<int:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
    # path('chat/api/stream/', views.chat_api_stream, name='chat_api_stream'),


    path('knowledge-graph/', views.knowledge_graph_view, name='knowledge_graph'),
    path('knowledge-graph/download/', views.download_knowledge_graph_csv, name='download_knowledge_graph'),
    path('knowledge-graph/upload/', views.upload_knowledge_graph, name='upload_knowledge_graph'),


# Dashboard Builder (NEW)
    path('dashboard-builder/<int:module_id>/', views.dashboard_builder_view, name='dashboard_builder'),
    path('dashboard/api/generate/', views.dashboard_generate_api, name='dashboard_generate'),
    path('dashboard/api/save/', views.dashboard_save_api, name='dashboard_save'),
    path('dashboard/api/list/', views.dashboard_list_api, name='dashboard_list'),
    path('dashboard/api/get/<int:dashboard_id>/', views.dashboard_get_api, name='dashboard_get'),
    path('dashboard/api/delete/<int:dashboard_id>/', views.dashboard_delete_api, name='dashboard_delete'),
    path('dashboard/api/filter-values/', views.dashboard_filter_values_api, name='dashboard_filter_values'),
]