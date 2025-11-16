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
]