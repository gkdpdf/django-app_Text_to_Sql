from django.urls import path
from . import views

urlpatterns = [
    # ---------- AUTH & DASHBOARD ----------
    path('', views.login_view, name='login'),
    path('dashboard/', views.dashboard_view, name='dashboard'),

    # ---------- MODULE CREATION & EDIT ----------
    path('new/', views.new_module_view, name='new_module'),
    path('edit/<int:module_id>/', views.edit_module_view, name='edit_module'),


    # path('api/chat/', views.chat_api, name='chat_api'),
    # ---------- CHATBOT PAGE (New) ----------
    # path('chat/<int:module_id>/', views.module_chat_view, name='chat_module'),

    path('knowledge-graph/', views.knowledge_graph_view, name='knowledge_graph'),
    path("upload-knowledge-graph/", views.upload_knowledge_graph, name="upload_knowledge_graph"),
    path("download-knowledge-graph/", views.download_knowledge_graph_csv, name="download_knowledge_graph"),
#    path("<str:chat_id>/", views.chatbot_view, name="chat_view"),
 
    path("chat/", views.chat_view, name="chat_view"),
    path("api/chat/", views.chat_api, name="chat_api"),

]
