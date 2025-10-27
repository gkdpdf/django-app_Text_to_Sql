from django.urls import path
from . import views

urlpatterns = [
    # ---------- AUTH & DASHBOARD ----------
    path('', views.login_view, name='login'),
    path('dashboard/', views.dashboard_view, name='dashboard'),

    # ---------- MODULE CREATION & EDIT ----------
    path('new/', views.new_module_view, name='new_module'),
    path('edit/<int:module_id>/', views.edit_module_view, name='edit_module'),

    # ---------- CHATBOT PAGE (New) ----------
    path('chat/<int:module_id>/', views.module_chat_view, name='chat_module'),
]
