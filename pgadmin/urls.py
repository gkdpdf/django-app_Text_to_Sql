from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('new/', views.new_module_view, name='new_module'),
    path('edit/<int:module_id>/', views.edit_module_view, name='edit_module'),
]
