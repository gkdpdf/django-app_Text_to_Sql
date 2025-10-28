from django import forms
from .models import KnowledgeGraph

class KnowledgeGraphForm(forms.ModelForm):
    class Meta:
        model = KnowledgeGraph
        fields = ['data']
        widgets = {
            'data': forms.Textarea(attrs={'rows': 20, 'cols': 100}),
        }