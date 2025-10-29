from django import forms
from .models import KnowledgeGraph,Metrics

class KnowledgeGraphForm(forms.ModelForm):
    class Meta:
        model = KnowledgeGraph
        fields = ['data']
        widgets = {
            'data': forms.Textarea(attrs={'rows': 20, 'cols': 100}),
        }


class Metrics(forms.ModelForm):
    class Meta:
        model = Metrics
        fields = ['data']
        widgets = {
            'data': forms.Textarea(attrs={'rows': 20, 'cols': 100}),
        }


class RCA(forms.ModelForm):
    class Meta:
        model = RCA
        fields = ['data']
        widgets = {
            'data': forms.Textarea(attrs={'rows': 20, 'cols': 100}),
        }

class extra_suggestion(forms.ModelForm):
    class Meta:
        model = RCA
        fields = ['data']
        widgets = {
            'data': forms.Textarea(attrs={'rows': 20, 'cols': 100}),
        }