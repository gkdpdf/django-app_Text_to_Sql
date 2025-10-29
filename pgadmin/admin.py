from django.contrib import admin
from .models import KnowledgeGraph,Metrics,RCA,Extra_suggestion

# Register your models here.
admin.site.register(KnowledgeGraph)
admin.site.register(Metrics)
admin.site.register(RCA)
admin.site.register(Extra_suggestion)