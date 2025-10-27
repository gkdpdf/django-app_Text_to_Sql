from django.db import models

class Module(models.Model):
    user_name = models.CharField(max_length=50)
    name = models.CharField(max_length=100)
    tables = models.JSONField()
    knowledge_graph = models.FileField(upload_to="knowledge_graphs/", null=True, blank=True)
    metrics = models.FileField(upload_to="metrics/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.user_name})"
