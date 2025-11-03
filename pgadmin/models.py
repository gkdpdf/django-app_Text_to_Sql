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

from django.db import models


class KnowledgeGraph(models.Model):
    data = models.JSONField(default=dict)  # stores all table-column-desc data in one object

    def __str__(self):
        return "Knowledge Graph Data"


class Metrics(models.Model):
    data = models.JSONField(default=dict)  # stores all table-column-desc data in one object

    def __str__(self):
        return "Metrics Data"


class RCA(models.Model):
    data = models.JSONField(default=dict)  # stores all table-column-desc data in one object

    def __str__(self):
        return "RCA Data"

class Extra_suggestion(models.Model):
    data = models.JSONField(default=dict)  # stores all table-column-desc data in one object

    def __str__(self):
        return "Extra Suggestion Data"

from django.db import models

class ChatSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title}"


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=[("user", "User"), ("bot", "Bot")])
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.role}: {self.content[:30]}"
