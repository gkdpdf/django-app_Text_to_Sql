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


from django.db import models
from django.contrib.auth.models import User
import json

class Conversation(models.Model):
    """Represents a chat conversation"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=255, db_index=True)
    title = models.CharField(max_length=255, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Store entities and context as JSON
    entities = models.JSONField(default=dict, blank=True)
    context = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['session_id', '-updated_at']),
        ]
    
    def __str__(self):
        return f"{self.title} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
    
    def get_first_message(self):
        """Get first user message for title generation"""
        first_msg = self.messages.filter(is_user=True).first()
        return first_msg.content if first_msg else "New Chat"


class Message(models.Model):
    """Represents a single message in a conversation"""
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    is_user = models.BooleanField(default=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Store message metadata
    metadata = models.JSONField(default=dict, blank=True)  # For clarifications, entities, etc.
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        sender = "User" if self.is_user else "AI"
        return f"{sender}: {self.content[:50]}..."