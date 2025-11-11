from django.db import models
from django.contrib.postgres.fields import ArrayField

class Module(models.Model):
    user_name = models.CharField(max_length=100)
    name = models.CharField(max_length=200)
    tables = ArrayField(models.CharField(max_length=200), blank=True, default=list)
    
    # Module-specific data
    knowledge_graph_data = models.JSONField(default=dict, blank=True)
    rca_list = models.JSONField(default=list, blank=True)
    pos_tagging = models.JSONField(default=list, blank=True)
    metrics_data = models.JSONField(default=dict, blank=True)
    extra_suggestions = models.TextField(blank=True, default="")
    
    # Flags
    kg_auto_generated = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.user_name}"

    class Meta:
        ordering = ['-created_at']


class KnowledgeGraph(models.Model):
    data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)


class Metrics(models.Model):
    data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)


class RCA(models.Model):
    data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)


class Extra_suggestion(models.Model):
    data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)


class Conversation(models.Model):
    session_id = models.CharField(max_length=100)
    module = models.ForeignKey(Module, on_delete=models.CASCADE, null=True, blank=True, related_name='conversations')
    title = models.CharField(max_length=200, default="New Chat")
    entities = models.JSONField(default=dict, blank=True)
    context = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_first_message(self):
        first_msg = self.messages.filter(is_user=True).first()
        return first_msg.content if first_msg else "New Chat"

    class Meta:
        ordering = ['-updated_at']


class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    is_user = models.BooleanField(default=True)
    metadata = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']