from django.db import models
from django.contrib.postgres.fields import ArrayField

class Module(models.Model):
    user_name = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    tables = ArrayField(models.CharField(max_length=255), blank=True, null=True)
    knowledge_graph_data = models.JSONField(default=dict, blank=True, null=True)
    metrics_data = models.JSONField(default=dict, blank=True, null=True)
    rca_list = models.JSONField(default=list, blank=True, null=True)
    pos_tagging = models.JSONField(default=list, blank=True, null=True)
    extra_suggestions = models.TextField(blank=True, null=True)
    kg_auto_generated = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.user_name})"

    class Meta:
        db_table = 'pgadmin_module'


class Conversation(models.Model):
    module = models.ForeignKey(Module, on_delete=models.CASCADE, related_name='conversations', null=True, blank=True)
    session_id = models.CharField(max_length=255, db_index=True)
    title = models.CharField(max_length=255, default="New Chat")
    entities = models.JSONField(default=dict, blank=True, null=True)
    context = models.JSONField(default=dict, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_first_message(self):
        first_msg = self.messages.filter(is_user=True).first()
        return first_msg.content if first_msg else "New Chat"

    def __str__(self):
        return f"{self.title} ({self.created_at.strftime('%Y-%m-%d')})"

    class Meta:
        db_table = 'pgadmin_conversation'
        ordering = ['-updated_at']


class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    is_user = models.BooleanField(default=True)
    metadata = models.JSONField(default=dict, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        role = "User" if self.is_user else "Assistant"
        return f"{role}: {self.content[:50]}"

    class Meta:
        db_table = 'pgadmin_message'
        ordering = ['timestamp']


class KnowledgeGraph(models.Model):
    data = models.JSONField(default=dict)
    class Meta:
        db_table = 'pgadmin_knowledgegraph'


class Metrics(models.Model):
    data = models.JSONField(default=dict)
    class Meta:
        db_table = 'pgadmin_metrics'


class RCA(models.Model):
    data = models.JSONField(default=dict)
    class Meta:
        db_table = 'pgadmin_rca'


class Extra_suggestion(models.Model):
    data = models.JSONField(default=dict)
    class Meta:
        db_table = 'pgadmin_extra_suggestion'