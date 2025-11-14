from django.db import models
from django.utils import timezone
from django.core.validators import MinLengthValidator


class Module(models.Model):
    """Main module for organizing database schemas and knowledge graphs"""
    id = models.BigAutoField(primary_key=True)
    user_name = models.CharField(max_length=255, db_index=True, validators=[MinLengthValidator(1)])
    name = models.CharField(max_length=255, validators=[MinLengthValidator(1)])
    tables = models.JSONField(default=list, blank=True)
    knowledge_graph_data = models.JSONField(default=dict, blank=True)
    rca_list = models.JSONField(default=list, blank=True)
    pos_tagging = models.JSONField(default=list, blank=True)
    metrics_data = models.JSONField(default=dict, blank=True)
    extra_suggestions = models.TextField(blank=True, default='')
    kg_auto_generated = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'pgadmin_module'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user_name', '-created_at']),
        ]

    def __str__(self):
        return f"{self.name} ({self.user_name})"

    def save(self, *args, **kwargs):
        """Ensure JSON fields are always proper types"""
        if not isinstance(self.tables, list):
            self.tables = []
        if not isinstance(self.knowledge_graph_data, dict):
            self.knowledge_graph_data = {}
        if not isinstance(self.rca_list, list):
            self.rca_list = []
        if not isinstance(self.pos_tagging, list):
            self.pos_tagging = []
        if not isinstance(self.metrics_data, dict):
            self.metrics_data = {}
        super().save(*args, **kwargs)


class Conversation(models.Model):
    """Chat conversation associated with a module"""
    id = models.BigAutoField(primary_key=True)
    module = models.ForeignKey(Module, on_delete=models.CASCADE, related_name='conversations')
    session_id = models.CharField(max_length=255, db_index=True)
    title = models.CharField(max_length=255, default='New Chat')
    entities = models.JSONField(default=dict, blank=True)
    context = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'pgadmin_conversation'
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['session_id', '-updated_at']),
            models.Index(fields=['module', '-updated_at']),
        ]

    def __str__(self):
        return f"Conversation: {self.title}"

    def get_first_message(self):
        """Get the first user message content"""
        first_msg = self.messages.filter(is_user=True).first()
        return first_msg.content if first_msg else "New Chat"

    def save(self, *args, **kwargs):
        """Ensure JSON fields are proper types"""
        if not isinstance(self.entities, dict):
            self.entities = {}
        if not isinstance(self.context, dict):
            self.context = {}
        super().save(*args, **kwargs)


class Message(models.Model):
    """Individual messages within a conversation"""
    id = models.BigAutoField(primary_key=True)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    is_user = models.BooleanField(default=True)
    metadata = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'pgadmin_message'
        ordering = ['timestamp']
        indexes = [
            models.Index(fields=['conversation', 'timestamp']),
        ]

    def __str__(self):
        msg_type = "User" if self.is_user else "Assistant"
        return f"{msg_type}: {self.content[:50]}"

    def save(self, *args, **kwargs):
        """Ensure metadata is a dict"""
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        super().save(*args, **kwargs)


# ============================================================================
# LEGACY MODELS (for backward compatibility)
# ============================================================================

class KnowledgeGraph(models.Model):
    """Legacy global knowledge graph"""
    id = models.BigAutoField(primary_key=True)
    data = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = 'pgadmin_knowledgegraph'

    def __str__(self):
        return f"Knowledge Graph (ID: {self.id})"


class Metrics(models.Model):
    """Legacy global metrics"""
    id = models.BigAutoField(primary_key=True)
    data = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = 'pgadmin_metrics'

    def __str__(self):
        return f"Metrics (ID: {self.id})"


class RCA(models.Model):
    """Legacy global RCA"""
    id = models.BigAutoField(primary_key=True)
    data = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = 'pgadmin_rca'

    def __str__(self):
        return f"RCA (ID: {self.id})"


class Extra_suggestion(models.Model):
    """Legacy global suggestions"""
    id = models.BigAutoField(primary_key=True)
    data = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = 'pgadmin_extra_suggestion'

    def __str__(self):
        return f"Extra Suggestion (ID: {self.id})"