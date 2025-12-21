import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

import django
django.setup()

from pgadmin.RAG_LLM.main import invoke_graph

result = invoke_graph("sales of bhujia", 64)
print(result)