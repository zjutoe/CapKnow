"""Minimal knowledge space lab package for phase 1."""

from .knowledge_space.tasks import Task, TaskUniverse
from .knowledge_space.state import KnowledgeState
from .knowledge_space.space import KnowledgeSpace

__all__ = ["Task", "TaskUniverse", "KnowledgeState", "KnowledgeSpace"]
