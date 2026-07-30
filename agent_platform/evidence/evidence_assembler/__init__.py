"""证据组装器模块"""

from .builder import ClaimSlot, EvidenceBuilder, EvidenceBundle, EvidenceItem
from .deduplicator import Deduplicator
from .parent_aggregator import ParentAggregator

__all__ = [
    "EvidenceBuilder",
    "EvidenceBundle",
    "EvidenceItem",
    "ClaimSlot",
    "Deduplicator",
    "ParentAggregator",
]
