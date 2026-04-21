"""
Memory schemas and interfaces.

Dataclasses for conversations, facts, episodes, procedures, and reminder intents.
Protocol definitions for stores, embedder, and clock contracts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict


@dataclass
class ConversationTurn:
    """
    A single turn in a conversation (user or assistant message).
    
    Stored in short-term memory (Supabase ``st_turns`` table) as a ring buffer.
    """
    user_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    ts: float  # epoch seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "ts": self.ts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            ts=data["ts"],
        )


@dataclass
class MemoryFact:
    """
    A distilled long-term memory fact extracted from conversations.
    
    Stored in Supabase Postgres (metadata + pgvector embeddings).
    """
    id: str
    user_id: str
    text: str
    score: float  # 0.0-1.0, composite of recency + repetition + explicitness
    tags: List[str] = field(default_factory=list)
    created_at: float = 0.0  # epoch seconds
    last_used_at: float = 0.0  # epoch seconds
    ttl_at: Optional[float] = None  # epoch seconds, None = no expiry
    pin: bool = False  # if True, do not shift reminders on collisions
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "text": self.text,
            "score": self.score,
            "tags": self.tags,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "ttl_at": self.ttl_at,
            "pin": self.pin,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryFact":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            text=data["text"],
            score=data["score"],
            tags=data.get("tags", []),
            created_at=data.get("created_at", 0.0),
            last_used_at=data.get("last_used_at", 0.0),
            ttl_at=data.get("ttl_at"),
            pin=data.get("pin", False),
        )