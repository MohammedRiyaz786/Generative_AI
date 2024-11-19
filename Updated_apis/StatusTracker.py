from typing import Dict, Any, Optional,List
import time
from dataclasses import dataclass
from enum import Enum
import asyncio
from threading import Lock
@dataclass
class StatusUpdate:
    message: str
    completed: int
    total: int
    details: Optional[Dict[str, Any]]
    timestamp: float

    def __init__(self, message: str, completed: int, total: int = 8, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.completed = completed
        self.total = total
        self.details = details
        self.timestamp = time.time()

class StatusTracker:
    def __init__(self):
        self._status_history: Dict[str, List[StatusUpdate]] = {}
        self._lock = Lock()
        self._processed_documents = set()
    
    def add_status(self, document_key: str, message: str, completed: int, 
                  total: int = 8, details: Optional[Dict[str, Any]] = None):
        with self._lock:
            if document_key not in self._status_history:
                self._status_history[document_key] = []
            
            status = StatusUpdate(
                message=message,
                completed=completed,
                total=total,
                details=details
            )
            self._status_history[document_key].append(status)
    
    def get_status_history(self, document_key: str) -> Optional[List[StatusUpdate]]:
        with self._lock:
            return self._status_history.get(document_key)
    
    def is_processed(self, document_key: str) -> bool:
        return document_key in self._processed_documents
    
    def mark_as_processed(self, document_key: str):
        self._processed_documents.add(document_key)
        # Clean up history after marking as processed
        with self._lock:
            if document_key in self._status_history:
                del self._status_history[document_key]

# Create global status tracker 
status_tracker = StatusTracker()