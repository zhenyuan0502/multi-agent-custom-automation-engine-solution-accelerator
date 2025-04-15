"""In-memory implementation of the CosmosMemoryContext for local development."""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Type, Tuple
import numpy as np

from semantic_kernel.memory.memory_record import MemoryRecord
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.contents import ChatMessageContent, ChatHistory, AuthorRole

from models.messages_kernel import BaseDataModel, Plan, Session, Step, AgentMessage


class InMemoryContext(MemoryStoreBase):
    """An in-memory implementation of the memory context for local development."""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        buffer_size: int = 100,
        initial_messages: Optional[List[ChatMessageContent]] = None,
    ) -> None:
        self._buffer_size = buffer_size
        self._messages = initial_messages or []
        self.session_id = session_id
        self.user_id = user_id
        
        # Storage for different data types
        self._storage = {
            "session": {},
            "plan": {},
            "step": {},
            "agent_message": {},
            "message": {},
            "memory": {},
        }
        
        # Collections for memory storage
        self._collections = {}
        
        self._initialized = asyncio.Event()
        self._initialized.set()  # Already initialized

    async def add_item(self, item: BaseDataModel) -> None:
        """Add a data model item to storage."""
        try:
            document = item.model_dump()
            item_type = document.get("data_type", "unknown")
            if item_type in self._storage:
                self._storage[item_type][document["id"]] = document
                logging.info(f"Item added to in-memory storage - {document['id']}")
        except Exception as e:
            logging.exception(f"Failed to add item to in-memory storage: {e}")

    async def update_item(self, item: BaseDataModel) -> None:
        """Update an existing item in storage."""
        try:
            document = item.model_dump()
            item_type = document.get("data_type", "unknown")
            if item_type in self._storage:
                self._storage[item_type][document["id"]] = document
        except Exception as e:
            logging.exception(f"Failed to update item in in-memory storage: {e}")

    async def get_item_by_id(
        self, item_id: str, partition_key: str, model_class: Type[BaseDataModel]
    ) -> Optional[BaseDataModel]:
        """Retrieve an item by its ID."""
        try:
            for storage_type, items in self._storage.items():
                if item_id in items:
                    return model_class.model_validate(items[item_id])
            return None
        except Exception as e:
            logging.exception(f"Failed to retrieve item from in-memory storage: {e}")
            return None

    async def query_items(
        self,
        query: str,
        parameters: List[Dict[str, Any]],
        model_class: Type[BaseDataModel],
    ) -> List[BaseDataModel]:
        """Query items from storage based on parameters."""
        try:
            # Extract parameters from the query
            data_type = None
            session_id = None
            id_filter = None
            plan_id = None
            
            for param in parameters:
                if param["name"] == "@data_type":
                    data_type = param["value"]
                elif param["name"] == "@session_id":
                    session_id = param["value"]
                elif param["name"] == "@id":
                    id_filter = param["value"]
                elif param["name"] == "@plan_id":
                    plan_id = param["value"]
            
            results = []
            
            # Basic filtering based on parameters
            if data_type and data_type in self._storage:
                for item_id, item in self._storage[data_type].items():
                    match = True
                    
                    if session_id is not None and item.get("session_id") != session_id:
                        match = False
                    if id_filter is not None and item.get("id") != id_filter:
                        match = False
                    if plan_id is not None and item.get("plan_id") != plan_id:
                        match = False
                    
                    if match:
                        item["ts"] = item.get("_ts", 0)  # Ensure ts field exists
                        results.append(model_class.model_validate(item))
                        
            return results
        except Exception as e:
            logging.exception(f"Failed to query items from in-memory storage: {e}")
            return []

    # Session methods
    
    async def add_session(self, session: Session) -> None:
        """Add a session."""
        await self.add_item(session)

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by session_id."""
        for item in self._storage["session"].values():
            if item.get("id") == session_id:
                return Session.model_validate(item)
        return None

    async def get_all_sessions(self) -> List[Session]:
        """Retrieve all sessions."""
        return [Session.model_validate(item) for item in self._storage["session"].values()]

    # Plan methods
    
    async def add_plan(self, plan: Plan) -> None:
        """Add a plan."""
        await self.add_item(plan)

    async def update_plan(self, plan: Plan) -> None:
        """Update a plan."""
        await self.update_item(plan)

    async def get_plan_by_session(self, session_id: str) -> Optional[Plan]:
        """Retrieve a plan by session."""
        for item in self._storage["plan"].values():
            if item.get("session_id") == session_id and item.get("user_id") == self.user_id:
                return Plan.model_validate(item)
        return None

    async def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Retrieve a plan by ID."""
        if plan_id in self._storage["plan"]:
            return Plan.model_validate(self._storage["plan"][plan_id])
        return None

    async def get_all_plans(self) -> List[Plan]:
        """Retrieve all plans."""
        return [Plan.model_validate(item) for item in self._storage["plan"].values() 
                if item.get("user_id") == self.user_id]

    # Step methods
    
    async def add_step(self, step: Step) -> None:
        """Add a step."""
        await self.add_item(step)

    async def update_step(self, step: Step) -> None:
        """Update a step."""
        await self.update_item(step)

    async def get_steps_for_plan(self, plan_id: str, session_id: Optional[str] = None) -> List[Step]:
        """Retrieve steps for a plan."""
        return [Step.model_validate(item) for item in self._storage["step"].values() 
                if item.get("plan_id") == plan_id and item.get("user_id") == self.user_id]

    async def get_step(self, step_id: str, session_id: str) -> Optional[Step]:
        """Retrieve a step by ID."""
        for item in self._storage["step"].values():
            if item.get("id") == step_id and item.get("session_id") == session_id:
                return Step.model_validate(item)
        return None

    # Agent message methods
    
    async def add_agent_message(self, message: AgentMessage) -> None:
        """Add an agent message."""
        await self.add_item(message)

    async def get_agent_messages_by_session(self, session_id: str) -> List[AgentMessage]:
        """Retrieve agent messages for a session."""
        return [AgentMessage.model_validate(item) for item in self._storage["agent_message"].values() 
                if item.get("session_id") == session_id]

    # Message methods
    
    async def add_message(self, message: ChatMessageContent) -> None:
        """Add a chat message."""
        self._messages.append(message)
        # Ensure buffer size is maintained
        while len(self._messages) > self._buffer_size:
            self._messages.pop(0)
            
        message_dict = {
            "id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "data_type": "message",
            "content": {
                "role": message.role.value,
                "content": message.content,
                "metadata": message.metadata
            },
            "source": message.metadata.get("source", ""),
            "_ts": 0,
        }
        self._storage["message"][message_dict["id"]] = message_dict

    async def get_messages(self) -> List[ChatMessageContent]:
        """Get messages for the session."""
        messages = []
        for item in self._storage["message"].values():
            if item.get("session_id") == self.session_id:
                content = item.get("content", {})
                role = content.get("role", "user")
                chat_role = AuthorRole.ASSISTANT
                if role == "user":
                    chat_role = AuthorRole.USER
                elif role == "system":
                    chat_role = AuthorRole.SYSTEM
                elif role == "tool":
                    chat_role = AuthorRole.TOOL
                
                message = ChatMessageContent(
                    role=chat_role,
                    content=content.get("content", ""),
                    metadata=content.get("metadata", {})
                )
                messages.append(message)
        return messages

    # Chat history methods
    
    def get_chat_history(self) -> ChatHistory:
        """Get chat history."""
        history = ChatHistory()
        for message in self._messages:
            history.add_message(message)
        return history
    
    async def save_chat_history(self, history: ChatHistory) -> None:
        """Save chat history."""
        for message in history.messages:
            await self.add_message(message)
    
    # Memory store methods
    
    async def upsert_memory_record(self, collection: str, record: MemoryRecord) -> str:
        """Store a memory record."""
        if collection not in self._collections:
            self._collections[collection] = {}
            
        record_id = record.id or str(uuid.uuid4())
        self._collections[collection][record.key] = record._replace(id=record_id)
        return record_id
    
    async def get_memory_record(self, collection: str, key: str, with_embedding: bool = False) -> Optional[MemoryRecord]:
        """Get a memory record."""
        if collection in self._collections and key in self._collections[collection]:
            record = self._collections[collection][key]
            if not with_embedding:
                return record._replace(embedding=None)
            return record
        return None
    
    async def remove_memory_record(self, collection: str, key: str) -> None:
        """Remove a memory record."""
        if collection in self._collections and key in self._collections[collection]:
            del self._collections[collection][key]

    # Implementation of abstract methods from MemoryStoreBase
    
    async def create_collection(self, collection_name: str) -> None:
        """Create a collection."""
        if collection_name not in self._collections:
            self._collections[collection_name] = {}

    async def get_collections(self) -> List[str]:
        """Get collection names."""
        return list(self._collections.keys())

    async def does_collection_exist(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return collection_name in self._collections

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        if collection_name in self._collections:
            del self._collections[collection_name]

    async def upsert(self, collection_name: str, record: MemoryRecord) -> str:
        """Upsert a memory record."""
        return await self.upsert_memory_record(collection_name, record)

    async def upsert_batch(self, collection_name: str, records: List[MemoryRecord]) -> List[str]:
        """Upsert multiple memory records."""
        results = []
        for record in records:
            record_id = await self.upsert_memory_record(collection_name, record)
            results.append(record_id)
        return results

    async def get(self, collection_name: str, key: str, with_embedding: bool = False) -> Optional[MemoryRecord]:
        """Get a memory record."""
        return await self.get_memory_record(collection_name, key, with_embedding)

    async def get_batch(self, collection_name: str, keys: List[str], with_embeddings: bool = False) -> List[MemoryRecord]:
        """Get multiple memory records."""
        results = []
        for key in keys:
            record = await self.get_memory_record(collection_name, key, with_embeddings)
            if record:
                results.append(record)
        return results

    async def remove(self, collection_name: str, key: str) -> None:
        """Remove a memory record."""
        await self.remove_memory_record(collection_name, key)

    async def remove_batch(self, collection_name: str, keys: List[str]) -> None:
        """Remove multiple memory records."""
        for key in keys:
            await self.remove_memory_record(collection_name, key)

    async def get_nearest_match(
        self, 
        collection_name: str, 
        embedding: np.ndarray, 
        limit: int = 1, 
        min_relevance_score: float = 0.0, 
        with_embeddings: bool = False
    ) -> Tuple[MemoryRecord, float]:
        """Get the nearest match to an embedding."""
        matches = await self.get_nearest_matches(
            collection_name, embedding, limit, min_relevance_score, with_embeddings
        )
        return matches[0] if matches else (None, 0.0)

    async def get_nearest_matches(
        self, 
        collection_name: str, 
        embedding: np.ndarray, 
        limit: int = 1, 
        min_relevance_score: float = 0.0, 
        with_embeddings: bool = False
    ) -> List[Tuple[MemoryRecord, float]]:
        """Get the nearest matches to an embedding."""
        if collection_name not in self._collections:
            return []
            
        results = []
        for record in self._collections[collection_name].values():
            if record.embedding is not None:
                # Compute cosine similarity
                similarity = np.dot(embedding, record.embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(record.embedding)
                )
                
                if similarity >= min_relevance_score:
                    if not with_embeddings:
                        record = record._replace(embedding=None)
                    results.append((record, float(similarity)))
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def get_memory_records(
        self, collection: str, limit: int = 1000, with_embeddings: bool = False
    ) -> List[MemoryRecord]:
        """Get all memory records from a collection."""
        if collection not in self._collections:
            return []
            
        records = list(self._collections[collection].values())
        if not with_embeddings:
            records = [record._replace(embedding=None) for record in records]
        return records[:limit]

    # Utility methods
    
    async def delete_item(self, item_id: str, partition_key: str) -> None:
        """Delete an item."""
        for storage_type, items in self._storage.items():
            if item_id in items:
                del items[item_id]
                break

    async def get_all_items(self) -> List[Dict[str, Any]]:
        """Get all items."""
        all_items = []
        for storage_type, items in self._storage.items():
            all_items.extend(items.values())
        return all_items[:100]  # Limit to 100 items

    async def close(self) -> None:
        """Close resources."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()