import asyncio
from typing import Callable, Dict, List, Awaitable
from cloudverse.backend.models.events import CloudVerseEvent

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[CloudVerseEvent], Awaitable[None]]]] = {}
        self.all_subscribers: List[Callable[[CloudVerseEvent], Awaitable[None]]] = []

    def subscribe(self, event_type: str, callback: Callable[[CloudVerseEvent], Awaitable[None]]):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def subscribe_all(self, callback: Callable[[CloudVerseEvent], Awaitable[None]]):
        self.all_subscribers.append(callback)

    async def publish(self, event: CloudVerseEvent):
        callbacks = self.subscribers.get(event.event_type, [])
        tasks = []
        for cb in callbacks:
            tasks.append(cb(event))
        for cb in self.all_subscribers:
            tasks.append(cb(event))
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# Global singleton event bus
event_bus = EventBus()
