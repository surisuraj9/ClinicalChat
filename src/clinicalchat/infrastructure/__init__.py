"""Infrastructure layer: everything that talks to the outside world — SQLite, ChromaDB, OpenAI, Tavily, Langfuse,
the audit log. Services depend on these classes; nothing here knows about chats, agents or the UI."""
