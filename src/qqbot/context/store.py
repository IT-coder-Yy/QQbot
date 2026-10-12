from qqbot.schema import (
    AgentRun,
    Conversation,
    Session,
)


class InMemoryContextStore:
    def __init__(self) -> None:
        self.conversations: dict[
            str,
            Conversation,
        ] = {}

        self.sessions: dict[
            str,
            Session,
        ] = {}

        self.runs: dict[
            str,
            AgentRun,
        ] = {}

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self.conversations.get(conversation_id)

    def save_conversation(self, conversation: Conversation) -> None:
        self.conversations[conversation.conversation_id] = conversation

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def save_session(self, session: Session) -> None:
        self.sessions[session.session_id] = session

    def save_run(self, run: AgentRun) -> None:
        self.runs[run.run_id] = run

    def delete_conversation(self, conversation_id: str) -> None:
        self.conversations.pop(conversation_id, None)
        for session_id, session in list(self.sessions.items()):
            if session.conversation_id == conversation_id:
                del self.sessions[session_id]
        for run_id, run in list(self.runs.items()):
            if run.conversation_id == conversation_id:
                del self.runs[run_id]
