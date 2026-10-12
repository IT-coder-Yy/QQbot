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
