from qqbot.schema import (
    AgentRun,
    Conversation,
    ModelContext,
    Session,
)


class ContextBuilder:
    def build(self, *, system_prompt: str, conversation: Conversation, session: Session, run: AgentRun) -> ModelContext:

        recent_messages = [message for message in session.recent_messages if not message.recalled and message.id != run.current_turn.id]

        return ModelContext(
            system_prompt=system_prompt,
            long_term_memory=conversation.long_term_memory,
            session_summary=session.summary,
            recent_messages=recent_messages,
            current_turn=run.current_turn,
            tool_trace=run.tool_trace,
        )
