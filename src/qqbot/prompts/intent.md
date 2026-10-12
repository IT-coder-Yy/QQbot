Determine whether the current_message is directed at the AI bot or clearly invites the AI bot to respond.

Use recent_messages only to understand the conversation flow, turn-taking, references, and who the current speaker is talking to.

Answer YES when:

- the speaker is continuing a conversation with the AI bot;
- the speaker is implicitly asking the AI bot a question;
- the speaker asks for information, help, advice, an opinion, or an action from the AI bot;
- the conversational context makes it clear that a response from the AI bot is expected.

Answer NO when:

- humans are talking to each other;
- the current message is a casual statement, reaction, joke, or comment not directed at the AI bot;
- a question is clearly directed at another human;
- the AI bot would be interrupting an ordinary human conversation;
- there is insufficient evidence that the speaker expects the AI bot to respond.

Be conservative. The bot should remain silent unless the current message is reasonably intended for it.
