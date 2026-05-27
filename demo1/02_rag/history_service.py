import os, json
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from typing import Sequence, List
import config_rag as config


def get_history(session_id):
    return FileChatHistory(session_id, config.history_storage_path)


class FileChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        super().__init__()
        self.session_id = session_id
        self.file_path = os.path.join(storage_path, self.session_id)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]):
        current_messages = self.messages
        current_messages.extend(messages)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([message_to_dict(msg) for msg in current_messages], f)

    @property
    def messages(self) -> List[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def clear(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
