import os, json
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from typing import Sequence, List


class FileChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        super().__init__()
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(self.storage_path, self.session_id)

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]):
        all_messages = list(self.messages)
        all_messages.extend(messages)

        new_messages = [message_to_dict(msg) for msg in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property
    def messages(self) -> List[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except FileNotFoundError:
            return []

    def clear(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)


f = FileChatHistory("u001", "chat_history")
# msg = [AIMessage("hello"), HumanMessage("你好吗"), AIMessage("我是大王")]
# f.add_messages(msg)
# print(f.messages)
# f.clear()
