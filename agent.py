from ollama import chat
from prompt import *
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []

        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, chunk):
        message = generate_prompt(chunk)
        # print("message :", message)
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        return result

    def execute(self):
        """
        Streaming + trả về toàn bộ chuỗi sau khi hoàn tất.
        """
        stream = chat(
            model='llama3.1:latest',
            messages=self.messages,
            options={
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "repeat_penalty": 1.0
            },
            stream=True
        )

        full_text = ""

        for chunk in stream:
            text = chunk["message"]["content"]
            full_text += text
            print(text, end='', flush=True)

        return full_text