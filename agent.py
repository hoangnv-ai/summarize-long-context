from ollama import chat
from prompt import *
class Agent:
    def __init__(self, system="", max_length=0.1):
        self.system = system
        self.max_length = max_length
        self.messages = []

        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, chunk):
        message = generate_prompt(chunk, self.max_length)
        print("message :", message)
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
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.15,
            },
            stream=True
        )

        full_text = ""

        for chunk in stream:
            text = chunk["message"]["content"]
            full_text += text
            print(text, end='', flush=True)

        return full_text