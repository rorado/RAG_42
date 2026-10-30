from transformers import pipeline


class LLM:
    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="Qwen/Qwen3-0.6B",
            device="cpu",
            dtype="auto",
        )

    def answer(self, question: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"""
                    You are a helpful AI assistant.

                    Answer the user's question using ONLY the provided context.

                    Rules:
                    1. Do not use outside knowledge.
                    2. Do not invent information.
                    3. If the context is insufficient, say:
                        "The provided context does not contain enough information."
                    4. Give clear and concise answers.

                    Context:
                    {context}
                """
            },
            {
                "role": "user",
                "content": question
            }
        ]

        output = self.generator(
            messages,
            max_new_tokens=512,
            do_sample=False,
            tokenizer_encode_kwargs={
                "enable_thinking": False,
            },
        )

        return output[0]["generated_text"][-1]["content"]
