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
                "role": "user",
                "content": f"""
                    Answer the question using only the provided context.

                    Rules:
                    1. Give a clear and understandable answer.
                    2. Do not invent information not found in the context.
                    3. Answer the exact question directly from contents.
                    Context:
                    {context}

                    Question:
                    {question}
                """
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
