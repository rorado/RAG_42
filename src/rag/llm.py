from transformers import pipeline


class LLM:
    def __init__(self) -> None:
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
                "content": """
                    Answer the user's question using only the provided context.

                    Use facts from only the context
                """
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}"
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

        return str(output[0]["generated_text"][-1]["content"])
