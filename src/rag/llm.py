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
                    Answer the user's question using only the provided context.

                    Use facts from only the context

                   Instructions:
                    - Identify the most relevant facts about the subject of the question.
                    - Prefer informative answers over vague statements.
                    - Ignore irrelevant context.
                    - You may paraphrase information from the context.
                    - Do not invent unsupported facts.
                    - If no relevant information is available, say:
                    "The provided context does not contain enough information."


                    Contexts:
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
