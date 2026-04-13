def format_question_for_family(question: str, family: str, detail_level: str = "moderate") -> str:
    question = question.strip()

    detail_map = {
        "short": "Answer in 1 to 2 sentences.",
        "moderate": "Answer clearly in 3 to 6 sentences.",
        "detailed": "Answer in a detailed explanation of about 6 to 10 sentences.",
    }

    instruction = detail_map.get(detail_level, detail_map["moderate"])

    if family == "gpt2":
        return (
            f"Question: {question}\n"
            f"Answer: {instruction}\n"
        )

    if family == "tinyllama":
        return (
            "<|system|>\n"
            "You are a helpful assistant. Answer accurately and clearly. "
            f"{instruction}\n"
            "<|user|>\n"
            f"{question}\n"
            "<|assistant|>\n"
        )

    if family == "qwen":
        return (
            "<|im_start|>system\n"
            f"You are a helpful assistant. {instruction}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    
    if family == "smollm2":
        return (
            "<|im_start|>system\n"
            f"You are a helpful assistant. {instruction}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    return question
