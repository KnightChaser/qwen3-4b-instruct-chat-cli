from vllm import LLM, SamplingParams


def main() -> None:
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,  # Qwen provides custom model code
        max_model_len=32768,
    )

    # How we want the model to sample
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,    # max tokens to generate
    )

    prompt = "What is the definition of artificial intelligence?"

    outputs = llm.generate([prompt], sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(generated_text.strip())


if __name__ == "__main__":
    main()
