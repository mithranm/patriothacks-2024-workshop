from openai import OpenAI


def chat_with_gpt():
    openai_completions_api_version = "v1"
    deployment_url = "https://mithranm--example-vllm-openai-compatible-serve-dev.modal.run"

    client = OpenAI(
        api_key="change_this",
        base_url=f"{deployment_url}/{openai_completions_api_version}",
    )

    conversation_history = []

    print("Welcome to the CLI Chat Application!")
    print("Type 'quit' to exit the conversation.")
    MODEL_NAME = client.models.list().data[0].id
    if MODEL_NAME == "Llama-3.2-3B-Instruct":
        print("Model is ready")
    else:
        print(f"Failed to load model: {MODEL_NAME}")
        exit(1)

    try:
        while True:
            user_input = input("You: ")

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            conversation_history.append({"role": "user", "content": user_input})
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=conversation_history
            )
            assistant_reply = response.choices[0].message.content
            conversation_history.append(
                {"role": "assistant", "content": assistant_reply}
            )

            print("Assistant:", assistant_reply)

    except KeyboardInterrupt:
        print("\nExiting the chat.")


if __name__ == "__main__":
    chat_with_gpt()
