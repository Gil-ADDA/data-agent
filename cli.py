import sys
from agent_with_tool import run_agent

def main():
    # One-shot mode: python cli.py "your question"
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
        print(f"\nRequest: {request}\n")
        result = run_agent(request)
        print("\n--- Answer ---")
        print(result)
        return

    # Interactive mode: python cli.py
    print("=== Data Agent CLI ===")
    print("Type your request and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Bye!")
            break

        result = run_agent(user_input)
        print(f"\nAgent: {result}\n")

if __name__ == "__main__":
    main()
