import sys
from service import add_name, load_names

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli.py [add <name> | list]")
        sys.exit(1)
    if sys.argv[1] == "add" and len(sys.argv) == 3:
        add_name(sys.argv[2])
        print(f"Added: {sys.argv[2]}")
    elif sys.argv[1] == "list":
        print("Names:", load_names())
    else:
        print("Invalid command.")
