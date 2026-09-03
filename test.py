from models.router import AttentionRouter


def main():
    router = AttentionRouter(d_input = 1024)
    print(f"{router=}")
if __name__ == "__main__":
    main() 