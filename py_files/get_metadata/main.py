from .scanner import scan
from .writer import write_csv


def main():
    data = scan()
    write_csv("runtimes.csv", data)


if __name__ == "__main__":
    main()