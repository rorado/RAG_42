import os

def search_files(dir="./"):

    with os.scandir(dir) as entries:
        for en in entries:
            if en.is_dir():
                search_files(en)
            elif en.is_file() and en.name.endswith((".py", ".md")):
                with open(en.path, "r") as file:
                    for line in file:
                        print(line)

def main():
    search_files("./vllm-0.10.1")

main()