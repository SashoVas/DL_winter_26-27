import argparse


def parse_command_line_arguments() -> tuple[int, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", type=int, default=1)
    parser.add_argument("-t", type=int, default=1)
    args = parser.parse_args()
    week, task = args.w, args.t
    return (week, task)


def main() -> int:
    return 0
