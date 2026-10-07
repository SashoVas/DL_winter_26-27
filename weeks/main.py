import argparse


def parse_command_line_arguments() -> tuple[int, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", type=int, default=1)
    args = parser.parse_args()
    week = args.w
    return (week, 1)


def main() -> int:
    return 0
