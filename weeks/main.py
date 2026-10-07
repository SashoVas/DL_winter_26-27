import argparse


def parse_command_line_arguments() -> tuple[int, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--week", type=int, default=1)
    parser.add_argument("-t", "--task", type=int, default=1)
    args = parser.parse_args()
    week, task = args.week, args.task
    return (week, task)


def main() -> int:
    return 0
