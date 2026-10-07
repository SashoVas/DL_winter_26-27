import argparse
import importlib


def parse_command_line_arguments() -> tuple[int, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--week", type=int, default=1)
    parser.add_argument("-t", "--task", type=int, default=1)
    args = parser.parse_args()
    week, task = args.week, args.task
    return (week, task)


def main() -> int:
    week, task = parse_command_line_arguments()
    week_str = f"{week:02d}"
    task_str = f"{task:02d}"
    try:
        mod = importlib.import_module(f"weeks.week{week_str}.task{task_str}")
    except ModuleNotFoundError:
        raise ValueError(f"Task {task_str} does not exist in week {week_str}!")
    return mod.main()
