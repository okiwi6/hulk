from collections.abc import Sized
from typing import Generator, Iterable, Union

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text


def default_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        IterationsPerSecondColumn(),
    )


class IterationsPerSecondColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.finished:
            rate = task.completed / task.elapsed
        else:
            elapsed = task.elapsed or 0
            rate = task.completed / elapsed if elapsed > 0 else 0

        if rate < 1.0 and rate > 0.0:
            return Text(f"{1.0 / rate:.2f} s/it")
        return Text(f"{rate:.2f} it/s")


class bar[T]:
    def __init__(self, iter: Union[Iterable[T], Sized], progress: Progress, name: str):
        self.iter = iter
        self.progress = progress
        self.task_id = progress.add_task(name, total=len(iter))

    def __len__(self) -> int:
        return len(self.iter)

    def __iter__(self):
        for item in self.iter:
            yield item
            self.progress.update(self.task_id, advance=1)
        self.progress.remove_task(self.task_id)
