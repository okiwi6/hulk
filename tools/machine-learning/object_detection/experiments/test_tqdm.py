from time import sleep
from typing import Sized

from rich.progress import Progress
from ssd.utils import bar
from tqdm import trange

with Progress() as progress:
    for i in bar(range(1000), progress, "Outer"):
        for j in bar(range(100), progress, "Inner"):
            sleep(0.001)
        for k in bar(range(200), progress, "Inner 2"):
            sleep(0.01)
