from pathlib import Path

from ssd.data_query import DvcImageQuery

DvcImageQuery(Path.cwd().joinpath(".."))
