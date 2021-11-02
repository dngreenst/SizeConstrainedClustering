import cProfile
import io
import pstats
from typing import Callable


def profile_func(func: Callable, calls_percent: float = 1.0, sortby: str = 'cumulative') -> any:
    pr = cProfile.Profile()
    pr.enable()
    result = func()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(calls_percent)
    print(s.getvalue())
    return result
