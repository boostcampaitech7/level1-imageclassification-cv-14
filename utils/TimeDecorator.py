import time
from datetime import timedelta

class TimeDecorator:
    def __init__(self) -> None:
        pass

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            end_time = time.time()

            td = timedelta(seconds = end_time - start_time)
            h, remainder = divmod(td.seconds, 3600)
            m, s = divmod(remainder, 60)
            ms = td.microseconds // 1000    

            print(f"{func.__name__} Running Time : {h:02d}h {m:02d}m {s:02d}s {ms:03d}ms")
            return res
        return wrapper