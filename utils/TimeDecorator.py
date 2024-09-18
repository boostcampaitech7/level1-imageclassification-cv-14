import time
from datetime import timedelta

class TimeDecorator:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        self.func(*args, **kwargs)
        end_time = time.time()

        td = timedelta(seconds = end_time - start_time)
        h, remainder = divmod(td.seconds, 3600)
        m, s = divmod(remainder, 60)
        ms = td.microseconds // 1000    

        print(f"{self.func.__name__} Running Time : {h:02d}h {m:02d}m {s:02d}s {ms:03d}ms")

