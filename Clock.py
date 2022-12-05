import time

class Clock:
    def __init__(self):
        self.reset_timer()

    def get_dt(self):
        self.time2 = time.perf_counter()
        dt = self.time2-self.time1
        self.time1 = self.time2
        return dt

    def reset_timer(self):
        self.time1 = time.perf_counter()
        self.time2 = None