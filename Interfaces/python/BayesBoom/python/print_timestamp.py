import time


def print_timestamp(iteration_number, ping):
    if ping <= 0:
        return
    if iteration_number % ping == 0:
        timestamp = time.asctime()
        sep = ' =-=-=-=-=-=-=-=-=-=-= '
        print(sep + timestamp + f" Iteration {iteration_number} " + sep)
