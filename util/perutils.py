import time


def TicTocGenerator():
    """Generator that returns time differences."""
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti

TicToc = TicTocGenerator()


def toc(tempBool=True):
    """Prints the time difference yielded by generator instance TicToc."""
    tempTimeInterval = next(TicToc)
    if tempBool:
        return tempTimeInterval


def tic():
    """Records a time in TicToc, marks the beginning of a time interval."""
    toc(False)