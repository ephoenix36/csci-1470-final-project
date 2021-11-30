from functools import wraps
from time import time

def timing(f):
    """
    Wraps the provided function and prints how long it took to to execute.
    """
    @wraps(f)
    def wrap(*args, **kw):
        t1 = time()
        result = f(*args, **kw)
        t2 = time()
        print('func:%r args:[%r, %r] took: %2.4f sec'.format(f.__name__, args, kw, t1-t2))
        return result
    return wrap