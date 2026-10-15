import functools
import time

_PROFILE_CALLBACK = None

def set_profile_callback(callback) -> None:
    global _PROFILE_CALLBACK
    _PROFILE_CALLBACK = callback


def _profile_backward(name):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(ctx, *args):
            callback = _PROFILE_CALLBACK
            if callback is None:
                return function(ctx, *args)
            started = time.perf_counter()
            try:
                return function(ctx, *args)
            finally:
                callback(
                    f"backward.{name}", time.perf_counter() - started
                )

        return wrapped

    return decorate
