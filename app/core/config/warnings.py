import functools
import inspect
import warnings


def deprecated(
    reason: str = "", category: type[Warning] = DeprecationWarning, stacklevel: int = 2
):
    """
    Lightweight deprecation decorator compatible with Python 3.12.

    - For functions/methods: emits a warning on call.
    - For classes: emits a warning on instantiation.
    """

    def _decorate(obj):
        message = reason or f"{getattr(obj, '__name__', obj)} is deprecated"

        if inspect.isclass(obj):
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def __init__(self, *args, **kwargs):
                warnings.warn(message, category=category, stacklevel=stacklevel)
                orig_init(self, *args, **kwargs)

            obj.__init__ = __init__
            return obj

        if callable(obj):

            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                warnings.warn(message, category=category, stacklevel=stacklevel)
                return obj(*args, **kwargs)

            return wrapper

        return obj

    return _decorate
