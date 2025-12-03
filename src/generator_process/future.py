"""
Future class for async result handling.

Reference: Dream Textures generator_process/future.py
Provides a way to handle results from subprocess operations asynchronously.
"""

import functools
import threading
from typing import Callable, Any, MutableSet, Optional


class Future:
    """
    Object that represents a value that has not completed processing, but will in the future.

    Add callbacks to be notified when values become available, or use `.result()` and `.exception()` 
    to wait for the value.
    """
    
    _response_callbacks: MutableSet[Callable[['Future', Any], None]]
    _exception_callbacks: MutableSet[Callable[['Future', BaseException], None]]
    _done_callbacks: MutableSet[Callable[['Future'], None]]
    _responses: list
    _exception: Optional[BaseException]
    _done_event: threading.Event
    done: bool
    cancelled: bool
    check_cancelled: Callable[[], bool]
    call_done_on_exception: bool

    def __init__(self):
        self._response_callbacks = set()
        self._exception_callbacks = set()
        self._done_callbacks = set()
        self._responses = []
        self._exception = None
        self._done_event = threading.Event()
        self.done = False
        self.cancelled = False
        self.call_done_on_exception = True
        self.check_cancelled = lambda: False

    def result(self, last_only: bool = False, timeout: Optional[float] = None):
        """
        Get the result value (blocking).
        
        Args:
            last_only: If True, return only the last response instead of all
            timeout: Maximum time to wait for result
            
        Returns:
            The result value(s)
            
        Raises:
            Exception: If the future completed with an exception
            TimeoutError: If timeout is reached before completion
        """
        def _response():
            match len(self._responses):
                case 0:
                    return None
                case 1:
                    return self._responses[0]
                case _:
                    return self._responses[-1] if last_only else self._responses
        
        if self._exception is not None:
            raise self._exception
        if self.done:
            return _response()
        else:
            completed = self._done_event.wait(timeout=timeout)
            if not completed:
                raise TimeoutError("Future did not complete within timeout")
            if self._exception is not None:
                raise self._exception
            return _response()
    
    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        """
        Get the exception if one occurred.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            The exception or None
        """
        if self.done:
            return self._exception
        else:
            self._done_event.wait(timeout=timeout)
            return self._exception
    
    def cancel(self):
        """Request cancellation of the operation."""
        self.cancelled = True

    def _run_on_main_thread(self, func: Callable):
        """
        Run a function on the main thread if possible.
        Uses bpy.app.timers if available.
        """
        if threading.current_thread() == threading.main_thread():
            func()
            return
        try:
            import bpy
            bpy.app.timers.register(func, persistent=True)
        except:
            func()

    def add_response(self, response: Any):
        """
        Add a response value and notify all consumers.
        
        Args:
            response: The response value to add
        """
        self._responses.append(response)
        
        def run_callbacks():
            for response_callback in self._response_callbacks:
                try:
                    response_callback(self, response)
                except Exception as e:
                    print(f"[Future] Response callback error: {e}")
        
        self._run_on_main_thread(run_callbacks)

    def set_exception(self, exception: BaseException):
        """
        Set the exception.
        
        Args:
            exception: The exception that occurred
        """
        self._exception = exception
        
        def run_callbacks():
            for exception_callback in self._exception_callbacks:
                try:
                    exception_callback(self, exception)
                except Exception as e:
                    print(f"[Future] Exception callback error: {e}")
        
        self._run_on_main_thread(run_callbacks)

    def set_done(self):
        """Mark the future as done."""
        assert not self.done, "Future is already done"
        self.done = True
        self._done_event.set()
        
        if self._exception is None or self.call_done_on_exception:
            def run_callbacks():
                for done_callback in self._done_callbacks:
                    try:
                        done_callback(self)
                    except Exception as e:
                        print(f"[Future] Done callback error: {e}")
            
            self._run_on_main_thread(run_callbacks)

    def add_response_callback(self, callback: Callable[['Future', Any], None]):
        """
        Add a callback to run whenever a response is received.
        Will be called multiple times by generator functions.
        
        Args:
            callback: Function taking (future, response)
        """
        self._response_callbacks.add(callback)
    
    def add_exception_callback(self, callback: Callable[['Future', BaseException], None]):
        """
        Add a callback to run when the future errors.
        Will only be called once at the first exception.
        
        Args:
            callback: Function taking (future, exception)
        """
        self._exception_callbacks.add(callback)
        if self._exception is not None:
            self._run_on_main_thread(functools.partial(callback, self, self._exception))

    def add_done_callback(self, callback: Callable[['Future'], None]):
        """
        Add a callback to run when the future is marked as done.
        Will only be called once.
        
        Args:
            callback: Function taking (future)
        """
        self._done_callbacks.add(callback)
        if self.done:
            self._run_on_main_thread(functools.partial(callback, self))
