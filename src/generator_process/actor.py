"""
Actor base class for subprocess-based computation.

Reference: Dream Textures generator_process/actor.py
Provides Queue-based IPC between Blender main process and computation subprocess.
"""

from multiprocessing import Queue, Lock, current_process, get_context
import multiprocessing.synchronize
import enum
import traceback
import threading
from typing import Type, TypeVar, Generator as GeneratorType, Optional, Callable
import sys
import os
from pathlib import Path

from .future import Future


def _sanitize_for_pickle(obj):
    """
    Convert objects to pure Python types to avoid importing torch on unpickle.
    This prevents TBB DLL conflicts in the main Blender process.
    
    Queue unpickling would otherwise import torch modules, causing:
    OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    if isinstance(obj, dict):
        return {_sanitize_for_pickle(k): _sanitize_for_pickle(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        sanitized = [_sanitize_for_pickle(item) for item in obj]
        return type(obj)(sanitized) if isinstance(obj, tuple) else sanitized
    # Convert numpy/torch arrays to lists
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    # Convert numpy/torch scalars
    if hasattr(obj, 'item'):
        return obj.item()
    # Fallback: convert to string representation
    try:
        return str(obj)
    except:
        return repr(obj)


def _get_addon_path() -> Path:
    """Get the path to this addon's directory."""
    return Path(__file__).parent.parent


def _absolute_path(path: str) -> str:
    """Get absolute path relative to addon directory."""
    return str(_get_addon_path() / path)


def _load_dependencies():
    """
    Load dependencies in subprocess.
    This is only called in the __actor__ process, NOT in Blender main process.
    """
    import site
    
    deps_path = _absolute_path(".python_dependencies")
    
    if not os.path.exists(deps_path):
        print(f"[Actor] Dependencies not found at: {deps_path}")
        return
    
    # Add to sys.path at the beginning
    if deps_path not in sys.path:
        sys.path.insert(0, deps_path)
    
    # Windows: Add DLL directories
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        dll_paths = [
            os.path.join(deps_path, "torch", "lib"),
            os.path.join(deps_path, "torch", "bin"),
            os.path.join(deps_path, "Library", "bin"),
            os.path.join(deps_path, "numpy.libs"),
            os.path.join(deps_path, "scipy.libs"),
            # NVIDIA CUDA libraries
            os.path.join(deps_path, "nvidia", "cublas", "bin"),
            os.path.join(deps_path, "nvidia", "cudnn", "bin"),
            os.path.join(deps_path, "nvidia", "cufft", "bin"),
            os.path.join(deps_path, "nvidia", "curand", "bin"),
            os.path.join(deps_path, "nvidia", "cusolver", "bin"),
            os.path.join(deps_path, "nvidia", "cusparse", "bin"),
            os.path.join(deps_path, "nvidia", "nccl", "bin"),
            os.path.join(deps_path, "nvidia", "nvtx", "bin"),
        ]
        
        for dll_path in dll_paths:
            if os.path.exists(dll_path):
                try:
                    os.add_dll_directory(dll_path)
                except (OSError, AttributeError):
                    pass
        
        # Also prepend to PATH
        existing_path = os.environ.get("PATH", "")
        valid_dll_paths = [p for p in dll_paths if os.path.exists(p)]
        if valid_dll_paths:
            os.environ["PATH"] = os.pathsep.join(valid_dll_paths) + os.pathsep + existing_path
    
    print(f"[Actor] Loaded dependencies from: {deps_path}")


# Check if we are in the actor subprocess
is_actor_process = current_process().name == "__actor__"

if is_actor_process:
    # Load dependencies ONLY in subprocess
    _load_dependencies()


class ActorContext(enum.IntEnum):
    """
    The context of an `Actor` object.
    
    One `Actor` instance is the `FRONTEND`, while the other instance is the backend, 
    which runs in a separate process.
    The `FRONTEND` sends messages to the `BACKEND`, which does work and returns a result.
    """
    FRONTEND = 0
    BACKEND = 1


class Message:
    """
    Represents a function signature with a method name, positional arguments, and keyword arguments.

    Note: All arguments must be picklable.
    """

    def __init__(self, method_name: str, args: tuple, kwargs: dict):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
    
    CANCEL = "__cancel__"
    END = "__end__"


def _start_backend(cls: Type['Actor'], message_queue: Queue, response_queue: Queue):
    """Start the backend actor loop."""
    cls(
        ActorContext.BACKEND,
        message_queue=message_queue,
        response_queue=response_queue
    ).start()


class TracedError(BaseException):
    """Exception with traceback information."""
    def __init__(self, base: BaseException, trace: str):
        self.base = base
        self.trace = trace


T = TypeVar('T', bound='Actor')


class Actor:
    """
    Base class for specialized actors.
    
    Uses queues to send actions to a background process and receive a response.
    Calls to any method declared by the frontend are automatically dispatched to the backend.

    All function arguments must be picklable.
    """

    _message_queue: Queue
    _response_queue: Queue
    _lock: multiprocessing.synchronize.Lock
    _shared_instance: Optional['Actor'] = None

    # Methods that are not used for message passing, and should not be overridden in `_setup`.
    _protected_methods = {
        "start",
        "close",
        "is_alive",
        "can_use",
        "shared"
    }

    def __init__(
        self, 
        context: ActorContext, 
        message_queue: Optional[Queue] = None, 
        response_queue: Optional[Queue] = None
    ):
        self.context = context
        self._message_queue = message_queue if message_queue is not None else get_context('spawn').Queue(maxsize=1)
        self._response_queue = response_queue if response_queue is not None else get_context('spawn').Queue(maxsize=1)
        self._setup()
        self.__class__._shared_instance = self
    
    def _setup(self):
        """Setup the Actor after initialization."""
        match self.context:
            case ActorContext.FRONTEND:
                self._lock = Lock()
                # Replace methods with message-sending wrappers
                for name in filter(
                    lambda name: callable(getattr(self, name)) and not name.startswith("_") and name not in self._protected_methods, 
                    dir(self)
                ):
                    setattr(self, name, self._send(name))
            case ActorContext.BACKEND:
                pass

    @classmethod
    def shared(cls: Type[T]) -> T:
        """Get or create the shared instance."""
        return cls._shared_instance or cls(ActorContext.FRONTEND).start()

    def start(self: T) -> T:
        """Start the actor process."""
        match self.context:
            case ActorContext.FRONTEND:
                self.process = get_context('spawn').Process(
                    target=_start_backend, 
                    args=(self.__class__, self._message_queue, self._response_queue), 
                    name="__actor__", 
                    daemon=True
                )
                
                # Fix for Blender not being able to start a subprocess
                # while previously installed addons are being initialized.
                main_module = sys.modules.get("__main__")
                main_file = getattr(main_module, "__file__", None) if main_module else None
                
                if main_file == "<blender string>":
                    try:
                        main_module.__file__ = None
                        self.process.start()
                    finally:
                        main_module.__file__ = main_file
                else:
                    self.process.start()
                    
            case ActorContext.BACKEND:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                self._backend_loop()
        
        return self
    
    def close(self):
        """Stop the actor process."""
        match self.context:
            case ActorContext.FRONTEND:
                if hasattr(self, 'process') and self.process.is_alive():
                    self.process.terminate()
                self._message_queue.close()
                self._response_queue.close()
            case ActorContext.BACKEND:
                pass
    
    @classmethod
    def shared_close(cls: Type[T]):
        """Close the shared instance if it exists."""
        if cls._shared_instance is None:
            return
        cls._shared_instance.close()
        cls._shared_instance = None
    
    def is_alive(self) -> bool:
        """Check if the actor process is alive."""
        match self.context:
            case ActorContext.FRONTEND:
                return hasattr(self, 'process') and self.process.is_alive()
            case ActorContext.BACKEND:
                return True

    def can_use(self) -> bool:
        """Check if the actor is available for use (not locked)."""
        if result := self._lock.acquire(block=False):
            self._lock.release()
        return result

    def _backend_loop(self):
        """Main loop for the backend process."""
        while True:
            message = self._message_queue.get()
            self._receive(message)

    def _receive(self, message: Message):
        """Process a received message in the backend."""
        try:
            response = getattr(self, message.method_name)(*message.args, **message.kwargs)
            
            if isinstance(response, GeneratorType):
                # Handle generator responses
                for res in iter(response):
                    # Check for cancellation
                    extra_message = None
                    try:
                        extra_message = self._message_queue.get(block=False)
                    except:
                        pass
                    
                    if extra_message == Message.CANCEL:
                        break
                    
                    if isinstance(res, Future):
                        def check_cancelled():
                            try:
                                return self._message_queue.get(block=False) == Message.CANCEL
                            except:
                                return False
                        
                        res.check_cancelled = check_cancelled
                        res.add_response_callback(lambda _, res: self._response_queue.put(_sanitize_for_pickle(res)))
                        res.add_exception_callback(lambda _, e: self._response_queue.put(RuntimeError(str(e))))
                        res.add_done_callback(lambda _: None)
                    else:
                        # Sanitize to avoid torch import on unpickle
                        self._response_queue.put(_sanitize_for_pickle(res))
            else:
                # Sanitize to avoid torch import on unpickle
                self._response_queue.put(_sanitize_for_pickle(response))
                
        except Exception as e:
            trace = traceback.format_exc()
            # Always convert to RuntimeError with string message to avoid pickle issues
            # This prevents torch exception types from being pickled
            error_msg = f"{type(e).__name__}: {str(e)}"
            self._response_queue.put(TracedError(RuntimeError(error_msg), trace))
        
        self._response_queue.put(Message.END)

    def _send(self, name: str) -> Callable:
        """Create a wrapper function that sends a message to the backend."""
        def _send_impl(*args, _block: bool = False, **kwargs) -> Future:
            future = Future()
            
            def _send_thread(future: Future):
                self._lock.acquire()
                self._message_queue.put(Message(name, args, kwargs))

                while not future.done:
                    if future.cancelled:
                        self._message_queue.put(Message.CANCEL)
                    
                    response = self._response_queue.get()
                    
                    if response == Message.END:
                        future.set_done()
                    elif isinstance(response, TracedError):
                        response.base.__cause__ = Exception(response.trace)
                        future.set_exception(response.base)
                    elif isinstance(response, Exception):
                        future.set_exception(response)
                    else:
                        future.add_response(response)
                
                self._lock.release()
            
            if _block:
                _send_thread(future)
            else:
                thread = threading.Thread(target=_send_thread, args=(future,), daemon=True)
                thread.start()
            
            return future
        
        return _send_impl
    
    def __del__(self):
        self.close()
