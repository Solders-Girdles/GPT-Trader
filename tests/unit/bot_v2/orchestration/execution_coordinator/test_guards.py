import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState

@pytest.mark.asyncio
async def test_ensure_order_lock_creates_lock(coordinator: ExecutionCoordinator) -> None:
    runtime_state = coordinator.context.runtime_state
    assert runtime_state.order_lock is None

    lock = coordinator.ensure_order_lock()

    assert lock is coordinator.context.runtime_state.order_lock


@pytest.mark.asyncio
async def test_run_runtime_guards_loop_handles_exceptions(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test _run_runtime_guards_loop handles exceptions gracefully."""
    exec_engine = Mock()
    exec_engine.run_runtime_guards = Mock(side_effect=Exception("guard_error"))
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
    runtime_state.exec_engine = exec_engine
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    # Run for a short time then cancel
    task = asyncio.create_task(coordinator._run_runtime_guards_loop())
    await asyncio.sleep(0.1)  # Let it run one iteration
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

    # Should have called run_runtime_guards despite exception
    base_context.runtime_state.exec_engine.run_runtime_guards.assert_called()


def test_ensure_order_lock_creates_lock_when_missing(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test ensure_order_lock creates lock when missing."""
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
    runtime_state.order_lock = None
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    lock = coordinator.ensure_order_lock()

    assert lock is not None
    assert base_context.runtime_state.order_lock is lock


def test_ensure_order_lock_returns_existing_lock(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test ensure_order_lock returns existing lock."""
    existing_lock = Mock()
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
    runtime_state.order_lock = existing_lock
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    lock = coordinator.ensure_order_lock()

    assert lock is existing_lock


def test_ensure_order_lock_handles_asyncio_error(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test ensure_order_lock handles asyncio initialization errors."""
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
    runtime_state.order_lock = None
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    # Mock asyncio.Lock to raise RuntimeError
    original_lock = asyncio.Lock
    asyncio.Lock = Mock(side_effect=RuntimeError("no event loop"))

    try:
        with pytest.raises(RuntimeError):
            coordinator.ensure_order_lock()
    finally:
        asyncio.Lock = original_lock


@pytest.mark.asyncio
async def test_run_runtime_guards_loop_handles_continuously(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
):
    """Test runtime guards loop runs continuously."""
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
    runtime_state.exec_engine = Mock()
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    call_count = 0

    def mock_run_runtime_guards():
        nonlocal call_count
        call_count += 1
        if call_count >= 3:  # Stop after 3 calls
            raise asyncio.CancelledError()
        # Simulate work
        import time
        time.sleep(0.01)

    exec_engine = Mock()
    exec_engine.run_runtime_guards = Mock(side_effect=mock_run_runtime_guards)
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
    runtime_state.exec_engine = exec_engine
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    # Should run continuously until cancelled
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        task = asyncio.create_task(coordinator._run_runtime_guards_loop())

        try:
            await asyncio.wait_for(task, timeout=0.1)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    assert call_count >= 2
