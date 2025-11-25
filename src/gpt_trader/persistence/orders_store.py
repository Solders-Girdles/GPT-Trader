from pathlib import Path


class OrdersStore:
    """
    Persists order state to disk.
    """

    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)

    # Add methods as needed by the application
