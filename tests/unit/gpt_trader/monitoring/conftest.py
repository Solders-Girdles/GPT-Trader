import pytest
from freezegun import freeze_time


@pytest.fixture
def frozen_time():
    with freeze_time("2024-01-01 12:00:00") as frozen:
        yield frozen
