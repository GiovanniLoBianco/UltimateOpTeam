import pytest
from ultimate_opteam.data import Player


@pytest.fixture
def sample_player_1():
    return Player(
        name="Lionel Messi",
        preferred_position="RW",
        alternate_position=["ST", "CF"],
        rating=93,
        nation="Argentina",
        league="Ligue 1",
        club="Paris SG",
    )


@pytest.fixture
def sample_player_2():
    return Player(
        name="Kylian Mbappé",
        preferred_position="ST",
        alternate_position=["LW"],
        rating=91,
        nation="France",
        league="Ligue 1",
        club="Paris SG",
    )


@pytest.fixture
def sample_players(sample_player_1, sample_player_2):
    return [sample_player_1, sample_player_2]
