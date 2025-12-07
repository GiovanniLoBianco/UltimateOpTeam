import pytest
from ultimate_opteam.data import Team, Player


@pytest.fixture
def sample_team(sample_players) -> Team:
    # sample_players has 2 players.
    # Messi (RW, alt: ST, CF), Mbappe (ST, alt: LW)
    # Let's put them in wrong positions to test optimize.
    # Messi at ST (alt), Mbappe at RW (bad)
    return Team(
        formation="4-4-2",
        composition=[("ST", sample_players[0]), ("RW", sample_players[1])],
    )


def test_team_rating(sample_team: Team):
    # (93 + 91) / 11 = 16.72...
    assert sample_team.rating == pytest.approx((93 + 91) / 11)


def test_team_chemistry(sample_team: Team):
    # Logic is complex, but let's test basic behavior.
    # Both same club (Paris SG), same league (Ligue 1).
    # Different nation (Argentina, France).
    # detailed_chemistry returns list of (pos, chem).
    # chem depends on thresholds.
    # 2 players from same club -> club count = 2.
    # 2 players from same league -> league count = 2.
    # 1 player from each nation -> nation count = 1.

    # _get_mode(score, thr):
    # league: score=2, thr=[2,4,7] -> mode 1 (>=2)
    # nation: score=1, thr=[2,5,8] -> mode 0
    # club: score=2, thr=[2,4,7] -> mode 1 (>=2)
    # total chem per player = 1 + 0 + 1 = 2.

    chem = sample_team.chemistry
    # 2 players * 2 chem = 4.
    assert chem == 4


def test_team_optimize(sample_players):
    # Messi (RW), Mbappe (ST)
    # Current: ST -> Messi (1), RW -> Mbappe (-30)
    # Optimal: RW -> Messi (2), ST -> Mbappe (2)

    team = Team(
        formation="4-4-2",
        composition=[("ST", sample_players[0]), ("RW", sample_players[1])],
    )

    opt_team = team.optimize()

    found_messi = False
    found_mbappe = False

    for pos, player in opt_team.composition:
        if player.name == "Lionel Messi":
            assert pos == "RW"
            found_messi = True
        if player.name == "Kylian Mbappé":
            assert pos == "ST"
            found_mbappe = True

    assert found_messi
    assert found_mbappe


def test_team_equals(sample_team: Team):
    assert sample_team.equals(sample_team)
    # Different formation
    other = Team("4-3-3", sample_team.composition)
    assert not sample_team.equals(other)


def test_team_chemistry_complex():
    # Create players with Icon and Hero status
    icon_player = Player(
        name="Icon Player",
        preferred_position="ST",
        alternate_position=[],
        rating=90,
        nation="Brazil",
        league="Icon League",
        club="Icon Club",
        icon=True,
    )
    hero_player = Player(
        name="Hero Player",
        preferred_position="CM",
        alternate_position=[],
        rating=88,
        nation="France",
        league="Ligue 1",
        club="Hero Club",
        hero=True,
    )

    # Icon and Hero always have 3 chemistry
    team = Team(
        formation="TEST",
        composition=[("ST", icon_player), ("CM", hero_player)],
    )

    detailed_chem = team.detailed_chemistry
    assert len(detailed_chem) == 2
    for _, chem in detailed_chem:
        assert chem == 3

    # Check if Icon boosts nation and Hero boosts league
    # Icon (Brazil) -> Nation count for Brazil should be +2 (Icon counts as 2? No, code says +1 if icon)
    # Wait, let's check code:
    # if player.icon and cat == "nation": category_score[cat][player.nation] += 1
    # So Icon adds 1 extra point to nation (total 2 for himself).

    # Hero (Ligue 1) -> League count for Ligue 1 should be +1 extra (total 2 for himself).

    # Let's add a normal player to benefit from this.
    normal_player_brazil = Player(
        name="Normal Brazil",
        preferred_position="CB",
        alternate_position=[],
        rating=80,
        nation="Brazil",
        league="Other League",
        club="Other Club",
    )

    team_boost = Team(
        formation="TEST",
        composition=[("ST", icon_player), ("CB", normal_player_brazil)],
    )

    # Brazil count: Icon (1) + Icon Bonus (1) + Normal (1) = 3.
    # Nation thresholds: [2, 5, 8]. Score 3 -> Mode 1 (>=2).
    # Normal player gets 1 chem from nation.

    chem_normal = [chem for pos, chem in team_boost.detailed_chemistry if pos == "CB"][
        0
    ]
    # League: 1 (Other) -> 0
    # Club: 1 (Other) -> 0
    # Nation: 3 -> 1
    # Total = 1
    assert chem_normal == 1


def test_optimize_impossible(sample_players):
    # 2 players, 2 positions, but players can't play there.
    # Messi (RW, ST, CF), Mbappe (ST, LW)
    # Positions: GK, CB

    team = Team(
        formation="TEST",
        composition=[("GK", sample_players[0]), ("CB", sample_players[1])],
    )

    opt_team = team.optimize()

    # Should still return a team with same players
    assert len(opt_team.composition) == 2
    # Positions should be GK and CB (swapped or not)
    positions = sorted([pos for pos, _ in opt_team.composition])
    assert positions == ["CB", "GK"]
