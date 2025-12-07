import pytest
from unittest.mock import patch, mock_open, MagicMock
from ultimate_opteam.optim.milp_model import UT_MILP_Model, extract_pareto_frontier
from ultimate_opteam.data import Team


@pytest.fixture
def sample_players_milp(sample_players):
    return sample_players


@patch("ultimate_opteam.optim.milp_model.json.load")
@patch("builtins.open", new_callable=mock_open)
def test_milp_model_init(mock_file, mock_json, sample_players_milp):
    # Mock formation with 2 positions matching our 2 sample players
    mock_json.return_value = {"TEST-2": ["ST", "RW"]}

    model = UT_MILP_Model(sample_players_milp, "TEST-2")

    assert model.formation == "TEST-2"
    assert len(model.players) == 2
    assert model.solver is not None

    # Check if variables are created
    # x variables: players * positions = 2 * 2 = 4
    assert len(model.x) == 4

    # Check positions property
    assert len(model.positions) == 2
    assert model.positions[0] == "ST"

    # Check constraints count (roughly)
    assert model.solver.NumConstraints() > 0
    assert model.solver.NumVariables() > 0


@patch("ultimate_opteam.optim.milp_model.json.load")
@patch("builtins.open", new_callable=mock_open)
def test_milp_properties(mock_file, mock_json, sample_players_milp):
    mock_json.return_value = {"TEST-2": ["ST", "RW"]}
    model = UT_MILP_Model(sample_players_milp, "TEST-2")

    # sample_players_milp has 2 players:
    # 1. Argentina, Ligue 1, Paris SG
    # 2. France, Ligue 1, Paris SG

    assert len(model.nation) == 2
    assert "Argentina" in model.nation
    assert "France" in model.nation

    assert len(model.league) == 1
    assert "Ligue 1" in model.league

    assert len(model.club) == 1
    assert "Paris SG" in model.club


@patch("ultimate_opteam.optim.milp_model.json.load")
@patch("builtins.open", new_callable=mock_open)
def test_milp_solve_mock(mock_file, mock_json, sample_players_milp):
    mock_json.return_value = {"TEST-2": ["ST", "RW"]}
    model = UT_MILP_Model(sample_players_milp, "TEST-2")

    # Mock solver.Solve() and solution retrieval
    with patch.object(model.solver, "Solve") as mock_solve:
        # Mock return value for Solve
        # We need to import pywraplp to access Solver.OPTIMAL
        from ortools.linear_solver import pywraplp

        mock_solve.return_value = pywraplp.Solver.OPTIMAL

        # Mock _extract_team_from_solution instead of variables
        with patch.object(model, "_extract_team_from_solution") as mock_extract:
            mock_extract.return_value = Team(
                "TEST-2",
                [("ST", sample_players_milp[0]), ("RW", sample_players_milp[1])],
            )

            result_team = model.solve()

            assert result_team is not None
            assert result_team.formation == "TEST-2"
            # optimize is called inside solve, so positions might change if optimize works.
            # Messi (0) prefers RW. Mbappe (1) prefers ST.
            # Initial (from extract): Messi->ST, Mbappe->RW.
            # Optimized: Messi->RW, Mbappe->ST.

            found_messi_rw = False
            for pos, player in result_team.composition:
                if player.name == "Lionel Messi" and pos == "RW":
                    found_messi_rw = True
            assert found_messi_rw


def test_extract_pareto_frontier():
    # Create 3 teams
    # T1: Rating 90, Chem 30
    # T2: Rating 80, Chem 20 (Dominated by T1)
    # T3: Rating 95, Chem 10 (Not dominated)

    m1 = MagicMock(spec=Team)
    m1.rating = 90
    m1.chemistry = 30
    m1.equals.return_value = False

    m2 = MagicMock(spec=Team)
    m2.rating = 80
    m2.chemistry = 20
    m2.equals.return_value = False

    m3 = MagicMock(spec=Team)
    m3.rating = 95
    m3.chemistry = 10
    m3.equals.return_value = False

    teams = [m1, m2, m3]
    pareto = extract_pareto_frontier(teams)

    assert m1 in pareto
    assert m2 not in pareto  # Dominated by m1
    assert m3 in pareto

