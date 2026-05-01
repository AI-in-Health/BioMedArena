import pytest

from harness import cli


def test_cli_list_commands_return_zero(capsys):
    assert cli.main(["list-benchmarks"]) == 0
    assert "quick_suite" in capsys.readouterr().out

    assert cli.main(["list-modes"]) == 0
    assert "heavy" in capsys.readouterr().out

    assert cli.main(["list-backbones"]) == 0
    assert "provider=" in capsys.readouterr().out


def test_cli_parser_rejects_missing_required_args():
    with pytest.raises(SystemExit):
        cli.main(["run", "--benchmark", "quick_suite"])


def test_cli_parser_rejects_invalid_web_tools_choice():
    with pytest.raises(SystemExit):
        cli.main([
            "run",
            "--benchmark", "quick_suite",
            "--backbone", "gpt-4o",
            "--web-tools", "bad",
        ])
