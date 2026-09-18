from jev_plays_pokemon.main import main


def test_main_prints_greeting(capsys):
    main()

    captured = capsys.readouterr()
    assert captured.out == "Hello from jev-plays-pokemon!\n"
