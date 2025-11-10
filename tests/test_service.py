from service import add_name, load_names


def test_add_and_load_names(tmp_path, monkeypatch):
    test_file = tmp_path / "names.json"
    monkeypatch.setattr("service.DATA_FILE", test_file)
    add_name("Jinwoo")
    add_name("Kevin")
    names = load_names()
    assert "Jinwoo" in names
    assert "Kevin" in names
