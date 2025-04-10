import pytest
import yaml

from alphabase.yaml_utils import load_yaml, save_yaml


def test_load_yaml_with_valid_file(tmp_path):
    file_path = tmp_path / "valid.yaml"
    file_path.write_text("key: value")
    result = load_yaml(file_path)
    assert result == {"key": "value"}


def test_oad_yaml_with_invalid_file(tmp_path):
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text("key: [unclosed")
    with pytest.raises(yaml.YAMLError):
        load_yaml(file_path)


def test_load_yaml_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_yaml("nonexistent.yaml")


def test_save_yaml_creates_file_with_correct_content(tmp_path):
    file_path = tmp_path / "output.yaml"
    save_yaml(file_path, {"key": "value"})
    assert file_path.read_text() == "key: value\n"
