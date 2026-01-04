import pytest

from daystrom.providers import ModelMetadata, Provider, ProviderMetadata


class TestModelMetadata:
    def test_init(self):
        metadata = ModelMetadata(
            name="gpt-4o",
            display_name="GPT-4o",
            input_cost=2.5,
            output_cost=10.0,
            context_limit=128000,
            output_limit=16384,
        )

        assert metadata.name == "gpt-4o"
        assert metadata.display_name == "GPT-4o"
        assert metadata.input_cost == 2.5
        assert metadata.output_cost == 10.0
        assert metadata.context_limit == 128000
        assert metadata.output_limit == 16384

    def test_init_with_none_values(self):
        metadata = ModelMetadata(
            name="unknown-model",
            display_name="Unknown",
            input_cost=None,
            output_cost=None,
            context_limit=None,
            output_limit=None,
        )

        assert metadata.input_cost is None
        assert metadata.output_cost is None
        assert metadata.context_limit is None
        assert metadata.output_limit is None


class TestProviderMetadata:
    def test_get_data_with_mock_response(self, mocker):
        """Test get_data parses API response correctly."""
        ProviderMetadata.get_data.cache_clear()

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "openai": {
                "models": {
                    "gpt-4o": {
                        "id": "gpt-4o",
                        "name": "GPT-4o",
                        "cost": {"input": 2.5, "output": 10.0},
                        "limit": {"context": 128000, "output": 16384},
                    }
                }
            }
        }
        mocker.patch("daystrom.providers.httpx.get", return_value=mock_response)

        data = ProviderMetadata.get_data()

        assert data is not None
        assert "openai" in data
        assert "gpt-4o" in data["openai"]["models"]

        ProviderMetadata.get_data.cache_clear()

    def test_get_data_handles_api_failure(self, mocker, capsys):
        """Test get_data handles API failures gracefully."""
        ProviderMetadata.get_data.cache_clear()

        mocker.patch(
            "daystrom.providers.httpx.get", side_effect=Exception("Network error")
        )

        data = ProviderMetadata.get_data()

        assert data is None

        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out
        assert "models.dev" in captured.out

        ProviderMetadata.get_data.cache_clear()

    def test_init_populates_models_from_api(self, mocker):
        """Test that __init__ populates models dict from API data."""
        ProviderMetadata.get_data.cache_clear()

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "testprovider": {
                "models": {
                    "test-model": {
                        "id": "test-model-id",
                        "name": "Test Model",
                        "cost": {"input": 1.0, "output": 2.0},
                        "limit": {"context": 8000, "output": 4000},
                    }
                }
            }
        }
        mocker.patch("daystrom.providers.httpx.get", return_value=mock_response)

        provider = ProviderMetadata(
            name="testprovider",
            display_name="Test Provider",
            base_url="https://test.api.com",
        )

        assert "test-model" in provider.models
        model = provider.models["test-model"]
        assert isinstance(model, ModelMetadata)
        assert model.name == "test-model-id"
        assert model.display_name == "Test Model"
        assert model.input_cost == 1.0
        assert model.output_cost == 2.0
        assert model.context_limit == 8000
        assert model.output_limit == 4000

        ProviderMetadata.get_data.cache_clear()

    def test_init_handles_missing_cost_and_limit(self, mocker):
        """Test that __init__ handles models without cost/limit data."""
        ProviderMetadata.get_data.cache_clear()

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "testprovider": {
                "models": {
                    "minimal-model": {
                        "id": "minimal-id",
                        "name": "Minimal Model",
                    }
                }
            }
        }
        mocker.patch("daystrom.providers.httpx.get", return_value=mock_response)

        provider = ProviderMetadata(
            name="testprovider",
            display_name="Test Provider",
        )

        assert "minimal-model" in provider.models
        model = provider.models["minimal-model"]
        assert model.input_cost is None
        assert model.output_cost is None
        assert model.context_limit is None
        assert model.output_limit is None

        ProviderMetadata.get_data.cache_clear()

    def test_init_handles_provider_not_in_data(self, mocker):
        """Test that __init__ handles provider not found in API data."""
        ProviderMetadata.get_data.cache_clear()

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"otherprovider": {"models": {}}}
        mocker.patch("daystrom.providers.httpx.get", return_value=mock_response)

        provider = ProviderMetadata(
            name="nonexistent",
            display_name="Nonexistent Provider",
        )

        assert provider.models == {}

        ProviderMetadata.get_data.cache_clear()
