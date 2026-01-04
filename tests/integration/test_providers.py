from daystrom.providers import ModelMetadata, Provider, ProviderMetadata


class TestProviderMetadata:
    def test_init_basic_attributes(self):
        """
        Test that basic attributes are set correctly.

        Choosing OpenAI as a likely reliable option for an integration test
        """
        provider = Provider.OPENAI.value

        assert provider.name == "openai"
        assert provider.display_name == "Open AI"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_models_dict_is_populated(self):
        """
        Test that models dict is populated from API.

        Choosing OpenAI as a likely reliable option for an integration test
        """
        provider = Provider.OPENAI.value

        assert isinstance(provider.models, dict)
        assert len(provider.models) > 0

    def test_models_contain_metadata(self):
        """
        Test that models contain proper ModelMetadata instances.

        Choosing OpenAI as a likely reliable option for an integration test
        """
        provider = Provider.OPENAI.value

        # Get any model from the dict
        model_name = next(iter(provider.models))
        model = provider.models[model_name]

        assert isinstance(model, ModelMetadata)
        assert model.name is not None
        assert model.display_name is not None

    def test_get_data_returns_dict(self):
        """Test that get_data returns API data."""
        data = ProviderMetadata.get_data()

        assert data is not None
        assert isinstance(data, dict)
        assert "openai" in data
        assert "openrouter" in data

    def test_get_data_is_cached(self):
        """Test that get_data uses caching."""
        data1 = ProviderMetadata.get_data()
        data2 = ProviderMetadata.get_data()

        assert data1 is data2


class TestProvider:
    def test_enum_values_are_provider_metadata(self):
        """Test that enum values are ProviderMetadata instances."""
        assert all(
            isinstance(provider.value, ProviderMetadata) for provider in Provider
        )

    def test_openai_has_models_with_costs(self):
        """
        Test that OpenAI provider has models with cost data.

        Choosing OpenAI as a likely reliable option for an integration test
        """
        openai = Provider.OPENAI.value

        # Find a model with cost data
        models_with_costs = [
            m
            for m in openai.models.values()
            if m.input_cost is not None and m.output_cost is not None
        ]

        assert len(models_with_costs) > 0
