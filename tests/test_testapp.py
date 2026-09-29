"""Smoke test for the test bench itself, not for the package."""

import pytest

from tests.testapp.models import Category, Page, Product, TextPlugin


@pytest.mark.django_db
def test_fixture_models_can_be_saved_and_related():
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    page = Page.objects.create(title="About", slug="about")
    TextPlugin.objects.create(page=page, body="Hello")

    assert category.products.get().name == "Hammer"
    assert page.text_plugins.get().body == "Hello"
