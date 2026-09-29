"""Smoke tests for the test bench itself, not for the package."""

import pytest
from django.core.management import call_command

from tests.testapp.models import AccordionItem, Category, Page, Product, TextPlugin


@pytest.mark.django_db
def test_fixture_models_can_be_saved_and_related():
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    page = Page.objects.create(title="About", slug="about")
    TextPlugin.objects.create(page=page, body="Hello")
    AccordionItem.objects.create(page=page, title="Question", body="Answer")

    assert category.products.get().name == "Hammer"
    assert page.text_plugins.get().body == "Hello"
    assert page.accordion_items.get().title == "Question"


@pytest.mark.django_db
def test_migration_matches_models():
    call_command("makemigrations", "testapp", "--check", "--dry-run")
