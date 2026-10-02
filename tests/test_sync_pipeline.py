from collections.abc import Callable

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_model_rag import AlreadyRegistered, NotRegistered, SyncPipeline, rag
from tests.testapp.models import Category, Product


def create_product(*, name: str, description: str, price: str) -> Product:
    """Create a product in a category of its own."""
    category = Category.objects.create(name="Tools")
    return Product.objects.create(
        name=name, description=description, price=price, category=category
    )


def test_pipeline_without_registered_model_produces_no_document() -> None:
    assert SyncPipeline().run() == []


@pytest.mark.django_db
def test_unregistered_model_produces_no_document() -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    rag.register(Product, fields=["name"])
    rag.unregister(Product)

    assert SyncPipeline().run() == []


@pytest.mark.django_db
def test_registered_model_produces_a_document_from_its_declared_field(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["name"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Hammer"]


@pytest.mark.django_db
def test_document_text_joins_declared_fields_with_a_blank_line(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.text == "Hammer\n\nDrives nails."


@pytest.mark.django_db
def test_document_text_follows_the_declared_field_order(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["description", "name"])

    [document] = SyncPipeline().run()

    assert document.text == "Drives nails.\n\nHammer"


@pytest.mark.django_db
def test_document_text_leaves_out_a_declared_field_with_an_empty_value(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="", price="9.90")
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.text == "Hammer"


@pytest.mark.django_db
def test_document_text_leaves_out_a_declared_field_with_a_blank_value(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="  \n ", price="9.90")
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.text == "Hammer"


@pytest.mark.django_db
def test_document_text_strips_the_surrounding_whitespace_of_each_value(
    register: Callable[..., None],
) -> None:
    create_product(name="  Hammer\n", description="\nDrives nails.  ", price="9.90")
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.text == "Hammer\n\nDrives nails."


@pytest.mark.django_db
def test_document_text_leaves_out_a_declared_field_whose_value_is_none(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["name", "subtitle"])

    [document] = SyncPipeline().run()

    assert document.text == "Hammer"


@pytest.mark.django_db
def test_instance_without_text_in_its_declared_fields_produces_no_document(
    register: Callable[..., None],
) -> None:
    named = Category.objects.create(name="Tools")
    Category.objects.create(name="")
    register(Category, fields=["name"])

    documents = SyncPipeline().run()

    assert [document.source_pk for document in documents] == [named.pk]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_documents_come_in_ascending_primary_key_order(
    register: Callable[..., None],
) -> None:
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    kitchen = Category.objects.create(name="Kitchen")
    register(Category, fields=["name"])

    documents = SyncPipeline().run()

    assert [document.source_pk for document in documents] == [
        tools.pk,
        garden.pk,
        kitchen.pk,
    ]


@pytest.mark.django_db
def test_documents_of_several_models_come_grouped_in_registration_order(
    register: Callable[..., None],
) -> None:
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    hammer = Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=tools
    )
    rake = Product.objects.create(
        name="Rake", description="Gathers leaves.", price="14.50", category=garden
    )
    register(Product, fields=["name"])
    register(Category, fields=["name"])

    documents = SyncPipeline().run()

    assert [(document.source_model, document.source_pk) for document in documents] == [
        ("product", hammer.pk),
        ("product", rake.pk),
        ("category", tools.pk),
        ("category", garden.pk),
    ]


@pytest.mark.django_db
def test_document_title_is_the_value_of_the_first_declared_field(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.title == "Hammer"


@pytest.mark.django_db
def test_document_title_follows_the_declared_field_order(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["description", "name"])

    [document] = SyncPipeline().run()

    assert document.title == "Drives nails."


@pytest.mark.django_db
def test_document_title_is_empty_when_the_first_declared_field_is_blank(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="   ", price="9.90")
    register(Product, fields=["description", "name"])

    [document] = SyncPipeline().run()

    assert document.title == ""


@pytest.mark.django_db
def test_document_title_is_empty_when_the_first_declared_field_is_none(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["subtitle", "name"])

    [document] = SyncPipeline().run()

    assert document.title == ""


@pytest.mark.django_db
def test_document_title_strips_the_surrounding_whitespace_of_its_value(
    register: Callable[..., None],
) -> None:
    create_product(name="  Hammer\n", description="Drives nails.", price="9.90")
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.title == "Hammer"


@pytest.mark.django_db
def test_document_title_is_the_value_of_the_declared_title_field(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["description", "name"], title_field="name")

    [document] = SyncPipeline().run()

    assert document.title == "Hammer"


@pytest.mark.django_db
def test_title_field_outside_the_declared_fields_gives_the_title_not_the_text(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["description"], title_field="name")

    [document] = SyncPipeline().run()

    assert (document.title, document.text) == ("Hammer", "Drives nails.")


@pytest.mark.django_db
def test_document_title_is_empty_when_the_declared_title_field_is_none(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["name"], title_field="subtitle")

    [document] = SyncPipeline().run()

    assert document.title == ""


@pytest.mark.django_db
def test_document_carries_the_source_of_its_product(
    register: Callable[..., None],
) -> None:
    product = create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["name"])

    [document] = SyncPipeline().run()

    assert (
        document.source_app_label,
        document.source_model,
        document.source_pk,
    ) == ("testapp", "product", product.pk)


@pytest.mark.django_db
def test_document_carries_the_source_of_its_own_model(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(pk=42, name="Tools")
    register(Category, fields=["name"])

    [document] = SyncPipeline().run()

    assert (
        document.source_app_label,
        document.source_model,
        document.source_pk,
    ) == ("testapp", "category", category.pk)


@pytest.mark.django_db
def test_document_text_holds_the_string_form_of_a_non_text_field(
    register: Callable[..., None],
) -> None:
    create_product(name="Drill", description="Bores holes.", price="149.00")
    register(Product, fields=["name", "price"])

    [document] = SyncPipeline().run()

    assert document.text == "Drill\n\n149.00"


@pytest.mark.django_db
def test_document_text_keeps_a_declared_field_whose_value_is_zero(
    register: Callable[..., None],
) -> None:
    create_product(name="Sticker", description="A free gift.", price="0")
    register(Product, fields=["name", "price"])

    [document] = SyncPipeline().run()

    assert document.text == "Sticker\n\n0.00"


@pytest.mark.django_db
def test_document_text_holds_the_label_of_a_field_with_choices(
    register: Callable[..., None],
) -> None:
    product = create_product(name="Hammer", description="Drives nails.", price="9.90")
    product.condition = "used"
    product.save()
    register(Product, fields=["name", "condition"])

    [document] = SyncPipeline().run()

    assert document.text == "Hammer\n\nSecond-hand"


def test_registering_a_field_the_model_does_not_have_names_it_in_the_error(
    register: Callable[..., None],
) -> None:
    with pytest.raises(ImproperlyConfigured, match="nmae"):
        register(Product, fields=["nmae"])


def test_registering_a_relation_field_names_it_in_the_error(
    register: Callable[..., None],
) -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bcategory\b"):
        register(Product, fields=["name", "category"])


def test_registering_a_reverse_relation_names_it_in_the_error(
    register: Callable[..., None],
) -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bproducts\b"):
        register(Category, fields=["name", "products"])


def test_registering_a_title_field_the_model_does_not_have_names_it_in_the_error(
    register: Callable[..., None],
) -> None:
    with pytest.raises(ImproperlyConfigured, match="nmae"):
        register(Product, fields=["name"], title_field="nmae")


def test_registering_a_relation_as_title_field_names_it_in_the_error(
    register: Callable[..., None],
) -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bcategory\b"):
        register(Product, fields=["name"], title_field="category")


def test_registering_a_model_without_any_declared_field_fails(
    register: Callable[..., None],
) -> None:
    with pytest.raises(ImproperlyConfigured):
        register(Product, fields=[])


def test_registering_a_single_field_name_instead_of_a_list_fails() -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bfields\b.*\blist\b"):
        # A bare string is the slip under test: the type checker rightly
        # rejects it, and rag.register is called directly so it does.
        rag.register(Product, fields="name")  # type: ignore[arg-type]


def test_registering_an_unordered_set_of_field_names_fails() -> None:
    with pytest.raises(ImproperlyConfigured):
        # A set has no stable order to give the text and the title: the type
        # checker rightly rejects it, and rag.register is called directly so
        # it does.
        rag.register(Product, fields={"name", "description"})  # type: ignore[arg-type]


def test_registering_a_one_shot_generator_of_field_names_fails() -> None:
    with pytest.raises(ImproperlyConfigured):
        # A generator is spent by the checks and would be registered empty:
        # the type checker rightly rejects it, and rag.register is called
        # directly so it does.
        rag.register(Product, fields=(name for name in ["name"]))  # type: ignore[arg-type]


@pytest.mark.django_db
def test_declared_fields_given_as_a_tuple_give_the_same_text_as_a_list() -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    # rag.register is called directly so the type checker sees the tuple.
    rag.register(Product, fields=("name", "description"))

    [document] = SyncPipeline().run()

    assert document.text == "Hammer\n\nDrives nails."


@pytest.mark.django_db
def test_registering_a_model_twice_fails_and_keeps_the_first_registration(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    register(Product, fields=["name"])

    with pytest.raises(AlreadyRegistered):
        register(Product, fields=["description"])

    [document] = SyncPipeline().run()
    assert document.text == "Hammer"


@pytest.mark.django_db
def test_changing_the_declared_fields_list_after_registering_has_no_effect(
    register: Callable[..., None],
) -> None:
    create_product(name="Hammer", description="Drives nails.", price="9.90")
    fields = ["name"]
    register(Product, fields=fields)

    fields.append("description")

    [document] = SyncPipeline().run()
    assert document.text == "Hammer"


def test_unregistering_a_model_that_is_not_registered_fails() -> None:
    with pytest.raises(NotRegistered):
        rag.unregister(Product)
