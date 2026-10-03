"""Test fixtures, copied from the prototype's demo app.

These models are the test bench, not code under test: they are copied as
they were, not re-derived test-first. Some deliberately have no ``__str__``,
as in the prototype.
"""

from django.db import models

# --- Simple case: an ordinary Django model -----------------------------


class Category(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self) -> str:
        return self.name

    def get_name_display(self) -> str:
        # Named like Django's display method, but written by hand for a field
        # without choices: the pipeline must not mistake it for a label. It
        # keeps a blank name blank, so only the case of the text tells it apart.
        return self.name.upper()


class Product(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    subtitle = models.CharField(max_length=200, blank=True, null=True)  # noqa: DJ001 -- the test bench needs a field whose value can be None
    price = models.DecimalField(max_digits=8, decimal_places=2)
    category = models.ForeignKey(
        Category, on_delete=models.CASCADE, related_name="products"
    )
    condition = models.CharField(
        max_length=10,
        choices=[("new", "New"), ("used", "Second-hand")],
        default="new",
    )

    def get_absolute_url(self) -> str:
        return f"/products/{self.pk}/"


# --- Scattered content: a Django CMS-like structure ---------------------
# A Page's content is not in a field of Page: it is spread across related
# plugin models.


class Page(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField()

    def get_absolute_url(self) -> str:
        return f"/pages/{self.slug}/"


class TextPlugin(models.Model):
    page = models.ForeignKey(
        Page, related_name="text_plugins", on_delete=models.CASCADE
    )
    body = models.TextField()


class AccordionItem(models.Model):
    page = models.ForeignKey(
        Page, related_name="accordion_items", on_delete=models.CASCADE
    )
    title = models.CharField(max_length=200)
    body = models.TextField()


# --- Third-party field: a subclass of a Django text field ---------------
# A rich-text package typically ships its field as a TextField subclass.


# type-arg: django-stubs makes the field classes generic, but Django's runtime
# classes are not subscriptable, so the base cannot take the type arguments.
class RichTextField(models.TextField):  # type: ignore[type-arg]
    """Stand in for a third-party rich-text field, without the dependency."""


class Article(models.Model):
    body = RichTextField()


# --- CharField subclasses held under neutral names ----------------------
# Each model has a plain CharField and one CharField subclass whose name says
# nothing about its type: only the type can tell the two fields apart.


class EmailHolder(models.Model):
    label = models.CharField(max_length=100)
    extra = models.EmailField()


class URLHolder(models.Model):
    label = models.CharField(max_length=100)
    extra = models.URLField()


class SlugHolder(models.Model):
    label = models.CharField(max_length=100)
    extra = models.SlugField()


# --- A project's own CharField subclass ---------------------------------
# A project or a third-party app may define its own CharField subclass for a
# code or an identifier, such as a country code.


# type-arg: django-stubs makes the field classes generic, but Django's runtime
# classes are not subscriptable, so the base cannot take the type arguments.
class CodeField(models.CharField):  # type: ignore[type-arg]
    """Stand in for a project's code field, such as a country code."""


class CodeHolder(models.Model):
    label = models.CharField(max_length=100)
    extra = CodeField(max_length=2)


# --- A plain CharField as the primary key --------------------------------
# A code such as "FR" held in a plain CharField that is the primary key: only
# its being the primary key says it is an identifier, not content.


class Country(models.Model):
    code = models.CharField(max_length=2, primary_key=True)
    name = models.CharField(max_length=100)


# --- A plain CharField the application fills in -------------------------
# A summary held in a plain CharField with editable=False: the application
# sets it, not the admin form, yet it is content all the same.


class Digest(models.Model):
    title = models.CharField(max_length=200)
    summary = models.CharField(max_length=200, editable=False)


# --- A title declared last ---------------------------------------------
# The title-like field comes after the body in declaration order: only its
# name can put it first.


class Note(models.Model):
    body = models.TextField()
    title = models.CharField(max_length=200)


# --- Title-like fields declared in reverse ------------------------------
# Every title-like name, declared after the body and in the reverse of the
# order they should come in: only a fixed order of names can sort them.


class Panel(models.Model):
    body = models.TextField()
    label = models.CharField(max_length=100)
    heading = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    title = models.CharField(max_length=200)


# --- No text at all -----------------------------------------------------
# Only a number, a date, a boolean and a relation: nothing to guess as text.


class StockLevel(models.Model):
    quantity = models.IntegerField()
    counted_on = models.DateField()
    in_stock = models.BooleanField(default=True)
    product = models.ForeignKey(
        Product, related_name="stock_levels", on_delete=models.CASCADE
    )


# --- No text at all, but a string form ----------------------------------
# Only a number, a date and a relation, like StockLevel, but with a __str__
# computed from the instance's own values.


class Delivery(models.Model):
    quantity = models.IntegerField()
    delivered_on = models.DateField()
    product = models.ForeignKey(
        Product, related_name="deliveries", on_delete=models.CASCADE
    )

    def __str__(self) -> str:
        return f"{self.quantity} delivered on {self.delivered_on.isoformat()}"


# --- Only CharField subclasses -----------------------------------------
# A slug, an e-mail address and a URL, but no plain CharField or TextField:
# text-ish fields, none of them content.


class ContactCard(models.Model):
    handle = models.SlugField()
    email = models.EmailField()
    website = models.URLField()


# --- A related model with several text fields ---------------------------
# A Lesson points to a Topic whose title comes after its summary in
# declaration order, next to a slug; the Topic's string form is its slug,
# not its text.


class Topic(models.Model):
    summary = models.TextField()
    title = models.CharField(max_length=200)
    slug = models.SlugField()

    def __str__(self) -> str:
        return self.slug


class Lesson(models.Model):
    title = models.CharField(max_length=200)
    topic = models.ForeignKey(Topic, related_name="lessons", on_delete=models.CASCADE)


# --- A related model with a field with choices --------------------------
# A Review points to a Product, whose condition is a text field with choices:
# its stored value and its label differ.


class Review(models.Model):
    title = models.CharField(max_length=200)
    product = models.ForeignKey(
        Product, related_name="reviews", on_delete=models.CASCADE
    )


# --- An optional relation -----------------------------------------------
# A Workshop may point to a Topic, or to nothing: its foreign key is nullable.


class Workshop(models.Model):
    title = models.CharField(max_length=200)
    topic = models.ForeignKey(
        Topic,
        related_name="workshops",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )


# --- A many-to-many relation --------------------------------------------
# A Course covers several Topics, and a Topic may belong to several Courses.


class Course(models.Model):
    title = models.CharField(max_length=200)
    topics = models.ManyToManyField(Topic, related_name="courses")


# --- A reverse relation without a related_name --------------------------
# A Remark points to a Note through a foreign key without related_name: the
# Note reaches its Remarks by the default accessor ``remark_set``, while the
# query name of the relation is ``remark``.


class Remark(models.Model):
    note = models.ForeignKey(Note, on_delete=models.CASCADE)
    body = models.TextField()


# --- A one-to-one relation ----------------------------------------------
# A Page may have one PageIntro, or none: the Page reaches it by the reverse
# one-to-one ``intro``, which raises when there is no PageIntro.


class PageIntro(models.Model):
    page = models.OneToOneField(Page, related_name="intro", on_delete=models.CASCADE)
    body = models.TextField()
