"""Test fixtures, copied from the prototype's demo app.

These models are the test bench, not code under test: they are copied as
they were, not re-derived test-first. Some deliberately have no ``__str__``,
as in the prototype.
"""

from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.db.models.manager import BaseManager
from django.urls import NoReverseMatch

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


# --- A foreign key to a model with an optional relation -----------------
# A Session points to a Workshop, which may point to a Topic through its
# SET_NULL foreign key: a lookup path from a Session reaches the Topic two
# links deep, the last one nulled when the Topic is deleted.


class Session(models.Model):
    title = models.CharField(max_length=200)
    workshop = models.ForeignKey(
        Workshop, related_name="sessions", on_delete=models.CASCADE
    )


# --- A many-to-many relation --------------------------------------------
# A Course covers several Topics, and a Topic may belong to several Courses.


class Course(models.Model):
    title = models.CharField(max_length=200)
    topics = models.ManyToManyField(Topic, related_name="courses")


# --- A multi-table child of a model with a many-to-many -----------------
# A MasterClass is a Course with an instructor of its own, in a table of its
# own: it inherits the Course's many-to-many ``topics``, whose links name its
# Course row, yet Django sends m2m_changed with the MasterClass, not a Course,
# as its instance.


class MasterClass(Course):
    instructor = models.CharField(max_length=200)


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


# --- A one-to-one relation with its own query name ----------------------
# A Supplier may have one SupplierProfile, or none: the Supplier reaches it by
# the reverse one-to-one accessor ``profile``, while the query name of the
# relation, set by related_query_name, is ``supplier_profile``.


class Supplier(models.Model):
    name = models.CharField(max_length=200)


class SupplierProfile(models.Model):
    supplier = models.OneToOneField(
        Supplier,
        related_name="profile",
        related_query_name="supplier_profile",
        on_delete=models.CASCADE,
    )
    body = models.TextField()


# --- A foreign key to a model with a reverse one-to-one -----------------
# A SupplierOrder points to a Supplier, which reaches its SupplierProfile by
# a reverse one-to-one whose query name differs from its accessor: a lookup
# path from a SupplierOrder crosses that relation past its first link.


class SupplierOrder(models.Model):
    reference = models.CharField(max_length=100)
    supplier = models.ForeignKey(
        Supplier, related_name="orders", on_delete=models.CASCADE
    )


# --- Multi-table inheritance --------------------------------------------
# A FeaturedProduct is a Product with a tagline of its own: it inherits the
# Product's fields and relations, and Django links it to its parent row by
# an automatic one-to-one field, ``product_ptr``.


class FeaturedProduct(Product):
    tagline = models.CharField(max_length=200)


# --- A foreign key to a multi-table child -------------------------------
# A Spotlight points to a FeaturedProduct, which reaches its Product row by
# the implicit parent link ``product_ptr``: a lookup path from a Spotlight
# crosses that link past its first one.


class Spotlight(models.Model):
    title = models.CharField(max_length=200)
    featured_product = models.ForeignKey(
        FeaturedProduct, related_name="spotlights", on_delete=models.CASCADE
    )


# --- Multi-table inheritance under a primary key of its own -------------
# A ClearanceProduct is a Product whose child row has a primary key of its
# own, a code such as "CLR-1", next to an explicit parent link, ``product``:
# its primary key is not the Product's, and a code can never equal the
# Product's integer primary key.


class ClearanceProduct(Product):
    code = models.CharField(max_length=20, primary_key=True)
    product = models.OneToOneField(
        Product,
        parent_link=True,
        related_name="clearance",
        on_delete=models.CASCADE,
    )


# --- A related model whose manager is not a Manager ---------------------
# A Step's default manager is built with BaseManager.from_queryset(), as some
# third-party apps do: its class derives from BaseManager, not from Manager. A
# Recipe reaches its Steps by the reverse foreign key ``steps``.


class StepQuerySet(models.QuerySet["Step"]):
    """Stand in for a third-party app's own queryset."""


class Recipe(models.Model):
    title = models.CharField(max_length=200)


class Step(models.Model):
    recipe = models.ForeignKey(Recipe, related_name="steps", on_delete=models.CASCADE)
    body = models.TextField()

    objects = BaseManager.from_queryset(StepQuerySet)()


# --- A generic relation -------------------------------------------------
# A Tag may be attached to an instance of any model, through a generic foreign
# key built from a content type and an object id. A Photo reaches its Tags by
# the generic relation ``tags``. A Tag's weight is a number, not text.


class Tag(models.Model):
    label = models.CharField(max_length=100)
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey("content_type", "object_id")
    # Not text, and not part of the link back: a followed generic relation
    # need never load it. Its default leaves tags created without it valid.
    weight = models.PositiveIntegerField(default=0)


class Photo(models.Model):
    title = models.CharField(max_length=200)
    tags = GenericRelation(Tag)


# --- Foreign keys to the models of a generic relation -------------------
# A Pin points to a Photo, which reaches its Tags by the generic relation
# ``tags``, and to a Tag, which reaches its object by the generic foreign key
# ``content_object``: a lookup path from a Pin crosses either past its first
# link.


class Pin(models.Model):
    note = models.CharField(max_length=200)
    photo = models.ForeignKey(Photo, related_name="pins", on_delete=models.CASCADE)
    tag = models.ForeignKey(Tag, related_name="pins", on_delete=models.CASCADE)


# --- A language held under a name of its own ----------------------------
# A Bulletin keeps its language code, such as "fr", in a field named
# ``locale``: not one of the names a language field would be guessed by.


class Bulletin(models.Model):
    title = models.CharField(max_length=200)
    locale = models.CharField(max_length=10)


# --- A language held in a field with choices ----------------------------
# A Leaflet keeps its language code in a field with choices, under a name of
# its own like Bulletin's: its stored code, such as "fr", and its label, such
# as "Français", differ.


class Leaflet(models.Model):
    title = models.CharField(max_length=200)
    locale = models.CharField(
        max_length=10,
        choices=[("fr", "Français"), ("en", "English")],
    )


# --- An optional language -----------------------------------------------
# A Memo keeps its language code under a name of its own like Bulletin's, but
# in a nullable field: its language may be unknown, stored as None.


class Memo(models.Model):
    title = models.CharField(max_length=200)
    locale = models.CharField(max_length=10, blank=True, null=True)  # noqa: DJ001 -- the test bench needs a language field whose value can be None


# --- A language held under the conventional name ------------------------
# A Notice keeps its language code, such as "fr", in its own field named
# ``language``: the name a language field would be guessed by.


class Notice(models.Model):
    title = models.CharField(max_length=200)
    language = models.CharField(max_length=10)


# --- A language held under the other conventional name ------------------
# A Circular keeps its language code, such as "fr", in its own field named
# ``language_code``, and has no field named ``language``: the other name a
# language field would be guessed by.


class Circular(models.Model):
    title = models.CharField(max_length=200)
    language_code = models.CharField(max_length=10)


# --- A language held under the short conventional name ------------------
# A Dispatch keeps its language code, such as "fr", in its own field named
# ``lang``, and has no field named ``language`` or ``language_code``: the
# short name a language field would be guessed by.


class Dispatch(models.Model):
    title = models.CharField(max_length=200)
    lang = models.CharField(max_length=10)


# --- A language held under two conventional names -----------------------
# A Gazette has both a field named ``language`` and one named
# ``language_code``: only an order of names can say which one is its language,
# whichever of the two an instance fills.


class Gazette(models.Model):
    title = models.CharField(max_length=200)
    language = models.CharField(max_length=10, blank=True)
    language_code = models.CharField(max_length=10, blank=True)


# --- A relation under the conventional name -----------------------------
# An Announcement's field named ``language`` is a foreign key to a Language,
# whose string form is its code, such as "fr": the name a language field
# would be guessed by, but held by a relation, not by an own value.


class Language(models.Model):
    code = models.CharField(max_length=10)

    def __str__(self) -> str:
        return self.code


class Announcement(models.Model):
    title = models.CharField(max_length=200)
    language = models.ForeignKey(
        Language, related_name="announcements", on_delete=models.CASCADE
    )


# --- A language held by a related model ---------------------------------
# An Excerpt has no language of its own: it may point to a Notice, whose own
# field ``language`` holds it, or to nothing, its foreign key being nullable.


class Excerpt(models.Model):
    title = models.CharField(max_length=200)
    notice = models.ForeignKey(
        Notice,
        related_name="excerpts",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )


# --- A language held by a related field with choices --------------------
# A Clipping has no language of its own: it may point to a Leaflet, whose
# field ``locale`` with choices holds it, or to nothing, its foreign key being
# nullable.


class Clipping(models.Model):
    title = models.CharField(max_length=200)
    leaflet = models.ForeignKey(
        Leaflet,
        related_name="clippings",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )


# --- A get_absolute_url that fails --------------------------------------
# A Brochure's get_absolute_url raises NoReverseMatch, as one reversing a URL
# name the project does not define would: a bug of the project's own.


class Brochure(models.Model):
    title = models.CharField(max_length=200)

    def get_absolute_url(self) -> str:
        message = "Reverse for 'brochure-detail' not found."
        raise NoReverseMatch(message)


# --- A get_absolute_url that gives nothing ------------------------------
# A Flyer's get_absolute_url returns None, as one for an instance with no page
# of its own might: a URL that is not there.


class Flyer(models.Model):
    title = models.CharField(max_length=200)

    def get_absolute_url(self) -> str | None:
        return None


# --- A URL held in an own field -----------------------------------------
# A Bookmark has no get_absolute_url: its URL, relative such as "/docs/a/" or
# absolute such as "https://example.com/b", is stored in its own plain
# CharField named ``link``, not one a URL would be guessed by.


class Bookmark(models.Model):
    title = models.CharField(max_length=200)
    link = models.CharField(max_length=200)


# --- A URL held in an own field, next to a get_absolute_url -------------
# A Pamphlet has both a get_absolute_url and its own nullable CharField named
# ``link``: its link may be filled, blank or unknown, stored as None, while
# get_absolute_url always gives a URL.


class Pamphlet(models.Model):
    title = models.CharField(max_length=200)
    link = models.CharField(max_length=200, blank=True, null=True)  # noqa: DJ001 -- the test bench needs a url field whose value can be None

    def get_absolute_url(self) -> str:
        return f"/pamphlets/{self.pk}/"


# --- A URL held by a related model --------------------------------------
# A Citation has no URL of its own and no get_absolute_url: it may point to a
# Bookmark, whose own field ``link`` holds it, or to nothing, its foreign key
# being nullable.


class Citation(models.Model):
    title = models.CharField(max_length=200)
    bookmark = models.ForeignKey(
        Bookmark,
        related_name="citations",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )


# --- A URL held in an own field with choices ----------------------------
# A Shortcut has no get_absolute_url: its URL is stored in its own CharField
# named ``link``, which has choices: its stored URL, such as "/docs/", and its
# label, such as "Documentation", differ.


class Shortcut(models.Model):
    title = models.CharField(max_length=200)
    link = models.CharField(
        max_length=200,
        choices=[("/docs/", "Documentation"), ("/faq/", "FAQ")],
    )


# --- A foreign key to a unique column other than the primary key --------
# A Shelf points to a Warehouse by the Warehouse's unique code, a slug, not by
# its primary key: the Warehouse reaches its Shelves by the reverse foreign key
# ``shelves``, matched to it by that code.


class Warehouse(models.Model):
    name = models.CharField(max_length=200)
    code = models.SlugField(unique=True)


class Shelf(models.Model):
    label = models.CharField(max_length=100)
    warehouse = models.ForeignKey(
        Warehouse,
        to_field="code",
        related_name="shelves",
        on_delete=models.CASCADE,
    )


# --- A foreign key to a unique, nullable column -------------------------
# A Bin may point to a Depot by the Depot's unique code, a slug, not by its
# primary key, or to nothing, its foreign key being nullable. A Depot's code
# is nullable too: a Depot may have none yet, stored as None. The Depot
# reaches its Bins by the reverse foreign key ``bins``, matched to it by that
# code.


class Depot(models.Model):
    name = models.CharField(max_length=200)
    code = models.SlugField(unique=True, blank=True, null=True)


class Bin(models.Model):
    label = models.CharField(max_length=100)
    depot = models.ForeignKey(
        Depot,
        to_field="code",
        related_name="bins",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )


# --- A default manager that already follows a foreign key --------------
# A Listing's default manager follows its foreign key ``category`` with
# select_related(), as a project may do so that every listing it shows comes
# with its category: whatever loads Listings through it joins the category.
# Its URL is stored in its own CharField named ``link``.


class ListingManager(models.Manager["Listing"]):
    def get_queryset(self) -> models.QuerySet["Listing"]:
        return super().get_queryset().select_related("category")


class Listing(models.Model):
    title = models.CharField(max_length=200)
    link = models.CharField(max_length=200)
    category = models.ForeignKey(
        Category, related_name="listings", on_delete=models.CASCADE
    )

    objects = ListingManager()


# --- A default manager that follows a foreign key two links deep -------
# An Offer's default manager follows its foreign key ``product``, then the
# Product's own foreign key ``category``, with select_related(), as a project
# may do so that every offer it shows comes with its product and the product's
# category: whatever loads Offers through it joins both.


class OfferManager(models.Manager["Offer"]):
    def get_queryset(self) -> models.QuerySet["Offer"]:
        return super().get_queryset().select_related("product__category")


class Offer(models.Model):
    title = models.CharField(max_length=200)
    product = models.ForeignKey(
        Product, related_name="offers", on_delete=models.CASCADE
    )

    objects = OfferManager()


# --- A related model whose default manager follows a foreign key --------
# A Showroom reaches its Exhibits by the reverse foreign key ``exhibits``. An
# Exhibit's default manager follows its other foreign key, ``category``, with
# select_related(), as a project may do so that every exhibit it shows comes
# with its category: whatever loads Exhibits through it joins the category.


class Showroom(models.Model):
    name = models.CharField(max_length=200)


class ExhibitManager(models.Manager["Exhibit"]):
    def get_queryset(self) -> models.QuerySet["Exhibit"]:
        return super().get_queryset().select_related("category")


class Exhibit(models.Model):
    showroom = models.ForeignKey(
        Showroom, related_name="exhibits", on_delete=models.CASCADE
    )
    label = models.CharField(max_length=200)
    category = models.ForeignKey(
        Category, related_name="exhibits", on_delete=models.CASCADE
    )

    objects = ExhibitManager()


# --- Two foreign keys to the same model ---------------------------------
# A Match points to a Team twice, by its home team and by its away team: the
# Team reaches its Matches by two reverse foreign keys, ``home_matches`` and
# ``away_matches``, one per foreign key. A Match may also belong to a
# Tournament, by a nullable foreign key: the Tournament reaches its Matches by
# a third reverse foreign key, ``matches``, from another model than the Team.


class Team(models.Model):
    name = models.CharField(max_length=200)


class Tournament(models.Model):
    name = models.CharField(max_length=200)


class Match(models.Model):
    title = models.CharField(max_length=200)
    home_team = models.ForeignKey(
        Team, related_name="home_matches", on_delete=models.CASCADE
    )
    away_team = models.ForeignKey(
        Team, related_name="away_matches", on_delete=models.CASCADE
    )
    tournament = models.ForeignKey(
        Tournament, null=True, related_name="matches", on_delete=models.SET_NULL
    )


# --- A many-to-many through a foreign key to a unique column ------------
# A Guild reaches its Craftsmen by the many-to-many ``members``, through a
# Membership model of its own whose foreign key points to the Guild by the
# Guild's unique code, a slug, not by its primary key: the join table matches
# each membership to its Guild by that code.


class Craftsman(models.Model):
    name = models.CharField(max_length=200)


class Guild(models.Model):
    name = models.CharField(max_length=200)
    code = models.SlugField(unique=True)
    members = models.ManyToManyField(
        Craftsman, through="Membership", related_name="guilds"
    )


class Membership(models.Model):
    guild = models.ForeignKey(Guild, to_field="code", on_delete=models.CASCADE)
    craftsman = models.ForeignKey(Craftsman, on_delete=models.CASCADE)


# --- A reverse many-to-many through a foreign key to a unique column ----
# A Musician reaches its Bands by the reverse many-to-many ``bands``: the Band
# declares the many-to-many ``musicians``, through an Engagement model of its
# own whose foreign key points to the Musician by the Musician's unique handle,
# a slug, not by its primary key: the join table matches each engagement to its
# Musician by that handle.


class Musician(models.Model):
    name = models.CharField(max_length=200)
    handle = models.SlugField(unique=True)


class Band(models.Model):
    name = models.CharField(max_length=200)
    musicians = models.ManyToManyField(
        Musician, through="Engagement", related_name="bands"
    )


class Engagement(models.Model):
    band = models.ForeignKey(Band, on_delete=models.CASCADE)
    musician = models.ForeignKey(Musician, to_field="handle", on_delete=models.CASCADE)


# --- A non-symmetrical many-to-many from a model to itself --------------
# A Person reaches the Persons mentoring them by the many-to-many ``mentors``,
# and those they mentor by its reverse side, ``mentees``: both foreign keys of
# the join table point to Person, and a link from one Person to another is not
# a link back.


class Person(models.Model):
    name = models.CharField(max_length=200)
    mentors = models.ManyToManyField("self", symmetrical=False, related_name="mentees")


# --- A symmetrical many-to-many from a model to itself ------------------
# A Friend reaches its Friends by the many-to-many ``friends``, symmetrical as
# Django makes a many-to-many to "self" by default: a link from one Friend to
# another is a link back, written as two rows of the join table, and its
# reverse side is Django's hidden relation, with no accessor of its own.


class Friend(models.Model):
    name = models.CharField(max_length=200)
    friends = models.ManyToManyField("self")


# --- A proxy model ------------------------------------------------------
# A CategoryProxy is a Category under another class, with no table of its own:
# saving one writes the Category's row, yet Django sends the save's signals
# with the proxy, not Category, as their sender.


class CategoryProxy(Category):
    class Meta:
        proxy = True


# --- A foreign key to a proxy model -------------------------------------
# A Banner points to a CategoryProxy, not to Category: its foreign key reaches
# the Category's row, yet names the proxy as its model, while saving a plain
# Category sends the save's signals with Category as their sender.


class Banner(models.Model):
    title = models.CharField(max_length=200)
    category = models.ForeignKey(
        CategoryProxy, related_name="banners", on_delete=models.CASCADE
    )


# --- A proxy of a followed model ----------------------------------------
# A TextPluginProxy is a TextPlugin under another class: saving or deleting
# one writes or deletes the TextPlugin's row, which its Page may follow, yet
# Django sends the signals with the proxy, not TextPlugin, as their sender.


class TextPluginProxy(TextPlugin):
    class Meta:
        proxy = True


# --- A proxy of a model followed through SET_NULL -----------------------
# A TopicProxy is a Topic under another class: deleting one deletes the
# Topic's row, which a Workshop may follow through its SET_NULL foreign key,
# yet Django sends the delete's signals with the proxy, not Topic, as their
# sender.


class TopicProxy(Topic):
    class Meta:
        proxy = True


# --- A many-to-many to a proxy model ------------------------------------
# A Curriculum covers several SubjectProxies, not Subjects: its many-to-many
# reaches the Subject's rows, yet names the proxy as its model, and Django
# sends m2m_changed with the proxy, not Subject, as the model of the links'
# other side. Subject has a model of its own, so that no other model's delete
# gains the delete of its links.


class Subject(models.Model):
    title = models.CharField(max_length=200)


class SubjectProxy(Subject):
    class Meta:
        proxy = True


class Curriculum(models.Model):
    title = models.CharField(max_length=200)
    subjects = models.ManyToManyField(SubjectProxy, related_name="curricula")


# --- A SET_NULL foreign key to a proxy model ----------------------------
# A Meetup may point to a ThemeProxy, not to Theme, or to nothing: its
# foreign key reaches the Theme's row and is set to null when that row is
# deleted, yet names the proxy as its model, while deleting a plain Theme
# sends the delete's signals with Theme as their sender. Theme has a model
# of its own, so that no other model's delete gains the Meetup's update.


class Theme(models.Model):
    name = models.CharField(max_length=200)


class ThemeProxy(Theme):
    class Meta:
        proxy = True


class Meetup(models.Model):
    title = models.CharField(max_length=200)
    theme = models.ForeignKey(
        ThemeProxy,
        related_name="meetups",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )


# --- A multi-table child of a followed model ----------------------------
# An Album reaches its Tracks by the reverse foreign key ``tracks``. A
# BonusTrack is a Track with a note of its own, in a table of its own: saving
# one writes a Track row, which its Album may follow, yet Django sends the
# save's signals with BonusTrack, not Track, as their sender.


class Album(models.Model):
    title = models.CharField(max_length=200)


class Track(models.Model):
    album = models.ForeignKey(Album, related_name="tracks", on_delete=models.CASCADE)
    title = models.CharField(max_length=200)


class BonusTrack(Track):
    note = models.CharField(max_length=200)


# --- A multi-column relation --------------------------------------------
# A Seminar points to a Venue by two columns of its own, matched to the
# Venue's city and name, unique together, through a ForeignObject: a relation
# with no database column of its own. The Venue reaches its Seminars by the
# reverse relation ``seminars``, matched to it by both columns.


class Venue(models.Model):
    city = models.CharField(max_length=100)
    name = models.CharField(max_length=200)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["city", "name"], name="testapp_venue_city_name_unique"
            )
        ]


class Seminar(models.Model):
    title = models.CharField(max_length=200)
    venue_city = models.CharField(max_length=100)
    venue_name = models.CharField(max_length=200)
    venue = models.ForeignObject(
        Venue,
        on_delete=models.CASCADE,
        from_fields=["venue_city", "venue_name"],
        to_fields=["city", "name"],
        related_name="seminars",
    )


# --- A foreign key to a model with a multi-column relation --------------
# A Talk points to a Seminar by a plain foreign key, and the Seminar points to
# its Venue through the multi-column ForeignObject ``venue``: a lookup path
# from a Talk crosses that relation past its first link.


class Talk(models.Model):
    title = models.CharField(max_length=200)
    seminar = models.ForeignKey(Seminar, related_name="talks", on_delete=models.CASCADE)


# --- A multi-column relation over integer columns -----------------------
# A Booking points to a Room by two integer columns of its own, matched to the
# Room's building and number, unique together, through a ForeignObject, like
# a Seminar to its Venue but over integers rather than text. The Room reaches
# its Bookings by the reverse relation ``bookings``, matched to it by both
# columns.


class Room(models.Model):
    building = models.IntegerField()
    number = models.IntegerField()
    name = models.CharField(max_length=200)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["building", "number"],
                name="testapp_room_building_number_unique",
            )
        ]


class Booking(models.Model):
    purpose = models.CharField(max_length=200)
    room_building = models.IntegerField()
    room_number = models.IntegerField()
    room = models.ForeignObject(
        Room,
        on_delete=models.CASCADE,
        from_fields=["room_building", "room_number"],
        to_fields=["building", "number"],
        related_name="bookings",
    )
