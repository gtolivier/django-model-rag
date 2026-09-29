"""Test fixtures, copied from the prototype's demo app.

These models are the test bench, not code under test: they are copied as
they were, not re-derived test-first. Some deliberately have no ``__str__``,
as in the prototype.
"""

from django.db import models

# --- Simple case: an ordinary Django model -----------------------------


class Category(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class Product(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=8, decimal_places=2)
    category = models.ForeignKey(
        Category, on_delete=models.CASCADE, related_name="products"
    )

    def get_absolute_url(self):
        return f"/products/{self.pk}/"


# --- Scattered content: a Django CMS-like structure ---------------------
# A Page's content is not in a field of Page: it is spread across related
# plugin models.


class Page(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField()

    def get_absolute_url(self):
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
