"""The test bench's ``model_rag`` module, found by autodiscovery at startup.

It registers nothing on purpose: the tests register the test bench's models
themselves, and a registration made here would make theirs fail with
``AlreadyRegistered``. Autodiscovery is observed through ``sys.modules``.
"""
