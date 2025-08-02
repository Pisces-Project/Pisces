.. _models_overview:
===================================
Astrophysics Models in Pisces
===================================

The most important part of the Pisces ecosystem is its models. Pisces
models are used to represent astrophysical systems ranging from simple
dark matter halos to fully realized disk galaxies, stellar models, and
more.

Behind the scenes, a model is generally an abstract wrapper around
a number of physics computations to allow a user to go from a set of
input quantities (parameters, profiles, etc.) to a set of **fields**,
representing the physical data of the model. The complexity of
this process and its numerical cost is dependent on the nature of the
model.

One of the most important elements of Pisces models is their extensibility.
Recognizing that

1. Astrophysical models are constantly changing and becoming more
   sophisticated to incorporate a larger set of relevant processes, and
2. Many models require domain-specific expertise which could not possibly
   be provided by a single developer,

we have designed the :mod:`models` module to be not only extremely flexible, but
also quite easy to develop new code from. In many cases, a user might begin
their scientific workflow by building a custom model class and relying on Pisces
model-interacting infrastructure to perform their science. For details on
writing new models, see :ref:`models_development`.

Modeling Basics
---------------

What Exactly is a Pisces Model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Structure of a Model
````````````````````````

Model Fields
````````````

Basic Usage
^^^^^^^^^^^

Creating and Opening Models
````````````````````````````

Accessing Model Data
````````````````````

Manipulating a Model
````````````````````

Extension Hooks
---------------
