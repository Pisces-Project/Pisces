"""Pisces model hook framework.

This module defines the infrastructure for attaching modular, reusable
components—called *hooks*—to Pisces models. Hooks encapsulate specialized
behavior such as sampling, diagnostics, or derived computation, and can be
selectively enabled or disabled on a per-model basis.

Usage
-----
To use a hook, mix it into a model class and optionally configure it via
class-level attributes. Hooks are automatically detected and can be accessed
via the tools in `_HookTools`.

Examples
--------
.. code-block:: python

    class MyModel(BaseModel, MyCustomHook):
        __MyCustomHook_HOOK_ENABLED__ = True
        _MyCustomHook_option = 42


    model = MyModel()
    model.get_active_hooks()  # -> ['my_custom_hook']

Design Notes
------------
Hooks are designed to be:

- **Non-intrusive**: They extend model behavior without modifying the base logic.
- **Declarative**: They rely on class-level flags and naming conventions for control.
- **Discoverable**: They support standardized metadata for reflection and documentation.

For more details on creating or customizing hooks, see the docstring for `BaseHook`.

"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pisces.particles.base import ParticleDataset


# -------------------------------------------------- #
# Template Class                                     #
# -------------------------------------------------- #
# This is the base class / template for all hook classes. This
# should be the starting point for implementing any new hook classes in
# this module.
class BaseHook:
    """Base class for all Pisces model hooks.

    This abstract base class defines a standardized interface and a set of
    conventions for modular "hooks" that can be mixed into Pisces model classes.
    Hooks encapsulate auxiliary logic such as sampling, initialization,
    diagnostics, or derived calculations, and are used to extend models
    non-intrusively without polluting the core behavior of the model.

    .. hint::

        An archetypal example of a hook is a particle sampler, which
        provides methods for generating or sampling particles from a model.
        This might not be possible in all models, but it is a common
        use-case and therefore can be mixed into models that support it.

    Hooks inheriting from this base class gain access to a standardized structure
    for declaration, configuration, and integration.

    Structure and Conventions
    -------------------------

    Hooks follow a set of naming and structural conventions to ensure that they
    are clearly organized, easily discoverable, and do not interfere with model
    internals unless explicitly intended.

    Feature Status Flag
    ^^^^^^^^^^^^^^^^^^^
    Each hook can be selectively enabled or disabled at the model level using a
    class-level flag with the format:

    .. code-block:: python

        __<HookClassName>_HOOK_ENABLED__ = True  # or False

    For example:

    .. code-block:: python

        class MyClusterModel(BaseModel):
            __SphericalGalaxyClusterParticleHook_HOOK_ENABLED__ = False

    This flag allows models to opt out of specific hook functionality without
    modifying the hook or base class definitions.

    At the level of the hook class, the hook class will define the feature status flag
    and set it generically to ``True`` so that any class which mixes in the hook
    will have the hook enabled by default. This is done to ensure that hooks
    are enabled by default, but can be disabled on a per-model basis.

    Metadata and Integration Support
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Hooks may optionally define metadata attributes:

    - ``_<HookClassName>_NAME``: A short identifier string (e.g., ``"particle_sampler"``)
    - ``_<HookClassName>_DOC``: A human-readable summary for documentation / summarizing.

    These are useful for introspection or automated tooling but are not required.

    Class Settings / Attributes
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Hooks can define class-level configuration settings using the pattern:

    .. code-block:: python

        _<HookClassName>_<setting_name> = <value>

    For example:

    .. code-block:: python

        class ParticleSamplerHook(BaseHook):
            _ParticleSamplerHook_max_attempts = 1000


        class MyCluster(
            SphericalGalaxyClusterModel,
            ParticleSamplerHook,
        ):
            _ParticleSamplerHook_max_attempts = (
                10  # Override the default
            )

    This enables per-model customization of hook behavior.

    Utility Methods
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Non-public helper methods in hooks should follow the naming pattern:

    .. code-block:: python

        def _<HookClassName>_<method_name>(self, ...):
            ...

    This avoids namespace pollution and clarifies which hook a utility belongs to.

    Public API Methods
    ------------------
    Hooks may expose methods intended for use by the model or end user.
    These should:

    - Follow standard Python naming conventions (no leading underscore)
    - Avoid overly generic names like ``apply`` or ``run`` unless context-specific
    - Be explicitly documented

    Invocation Lifecycle
    --------------------
    Hook methods may be:

    - Explicitly called by model methods
    - Automatically invoked by Pisces during construction or export
    - Used by other hooks

    Hook logic should be self-contained and avoid side effects or global state.
    """

    # ----------------------------------- #
    # Feature Status Flag                 #
    # ----------------------------------- #
    # The feature status flag is a variable which indicates to pisces whether
    # or not this hook is enabled for a given model class. It may be explicitly
    # set to `False` in a specific model class to disable the hook for that class.
    #
    # The hook flag should be named ``__<HookClassName>_HOOK_ENABLED__``, where
    # ``<HookClassName>`` is the name of the hook class.
    #
    # The ``_IS_TEMPLATE__`` flag is used to indicate that this hook is a template
    # and should not be used directly.

    # __HOOK_ENABLED__ = True
    # __HOOK_IS_TEMPLATE__ = True # Mark this as a template hook.

    # ----------------------------------- #
    # Metadata and Integration Support    #
    # ----------------------------------- #
    # Hooks may optionally define metadata attributes such as:
    #
    # - `_<HookClassName>_NAME`: A short name or identifier for introspection.
    # - `_<HookClassName>_DOC`: A short description for use in automated docs or GUIs.
    #
    # These are not required, but help standardize discovery and documentation.

    # ----------------------------------- #
    # Class Settings / Attributes         #
    # ----------------------------------- #
    # Hooks often need special information / configuration about
    # how to work for a particular model class. This can be managed
    # using class attributes that are defined in the hook class.
    #
    # Each attribute in this section should be __<HookClassName>_<setting_name>__ and
    # should be documented in the class docstring. Model subclasses which mixin
    # this hook can then override these settings to modify behavior of the hook for
    # their specific model cases.

    # _<HookClassName>_hook_example_setting__ = "default_value"

    # ----------------------------------- #
    # Utility methods / sub-methods       #
    # ----------------------------------- #
    # To avoid cluttering the namespace of models, any methods in
    # the hook that are not directly exposed to the user should be
    # private (prefixed with _) and should have the following naming
    # convention: ``_<HookClassName>_<method_name>``. This ensures that
    # the methods are clearly associated with the hook and avoids
    # potential naming conflicts with other methods in the model class.
    #
    # Hooks may have any number of utility methods.

    # ----------------------------------- #
    # Public API Methods                  #
    # ----------------------------------- #
    # Any methods intended for use by the model or end-user should follow standard
    # Python naming conventions (no leading underscores) and should be explicitly
    # documented. Avoid overly generic names (e.g., `apply`, `update`) to reduce
    # name conflicts when hooks are mixed into multiple model classes.


# -------------------------------------------------- #
# Hook Tools                                         #
# -------------------------------------------------- #
class _HookTools:
    """Mixin class providing introspection utilities for Pisces model hooks.

    This class is intended to be inherited by all Pisces models to provide
    reflection-based tools for discovering, filtering, and interacting with
    attached hooks.

    It assumes that hook mixins are subclasses of :class:`BaseHook` and that
    each hook follows the naming conventions described in :class:`BaseHook`.
    """

    # --- Hook Retrieval Methods --- #
    @classmethod
    def get_all_hooks(cls, names: bool = True) -> list[BaseHook] | list[str]:
        """Return all hook classes mixed into the model, regardless of activation status.

        If `names` is True, return a list of hook class names instead of class objects.

        Parameters
        ----------
        names : bool, optional
            If True, return a list of hook class names. If False, return the actual hook classes.
            Defaults to True.

        Returns
        -------
        list of type
            List of hook classes (subclasses of BaseHook) in MRO.

        """
        # Retrieve the relevant hook classes by seeking the subclasses of
        # BaseHook in the method resolution order (MRO) of the class.
        _hook_classes = [
            hook_class
            for hook_class in cls.__mro__
            if issubclass(hook_class, BaseHook)
            and (hook_class is not BaseHook)
            and (getattr(hook_class, f"__{hook_class.__name__}_IS_TEMPLATE__", False) is False)
        ]

        # if `names` is True, we need to parse for the names of
        # the hooks, defaulting to the class name. Otherwise, we return the classes themselves.
        if not names:
            return _hook_classes
        else:
            _hook_names = [
                getattr(hook_class, f"_{hook_class.__name__}_NAME", hook_class.__name__) for hook_class in _hook_classes
            ]
            return _hook_names

    @classmethod
    def get_active_hooks(cls, names: bool = True) -> list[BaseHook] | list[str]:
        """Return all active hook classes mixed into the model.

        A hook is considered active if the model class has not explicitly disabled it
        via a ``__<HookClassName>_HOOK_ENABLED__ = False`` flag.

        Parameters
        ----------
        names : bool, optional
            If True, return a list of hook names (via ``_<HookClassName>_NAME`` if available,
            or fall back to class name). If False, return the hook class objects themselves.
            Defaults to True.

        Returns
        -------
        list of str or list of type
            List of active hook names or hook class objects.

        """
        # Begin by fetching ALL of the hooks using .get_all_hooks.
        all_hooks = cls.get_all_hooks(names=False)

        # Now, for each of these classes, we check whether the flag
        # is enabled or disabled in self.
        active_hooks = []
        for hook_class in all_hooks:
            hook_flag_value = getattr(cls, f"__{hook_class.__name__}_HOOK_ENABLED__", False)

            if hook_flag_value is True:
                active_hooks.append(hook_class)

        # Now handle names the same way it was done
        # in the all_hooks method.
        if not names:
            return active_hooks
        else:
            active_hook_names = [
                getattr(hook_class, f"_{hook_class.__name__}_NAME", hook_class.__name__) for hook_class in active_hooks
            ]
            return active_hook_names

    @classmethod
    def get_hook_class(cls, hook_name: str) -> type:
        """Return the hook class associated with a given hook name.

        This resolves against the symbolic name stored in ``_<HookClassName>_NAME``
        if present, and falls back to the class name.

        Parameters
        ----------
        hook_name : str
            The name of the hook, either the symbolic name or class name.

        Returns
        -------
        type
            The hook class object.

        Raises
        ------
        ValueError
            If no hook matches the provided name.

        """
        for hook_class in cls.get_all_hooks(names=False):
            name = getattr(hook_class, f"_{hook_class.__name__}_NAME", hook_class.__name__)
            if name == hook_name:
                return hook_class

        raise ValueError(f"No hook found with name '{hook_name}'.")

    @staticmethod
    def get_hook_name(hook_cls: type) -> str:
        """Return the symbolic name for a given hook class.

        If ``_<HookClassName>_NAME`` is defined, it is used;
        otherwise the class name is returned.

        Parameters
        ----------
        hook_cls : type
            The hook class.

        Returns
        -------
        str
            The symbolic name or class name.

        """
        return getattr(hook_cls, f"_{hook_cls.__name__}_NAME", hook_cls.__name__)

    @staticmethod
    def get_hook_description(hook_cls: type) -> str:
        """Return the human-readable description for a given hook class.

        If ``_<HookClassName>_DOC`` is defined, it is used; otherwise an empty string.

        Parameters
        ----------
        hook_cls : type
            The hook class.

        Returns
        -------
        str
            The description or an empty string.

        """
        return getattr(hook_cls, f"_{hook_cls.__name__}_DOC", "")

    @classmethod
    def has_hook(cls, hook: BaseHook | str) -> bool:
        """Check if a given hook is mixed into the model (regardless of activation status).

        The hook may be specified either as a class (subclass of BaseHook) or as a string name.
        When a string is provided, it is matched against both the symbolic name
        (``_<HookClassName>_NAME``) and the class name.

        Parameters
        ----------
        hook : type or str
            A hook class or hook name to test for.

        Returns
        -------
        bool
            True if the hook is present in the model's MRO, False otherwise.

        """
        if isinstance(hook, str):
            try:
                cls.get_hook_class(hook)
                return True
            except ValueError:
                return False

        elif isinstance(hook, type) and issubclass(hook, BaseHook):
            return hook in cls.get_all_hooks(names=False)

        else:
            raise TypeError("hook must be a subclass of BaseHook or a string name")


# -------------------------------------------------- #
# Custom Hooks                                       #
# -------------------------------------------------- #
class ParticleGenerationHook(BaseHook, ABC):
    """Abstract hook for converting models into particle datasets.

    This hook defines a standardized interface for models that can be
    transformed into particle-based representations. This hook defines a
    single public method, :meth:`generate_particles`, which tells the model
    how to convert itself into a particle dataset.

    Models which implement this hook may override the default
    implementation to provide custom logic for generating particles. Subclasses
    of the hook may also specify the logic for particle generation. The resulting
    particle datasets meet all of the standards and requirements for Pisces particle datasets,
    see :ref:`particles_overview` for more information.

    This hook is useful for models that support Monte Carlo sampling,
    numerical realizations, or forward modeling of observables in a
    particle-based format.

    .. note::

        This class is a **template hook** and should not be directly mixed into
        user-facing models. Instead, it should be subclassed to define concrete
        implementations with appropriate sampling logic and class-level settings.
        It is marked as a template using the flag:

        .. code-block:: python

            __ParticleGenerationHook_IS_TEMPLATE__ = True

        This flag ensures that automated discovery tools will skip this hook
        when scanning for active hook implementations.

    Hook Settings and Configuration
    -------------------------------
    Concrete implementations of this hook may define additional class-level
    configuration options, such as the number of particles to generate,
    sampling accuracy, or physical constraints. These options should follow
    the naming pattern:

    .. code-block:: python

        _<HookClassName>_<setting_name> = <default_value>

    Example
    -------
    A concrete hook might look like:

    .. code-block:: python

        class MyParticleHook(ParticleGenerationHook):
            __MyParticleHook_HOOK_ENABLED__ = True
            _MyParticleHook_max_particles = 10000

            def generate_particles(
                self, resolution: int = 1000
            ):
                # sampling logic here
                return ParticleDataset(...)

    Integration
    -----------
    When mixed into a model, Pisces will detect this hook using the standard
    hook introspection tools defined in :class:`_HookTools`. Models using this
    hook will typically support conversion to particle-based outputs for use
    in pipelines such as mock observables, synthetic catalogs, or simulations.

    """

    # ----------------------------------- #
    # Class Settings / Attributes         #
    # ----------------------------------- #
    # Feature status flag: this flag ensures that we have a way
    # of inspecting to detect whether or not a particular hook is enabled
    # for a given model class.
    __ParticleGenerationHook_ENABLED__ = True
    __ParticleGenerationHook_IS_TEMPLATE__ = True  # Mark this as a template hook.

    # --- Particle Generation Settings --- #
    # This section of the class should define any
    # relevant, class-level settings that are required
    # for the particle generation hook to function correctly.

    # ----------------------------------- #
    # Generator Methods                   #
    # ----------------------------------- #
    # This section of the hook should be used to
    # encapsulate the logic for generating the particle dataset.
    @abstractmethod
    def generate_particles(self, *args, **kwargs) -> "ParticleDataset":
        """Convert this model into a particle-based representation.

        This method creates a synthetic particle dataset that captures the physical
        structure described by the model. The returned dataset contains positions,
        velocities, masses, and other relevant fields for each particle, and is
        suitable for use in mock observations, simulations, or analysis pipelines.

        The specific method of generation—such as random sampling, grid population,
        or numerical evaluation—depends on the model and the hook subclass providing
        the implementation.

        Common use cases include:

        - Creating Monte Carlo realizations of galaxy clusters
        - Exporting a model as particles for visualization or N-body input
        - Generating mock data for pipelines expecting discrete samples

        Parameters
        ----------
        *args : tuple
            Optional positional arguments passed to the model’s particle generator.
            These depend on the specific implementation (e.g., number of particles,
            coordinate frame, sampling mode).

        **kwargs : dict
            Optional keyword arguments that control sampling resolution, included
            components, or output structure. Supported options vary by model.

        Returns
        -------
        ~pisces.particles.base.ParticleDataset
            A particle dataset containing the generated realization of this model.
            Includes physical fields (e.g., positions, velocities, mass) and
            metadata specific to the model type.

        """
        return NotImplemented
