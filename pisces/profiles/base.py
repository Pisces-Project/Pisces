"""
Profile base class module.

This module provides the core base classes from which all other
profiles are constructed and includes skeletons for developers to
use when adding new profiles to the package.
"""
from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import sympy as sp
import unyt

from pisces._registries import __default_profile_registry__
from pisces.utilities.symbols import lambdify_expression

from ._exceptions import ProfileClassSetupError

# ============================= #
# Core Structures               #
# ============================= #
# These are the core structures for defining
# the behavior of profiles in Pisces across a
# variety of different scenarios.


def derived_profile(name: Optional[str] = None) -> classmethod:
    """
    Mark a class method as a derived profile generator.

    Derived profiles define secondary symbolic profiles (e.g., gradients, potentials)
    associated with this class. When decorated, the method is automatically registered
    at class construction and the profile can be instantiated on demand via
    :meth:`BaseProfile.get_derived_profile`.

    The decorated method must be a ``@classmethod`` with the following signature:

    .. code-block:: python

        @classmethod
        def my_derived(cls, *variable_symbols, **parameter_symbols):
            return symbolic_func, units_func, variables, parameters

    The method must return a tuple:

    - ``symbolic_func(*variable_symbols, **parameter_symbols)`` — Callable generating a SymPy expression
    - ``units_func(*variable_units, **parameter_units)`` — Callable computing units with unyt
    - ``variables`` — List of str, independent variables for the derived profile
    - ``parameters`` — Dict of str: default parameters for the derived profile

    Parameters
    ----------
    name : str, optional
        A custom name for the derived profile. Defaults to the method name.

    Returns
    -------
    classmethod
        The decorated classmethod, with registration metadata.

    Notes
    -----
    - Derived profiles are instantiated as dynamic subclasses of :attr:`__DERIVED_BASE__`, or
      :class:`BaseProfile` if unspecified.
    - Use :meth:`BaseProfile.get_derived_profile` to access an instantiated version.

    Raises
    ------
    TypeError
        If applied to a method that is not a @classmethod.
    """

    def decorator(func):
        """Add the wrapper around the class expression."""
        if not isinstance(func, classmethod):
            raise TypeError(
                "The @derived_profile decorator must be applied to a @classmethod."
            )

        original_func = func.__func__  # Extract underlying function from classmethod

        @wraps(original_func)
        def wrapper(*args, **kwargs):
            """Wrap original function."""
            return original_func(*args, **kwargs)

        # Rewrap as a classmethod
        wrapped_method = classmethod(wrapper)

        # Attach metadata
        wrapped_method.derived_profile = True
        wrapped_method.expression_name = name or original_func.__name__

        return wrapped_method

    return decorator


class _ProfileMeta(ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Generate a new ProfileMeta class. This procedure creates a generic
        object subclass before performing the following 3 procedures:

        1. Check for abstraction: if the class is abstract, we just return the
           generic ``object`` baseclass.
        2. Validate the class: Check for required profile structures.
        3. Setup the class: Construct all of the relevant symbolics, etc.
        """
        # --- Generate the generic class --- #
        # Perform the standard operation to create an `object` descended
        # base class which can then be altered as needed.
        cls_object: Type[BaseProfile] = super().__new__(
            mcs, name, bases, namespace, **kwargs
        )

        # --- Check for abstraction --- #
        _cls_is_abstract = getattr(cls_object, "__IS_ABSTRACT__", False)
        if _cls_is_abstract:
            # We do not process this class at all.
            return cls_object

        # --- Validate Class --- #
        mcs.validate_profile_class(cls_object)

        # -- Setup the Class -- #
        # We now setup the class and register it.
        if cls_object.__REGISTER__ and not __default_profile_registry__.has(
            cls_object.__name__
        ):
            __default_profile_registry__.register(cls_object.__name__, cls_object)

        if cls_object.__SETUP_AT__ == "import":
            cls_object.__cls_setup__()

        return cls_object

    @staticmethod
    def validate_profile_class(cls):
        """
        Validate a new profile class.

        This includes determining the number of dimensions and ensuring
        that bounds and coordinates are all accurate.
        """
        # Check the new class for the required attributes that all classes should have.
        __required_elements__ = [
            "__VARIABLES__",
            "__PARAMETERS__",
            "__cls_var_symbols__",
            "__cls_param_symbols__",
        ]
        for _re_ in __required_elements__:
            if not hasattr(cls, _re_):
                raise ProfileClassSetupError(
                    f"Profile class {cls.__name__} does not define or inherit an expected "
                    f"class attribute: `{_re_}`."
                )

        # Ensure that we have specified axes and that they have the correct length.
        # The AXES_BOUNDS need to be validated to ensure that they have the correct
        # structure and only specify valid conventions for boundaries.
        if cls.__VARIABLES__ is None:
            raise ProfileClassSetupError(
                f"Profile class {cls.__name__} does not define a set of variables"
                "using the `__VARIABLES__` attribute."
            )


class BaseProfile(ABC, metaclass=_ProfileMeta):
    """
    Abstract base class for constructing symbolic profile functions.

    :class:`BaseProfile` provides the infrastructure for defining parameterized, symbolic expressions
    that can be evaluated numerically with units. Subclasses define their behavior by specifying
    independent variables, parameters, and symbolic expressions. Additional derived profiles,
    such as gradients or potentials, can be attached via :func:`derived_profile`.

    This class integrates:

    - SymPy for symbolic construction
    - unyt for unit-aware parameter handling
    - Automated symbolic setup and expression management via a custom metaclass
    - Optional serialization to and from minimal dictionaries
    - Dynamic construction of secondary (derived) profiles

    Intended Usage
    --------------
    Subclasses must define the following:

    - `__VARIABLES__` : list of str
      Independent variables of the profile.

    - `__PARAMETERS__` : dict of str, default values
      Named parameters for the profile function. These are provided at instantiation with units.

    - `__function__` : abstractmethod
      Returns the symbolic expression as a SymPy object.

    - `__function_units__` : abstractmethod
      Returns the dimensional units of the profile output.

    Additional customization is possible using:

    - `__IS_ABSTRACT__` : bool
      Prevents symbolic setup and registration for base or template classes.

    - `__REGISTER__` : bool
      Controls whether the class is added to the global profile registry.

    - `__SETUP_AT__` : "init" or "import"
      Controls when symbolic setup occurs.

    - `__DERIVED_BASE__` : type
      Specifies the base class for any derived profiles.

    Derived Profiles
    ----------------
    Developers can attach secondary symbolic profiles using the `@derived_profile` decorator.
    These are dynamically constructed as subclasses and accessed via `get_derived_profile`.

    Example
    -------
    .. code-block:: python

        class MyProfile(BaseProfile):
            __IS_ABSTRACT__ = False
            __VARIABLES__ = ["x"]
            __PARAMETERS__ = {"a": 1.0}

            def __function__(self, x, a):
                return a * x ** 2

            def __function_units__(self, x_unit, a_unit):
                return a_unit * x_unit ** 2

            @derived_profile()
            @classmethod
            def gradient(cls, x, a):
                def func(x, a):
                    return 2 * a * x
                def units(x_unit, a_unit):
                    return a_unit * x_unit
                return func, units, ["x"], {"a": 1.0}

    Notes
    -----
    - Subclasses must explicitly disable `__IS_ABSTRACT__` to trigger full setup.
    - Symbolic variables and parameter symbols are generated during class setup.
    - Parameters provided at initialization can include units and are stored internally as magnitude/unit pairs.
    - Numerical evaluation supports unit-aware input and output via `__call__`.
    """

    # =============================== #
    # Class Level Flags               #
    # =============================== #
    # Class flags for BaseProfile descendants
    # allow developers to control and customize
    # subclass behavior during metaclass processing,
    # symbolic setup, and profile registration.

    # CONVENTION: All class-level flags are defined
    # with ALL_CAPS naming for clarity and consistency.

    __IS_ABSTRACT__: bool = True
    """
    bool : Marks this profile class as abstract.

    If set to ``True``, the class is treated as an abstract template.
    The metaclass will skip validation, symbolic setup, and registration steps
    during class construction.

    This flag is intended for base classes or incomplete implementations that
    define a structural framework but should not be directly instantiated or
    registered in the profile registry.

    Notes
    -----
    - Concrete subclasses **must** set ``__IS_ABSTRACT__ = False`` to enable full setup.
    - Failure to do so results in bypassing all symbolic infrastructure.
    """

    __REGISTER__: bool = True
    """
    bool : Controls automatic registration of the profile.

    If ``True``, the profile class is added to the global
    ``__default_profile_registry__`` during class construction,
    allowing it to be retrieved dynamically by name.

    If ``False``, the profile class is fully constructed and functional
    but omitted from the registry. This is useful for:

    - Internal helper classes
    - Dynamically generated derived profiles
    - Experimental or private implementations

    Notes
    -----
    - Abstract classes are never registered, regardless of this flag.
    - To prevent registry pollution, most temporary or internal profiles
      should explicitly set ``__REGISTER__ = False``.
    """

    __SETUP_AT__: Literal["init", "import"] = "init"
    """
    str : Controls when symbolic setup occurs for the profile class.

    Valid options are:

    - ``"import"`` : Symbolic setup runs during class construction (at import time).
    - ``"init"``   : Symbolic setup is deferred until the first instance is initialized.

    Use ``"import"`` for:

    - Profiles that require immediate symbolic availability
    - Classes where derived profiles or symbolic methods depend on setup at import time

    Use ``"init"`` for:

    - Profiles that should minimize startup cost
    - Cases where symbolic setup is only required on first use

    Notes
    -----
    - Deferred setup with ``"init"`` is idempotent; setup only runs once, even across instances.
    - Changing this flag does not affect already constructed classes.
    """

    __DERIVED_BASE__: Optional[Type["BaseProfile"]] = None
    """
    type or None : Specifies the base class for dynamically generated derived profiles.

    When using the ``@derived_profile`` decorator, derived profiles are constructed
    as subclasses of this type.

    Defaults to ``BaseProfile`` if not specified, but developers can override to:

    - Enforce specific behavior in derived profiles
    - Introduce specialized methods or infrastructure
    - Provide alternative symbolic backends

    Examples
    --------
    To force all derived profiles of a custom base to inherit from a shared subclass:

    .. code-block:: python

        class MyBaseProfile(BaseProfile):
            __IS_ABSTRACT__ = True
            __DERIVED_BASE__ = MyDerivedBase

    Notes
    -----
    - Derived profiles respect the ``__DERIVED_BASE__`` of the defining class,
      not necessarily the base-most ancestor.
    - The provided type must itself be a subclass of ``BaseProfile`` or compatible.
    """

    # =============================== #
    # Class Attributes                #
    # =============================== #
    # These settings are used to determine how the
    # profile will behave from a mathematical standpoint. These
    # should be set in almost all subclasses to define the
    # relevant variables, parameters, etc.
    __VARIABLES__: List[str] = None
    """list of str: The independent variables of the profile function.
    These are converted at instantiation to sympy variables in order to construct
    the relevant profile.
    """
    __PARAMETERS__: Dict[str, Any] = dict()
    """ dict of str, Any: The parameters which define this profile and its
    behavior. This should include any scale lengths, masses, etc. Each of the parameters
    is set by specifying the corresponding kwarg when initializing the class.
    """
    # ============================== #
    # Class Initialization Vars      #
    # ============================== #
    # These components of the class should always be left as
    # sentinels and then filled during the metaclass initialization.
    # This is where sympy symbols are stored.
    #
    # This section also includes the methods accessed during
    # class construction so that they can be overwritten as needed
    # in subclasses.
    __cls_var_symbols__: List[sp.Symbol] = None
    __cls_param_symbols__: Dict[str, sp.Symbol] = dict()
    __cls_derived_profiles__: Dict[str, Any] = dict()
    __cls_is_setup_flag__: bool = False

    @classmethod
    def __cls_setup_symbols__(cls):
        """
        Create the symbols for the parameters and the variables
        of the class.

        This is executed by the metaclass when the class is generated vis-a-vis
        the ``__setup_class__`` method. It is the first step in class creation.

        Requirements
        ------------
        Whatever form this method takes, it must fill the ``__cls_var_symbols__``
        and ``__cls_param_symbols__`` successfully with SymPy symbols representing
        each of the relevant variables and parameters. ``__cls_var_symbols__`` should
        be a list of symbols while ``__cls_param_symbols__`` should be a dictionary
        with the string of the parameter and then the symbol as the value.
        """
        # --- DEFAULT --- #
        cls.__cls_var_symbols__ = [sp.Symbol(_var_) for _var_ in cls.__VARIABLES__]
        cls.__cls_param_symbols__ = {
            _param_name_: sp.Symbol(_param_name_) for _param_name_ in cls.__PARAMETERS__
        }

    @classmethod
    def __cls_setup_implicit_symbolic_attributes__(cls):
        """
        Register all class-level symbolic expressions defined with @derived_profile.

        Scans the method resolution order (MRO) of the class and identifies class methods
        tagged as symbolic expressions.

        Adds them to the `__cls_symbolic_attrs__` dictionary. The expressions are evaluated
        on demand when requested via `get_derived_profile()`.

        Notes
        -----
        This only registers the expression. Evaluation is deferred until the first access.
        """
        # Set the derived profiles blank so that we do not
        # get overlap between constructors / behavior between classes.
        cls.__cls_derived_profiles__ = dict()

        # begin the iteration through the class __mro__ to find objects
        # in the entire inheritance structure.
        seen = set()
        for base in reversed(cls.__mro__):  # reversed to ensure subclass -> baseclass
            # Check if we need to search this element of the __mro__. We only exit if we find
            # `object` because it's not going to have any worthwhile symbolics.
            if base is object:
                continue

            # Check this element of the __mro__ for any relevant elements that
            # we might want to attach to this class.
            for attr_name, method in base.__dict__.items():
                # Check if we have any interest in processing these methods. If the method is already
                # seen, then we skip it. Additionally, if the class expression is missing the correct
                # attributes, we skip it.
                if (base, attr_name) in seen:
                    continue
                if (not isinstance(method, classmethod)) and not (
                    callable(method) and getattr(method, "derived_profile", False)
                ):
                    seen.add((base, attr_name))
                    continue
                elif (isinstance(method, classmethod)) and not (
                    callable(method.__func__)
                    and getattr(method, "derived_profile", False)
                ):
                    seen.add((base, attr_name))  # type: ignore
                    continue
                seen.add((base, attr_name))  # type: ignore

                # At this point, any remaining methods are relevant class expressions which should
                # be registered. We create the dynamic class at this stage and then only instantiate
                # when required by the user.
                expression_name = getattr(method, "expression_name", attr_name)
                _func, _ufunc, _vars, _params = getattr(cls, method.__name__)()
                cls.__cls_derived_profiles__[
                    expression_name
                ] = build_dynamic_profile_class(
                    _func,
                    _ufunc,
                    variables=_vars,
                    parameters=_params,
                    base=cls.__DERIVED_BASE__,
                )

    @classmethod
    def __cls_setup__(cls):
        """
        Orchestrates the symbolic setup for a profile class.

        This is the main entry point used during class construction. It performs the following steps:

        1. Initializes variables and parameter symbols.
        2. Builds explicit class symbols (things like the metric and metric density)
        3. Registers class expressions.
        4. Sets up internal flags to avoid re-processing.

        Raises
        ------
        CoordinateClassException
            If any part of the symbolic setup fails (e.g., axes, metric, or expressions).
        """
        # Check for abstraction or existing configuration. If either of these has
        # occurred, the execution should stop.
        if cls.__IS_ABSTRACT__:
            raise TypeError(
                f"CoordinateSystem class {cls.__name__} is abstract and cannot be instantiated or constructed."
            )
        if cls.__cls_is_setup_flag__:
            return

        # Step 1: Construct the symbols.
        try:
            cls.__cls_setup_symbols__()
        except Exception as e:
            raise ProfileClassSetupError(
                f"Failed to setup the variable symbols for profile class {cls.__name__} due to"
                f" an error: {e}."
            ) from e

        # Step 2: Construct implicit symbolic attributes.
        try:
            cls.__cls_setup_implicit_symbolic_attributes__()
        except Exception as e:
            raise ProfileClassSetupError(
                f"Failed to setup derived class expressions for profile class {cls.__name__} due to"
                f" an error: {e}."
            ) from e

    # ============================== #
    # Initialization                 #
    # ============================== #
    def _setup_parameters(self, **kwargs):
        # Create the storage buffers for the parameter
        # values and units.
        _pvalues, _punits = dict(), dict()

        for _parameter_name in kwargs:
            if _parameter_name not in self.__PARAMETERS__:
                raise ValueError(
                    f"Parameter `{_parameter_name}` is not a recognized parameter of the {self.__class__.__name__} profile."
                )

        # Iterate through the parameters, strip the units,
        # and assign.
        for _parameter_name in self.__PARAMETERS__:
            # Check that the parameter is valid.
            if _parameter_name in kwargs:
                _parameter_value = kwargs[_parameter_name]
            else:
                _parameter_value = self.__PARAMETERS__[_parameter_name]

            # Now set them in
            if hasattr(_parameter_value, "units"):
                _pvalues[_parameter_name] = _parameter_value.value
                _punits[_parameter_name] = _parameter_value.units
            else:
                _pvalues[_parameter_name] = _parameter_value
                _punits[_parameter_name] = unyt.Unit("")

        return _pvalues, _punits

    @classmethod
    @abstractmethod
    def __function__(cls, *args, **kwargs) -> Any:
        ...

    @classmethod
    @abstractmethod
    def __function_units__(cls, *args, **kwargs) -> unyt.Unit:
        ...

    def __init__(self, **kwargs):
        """
        Initialize a profile instance with specific parameter values.

        This constructor sets up the symbolic and numerical infrastructure for the profile
        by performing the following steps:

        1. If the class has not already been set up, trigger symbolic construction of metric tensors,
           symbols, and expressions.
        2. Validate and store user-provided parameter values, overriding class defaults.
        3. Substitute parameter values into symbolic expressions to produce instance-specific forms.
        4. Lambdify key expressions (metric tensor, inverse metric, metric density) for numerical evaluation.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments specifying values for profile parameters. Each key should match
            a parameter name defined in ``__PARAMETERS__``. Any unspecified parameters will use the class-defined
            default values.

        Raises
        ------
        ValueError
            If a provided parameter name is not defined in the profile.

        """
        # -- Class Initialization -- #
        # For profiles with setup flags for 'init', it is necessary to process
        # symbolics at this point if the class is not initialized.
        self.__class__.__cls_setup__()

        # -- Parameter Creation -- #
        # The profile takes a set of kwargs (potentially empty) which specify
        # the parameters of the profile. Each should be adapted into a self.__parameters__ dictionary.
        self.__parameters__, self.__parameter_units__ = self._setup_parameters(**kwargs)

        # -- Numerical Implementation -- #
        # We now realize the symbolic version of this particular
        # instance as well as the numerical version.
        self.__symbolic_profile__ = self.__function__(
            *self.__cls_var_symbols__, **self.__cls_param_symbols__
        )
        self.__numeric_profile__ = self.lambdify_expression(self.__symbolic_profile__)

        # -- Expression Management -- #
        self.__derived_profiles__: Dict[str, Any] = dict()

    # =============================== #
    # Dunder Methods                  #
    # =============================== #
    def __repr__(self):
        return f"<{self.__class__.__name__} - Parameters={self.__parameters__}> "

    def __str__(self):
        return f"<{self.__class__.__name__}>"

    def __hash__(self):
        r"""
        Compute a hash value for the profile instance.

        The hash is based on the class name and keyword arguments (``__parameters__``).
        This ensures that two instances with the same class and initialization parameters produce the same hash.

        Returns
        -------
        int
            The hash value of the instance.
        """
        # Create a parameter tuple with each parameter linked
        # to the string of its unit in a tuple.
        _parameter_linked_tuple = tuple(
            (pname, p_value.value, str(p_value.units))
            for pname, p_value in sorted(self.parameters.items())
        )
        return hash((self.__class__.__name__, _parameter_linked_tuple))

    def __getitem__(self, item: str) -> unyt.unyt_quantity:
        """
        Return the value of a parameter with a given name.

        Parameters
        ----------
        item : str
            The parameter to fetch.

        Returns
        -------
        ~unyt.array.unyt_quantity
            The resulting parameter.

        Raises
        ------
        KeyError
            If the parameter doesn't exist.

        """
        return self.__parameters__[item] * self.__parameter_units__[item]

    def __contains__(self, parameter: str) -> bool:
        """
        Check whether a given parameter exists for this profile.

        Parameters
        ----------
        parameter : str
            The parameter name to check.

        Returns
        -------
        bool
            True if the parameter name is present; False otherwise.

        """
        return parameter in self.__PARAMETERS__

    def __copy__(self):
        """
        Create a shallow copy of the coordinate system.

        Returns
        -------
        _CoordinateSystemBase
            A new instance of the same class, initialized with the same parameters.

        """
        # Shallow copy: re-init with same parameters
        cls = self.__class__
        new_obj = cls(**self.parameters)
        return new_obj

    def __call__(
        self, *x, units: Optional[Union[str, unyt.Unit]] = None, no_units: bool = False
    ) -> Union[unyt.unyt_quantity, unyt.unyt_array]:
        # separate the x args from their units.
        xnum, xunit = (
            [_x.value if hasattr(_x, "units") else _x for _x in x],
            [_x.units if hasattr(_x, "units") else unyt.Unit("") for _x in x],
        )

        # Evaluate the numerical function using the xnum
        output_value = self.__numeric_profile__(*xnum)
        output_units = self.__function_units__(*xunit, **self.__parameter_units__)

        if no_units:
            return output_value
        elif units is None:
            return output_value * output_units
        else:
            return (output_value * output_units).to(units)

    def __call_no_units__(self, *x):
        return self.__numeric_profile__(*x)

    # =============================== #
    # Properties                      #
    # =============================== #
    @property
    def variables(self) -> List[str]:
        """The axes names present in this coordinate system."""
        return self.__VARIABLES__[:]

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        The parameters of this coordinate system. Note that modifications made to the returned dictionary
        are not reflected in the class itself. To change a parameter value, the class must be re-instantiated.
        """
        return {
            k: self.__parameters__[k] * self.__parameter_units__[k]
            for k in self.__parameters__
        }

    @property
    def variable_symbols(self) -> List[sp.Symbol]:
        """
        The symbols representing each of the coordinate axes in this coordinate system.
        """
        return self.__class__.__cls_var_symbols__[:]

    @property
    def parameter_symbols(self):
        """
        Get the symbolic representations of the coordinate system parameters.

        Returns
        -------
        dict of str, ~sympy.core.symbol.Symbol
            A dictionary mapping parameter names to their corresponding SymPy symbols.
            These symbols are used in all symbolic expressions defined by the coordinate system.

        Notes
        -----
        - The returned dictionary is a copy, so modifying it will not affect the internal state.
        - These symbols are created during class setup and correspond to keys in `self.parameters`.
        """
        return self.__cls_param_symbols__.copy()

    @property
    def derived_profile_classes(self) -> Dict[str, Type["BaseProfile"]]:
        """
        Get the available derived profile classes for this instance.

        Returns
        -------
        dict of str, Type[BaseProfile]
            A dictionary mapping derived profile names to their dynamically
            generated profile classes.
        """
        return self.__class__.__cls_derived_profiles__.copy()

    # ==================================== #
    # Expressions Management               #
    # ==================================== #
    def substitute_expression(self, expression: Any) -> Any:
        """
        Replace symbolic parameters with numerical values in an expression.

        This method takes a symbolic expression that may include parameter symbols and
        substitutes them with the numerical values assigned at instantiation.

        Parameters
        ----------
        expression : str or sympy expression
            The symbolic expression to substitute parameter values into.

        Returns
        -------
        sympy expression
            The expression with parameters replaced by their numeric values.

        Notes
        -----
        - Only parameters defined in ``self.__parameters__`` are substituted.
        - If an expression does not contain any parameters, it remains unchanged.
        - This method is useful for obtaining instance-specific symbolic representations.

        Example
        -------

        .. code-block:: python

            from sympy import Symbol
            expr = Symbol('a') * Symbol('x')
            coords = MyCoordinateSystem(a=3)
            print(coords.substitute_expression(expr))
            3*x

        """
        # Substitute in each of the parameter values.
        _params = {k: v for k, v in self.__parameters__.items()}
        return sp.simplify(sp.sympify(expression).subs(_params))

    def lambdify_expression(self, expression: Union[str, sp.Basic]) -> Callable:
        """
        Convert a symbolic expression into a callable function.

        Parameters
        ----------
        expression : :py:class:`str` or sp.Basic
            The symbolic expression to lambdify.

        Returns
        -------
        Callable
            A callable numerical function.
        """
        return lambdify_expression(
            expression, self.__cls_var_symbols__, self.__parameters__
        )

    def get_derived_profile(self, profile_name: str, **kwargs):
        """
        Access and instantiate a derived profile by name.

        If the profile has already been instantiated and no additional kwargs
        are provided, the cached version is returned.

        Parameters
        ----------
        profile_name : str
            The name of the derived profile as registered via @derived_profile.
        **kwargs : dict
            Optional parameter overrides for the derived profile.

        Returns
        -------
        BaseProfile
            An instance of the derived profile, initialized with inherited and/or overridden parameters.

        Raises
        ------
        KeyError
            If the requested derived profile is not defined.
        """
        if profile_name not in self.__class__.__cls_derived_profiles__:
            raise KeyError(
                f"Derived profile '{profile_name}' not found for class {self.__class__.__name__}."
            )

        # If already cached and no new kwargs, return cached version
        if profile_name in self.__derived_profiles__ and not kwargs:
            return self.__derived_profiles__[profile_name]

        DerivedCls = self.__class__.__cls_derived_profiles__[profile_name]

        # Combine inherited parameters with any overrides
        init_params = DerivedCls.__PARAMETERS__.copy()
        init_params.update(
            {k: v for k, v in self.parameters.items() if k in init_params}
        )
        init_params.update(kwargs)

        derived_instance = DerivedCls(**init_params)
        self.__derived_profiles__[profile_name] = derived_instance
        return derived_instance

    def list_derived_profiles(self) -> List[str]:
        """
        List all available derived profiles for this instance.

        Returns
        -------
        list of str
            The names of all registered derived profiles.
        """
        return list(self.__class__.__cls_derived_profiles__.keys())

    # ==================================== #
    # Utilities                            #
    # ==================================== #
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this profile to a minimal dictionary representation.

        The dictionary contains:

        - ``class`` : The profile class name (for reconstruction)
        - ``parameters`` : A nested dictionary of parameters with explicit unit strings

        Example output:

        .. code-block:: python

            {
                "class": "MyProfile",
                "parameters": {
                    "a": {"value": 3.0, "unit": "kpc"},
                    "b": {"value": 5.0, "unit": ""}
                }
            }

        Returns
        -------
        dict
            Serialized profile representation suitable for storage or reconstruction.
        """
        return {
            "class": self.__class__.__name__,
            "parameters": {
                pname: {
                    "value": self.__parameters__[pname],
                    "unit": str(self.__parameter_units__[pname]),
                }
                for pname in self.__parameters__
            },
        }

    @classmethod
    def from_dict(cls, data):
        """
        Reconstruct a profile instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing ``parameters`` in the format produced by :meth:`to_dict`.

        Returns
        -------
        BaseProfile
            A new instance of the profile with the specified parameters.

        Raises
        ------
        ValueError
            If required keys are missing or invalid parameter names are provided.
        """
        params = {
            pname: unyt.unyt_quantity(pdata["value"], pdata["unit"])
            for pname, pdata in data.get("parameters", {}).items()
        }
        return cls(**params)

    def to_yaml(self, filepath: str, **kwargs):
        """
        Serialize the profile to a YAML file.

        Parameters
        ----------
        filepath : str
            Path to the output YAML file.
        kwargs : dict
            Additional arguments passed to yaml.dump.
        """
        import yaml

        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, **kwargs)

    def to_json(self, filepath: str, **kwargs):
        """
        Serialize the profile to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the output JSON file.
        kwargs : dict
            Additional arguments passed to json.dump.
        """
        import json

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, **kwargs)

    @classmethod
    def from_yaml(cls, filepath: str) -> "BaseProfile":
        """
        Reconstruct a profile from a YAML file.

        Parameters
        ----------
        filepath : str
            Path to the YAML file.

        Returns
        -------
        BaseProfile
            Reconstructed profile instance.
        """
        import yaml

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return profile_from_dict(data)

    @classmethod
    def from_json(cls, filepath: str) -> "BaseProfile":
        """
        Reconstruct a profile from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.

        Returns
        -------
        BaseProfile
            Reconstructed profile instance.
        """
        import json

        with open(filepath, "r") as f:
            data = json.load(f)
        return profile_from_dict(data)

    def to_hdf5(self, h5obj, name=None):
        """
        Store profile metadata into an HDF5 object as attributes.

        Parameters
        ----------
        h5obj : ~h5py.File, ~h5py.Group, or ~h5py.Dataset
            The HDF5 container to attach metadata to. All of the metadata
            representing this profile is attached with `name` prefixed.
        name : str, optional
            Namespace prefix for attribute keys. Defaults to none, in which
            case parameters are given no prefix. This can be unsafe if saving
            multiple profiles to the same object.

        Notes
        -----
        Only metadata is stored. This does not save large arrays or derived results.
        """
        data = self.to_dict()

        if name is not None:
            prefix = f"{name}."
        else:
            prefix = ""

        h5obj.attrs[f"{prefix}class"] = data["class"]

        for pname, pinfo in data["parameters"].items():
            h5obj.attrs[f"{prefix}param.{pname}.value"] = pinfo["value"]
            h5obj.attrs[f"{prefix}param.{pname}.unit"] = pinfo["unit"]

    @classmethod
    def from_hdf5(cls, h5obj, name=None) -> "BaseProfile":
        """
        Reconstruct a profile from HDF5 attributes.

        Parameters
        ----------
        h5obj : ~h5py.File, ~h5py.Group, or ~h5py.Dataset
            HDF5 object containing profile metadata as attributes.
        name : str, optional
            Namespace prefix for attribute keys. If None, parameters are read without a prefix.

        Returns
        -------
        BaseProfile
            Reconstructed profile instance.

        Raises
        ------
        ValueError
            If required metadata is missing or incomplete.
        """
        if name is not None:
            prefix = f"{name}."
        else:
            prefix = ""

        # Validate required class entry
        if f"{prefix}class" not in h5obj.attrs:
            raise ValueError(
                f"HDF5 object does not contain profile metadata with prefix '{prefix}'."
            )

        cls_name = h5obj.attrs[f"{prefix}class"]

        # Extract parameter values and units
        params = {}
        for attr in h5obj.attrs:
            if attr.startswith(f"{prefix}param.") and attr.endswith(".value"):
                pname = attr[len(f"{prefix}param.") : -len(".value")]
                pval = h5obj.attrs[attr]
                punit = h5obj.attrs.get(f"{prefix}param.{pname}.unit", "")
                params[pname] = unyt.unyt_quantity(pval, punit)

        return profile_from_dict(
            {
                "class": cls_name,
                "parameters": {
                    pname: {"value": param.value, "unit": str(param.units)}
                    for pname, param in params.items()
                },
            }
        )


# ============================= #
# Special Case Subclasses       #
# ============================= #
# Special subclasses with general purpose structures.


class BaseSphericalRadialProfile(BaseProfile, ABC):
    """
    Abstract base class for spherically symmetric, 1D radial profiles.

    This class provides standard conventions for radial profiles, defining:

    - A single independent variable ``r`` (radius)
    - Automatic symbolic setup for ``r``
    - A derived profile for the radial derivative

    Subclasses define specific profile behavior by implementing:

    - ``__function__`` : returns the symbolic expression as a SymPy object
    - ``__function_units__`` : returns the dimensional units of the profile

    Notes
    -----
    - This class is abstract. Concrete subclasses must set ``__IS_ABSTRACT__ = False``.
    - Units for ``r`` are propagated through the derivative profile.
    - The radial derivative is accessible via :meth:`get_derived_profile("derivative")`.

    Example
    -------
    .. code-block:: python

        class DensityProfile(BaseRadialProfile):
            __IS_ABSTRACT__ = False
            __PARAMETERS__ = {"rho0": 1.0}

            def __function__(self, r, rho0):
                return rho0 / (r**2)

            def __function_units__(self, r_unit, rho0_unit):
                return rho0_unit / r_unit**2
    """

    __IS_ABSTRACT__ = True
    __VARIABLES__ = ["r"]

    @derived_profile("derivative")
    @classmethod
    def _radial_derivative(cls):
        """
        Construct the symbolic radial derivative of the profile.
        """

        # define the new function as the sp.diff of the
        # class profile.
        def _func(_r, **_params):
            return sp.diff(cls.__function__(_r, **_params), _r)

        # Define the unit propagation function as
        # a simple extension of the class's unit function.
        def _unit_func(_r_unit, **_param_units):
            return cls.__function_units__(_r_unit, **_param_units) / _r_unit

        return _func, _unit_func, ["r"], cls.__PARAMETERS__.copy()


# ============================= #
# Utility Functions             #
# ============================= #
def profile_from_dict(
    data: Dict[str, Any], registry: Optional[Dict[str, Type[BaseProfile]]] = None
) -> BaseProfile:
    """
    Reconstruct a profile instance from a dictionary with optional registry support.

    Parameters
    ----------
    data : dict
        Dictionary with ``class`` and ``parameters`` fields, as produced by :meth:`BaseProfile.to_dict`.
    registry : dict, optional
        Mapping of class names to profile classes. Defaults to the global ``__default_profile_registry__``.

    Returns
    -------
    BaseProfile
        The reconstructed profile instance.

    Raises
    ------
    ValueError
        If the class name is missing or the class cannot be resolved.
    """
    if registry is None:
        registry = __default_profile_registry__

    cls_name = data.get("class")
    if cls_name is None:
        raise ValueError("Serialized profile is missing the 'class' field.")

    if cls_name not in registry:
        raise ValueError(f"Profile class '{cls_name}' not found in registry.")

    cls = registry[cls_name]
    return cls.from_dict(data)


def build_dynamic_profile_class(
    func: Callable,
    unit_func: Callable,
    variables: List[str],
    parameters: Optional[Dict[str, str]] = None,
    base: Optional[Type[BaseProfile]] = None,
) -> Type[BaseProfile]:
    """
    Dynamically construct a profile class from a specified function and other metadata.

    Parameters
    ----------
    func : Callable
        Function to generate the symbolic expression for the profile. This must be a callable with
        signature ``func(*variables,**parameters)`` which returns a valid SymPy expression.
    unit_func : Callable
        Function to determine the result units of the profile. This must be a callable with
        signature ``func(*variable_units,**parameter_units)`` which returns a :class:`~unyt.unit_object.Unit`
        instance with the resulting units.
    variables : list of str
        The independent variables of the profile. These should be provided as strings in the order
        in which they appear in `func` and `unit_func`.
    parameters : dict of str, float, optional
        Dictionary mapping parameter names to their default values. These
        parameters define required inputs for the derived profile. By default,
        no parameters are required. Derived profiles do **not** inherit parameters
        from the base class automatically — all parameters must be explicitly defined here.
    base : type, optional
        The base class from which to construct the profile. By default, this is the base class
        :class:`BaseProfile`; however, it may be changed to allow general subclassing behavior.

    Returns
    -------
    type
        A dynamically constructed, ready-to-instantiate profile class for the derived quantity.

    Notes
    -----
    - The ``variables`` list defines the symbolic input order for the derived profile. Must match
      the argument structure expected by ``func`` and ``unit_func``.
    - Derived profiles inherit from ``base`` (defaults to :class:`BaseProfile`), unless
      overridden via :attr:`__DERIVED_BASE__`.
    """
    # Determine the base class to use for the
    # profile class. By default, this is the
    # base class.
    if base is None:
        base = BaseProfile

    # Fix the parameters to be a
    # fully realized dictionary.
    if parameters is None:
        parameters = {}

    # Generate the dynamic class using the provided
    # data.
    class _DynamicProfile(base):
        __IS_ABSTRACT__ = False
        __REGISTER__ = False  # Derived profiles should not register globally
        __SETUP_AT__ = "init"
        __VARIABLES__ = variables
        __PARAMETERS__ = parameters

        @classmethod
        def __function__(cls, *args, **kwargs) -> Any:
            return func(*args, **kwargs)

        @classmethod
        def __function_units__(cls, *argu, **kwargu) -> unyt.Unit:
            return unit_func(*argu, **kwargu)

    return _DynamicProfile
