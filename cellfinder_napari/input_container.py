from abc import abstractmethod
from dataclasses import asdict
from pathlib import Path


class InputContainer:
    """Base for classes that contain inputs

    Intended to be derived to group specific related widget inputs (e.g from the same widget section)
    into a container. Derived classes should be Python data classes.

    Enforces common interfaces for
    - how to get default values for the inputs
    - how inputs are passed to cellfinder core
    - how the inputs are shown in the widget
    """

    @classmethod
    def defaults(cls) -> dict:
        """Returns default values of this class's fields as a dict."""
        # Derived classes are not expected to be particularly
        # slow to instantiate, so use the default constructor
        # to avoid code repetition.
        return asdict(cls())

    @abstractmethod
    def as_core_arguments(self) -> dict:
        """Determines how dataclass fields are passed to cellfinder-core.

        The implementation provided here can be re-used in derived classes, if convenient.
        """
        # note that asdict returns a new instance of a dict,
        # so any subsequent modifications of this dict won't affect the class instance
        return asdict(self)

    @classmethod
    def _custom_widget(
        cls, key: str, custom_label: str = None, **kwargs
    ) -> dict:
        """Represents a field, given by key, as a formatted widget with the field's default value.

        The widget label is the capitalized key by default, with underscores replaced by spaces, unless custom_label is specified.
        Keyword arguments like step, min, max, ... are passed to napari underneath.
        """
        label = (
            key.replace("_", " ").capitalize()
            if custom_label is None
            else custom_label
        )
        value = cls.defaults()[key]
        return dict(value=value, label=label, **kwargs)

    @classmethod
    @abstractmethod
    def widget_representation(cls) -> dict:
        """What the class will look like as a napari widget"""
        pass
