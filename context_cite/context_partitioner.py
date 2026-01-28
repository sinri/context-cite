import re
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Union
from abc import ABC, abstractmethod
from .utils import split_text


class BaseContextPartitioner(ABC):
    """
    A base class for partitioning a context into sources.

    Attributes:
        context (str):
            The context to partition.

    Methods:
        num_sources(self) -> int:
            Property. The number of sources within the context.
        split_context(self) -> None:
            Split the context into sources.
        get_source(self, index: int) -> str:
            Get a represention of the source corresponding to a given index.
        get_context(self, mask: Optional[NDArray] = None) -> str:
            Get a version of the context ablated according to the given mask.
        sources(self) -> List[str]:
            Property. A list of all sources within the context.
    """

    def __init__(self, context: str) -> None:
        self.context = context

    @property
    @abstractmethod
    def num_sources(self) -> int:
        """The number of sources."""

    @abstractmethod
    def split_context(self) -> None:
        """Split the context into sources."""

    @abstractmethod
    def get_source(self, index: int) -> str:
        """Get a represention of the source corresponding to a given index."""

    @abstractmethod
    def get_context(self, mask: Optional[NDArray] = None):
        """Get a version of the context ablated according to the given mask."""

    @property
    def sources(self) -> List[str]:
        """A list of all sources."""
        return [self.get_source(i) for i in range(self.num_sources)]


class SimpleContextPartitioner(BaseContextPartitioner):
    """
    A simple context partitioner that splits the context into sources based on
    a separator.
    """

    def __init__(self, context: str, source_type: str = "sentence") -> None:
        super().__init__(context)
        self.source_type = source_type
        self._cache = {}

    def split_context(self):
        """Split text into parts and cache the parts and separators."""
        parts, separators, _ = split_text(self.context, self.source_type)
        self._cache["parts"] = parts
        self._cache["separators"] = separators

    @property
    def parts(self):
        if self._cache.get("parts") is None:
            self.split_context()
        return self._cache["parts"]

    @property
    def separators(self):
        if self._cache.get("separators") is None:
            self.split_context()
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        return len(self.parts)

    def get_source(self, index: int) -> str:
        return self.parts[index]

    def get_context(self, mask: Optional[NDArray] = None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        separators = np.array(self.separators)[mask]
        parts = np.array(self.parts)[mask]
        context = ""
        for i, (separator, part) in enumerate(zip(separators, parts)):
            if i > 0:
                context += separator
            context += part
        return context


class SeparatorContextPartitioner(BaseContextPartitioner):
    """
    A context partitioner that splits the context into sources based on a
    custom separator (string or regex pattern).

    This partitioner allows users to specify custom separators, such as
    standalone line markers (e.g., "====") or regular expressions, to
    partition the context into sources.

    Attributes:
        context (str):
            The context to partition.
        separator (str | re.Pattern):
            The separator pattern used to split the context.
        match_whole_line (bool):
            If True, the separator must match a whole line (for standalone
            line separators).
        strip_sources (bool):
            If True, strip whitespace from each source.
    """

    def __init__(
        self,
        context: str,
        separator: Union[str, re.Pattern],
        match_whole_line: bool = False,
        strip_sources: bool = True,
    ) -> None:
        """
        Initialize the separator-based context partitioner.

        Arguments:
            context (str):
                The context to partition.
            separator (str | re.Pattern):
                The separator pattern. Can be a string or a compiled regex pattern.
                If match_whole_line is True, the separator will be converted to
                match a whole line.
            match_whole_line (bool, optional):
                If True, the separator must match a whole line. This is useful
                for standalone line separators like "====". Defaults to False.
            strip_sources (bool, optional):
                If True, strip leading and trailing whitespace from each source.
                Defaults to True.
        """
        super().__init__(context)
        self.separator = separator
        self.match_whole_line = match_whole_line
        self.strip_sources = strip_sources
        self._cache = {}

        # Compile the regex pattern
        if isinstance(separator, re.Pattern):
            self._pattern = separator
        else:
            # If match_whole_line is True, convert separator to match whole line
            if match_whole_line:
                # Escape the separator and wrap it to match whole line
                escaped_separator = re.escape(separator)
                pattern_str = f"^{escaped_separator}$"
            else:
                # Use separator as-is (will be escaped if needed)
                pattern_str = separator

            try:
                self._pattern = re.compile(pattern_str, re.MULTILINE)
            except re.error as e:
                raise ValueError(f"Invalid separator pattern: {separator}. Error: {e}")

    def split_context(self) -> None:
        """
        Split the context into sources based on the separator pattern.
        Results are cached in _cache.
        """
        if self._cache.get("parts") is not None:
            return

        parts = []
        separators = []

        # Split the context using the pattern
        # re.split() includes the separator in the result if it's in a capturing group
        # We use a non-capturing group to split but not include the separator
        split_pattern = f"(?:{self._pattern.pattern})" if not self.match_whole_line else self._pattern.pattern

        # For whole line matching, we need to handle line boundaries differently
        if self.match_whole_line:
            # Split by lines first, then check each line against the pattern
            lines = self.context.splitlines(keepends=True)
            current_part = []
            current_separator = ""

            for i, line in enumerate(lines):
                # Check if this line matches the separator pattern
                # Remove newline for matching, but keep it for reconstruction
                line_content = line.rstrip("\n\r")
                if self._pattern.match(line_content):
                    # This is a separator line
                    if current_part:
                        # Save the accumulated part
                        part_text = "".join(current_part)
                        if self.strip_sources:
                            part_text = part_text.strip()
                        parts.append(part_text)
                        separators.append(current_separator)
                        current_part = []
                    # The separator itself (with newline) becomes the separator
                    current_separator = line
                else:
                    # This is content, accumulate it
                    current_part.append(line)
                    # Reset separator for next part
                    if current_separator and i > 0:
                        current_separator = ""

            # Add the last part if any
            if current_part:
                part_text = "".join(current_part)
                if self.strip_sources:
                    part_text = part_text.strip()
                parts.append(part_text)
                separators.append("")
        else:
            # For non-whole-line matching, use re.split
            # We need to capture the separators for reconstruction
            # Create a pattern that captures the separator
            # Escape the pattern if it's not already a regex pattern
            if isinstance(self.separator, re.Pattern):
                pattern_str = self._pattern.pattern
            else:
                # For string separators, we need to escape special regex chars
                # But if match_whole_line was False, user might want regex, so check
                # Actually, if separator is a string and match_whole_line is False,
                # we should treat it as a literal string (escape it)
                pattern_str = re.escape(self.separator)
            
            capturing_pattern = re.compile(f"({pattern_str})")
            split_result = capturing_pattern.split(self.context)

            # split_result alternates between parts and separators
            # Format: [part1, sep1, part2, sep2, part3, ...]
            # If text starts with separator: [empty_part, sep1, part2, ...]
            # If text ends with separator: [..., partN, sepN, empty_part]
            
            # Process the split result
            # The separator for each part is the one BEFORE it (for reconstruction)
            for i in range(0, len(split_result), 2):
                # Even indices are parts
                part = split_result[i] if i < len(split_result) else ""
                if self.strip_sources:
                    part = part.strip()
                parts.append(part)
                
                # Separator for this part is the one before it (if any)
                # For the first part (i=0), separator is empty
                # For subsequent parts, separator is at index i-1
                if i > 0 and (i - 1) < len(split_result):
                    separators.append(split_result[i - 1])
                else:
                    separators.append("")
            
            # Handle case where text ends with a separator
            # If the last element is a separator (even length), add empty part
            if len(split_result) > 0 and len(split_result) % 2 == 0:
                # Last element is a separator, add empty part
                parts.append("")
                separators.append(split_result[-1])

        # Handle edge case: if no separators found, the whole context is one source
        if not parts:
            part_text = self.context
            if self.strip_sources:
                part_text = part_text.strip()
            parts.append(part_text)
            separators.append("")

        # Filter out empty parts at the beginning/end if they're just whitespace
        # But keep them if they're actual content
        self._cache["parts"] = parts
        self._cache["separators"] = separators

    @property
    def parts(self) -> List[str]:
        """Get the split parts. Automatically calls split_context() if needed."""
        if self._cache.get("parts") is None:
            self.split_context()
        return self._cache["parts"]

    @property
    def separators(self) -> List[str]:
        """Get the separators. Automatically calls split_context() if needed."""
        if self._cache.get("separators") is None:
            self.split_context()
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        """The number of sources."""
        return len(self.parts)

    def get_source(self, index: int) -> str:
        """
        Get the source at the given index.

        Arguments:
            index (int): The index of the source (0-based).

        Returns:
            str: The source text at the given index.
        """
        return self.parts[index]

    def get_context(self, mask: Optional[NDArray] = None) -> str:
        """
        Get a version of the context ablated according to the given mask.

        Arguments:
            mask (Optional[NDArray]): Boolean array specifying which sources to keep.
                If None, returns the full context. The array length should equal
                the number of sources. True means keep the source, False means remove it.

        Returns:
            str: The ablated context with only the sources specified by the mask.
        """
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)

        if len(mask) != self.num_sources:
            raise ValueError(
                f"Mask length ({len(mask)}) does not match number of sources ({self.num_sources})"
            )

        separators = np.array(self.separators, dtype=object)[mask]
        parts = np.array(self.parts, dtype=object)[mask]

        # Reconstruct the context
        context_parts = []
        for i, (separator, part) in enumerate(zip(separators, parts)):
            if i > 0 and separator:
                context_parts.append(separator)
            context_parts.append(part)

        return "".join(context_parts)
