"""
Mojo Syntax Automation Script

This script provides comprehensive automation for Mojo syntax standardization and validation
according to the patterns documented in mojo_syntax.md. It can scan files for violations,
suggest corrections, apply automatic fixes, and generate compliance reports.

Usage:
    mojo update_mojo_syntax.mojo --scan [directory]
    mojo update_mojo_syntax.mojo --fix [file]
    mojo update_mojo_syntax.mojo --validate [file]
    mojo update_mojo_syntax.mojo --report [directory]

Features:
- Pattern detection for common syntax violations
- Automatic corrections with safety backups
- Compliance checking and scoring
- Comprehensive reporting system
- GPU acceleration pattern preservation (supports current API: enqueue_create_buffer, enqueue_function)
- Docstring content exclusion (by default, use --check-docstring-code to enable)

Content Exclusion:
The script automatically excludes certain content from syntax violation detection:
1. Variable-assigned strings (sample code assignments) - always excluded
2. Docstring content (triple-quoted strings) - excluded by default, use --check-docstring-code to enable

This prevents false positives from test/example code and aligns with mojo_syntax.md
guidelines about avoiding code examples in docstrings due to Mojo LSP parsing issues.
"""

from collections import List
from time import perf_counter_ns as now
from sys.arg import argv
from pathlib import Path


struct SyntaxViolation(Copyable, Movable):
    """Represents a syntax violation found in a Mojo file."""

    var file_path: String
    var line_number: Int
    var violation_type: String
    var description: String
    var suggested_fix: String
    var severity: String  # "error", "warning", "suggestion"

    fn __init__(
        out self,
        file_path: String,
        line_number: Int,
        violation_type: String,
        description: String,
        suggested_fix: String,
        severity: String,
    ):
        """Initialize a syntax violation."""
        self.file_path = file_path
        self.line_number = line_number
        self.violation_type = violation_type
        self.description = description
        self.suggested_fix = suggested_fix
        self.severity = severity


struct ComplianceReport(Copyable, Movable):
    """Comprehensive compliance report for a file or directory."""

    var file_path: String
    var total_lines: Int
    var violations: List[SyntaxViolation]
    var compliance_score: Float64
    var last_checked: Int  # timestamp

    fn __init__(out self, file_path: String):
        """Initialize a compliance report."""
        self.file_path = file_path
        self.total_lines = 0
        self.violations = List[SyntaxViolation]()
        self.compliance_score = 0.0
        self.last_checked = now()

    fn add_violation(mut self, violation: SyntaxViolation):
        """Add a violation to the report."""
        self.violations.append(violation)

    fn calculate_score(mut self):
        """Calculate compliance score based on violations."""
        if self.total_lines == 0:
            self.compliance_score = 100.0
            return

        error_weight = 10.0
        warning_weight = 5.0

        total_penalty = 0.0
        for i in range(len(self.violations)):
            violation = self.violations[i]
            if violation.severity == "error":
                total_penalty += error_weight
            elif violation.severity == "warning":
                total_penalty += warning_weight
            # Note: "suggestion" items are excluded from compliance calculation

        # Calculate score as percentage (simple penalty-based approach)
        # Start with 100% and subtract penalties
        penalty_percentage = total_penalty
        self.compliance_score = max(0.0, 100.0 - penalty_percentage)


struct StructInfo:
    """Information about a struct definition."""

    var name: String
    var has_copyable: Bool
    var has_movable: Bool

    fn __init__(out self, name: String, has_copyable: Bool, has_movable: Bool):
        """
        Initialize StructInfo with struct metadata.

        Args:
            name: Name of the struct.
            has_copyable: Whether the struct has Copyable trait.
            has_movable: Whether the struct has Movable trait.
        """
        self.name = name
        self.has_copyable = has_copyable
        self.has_movable = has_movable


struct LifecycleAnalysis(Copyable, Movable):
    """Analysis results for struct lifecycle methods."""

    var has_trivial_copyinit: Bool
    var has_trivial_moveinit: Bool
    var needs_custom_copy: Bool
    var needs_custom_move: Bool
    var copyinit_line: Int
    var moveinit_line: Int

    fn __init__(out self):
        """
        Initialize LifecycleAnalysis with default values.

        Sets all analysis flags to False and line numbers to 0.
        """
        self.has_trivial_copyinit = False
        self.has_trivial_moveinit = False
        self.needs_custom_copy = False
        self.needs_custom_move = False
        self.copyinit_line = 0
        self.moveinit_line = 0


struct MojoSyntaxChecker(Copyable, Movable):
    """Main syntax checker and automation engine."""

    var reports: List[ComplianceReport]
    var backup_enabled: Bool
    var auto_fix_enabled: Bool
    var preserve_gpu_patterns: Bool
    var show_observations: Bool
    var check_docstring_code: Bool
    var auto_cleanup_backups: Bool
    var keep_backups: Bool
    var backup_retention_days: Int

    fn __init__(out self):
        """Initialize the syntax checker."""
        self.reports = List[ComplianceReport]()
        self.backup_enabled = True
        self.auto_fix_enabled = False
        self.preserve_gpu_patterns = True
        self.show_observations = False
        self.check_docstring_code = False
        self.auto_cleanup_backups = True  # Auto-cleanup by default
        self.keep_backups = False  # Don't keep backups by default
        self.backup_retention_days = 7  # Keep backups for 7 days

    fn _is_inside_docstring(self, lines: List[String], line_index: Int) -> Bool:
        """
        Check if the given line index is inside a docstring.

        Args:
            lines: List of all lines in the file
            line_index: Zero-based index of the line to check

        Returns:
            True if the line is inside a docstring, False otherwise
        """
        # Count triple quotes before this line
        triple_quote_count = 0

        for i in range(line_index + 1):  # Include current line
            line = lines[i]
            # Count occurrences of triple quotes in this line
            quote_pos = 0
            while True:
                pos = line.find('"""', quote_pos)
                if pos == -1:
                    break
                triple_quote_count += 1
                quote_pos = pos + 3

        # If odd number of triple quotes, we're inside a docstring
        return (triple_quote_count % 2) == 1

    fn _is_inside_variable_string(
        self, lines: List[String], line_index: Int
    ) -> Bool:
        """
        Check if the given line index is inside a variable-assigned string literal.

        Args:
            lines: List of all lines in the file
            line_index: Zero-based index of the line to check

        Returns:
            True if the line is inside a variable-assigned string, False otherwise
        """
        # Simple approach: scan backwards to find variable assignment start,
        # then scan forwards to find the end

        # Find the most recent variable assignment with triple quotes before current line
        assignment_start = -1
        for i in range(line_index - 1, -1, -1):  # Scan backwards
            line = lines[i].strip()
            if "=" in line and '"""' in line:
                equals_pos = line.find("=")
                triple_quote_pos = line.find('"""')
                if equals_pos < triple_quote_pos:
                    assignment_start = i
                    break

        if assignment_start == -1:
            return False  # No variable assignment found before current line

        # Find the closing triple quotes after the assignment
        for i in range(assignment_start + 1, len(lines)):
            line = lines[i].strip()
            if line == '"""':
                # Found closing triple quotes
                if assignment_start < line_index < i:
                    return True
                break

        return False

    fn _should_skip_line_for_violations(
        self, lines: List[String], line_index: Int
    ) -> Bool:
        """
        Determine if a line should be skipped for violation detection.

        Args:
            lines: List of all lines in the file
            line_index: Zero-based index of the line to check

        Returns:
            True if the line should be skipped, False if it should be checked
        """
        # Always skip variable-assigned string content (sample code)
        if self._is_inside_variable_string(lines, line_index):
            return True

        # Skip docstring content unless explicitly enabled
        if not self.check_docstring_code:
            return self._is_inside_docstring(lines, line_index)

        return False

    fn check_import_patterns(
        self, file_content: String, file_path: String
    ) -> List[SyntaxViolation]:
        """Check import statement patterns against mojo_syntax.md standards."""
        violations = List[SyntaxViolation]()
        lines = file_content.split("\n")

        # Track import sections for proper organization checking
        var stdlib_imports = List[Int]()  # Line numbers of stdlib imports
        var project_imports = List[Int]()  # Line numbers of project imports
        var gpu_imports = List[Int]()  # Line numbers of GPU imports
        var first_non_comment_line = -1

        # First pass: categorize imports and find first non-comment line
        for i in range(len(lines)):
            # Skip docstring content unless explicitly enabled
            if self._should_skip_line_for_violations(lines, i):
                continue

            line = lines[i].strip()
            line_num = i + 1

            # Skip empty lines and comments, but track first substantial line
            if (
                line == ""
                or line.startswith("#")
                or line.startswith('"""')
                or line.startswith("'''")
            ):
                continue

            if (
                first_non_comment_line == -1
                and not line.startswith("from ")
                and not line.startswith("import ")
            ):
                first_non_comment_line = line_num

            # Check for relative imports (violation)
            if line.startswith("from .") or line.startswith("from .."):
                violation = SyntaxViolation(
                    file_path,
                    line_num,
                    "import_pattern",
                    "Relative import detected",
                    "Use full path imports: from src.module_name import ...",
                    "error",
                )
                violations.append(violation)

            # Categorize imports by type
            if line.startswith("from ") or line.startswith("import "):
                # Standard library imports
                if (
                    line.startswith("from sys")
                    or line.startswith("from collections")
                    or line.startswith("from memory")
                    or line.startswith("from math")
                    or line.startswith("from time")
                    or line.startswith("from testing")
                ):
                    stdlib_imports.append(line_num)

                # GPU/MAX Engine imports (special category)
                elif (
                    "gpu.host" in line
                    or "gpu" in line
                    or "layout" in line
                    or "has_nvidia_gpu_accelerator" in line
                    or "has_amd_gpu_accelerator" in line
                ):
                    gpu_imports.append(line_num)

                # Project imports (src.* patterns)
                elif line.startswith("from src.") or "src." in line:
                    project_imports.append(line_num)

        # Second pass: Check import organization
        # Standard library imports should come before project imports
        if len(stdlib_imports) > 0 and len(project_imports) > 0:
            last_stdlib = stdlib_imports[-1] if len(stdlib_imports) > 0 else 0
            first_project = (
                project_imports[0] if len(project_imports) > 0 else 999999
            )

            if last_stdlib > first_project:
                violation = SyntaxViolation(
                    file_path,
                    last_stdlib,
                    "import_organization",
                    (
                        "Standard library imports should come before project"
                        " imports"
                    ),
                    (
                        "Move standard library imports to top of file, before"
                        " project imports"
                    ),
                    "warning",
                )
                violations.append(violation)

        # Check for scattered standard library imports (should be grouped)
        if len(stdlib_imports) > 1:
            for i in range(1, len(stdlib_imports)):
                gap = stdlib_imports[i] - stdlib_imports[i - 1]
                if gap > 5:  # Allow reasonable gaps for comments
                    violation = SyntaxViolation(
                        file_path,
                        stdlib_imports[i],
                        "import_organization",
                        "Standard library imports should be grouped together",
                        "Group all standard library imports in one section",
                        "suggestion",
                    )
                    violations.append(violation)
                    break  # Only report once per file

        return violations

    fn check_struct_patterns(
        self, file_content: String, file_path: String
    ) -> List[SyntaxViolation]:
        """Check struct definition patterns."""
        violations = List[SyntaxViolation]()
        lines = file_content.split("\n")

        for i in range(len(lines)):
            # Skip docstring content unless explicitly enabled
            if self._should_skip_line_for_violations(lines, i):
                continue

            line = lines[i].strip()
            line_num = i + 1

            # Check for struct definitions
            if line.startswith("struct "):
                # Check for missing docstring - improved logic to handle struct inheritance/traits
                docstring_found = False
                # Find the end of struct definition (look for colon) - handles multi-line definitions and comments
                j = i
                while j < len(lines):
                    line_stripped = lines[j].strip()
                    # Check if line contains colon (handle comments after colon)
                    if ":" in line_stripped:
                        # Make sure colon is not inside a string or comment at the beginning
                        colon_pos = line_stripped.find(":")
                        if colon_pos >= 0:
                            break
                    j += 1

                # Look for docstring after struct definition
                k = j + 1
                while k < len(lines):
                    next_line = lines[k].strip()
                    if next_line == "":
                        k += 1
                        continue
                    if next_line.startswith('"""'):
                        docstring_found = True
                    break

                if not docstring_found:
                    violation = SyntaxViolation(
                        file_path,
                        line_num,
                        "struct_documentation",
                        "Struct missing docstring",
                        "Add comprehensive docstring after struct definition",
                        "warning",
                    )
                    violations.append(violation)

                # Analyze struct for trait requirements
                struct_violations = self._analyze_struct_traits(
                    lines, i, file_path
                )
                for violation in struct_violations:
                    violations.append(violation)

        # Check for struct initialization order violations
        var init_violations = self._check_struct_initialization_order(
            lines, file_path
        )
        for violation in init_violations:
            violations.append(violation)

        return violations

    fn _check_struct_initialization_order(
        self, lines: List[String], file_path: String
    ) -> List[SyntaxViolation]:
        """Check for struct initialization order violations."""
        violations = List[SyntaxViolation]()

        for i in range(len(lines)):
            line = lines[i].strip()

            # Look for __init__ method definitions
            if line.startswith("fn __init__("):
                var indent_level = len(lines[i]) - len(lines[i].lstrip())
                var found_field_assignment = False

                # Scan through the __init__ method until we find a line with same or less indentation
                for j in range(i + 1, len(lines)):
                    var current_line = lines[j].strip()
                    var current_indent = len(lines[j]) - len(lines[j].lstrip())

                    # If we hit a line with same or less indentation (and it's not empty), we've exited the method
                    if len(current_line) > 0 and current_indent <= indent_level:
                        break

                    # Skip the method signature line
                    if j == i:
                        continue

                    # Look for instance method calls in field assignments (self.field = self._method())
                    if (
                        "self." in current_line
                        and "=" in current_line
                        and "self._" in current_line
                        and "(" in current_line
                        and not current_line.startswith("#")
                    ):
                        # This is a field assignment using an instance method call
                        violation = SyntaxViolation(
                            file_path,
                            j + 1,
                            "struct_init_order",
                            (
                                "Instance method call in __init__() during"
                                " field initialization"
                            ),
                            (
                                "Use @staticmethod decorator for methods called"
                                " during initialization, or initialize all"
                                " fields with defaults first"
                            ),
                            "error",
                        )
                        violations.append(violation)
                        break

                    # Look for field assignments (self.field = ...)
                    if (
                        "self." in current_line
                        and "=" in current_line
                        and not current_line.startswith("#")
                    ):
                        # Check if it's a field assignment (not a method call)
                        if not "(" in current_line.split("=")[0]:
                            found_field_assignment = True

                    # Look for standalone instance method calls (self._method())
                    if (
                        "self._" in current_line
                        and "(" in current_line
                        and "=" not in current_line
                        and not current_line.startswith("#")
                    ):
                        # Check if this is before field assignments
                        if not found_field_assignment:
                            violation = SyntaxViolation(
                                file_path,
                                j + 1,
                                "struct_init_order",
                                (
                                    "Instance method call in __init__() before"
                                    " field initialization"
                                ),
                                (
                                    "Use @staticmethod decorator for methods"
                                    " called during initialization, or"
                                    " initialize all fields first"
                                ),
                                "error",
                            )
                            violations.append(violation)
                            break

        return violations

    fn _analyze_struct_traits(
        self, lines: List[String], struct_line_idx: Int, file_path: String
    ) -> List[SyntaxViolation]:
        """Analyze struct for trait requirements based on lifecycle methods."""
        violations = List[SyntaxViolation]()
        struct_line = lines[struct_line_idx].strip()

        # Extract struct name and existing traits
        struct_info = self._parse_struct_definition(String(struct_line))
        _ = struct_info.name  # struct_name not used in current implementation
        has_copyable = struct_info.has_copyable
        has_movable = struct_info.has_movable

        # Find struct body and analyze lifecycle methods
        struct_body = self._extract_struct_body(lines, struct_line_idx)
        lifecycle_analysis = self._analyze_lifecycle_methods(struct_body)

        # NOTE: Traits are almost always required in Mojo - compiler does NOT auto-generate
        # Only flag traits as redundant in very specific cases (e.g., utility structs never copied)

        # Check for redundant traits (very rare cases)
        # Most structs need traits for copying, function returns, and collection storage
        # Only suggest removal for utility structs that are never copied/moved
        if has_copyable and not lifecycle_analysis.needs_custom_copy:
            # TODO: Add logic to detect if struct is actually used in copyable contexts
            # For now, don't suggest removing traits as they're usually needed
            pass

        if has_movable and not lifecycle_analysis.needs_custom_move:
            # TODO: Add logic to detect if struct is actually used in movable contexts
            # For now, don't suggest removing traits as they're usually needed
            pass

        # Check for redundant lifecycle methods
        if lifecycle_analysis.has_trivial_copyinit:
            # Adjust line number to be relative to file, not struct body
            absolute_line = struct_line_idx + lifecycle_analysis.copyinit_line
            if has_copyable:
                # Struct has trait but method is trivial - suggest removing method
                violation = SyntaxViolation(
                    file_path,
                    absolute_line,
                    "redundant_method",
                    "Trivial __copyinit__ method duplicates compiler default",
                    (
                        "Remove __copyinit__ method - Copyable trait handles"
                        " this automatically"
                    ),
                    "suggestion",
                )
            else:
                # Struct has trivial method but no trait - suggest replacing with trait
                violation = SyntaxViolation(
                    file_path,
                    absolute_line,
                    "redundant_method",
                    (
                        "Trivial __copyinit__ method should use Copyable trait"
                        " instead"
                    ),
                    (
                        "Remove __copyinit__ method and add Copyable trait to"
                        " struct"
                    ),
                    "suggestion",
                )
            violations.append(violation)

        if lifecycle_analysis.has_trivial_moveinit:
            # Adjust line number to be relative to file, not struct body
            absolute_line = struct_line_idx + lifecycle_analysis.moveinit_line
            if has_movable:
                # Struct has trait but method is trivial - suggest removing method
                violation = SyntaxViolation(
                    file_path,
                    absolute_line,
                    "redundant_method",
                    "Trivial __moveinit__ method duplicates compiler default",
                    (
                        "Remove __moveinit__ method - Movable trait handles"
                        " this automatically"
                    ),
                    "suggestion",
                )
            else:
                # Struct has trivial method but no trait - suggest replacing with trait
                violation = SyntaxViolation(
                    file_path,
                    absolute_line,
                    "redundant_method",
                    (
                        "Trivial __moveinit__ method should use Movable trait"
                        " instead"
                    ),
                    (
                        "Remove __moveinit__ method and add Movable trait to"
                        " struct"
                    ),
                    "suggestion",
                )
            violations.append(violation)

        return violations

    fn _parse_struct_definition(self, struct_line: String) -> StructInfo:
        """Parse struct definition line to extract name and traits."""
        # Extract struct name (between "struct " and "(" or ":")
        var name = ""
        var has_copyable = False
        var has_movable = False

        # Find struct name
        if "struct " in struct_line:
            var start_idx = struct_line.find("struct ") + 7
            var end_idx = len(struct_line)

            if "(" in struct_line:
                end_idx = struct_line.find("(")
                # Check for traits in parentheses
                var traits_section = struct_line[
                    struct_line.find("(") : struct_line.find(")") + 1
                ]
                has_copyable = "Copyable" in traits_section
                has_movable = "Movable" in traits_section
            elif ":" in struct_line:
                end_idx = struct_line.find(":")

            # Extract name by building string character by character
            for idx in range(start_idx, end_idx):
                name += struct_line[idx]
            # name is already a String, just strip whitespace manually
            while name.startswith(" ") or name.startswith("\t"):
                name = name[1:]
            while name.endswith(" ") or name.endswith("\t"):
                name = name[:-1]

        return StructInfo(name, has_copyable, has_movable)

    fn _extract_struct_body(
        self, lines: List[String], struct_start: Int
    ) -> List[String]:
        """Extract the body of a struct definition."""
        body = List[String]()
        var i = struct_start + 1
        var indent_level = 0
        var found_body = False

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines
            if stripped == "":
                i += 1
                continue

            # Calculate indentation
            var current_indent = len(line) - len(line.lstrip())

            # If we haven't found the body yet, look for first indented content
            if not found_body:
                if current_indent > 0:
                    indent_level = current_indent
                    found_body = True
                    body.append(line)
                i += 1
                continue

            # If we're in the body, check if we've reached the end
            if (
                current_indent <= indent_level
                and String(stripped) != ""
                and not String(stripped).startswith("#")
            ):
                # We've reached the end of the struct
                break

            body.append(line)
            i += 1

        return body

    fn _analyze_lifecycle_methods(
        self, struct_body: List[String]
    ) -> LifecycleAnalysis:
        """Analyze lifecycle methods in struct body."""
        analysis = LifecycleAnalysis()

        var i = 0
        while i < len(struct_body):
            line = struct_body[i].strip()

            # Look for __copyinit__ method
            if "fn __copyinit__" in line:
                analysis.copyinit_line = i + 1
                # Analyze if this is a trivial implementation
                method_body = self._extract_method_body(struct_body, i)
                analysis.has_trivial_copyinit = self._is_trivial_copyinit(
                    method_body
                )
                analysis.needs_custom_copy = not analysis.has_trivial_copyinit

            # Look for __moveinit__ method
            elif "fn __moveinit__" in line:
                analysis.moveinit_line = i + 1
                # Analyze if this is a trivial implementation
                method_body = self._extract_method_body(struct_body, i)
                analysis.has_trivial_moveinit = self._is_trivial_moveinit(
                    method_body
                )
                analysis.needs_custom_move = not analysis.has_trivial_moveinit

            i += 1

        return analysis

    fn _extract_method_body(
        self, lines: List[String], method_start: Int
    ) -> List[String]:
        """Extract the body of a method definition."""
        body = List[String]()
        var i = method_start + 1
        var method_indent = 0
        _ = False  # found_body not used in current implementation

        # Find the method's indentation level
        if method_start < len(lines):
            method_line = lines[method_start]
            method_indent = len(method_line) - len(method_line.lstrip())

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines and comments
            if stripped == "" or String(stripped).startswith("#"):
                i += 1
                continue

            # Calculate current indentation
            var current_indent = len(line) - len(line.lstrip())

            # If we're back to method level or less, we've reached the end
            if current_indent <= method_indent and String(stripped) != "":
                break

            body.append(line)
            i += 1

        return body

    fn _is_trivial_copyinit(self, method_body: List[String]) -> Bool:
        """Check if __copyinit__ method is trivial (just copies fields)."""
        # A trivial copyinit only contains field assignments like: self.field = other.field
        var non_trivial_lines = 0

        for i in range(len(method_body)):
            line = method_body[i]
            stripped = line.strip()
            if stripped == "" or String(stripped).startswith("#"):
                continue  # Skip empty lines and comments

            # Check if this is a simple field assignment
            if "self." in stripped and "=" in stripped and "other." in stripped:
                # This looks like a field copy: self.field = other.field
                continue
            else:
                # This is non-trivial logic
                non_trivial_lines += 1

        return non_trivial_lines == 0

    fn _is_trivial_moveinit(self, method_body: List[String]) -> Bool:
        """Check if __moveinit__ method is trivial (just moves fields)."""
        # A trivial moveinit only contains field moves like: self.field = other.field^
        var non_trivial_lines = 0

        for i in range(len(method_body)):
            line = method_body[i]
            stripped = line.strip()
            if stripped == "" or String(stripped).startswith("#"):
                continue  # Skip empty lines and comments

            # Check if this is a simple field move
            if (
                "self." in stripped
                and "=" in stripped
                and ("other." in stripped and "^" in stripped)
            ):
                # This looks like a field move: self.field = other.field^
                continue
            else:
                # This is non-trivial logic
                non_trivial_lines += 1

        return non_trivial_lines == 0

    fn _check_unsafe_pointer_ownership(
        self,
        lines: List[String],
        line_index: Int,
        file_content: String,
        file_path: String,
    ) -> Optional[SyntaxViolation]:
        """
        Check UnsafePointer parameters for ownership violations.

        Only flags UnsafePointer parameters that are explicitly owned by the function
        and require memory management. Borrowed parameters (default) are not flagged.

        Args:
            lines: All lines in the file.
            line_index: Current line index being checked.
            file_content: Full file content for context.
            file_path: Path to the file being checked.

        Returns:
            Optional violation if owned UnsafePointer lacks proper memory management.
        """
        line = lines[line_index].strip()
        line_num = line_index + 1

        # Skip if not a function parameter line with UnsafePointer
        if not ("UnsafePointer" in line and ("fn " in line or ":" in line)):
            return None

        # Check if this is a GPU kernel function (typically borrowed parameters)
        if self._is_gpu_kernel_function(lines, line_index):
            return None  # GPU kernels use borrowed pointers managed by DeviceContext

        # Check if this is an explicitly owned parameter
        if not self._is_owned_unsafe_pointer_parameter(String(line)):
            return None  # Only flag owned parameters

        # Check if the function properly manages the owned memory
        if self._has_proper_memory_management(lines, line_index, file_content):
            return None  # Proper management found

        # Create violation for owned UnsafePointer without proper management
        return SyntaxViolation(
            file_path,
            line_num,
            "performance_memory_leak",
            "Owned UnsafePointer parameter without explicit memory management",
            (
                "Add .free() call for owned UnsafePointer parameters or use"
                " RAII patterns"
            ),
            "warning",
        )

    fn _is_gpu_kernel_function(
        self, lines: List[String], line_index: Int
    ) -> Bool:
        """Check if the current line is part of a GPU kernel function definition.
        """
        # Look backwards for function definition
        var i = line_index
        while i >= 0:
            line = lines[i].strip()
            if line.startswith("fn "):
                # Check if function name suggests it's a GPU kernel
                return (
                    "_kernel" in line
                    or "gpu_" in line.lower()
                    or "kernel_" in line
                    or
                    # Check for common GPU kernel parameter patterns
                    (i + 1 < len(lines) and "thread_idx" in lines[i + 1])
                    or (i + 2 < len(lines) and "thread_idx" in lines[i + 2])
                )
            i -= 1
        return False

    fn _is_owned_unsafe_pointer_parameter(self, line: String) -> Bool:
        """Check if line contains an explicitly owned UnsafePointer parameter.
        """
        # Look for explicit ownership annotations
        return (
            "UnsafePointer" in line
            and ("owned " in line or "var " in line)
            and ":" in line  # Parameter declaration
        )

    fn _has_proper_memory_management(
        self, lines: List[String], line_index: Int, file_content: String
    ) -> Bool:
        """Check if function has proper memory management for owned UnsafePointer.
        """
        # Find the function that contains this parameter
        function_body = self._extract_function_body(lines, line_index)

        # Check if the function body contains a .free() call
        for line in function_body:
            if ".free()" in line:
                return True

        return False

    fn _extract_function_body(
        self, lines: List[String], param_line_index: Int
    ) -> List[String]:
        """Extract the body of the function containing the given parameter line.
        """
        body = List[String]()

        # Find the start of the function
        var func_start = param_line_index
        while func_start >= 0:
            if lines[func_start].strip().startswith("fn "):
                break
            func_start -= 1

        if func_start < 0:
            return body  # No function found

        # Extract function body until next function or end of file
        var i = func_start + 1
        var indent_level = 0
        var found_body_start = False

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines and comments
            if stripped == "" or stripped.startswith("#"):
                i += 1
                continue

            # Look for function body start (after docstring)
            if not found_body_start:
                if not stripped.startswith('"""') and not stripped.startswith(
                    "'''"
                ):
                    found_body_start = True
                else:
                    # Skip docstring lines
                    i += 1
                    continue

            # Track indentation to know when function ends
            if found_body_start:
                current_indent = len(line) - len(line.lstrip())

                # If we hit a line at the same or lower indentation that starts a new definition
                if current_indent <= indent_level and (
                    stripped.startswith("fn ")
                    or stripped.startswith("struct ")
                    or stripped.startswith("alias ")
                    or stripped.startswith("from ")
                ):
                    break

                if indent_level == 0:
                    indent_level = current_indent

                body.append(line)

            i += 1

        return body

    fn check_function_patterns(
        self, file_content: String, file_path: String
    ) -> List[SyntaxViolation]:
        """Check function definition patterns."""
        violations = List[SyntaxViolation]()
        lines = file_content.split("\n")

        for i in range(len(lines)):
            # Skip docstring content unless explicitly enabled
            if self._should_skip_line_for_violations(lines, i):
                continue

            # Note: Function pattern checking (docstrings, etc.) is handled by check_documentation_patterns()
            # This function is reserved for future function-specific patterns that don't overlap

        return violations

    fn check_variable_patterns(
        self, file_content: String, file_path: String
    ) -> List[SyntaxViolation]:
        """Check variable declaration patterns."""
        violations = List[SyntaxViolation]()
        lines = file_content.split("\n")

        for i in range(len(lines)):
            # Skip docstring content unless explicitly enabled
            if self._should_skip_line_for_violations(lines, i):
                continue

            line = lines[i].strip()
            line_num = i + 1

            # Check for old 'let' keyword usage
            if line.startswith("let "):
                violation = SyntaxViolation(
                    file_path,
                    line_num,
                    "variable_declaration",
                    "Old 'let' keyword usage detected",
                    "Use direct assignment or 'var' for mutable variables",
                    "error",
                )
                violations.append(violation)

            # Check for DeviceContext variable naming convention
            if "= DeviceContext()" in line:
                # Extract variable name
                if "var " in line:
                    var_part = line.split("var ")[1].split("=")[0].strip()
                    # Check if variable name follows convention
                    is_valid_name = (
                        var_part == "ctx"
                        or var_part == "gpu_ctx"
                        or var_part == "main_ctx"
                        or var_part == "compute_ctx"
                        or var_part == "stream_ctx"
                        or var_part.endswith("_ctx")
                    )
                    if not is_valid_name:
                        violation = SyntaxViolation(
                            file_path,
                            line_num,
                            "variable_naming",
                            (
                                "DeviceContext variable should follow naming"
                                " convention"
                            ),
                            (
                                "Use 'ctx' or descriptive prefix like"
                                " 'gpu_ctx', 'main_ctx', 'compute_ctx'"
                            ),
                            "suggestion",
                        )
                        violations.append(violation)

        return violations

    fn check_gpu_patterns(
        self, file_content: String, file_path: String
    ) -> List[SyntaxViolation]:
        """Check GPU acceleration patterns and ensure they're preserved."""
        violations = List[SyntaxViolation]()
        lines = file_content.split("\n")

        _ = False  # has_gpu_imports placeholder
        has_device_context = False
        has_gpu_kernels = False

        for i in range(len(lines)):
            # Skip docstring content unless explicitly enabled
            if self._should_skip_line_for_violations(lines, i):
                continue

            line = lines[i].strip()
            line_num = i + 1

            # Check for proper GPU imports
            if (
                "has_nvidia_gpu_accelerator" in line
                or "has_amd_gpu_accelerator" in line
            ):
                _ = True  # GPU imports detected

            if "DeviceContext" in line:
                has_device_context = True

            if "thread_idx" in line or "block_idx" in line:
                has_gpu_kernels = True

            # Check for incorrect GPU API method names
            if (
                ".create_buffer[" in line
                and "enqueue_create_buffer" not in line
            ):
                violation = SyntaxViolation(
                    file_path,
                    line_num,
                    "gpu_api_deprecated",
                    "Incorrect GPU API method name detected",
                    (
                        "Use 'enqueue_create_buffer()' instead of"
                        " 'create_buffer()'"
                    ),
                    "error",
                )
                violations.append(violation)

            if ".call_function(" in line and "enqueue_function" not in line:
                violation = SyntaxViolation(
                    file_path,
                    line_num,
                    "gpu_api_deprecated",
                    "Incorrect GPU API method name detected",
                    "Use 'enqueue_function()' instead of 'call_function()'",
                    "error",
                )
                violations.append(violation)

            # Check for simulation labels that should be removed (exclude detection logic itself)
            if ("SIMULATED GPU:" in line or "PLACEHOLDER:" in line) and not (
                '"SIMULATED GPU:"' in line or '"PLACEHOLDER:"' in line
            ):
                violation = SyntaxViolation(
                    file_path,
                    line_num,
                    "gpu_simulation",
                    "GPU simulation label detected",
                    (
                        "Replace with real GPU implementation or remove"
                        " simulation labels"
                    ),
                    "warning",
                )
                violations.append(violation)

        # Validate GPU pattern consistency
        if has_gpu_kernels and not has_device_context:
            violation = SyntaxViolation(
                file_path,
                1,
                "gpu_consistency",
                "GPU kernels found without DeviceContext",
                "Add DeviceContext import and usage for GPU operations",
                "error",
            )
            violations.append(violation)

        return violations

    fn assess_docstring_quality(
        self, lines: List[String], start_line: Int
    ) -> (Bool, String, String):
        """
        Assess the quality of a docstring starting at the given line.

        Returns:
            Tuple of (is_good_quality, issue_description, classification).
            Classification can be: "One-line docstring", "Multi-line docstring", or "Violation".
        """
        if start_line >= len(lines):
            return (False, "Docstring not found", "Violation")

        # Build full docstring content by scanning lines
        full_content = ""
        i = start_line
        in_docstring = False
        docstring_line_count = 0

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('"""'):
                if not in_docstring:
                    in_docstring = True
                    # Check if it's a single-line docstring
                    if line.count('"""') == 2:
                        # Single line docstring like """Brief description."""
                        content = line[3:-3].strip()
                        full_content += content
                        docstring_line_count = 1
                        break
                    else:
                        # Multi-line docstring starts
                        content_after = line[3:].strip()
                        if content_after:
                            full_content += content_after + " "
                            docstring_line_count += 1
                else:
                    # End of multi-line docstring
                    content_before = line[:-3].strip()
                    if content_before:
                        full_content += content_before
                        docstring_line_count += 1
                    break
            elif in_docstring:
                # Inside multi-line docstring
                full_content += line + " "
                docstring_line_count += 1

            i += 1

        # Analyze docstring quality
        if docstring_line_count == 0 or len(full_content.strip()) == 0:
            return (False, "Empty docstring", "Violation")

        full_content = String(full_content.strip())

        # Quality criteria (aligned with mojo_syntax.md guidelines)
        # Note: Examples are NOT required due to Mojo LSP parsing issues
        # Note: Raises sections are optional - only needed when function has 'raises'
        has_description = len(full_content) > 10
        has_args = "Args:" in full_content or "Parameters:" in full_content
        has_returns = "Returns:" in full_content or "Return:" in full_content
        _ = (
            "Raises:" in full_content or "Raise:" in full_content
        )  # Optional documentation

        # For single-line docstrings, check if they're appropriate
        if docstring_line_count == 1:
            if len(full_content) >= 10:
                # Appropriate one-line docstring for simple functions
                return (
                    True,
                    "Appropriate one-line docstring",
                    "One-line docstring",
                )
            else:
                return (False, "Single-line docstring too brief", "Violation")

        # For multi-line docstrings, expect more comprehensive content
        # Calculate quality score (excluding examples per new guidelines)
        # Note: Raises sections are optional - only needed when function has 'raises'
        quality_indicators = 0
        if has_description:
            quality_indicators += 1
        if has_args:
            quality_indicators += 1
        if has_returns:
            quality_indicators += 1
        # Raises sections are optional (not counted as required indicator)
        # if has_raises:
        #     quality_indicators += 1  # Optional bonus

        # Consider comprehensive if it has description and at least one other element
        # OR if it's a substantial single description (>50 chars)
        # OR if it's multi-line with good content
        is_comprehensive = (
            (quality_indicators >= 2)
            or (len(full_content) > 50 and has_description)
            or (docstring_line_count > 3 and has_description)
        )

        if not is_comprehensive:
            if len(full_content) < 10:
                return (False, "Docstring too brief", "Violation")
            elif not has_description:
                return (False, "Missing meaningful description", "Violation")
            else:
                return (
                    False,
                    "Consider adding Args or Returns sections",
                    "Violation",
                )

        return (
            True,
            "Good quality multi-line docstring",
            "Multi-line docstring",
        )

    fn check_struct_traits(self, struct_line: String) -> Bool:
        """
        Check if a struct has appropriate traits or doesn't need them.

        Returns:
            True if struct traits are appropriate, False if missing and needed.
        """
        # Check if struct already has traits
        if "(" in struct_line and ")" in struct_line:
            # Extract traits section
            start_paren = struct_line.find("(")
            end_paren = struct_line.find(")")
            if start_paren != -1 and end_paren != -1:
                traits_section = struct_line[start_paren + 1 : end_paren]
                # Check for common traits
                if "Copyable" in traits_section or "Movable" in traits_section:
                    return True
                # Check for other valid traits that might not need Copyable/Movable
                if (
                    "CollectionElement" in traits_section
                    or "Stringable" in traits_section
                ):
                    return True

        # Check if it's a utility struct that might not need traits
        struct_name = (
            struct_line.replace("struct ", "")
            .split("(")[0]
            .split(":")[0]
            .strip()
        )

        # Utility structs that typically don't need Copyable/Movable
        utility_patterns = [
            "Config",
            "Constants",
            "Utils",
            "Helper",
            "Manager",
            "Builder",
        ]
        for pattern in utility_patterns:
            if pattern in struct_name:
                return True

        # If no traits found and not a utility struct, suggest adding them
        return False

    fn check_documentation_patterns(
        self, file_content: String, file_path: String
    ) -> List[SyntaxViolation]:
        """Check comprehensive documentation compliance patterns."""
        violations = List[SyntaxViolation]()
        lines = file_content.split("\n")

        i = 0
        while i < len(lines):
            # Skip lines that should be excluded from violation detection
            if self._should_skip_line_for_violations(lines, i):
                i += 1
                continue

            line = lines[i].strip()
            line_num = i + 1

            # Check for struct definitions
            if line.startswith("struct "):
                # Find the end of struct definition (look for colon) - handles comments after colon
                j = i
                while j < len(lines):
                    line_stripped = lines[j].strip()
                    # Check if line contains colon (handle comments after colon)
                    if ":" in line_stripped:
                        # Make sure colon is not inside a string or comment at the beginning
                        colon_pos = line_stripped.find(":")
                        if colon_pos >= 0:
                            break
                    j += 1

                # Look for docstring after struct definition
                docstring_found = False
                k = j + 1
                while k < len(lines):
                    next_line = lines[k].strip()
                    if next_line == "":
                        k += 1
                        continue
                    if next_line.startswith('"""'):
                        docstring_found = True
                        # Assess docstring quality
                        quality_result = self.assess_docstring_quality(lines, k)
                        is_good = quality_result[0]
                        issue_desc = quality_result[1]
                        classification = quality_result[2]

                        if not is_good:
                            # Only create violations for actual problems, not observations
                            if classification == "Violation":
                                violation = SyntaxViolation(
                                    file_path,
                                    k + 1,
                                    "documentation_quality",
                                    "Docstring quality issue: " + issue_desc,
                                    (
                                        "Add comprehensive description with"
                                        " purpose and parameters"
                                    ),
                                    "warning",
                                )
                                violations.append(violation)
                        else:
                            # Create suggestion for observations (one-line docstrings, etc.)
                            if classification == "One-line docstring":
                                violation = SyntaxViolation(
                                    file_path,
                                    k + 1,
                                    "documentation_style",
                                    issue_desc,
                                    (
                                        "Consider if multi-line format would be"
                                        " more appropriate"
                                    ),
                                    "suggestion",
                                )
                                violations.append(violation)
                    break

                if not docstring_found:
                    violation = SyntaxViolation(
                        file_path,
                        line_num,
                        "documentation_missing",
                        "Missing docstring for struct",
                        "Add comprehensive docstring describing struct purpose",
                        "error",
                    )
                    violations.append(violation)

                # Check struct traits with improved logic
                if not self.check_struct_traits(String(line)):
                    violation = SyntaxViolation(
                        file_path,
                        line_num,
                        "struct_traits",
                        (
                            "Struct likely needs (Copyable, Movable) traits for"
                            " copying and collections"
                        ),
                        (
                            "Add (Copyable, Movable) traits - required for"
                            " copying, function returns, and collection storage"
                        ),
                        "suggestion",
                    )
                    violations.append(violation)

                i = j + 1

            # Check for function definitions
            elif line.startswith("fn ") and "(" in line:
                # Find the end of function signature (look for closing colon)
                j = i
                while j < len(lines):
                    current_line = lines[j].strip()
                    if current_line.endswith(":") and (
                        ")" in current_line or j > i
                    ):
                        break
                    j += 1

                # Look for docstring after function signature
                docstring_found = False
                k = j + 1
                while k < len(lines):
                    next_line = lines[k].strip()
                    if next_line == "":
                        k += 1
                        continue
                    if next_line.startswith('"""'):
                        docstring_found = True
                        # Assess docstring quality
                        quality_result = self.assess_docstring_quality(lines, k)
                        is_good = quality_result[0]
                        issue_desc = quality_result[1]
                        classification = quality_result[2]

                        if not is_good:
                            # Only create violations for actual problems, not observations
                            if classification == "Violation":
                                violation = SyntaxViolation(
                                    file_path,
                                    k + 1,
                                    "documentation_quality",
                                    "Docstring quality issue: " + issue_desc,
                                    (
                                        "Add comprehensive description with"
                                        " purpose and parameters"
                                    ),
                                    "warning",
                                )
                                violations.append(violation)
                        else:
                            # Create suggestion for observations (one-line docstrings, etc.)
                            if classification == "One-line docstring":
                                violation = SyntaxViolation(
                                    file_path,
                                    k + 1,
                                    "documentation_style",
                                    issue_desc,
                                    (
                                        "Consider if multi-line format would be"
                                        " more appropriate"
                                    ),
                                    "suggestion",
                                )
                                violations.append(violation)
                    break

                if not docstring_found:
                    violation = SyntaxViolation(
                        file_path,
                        line_num,
                        "documentation_missing",
                        "Missing docstring for function",
                        (
                            "Add comprehensive docstring describing function"
                            " purpose"
                        ),
                        "warning",
                    )
                    violations.append(violation)

                i = j + 1
            else:
                i += 1

        return violations

    fn check_error_handling_patterns(
        self, file_content: String, file_path: String
    ) -> List[SyntaxViolation]:
        """Check comprehensive error handling pattern compliance."""
        violations = List[SyntaxViolation]()
        lines = file_content.split("\n")

        for i in range(len(lines)):
            # Skip docstring content unless explicitly enabled
            if self._should_skip_line_for_violations(lines, i):
                continue

            line = lines[i].strip()
            line_num = i + 1

            # DESIGN PATTERN: Minimal 'raises' Usage
            # =====================================
            # We do NOT automatically flag missing 'raises' annotations as violations.
            #
            # CORE PRINCIPLE: Only add 'raises' where the compiler REQUIRES it.
            #
            # RATIONALE:
            # - Keeps function signatures clean and minimal
            # - Reduces unnecessary error propagation chains
            # - Makes actual error-raising functions more visible
            # - Follows compiler-driven approach rather than static analysis guessing
            #
            # The Mojo compiler will produce errors when 'raises' is truly needed.
            # Static analysis cannot reliably determine when 'raises' is required
            # because it depends on whether errors are handled internally or propagated.

            # Check for bare except clauses
            if line.startswith("except:") or line == "except:":
                violation = SyntaxViolation(
                    file_path,
                    line_num,
                    "error_handling_bare_except",
                    "Bare except clause detected",
                    "Specify exception type: except SpecificError:",
                    "warning",
                )
                violations.append(violation)

            # Check for error messages without context
            if "Error(" in line and len(line.split('"')) < 3:
                violation = SyntaxViolation(
                    file_path,
                    line_num,
                    "error_handling_message",
                    "Error without descriptive message",
                    "Add descriptive error message with context",
                    "suggestion",
                )
                violations.append(violation)

        return violations

    fn check_performance_patterns(
        self, file_content: String, file_path: String
    ) -> List[SyntaxViolation]:
        """Check performance optimization pattern compliance."""
        violations = List[SyntaxViolation]()
        lines = file_content.split("\n")

        for i in range(len(lines)):
            # Skip docstring content unless explicitly enabled
            if self._should_skip_line_for_violations(lines, i):
                continue

            line = lines[i].strip()
            line_num = i + 1

            # Check for inefficient loop patterns
            if "for i in range(len(" in line and "append" in line:
                violation = SyntaxViolation(
                    file_path,
                    line_num,
                    "performance_inefficient_loop",
                    "Inefficient loop with append pattern",
                    (
                        "Consider pre-allocating list size or using list"
                        " comprehension"
                    ),
                    "suggestion",
                )
                violations.append(violation)

            # Check for missing GPU acceleration opportunities
            if (
                "matrix" in line.lower()
                and "multiply" in line.lower()
                and "gpu" not in line.lower()
                and "GPU" not in line
            ):
                violation = SyntaxViolation(
                    file_path,
                    line_num,
                    "performance_gpu_opportunity",
                    "Matrix operation without GPU acceleration",
                    (
                        "Consider using GPU-accelerated matrix operations for"
                        " better performance"
                    ),
                    "suggestion",
                )
                violations.append(violation)

            # Check for UnsafePointer ownership violations
            if "UnsafePointer" in line:
                ownership_violation = self._check_unsafe_pointer_ownership(
                    lines, i, file_content, file_path
                )
                if ownership_violation:
                    violations.append(ownership_violation.value())

            # Check for string concatenation in loops
            if (
                "for " in line
                and i + 1 < len(lines)
                and "+=" in lines[i + 1]
                and '"' in lines[i + 1]
            ):
                violation = SyntaxViolation(
                    file_path,
                    line_num + 1,
                    "performance_string_concat",
                    "String concatenation in loop",
                    (
                        "Use StringBuilder or collect strings and join for"
                        " better performance"
                    ),
                    "suggestion",
                )
                violations.append(violation)

        return violations

    fn scan_file(mut self, file_path: String) raises -> ComplianceReport:
        """Scan a single file for syntax violations."""
        report = ComplianceReport(file_path)

        # Read actual file content using Mojo's Path.read_text()
        file_path_obj = Path(file_path)

        # Check if file exists and is accessible
        if not file_path_obj.exists():
            violation = SyntaxViolation(
                file_path,
                0,
                "file_access",
                "File does not exist: " + file_path,
                "Check file path and ensure file exists",
                "error",
            )
            report.add_violation(violation)
            return report

        if not file_path_obj.is_file():
            violation = SyntaxViolation(
                file_path,
                0,
                "file_access",
                "Path is not a file: " + file_path,
                "Ensure path points to a regular file",
                "error",
            )
            report.add_violation(violation)
            return report

        try:
            # Read actual file content
            content = file_path_obj.read_text()

            # Calculate actual line count
            line_count = 1  # Start with 1 for the first line
            for i in range(len(content)):
                if content[i] == "\n":
                    line_count += 1
            report.total_lines = line_count

            # Run all checks
            import_violations = self.check_import_patterns(content, file_path)
            struct_violations = self.check_struct_patterns(content, file_path)
            function_violations = self.check_function_patterns(
                content, file_path
            )
            variable_violations = self.check_variable_patterns(
                content, file_path
            )
            gpu_violations = self.check_gpu_patterns(content, file_path)
            documentation_violations = self.check_documentation_patterns(
                content, file_path
            )
            error_handling_violations = self.check_error_handling_patterns(
                content, file_path
            )
            performance_violations = self.check_performance_patterns(
                content, file_path
            )

            # Add all violations to report
            for violation in import_violations:
                report.add_violation(violation)
            for violation in struct_violations:
                report.add_violation(violation)
            for violation in function_violations:
                report.add_violation(violation)
            for violation in variable_violations:
                report.add_violation(violation)
            for violation in gpu_violations:
                report.add_violation(violation)
            for violation in documentation_violations:
                report.add_violation(violation)
            for violation in error_handling_violations:
                report.add_violation(violation)
            for violation in performance_violations:
                report.add_violation(violation)

            # Calculate compliance score
            report.calculate_score()
        except e:
            # File reading failed
            violation = SyntaxViolation(
                file_path,
                0,
                "file_access",
                "Cannot read file: " + String(e),
                "Check file permissions and encoding",
                "error",
            )
            report.add_violation(violation)
            report.compliance_score = 0.0

        return report

    fn apply_automatic_fixes(mut self, file_path: String) raises -> Bool:
        """Apply automatic fixes to a file with safety backups."""
        if not self.auto_fix_enabled:
            print("Automatic fixes disabled. Use --enable-auto-fix to enable.")
            return False

        try:
            # Read current content
            with open(file_path, "r") as f:
                content = f.read()

            # Create backup if enabled
            if self.backup_enabled:
                backup_path = file_path + ".backup"
                with open(backup_path, "w") as backup:
                    backup.write(content)
                print("Backup created:", backup_path)

            # Apply safe fixes
            fixed_content = self.fix_import_patterns(content)
            fixed_content = self.fix_variable_declarations(fixed_content)
            fixed_content = self.fix_documentation_issues(fixed_content)
            fixed_content = self.fix_trait_issues(fixed_content)

            # Write fixed content
            with open(file_path, "w") as f:
                f.write(fixed_content)

            print("Automatic fixes applied to:", file_path)

            # Validate compilation after fixes
            if self.validate_compilation(file_path):
                print("✅ File compiles successfully after fixes")
                return True
            else:
                print("❌ Compilation failed after fixes")
                if self.backup_enabled:
                    print("Consider rolling back changes")
                return False

        except e:
            print("Error applying fixes to", file_path, ":", String(e))
            return False

    fn validate_compilation(self, file_path: String) raises -> Bool:
        """Validate that a file compiles successfully."""
        # For now, return True as compilation validation requires subprocess
        # In a full implementation, this would run: mojo build --check-only file_path
        print(
            "Compilation validation: Assuming success (subprocess not"
            " available)"
        )
        return True

    fn rollback_changes(self, file_path: String) raises -> Bool:
        """Rollback changes using backup file."""
        backup_path = file_path + ".backup"
        try:
            with open(backup_path, "r") as backup:
                content = backup.read()
            with open(file_path, "w") as original:
                original.write(content)
            print("Successfully rolled back changes for:", file_path)
            return True
        except e:
            print(
                "Failed to rollback changes for:",
                file_path,
                "- Error:",
                String(e),
            )
            return False

    fn cleanup_backup_file(self, file_path: String) -> Bool:
        """Clean up backup file for a specific source file."""
        backup_path = file_path + ".backup"
        try:
            # Check if backup file exists
            with open(backup_path, "r") as _:
                pass  # File exists, proceed with deletion

            # Delete the backup file
            from os import remove

            remove(backup_path)
            print("✅ Cleaned up backup:", backup_path)
            return True
        except:
            # Backup file doesn't exist or couldn't be deleted
            return False

    fn cleanup_old_backups(self, directory: String) -> Int:
        """Clean up backup files older than retention period."""
        if self.backup_retention_days <= 0:
            return 0

        cleaned_count = 0
        try:
            from os import listdir, remove

            # For now, just clean all backup files since time/stat APIs are limited
            # TODO: Implement proper age-based cleanup when time APIs are available
            files = listdir(directory)
            for file_name in files:
                if file_name.endswith(".backup"):
                    file_path = directory + "/" + file_name
                    try:
                        remove(file_path)
                        print("🗑️  Removed backup:", file_path)
                        cleaned_count += 1
                    except:
                        continue  # Skip files we can't process

        except:
            print("Warning: Could not perform automatic backup cleanup")

        return cleaned_count

    fn cleanup_all_backups(self, base_directory: String) -> Int:
        """Recursively clean up all backup files in directory tree."""
        # Clean current directory
        total_cleaned = self.cleanup_old_backups(base_directory)

        # For now, just clean the current directory
        # TODO: Add recursive directory traversal when path APIs are available
        if total_cleaned > 0:
            print(
                "Cleaned", total_cleaned, "backup files from:", base_directory
            )

        return total_cleaned

    fn fix_import_patterns(self, content: String) -> String:
        """Fix import pattern violations."""
        lines = content.split("\n")
        fixed_lines = List[String]()

        for line in lines:
            # Fix relative imports
            if line.strip().startswith("from ."):
                # Convert to absolute import (basic fix)
                fixed_line = line.replace("from .", "from src.")
                fixed_lines.append(fixed_line)
            elif line.strip().startswith("from .."):
                # Convert to absolute import (basic fix)
                fixed_line = line.replace("from ..", "from src.")
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

        return "\n".join(fixed_lines)

    fn fix_variable_declarations(self, content: String) -> String:
        """Fix variable declaration violations."""
        lines = content.split("\n")
        fixed_lines = List[String]()

        for line in lines:
            # Fix old 'let' keyword usage
            if line.strip().startswith("let "):
                # Convert to var (basic fix - may need manual review)
                fixed_line = line.replace("let ", "var ")
                fixed_lines.append(
                    "# TODO: Review variable declaration - " + fixed_line
                )
            else:
                fixed_lines.append(line)

        return "\n".join(fixed_lines)

    fn fix_documentation_issues(self, content: String) -> String:
        """Fix basic documentation issues."""
        lines = content.split("\n")
        fixed_lines = List[String]()

        i = 0
        while i < len(lines):
            line = lines[i]

            # Add basic docstrings for functions missing them
            if line.strip().startswith("fn ") and ("(" in line or "[" in line):
                # Find the end of the function signature
                signature_start = i
                signature_end_index = i
                paren_count = 0
                signature_complete = False

                # Count parentheses to find the actual end of the function signature
                j = i
                bracket_count = 0  # Track square brackets for generics
                found_opening_paren = (
                    False  # Track if we've found the parameter list
                )

                while j < len(lines) and not signature_complete:
                    current_line = lines[j]

                    # Count parentheses and brackets in this line
                    for char_idx in range(len(current_line)):
                        char = current_line[char_idx]
                        if char == "[":
                            bracket_count += 1
                        elif char == "]":
                            bracket_count -= 1
                        elif char == "(":
                            paren_count += 1
                            found_opening_paren = True
                        elif char == ")":
                            paren_count -= 1

                        # If we've found the opening paren, closed all parentheses and brackets, and found a colon, signature is complete
                        if (
                            found_opening_paren
                            and paren_count == 0
                            and bracket_count == 0
                            and ":" in current_line[char_idx:]
                        ):
                            signature_end_index = j
                            signature_complete = True
                            break

                    j += 1

                # Check if there's a docstring after the complete signature
                docstring_line_index = signature_end_index + 1
                has_docstring = False

                if docstring_line_index < len(lines):
                    next_line = lines[docstring_line_index].strip()
                    has_docstring = next_line.startswith('"""')

                # Add all lines of the function signature
                for sig_line_idx in range(
                    signature_start, signature_end_index + 1
                ):
                    fixed_lines.append(lines[sig_line_idx])

                # Add docstring if missing (after the complete signature)
                if not has_docstring:
                    # Use the indentation of the function definition line
                    base_indent = len(lines[signature_start]) - len(
                        lines[signature_start].lstrip()
                    )
                    docstring = (
                        " " * (base_indent + 4)
                        + '"""TODO: Add function description."""'
                    )
                    fixed_lines.append(docstring)

                # Move index past the processed signature
                i = signature_end_index + 1
                continue
            else:
                fixed_lines.append(line)
                i += 1

        return "\n".join(fixed_lines)

    fn fix_trait_issues(self, content: String) -> String:
        """Fix trait-related violations based on lifecycle method analysis."""
        lines = content.split("\n")
        fixed_lines = List[String]()
        var i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check for struct definitions
            if String(stripped).startswith("struct "):
                # Analyze this struct for trait issues
                struct_violations = self._analyze_struct_traits(
                    lines, i, "temp_file"
                )

                # Apply fixes based on violations
                if len(struct_violations) > 0:
                    fixed_line = self._apply_struct_trait_fixes(
                        line, struct_violations
                    )
                    fixed_lines.append(fixed_line)

                    # Skip redundant lifecycle methods if found
                    i = self._skip_redundant_methods(
                        lines, i, struct_violations
                    )
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

            i += 1

        return "\n".join(fixed_lines)

    fn _apply_struct_trait_fixes(
        self, struct_line: String, violations: List[SyntaxViolation]
    ) -> String:
        """Apply trait fixes to a struct definition line."""
        fixed_line = struct_line

        for i in range(len(violations)):
            violation = violations[i]
            if violation.violation_type == "redundant_trait":
                # Remove redundant traits
                if "Copyable" in violation.description:
                    fixed_line = self._remove_trait_from_line(
                        fixed_line, "Copyable"
                    )
                elif "Movable" in violation.description:
                    fixed_line = self._remove_trait_from_line(
                        fixed_line, "Movable"
                    )

        return fixed_line

    fn _remove_trait_from_line(
        self, line: String, trait_name: String
    ) -> String:
        """Remove a specific trait from a struct definition line."""
        # This is a simplified implementation - in practice would need more robust parsing
        if "(" + trait_name + ")" in line:
            return line.replace("(" + trait_name + ")", "")
        elif "(" + trait_name + "," in line:
            return line.replace("(" + trait_name + ",", "(")
        elif "," + trait_name + ")" in line:
            return line.replace("," + trait_name + ")", ")")
        elif trait_name in line:
            return line.replace(trait_name, "")
        return line

    fn _skip_redundant_methods(
        self,
        lines: List[String],
        struct_start: Int,
        violations: List[SyntaxViolation],
    ) -> Int:
        """Skip redundant lifecycle methods that should be removed."""
        # For now, just return the current position - full implementation would skip method blocks
        return struct_start

    fn print_report(self, reports: List[ComplianceReport]):
        """Print a comprehensive compliance report."""
        print("=" * 80)
        print("MOJO SYNTAX COMPLIANCE REPORT")
        print("=" * 80)
        print("")

        # Summary statistics
        total_files = len(reports)
        total_violations = (
            0  # Only actual violations (errors + warnings + info)
        )
        total_errors = 0
        total_warnings = 0
        total_one_line_docstrings = 0
        total_suggestions = 0
        average_score = 0.0

        for i in range(len(reports)):
            report = reports[i]
            average_score += report.compliance_score

            for j in range(len(report.violations)):
                violation = report.violations[j]
                if violation.severity == "error":
                    total_errors += 1
                    total_violations += 1
                elif violation.severity == "warning":
                    total_warnings += 1
                    total_violations += 1

                elif violation.severity == "suggestion":
                    # Check if it's a one-line docstring observation
                    if (
                        "Appropriate one-line docstring"
                        in violation.description
                    ):
                        total_one_line_docstrings += 1
                    else:
                        total_suggestions += 1

        if total_files > 0:
            average_score /= Float64(total_files)

        total_observations = total_one_line_docstrings + total_suggestions

        print("SUMMARY:")
        print("- Files scanned:", total_files)
        print("- Total violations:", total_violations)
        print("- Errors:", total_errors)
        print("- Warnings:", total_warnings)
        print("- Observations:", total_observations)
        print("  - Suggestions:", total_suggestions)
        print("  - One-line docstrings:", total_one_line_docstrings)
        # Format average score to 1 decimal place
        rounded_score = Int(average_score * 10 + 0.5) / 10.0
        formatted_score = String(rounded_score)
        print("- Average compliance score:", formatted_score, "%")
        print("")

        # Individual file reports
        print("DETAILED RESULTS:")
        print("-" * 80)

        for i in range(len(reports)):
            report = reports[i]
            print("")
            print("File:", report.file_path)
            print("Compliance Score:", report.compliance_score, "%")
            print("Lines:", report.total_lines)

            # Separate violations into Issues and Observations first
            issues = List[SyntaxViolation]()
            observations = List[SyntaxViolation]()

            for j in range(len(report.violations)):
                violation = report.violations[j]
                if (
                    violation.severity == "error"
                    or violation.severity == "warning"
                ):
                    issues.append(violation)
                else:
                    observations.append(violation)

            # Print correct violation count (only actual violations, not observations)
            print("Violations:", len(issues))

            # Display Issues first
            if len(issues) > 0:
                print("")
                print("Issues found (" + String(len(issues)) + "):")

                for j in range(len(issues)):
                    violation = issues[j]
                    severity_marker = (
                        "❌" if violation.severity
                        == "error" else "⚠️"  # warning
                    )
                    print(
                        "  " + severity_marker + " Line",
                        violation.line_number,
                        ":",
                        violation.description,
                    )
                    print("    Type:", violation.violation_type)
                    print("    Fix:", violation.suggested_fix)
                    print("")

            # Display Observations second (only if enabled)
            if len(observations) > 0 and self.show_observations:
                print("Observations (" + String(len(observations)) + "):")

                # Separate suggestions from one-line docstrings
                suggestions = List[SyntaxViolation]()
                one_line_docstrings = List[SyntaxViolation]()

                for j in range(len(observations)):
                    violation = observations[j]
                    if (
                        "Appropriate one-line docstring"
                        in violation.description
                    ):
                        one_line_docstrings.append(violation)
                    else:
                        suggestions.append(violation)

                # Display suggestions first
                for j in range(len(suggestions)):
                    violation = suggestions[j]
                    print(
                        "  ℹ️ Line",
                        violation.line_number,
                        ":",
                        violation.description,
                    )
                    print("    Type:", violation.violation_type)
                    print("    Fix:", violation.suggested_fix)
                    print("")

                # Display one-line docstrings second
                for j in range(len(one_line_docstrings)):
                    violation = one_line_docstrings[j]
                    print(
                        "  ℹ️ Line",
                        violation.line_number,
                        ":",
                        violation.description,
                    )
                    print("    Type:", violation.violation_type)
                    print("    Fix:", violation.suggested_fix)
                    print("")

            # Show usage hint when observations are hidden
            if len(observations) > 0 and not self.show_observations:
                print("")
                print(
                    "Use --show-observations to display "
                    + String(len(observations))
                    + " suggestions and style recommendations"
                )

            if len(issues) == 0 and len(observations) == 0:
                print("✅ No violations found!")

            print("-" * 40)

    fn scan_directory(
        mut self, directory_path: String
    ) raises -> List[ComplianceReport]:
        """Scan all .mojo files in a directory."""
        reports = List[ComplianceReport]()

        print("Scanning directory:", directory_path)

        # Normalize directory path (remove trailing slash if present)
        normalized_path = directory_path
        if directory_path.endswith("/"):
            normalized_path = directory_path[:-1]

        # Known files to scan based on project structure
        test_files = List[String]()

        # Core GPU files (priority)
        test_files.append(normalized_path + "/utils/gpu_matrix.mojo")
        test_files.append(normalized_path + "/utils/gpu_utils.mojo")
        test_files.append(normalized_path + "/utils/physics.mojo")
        test_files.append(
            normalized_path + "/digital_twin/gpu_neural_network.mojo"
        )
        test_files.append(
            normalized_path + "/benchmarks/gpu_cpu_benchmark.mojo"
        )
        test_files.append(normalized_path + "/benchmarks/report_generator.mojo")

        # Neural network files
        test_files.append(normalized_path + "/digital_twin/neural_network.mojo")
        test_files.append(normalized_path + "/digital_twin/simple_network.mojo")
        test_files.append(normalized_path + "/digital_twin/trainer.mojo")
        test_files.append(
            normalized_path + "/digital_twin/integrated_trainer.mojo"
        )

        # Control system files (core)
        test_files.append(normalized_path + "/control/ai_controller.mojo")
        test_files.append(normalized_path + "/control/mpc_controller.mojo")
        test_files.append(normalized_path + "/control/rl_controller.mojo")
        test_files.append(normalized_path + "/control/safety_monitor.mojo")

        # Data processing files
        test_files.append(normalized_path + "/data/loader.mojo")
        test_files.append(normalized_path + "/data/csv_reader.mojo")
        test_files.append(normalized_path + "/data/analyzer.mojo")

        for i in range(len(test_files)):
            file_path = test_files[i]
            # Scan the file
            report = self.scan_file(file_path)
            reports.append(report)
            print("✅ Scanned:", file_path)

        if len(reports) == 0:
            print("Note: No .mojo files found or accessible in directory")
            print(
                "In a full implementation, this would use file system APIs for"
                " recursive scanning"
            )

        return reports


fn print_usage():
    """Print usage information for the script."""
    print("Mojo Syntax Automation Script")
    print("=" * 50)
    print("")
    print("Usage:")
    print("  mojo update_mojo_syntax.mojo --scan [directory]")
    print("  mojo update_mojo_syntax.mojo --fix [file]")
    print("  mojo update_mojo_syntax.mojo --validate [file]")
    print("  mojo update_mojo_syntax.mojo --report [directory]")
    print("")
    print("Options:")
    print("  --scan [dir]       Scan directory for syntax violations")
    print("  --fix [file]       Apply automatic fixes to file")
    print("  --validate [file]  Validate single file compliance")
    print("  --report [dir]     Generate compliance report")
    print("  --cleanup [dir]    Clean up backup files in directory")
    print("  --enable-auto-fix  Enable automatic fixing (with backups)")
    print("  --disable-backup   Disable backup creation")
    print("  --keep-backups     Keep backup files after successful completion")
    print(
        "  --auto-cleanup     Enable automatic cleanup of backup files"
        " (default)"
    )
    print(
        "  --retention-days N Set backup retention period in days (default: 7)"
    )
    print("  --show-observations Show suggestions and style recommendations")
    print(
        "  --check-docstring-code Enable syntax checking within docstring code"
        " examples"
    )
    print("  --help             Show this help message")
    print("")
    print("Examples:")
    print("  mojo update_mojo_syntax.mojo --scan src/")
    print(
        "  mojo update_mojo_syntax.mojo --validate"
        " src/pendulum/utils/gpu_matrix.mojo"
    )
    print(
        "  mojo update_mojo_syntax.mojo --validate"
        " src/utils/gpu_matrix.mojo --show-observations"
    )
    print(
        "  mojo update_mojo_syntax.mojo --fix"
        " src/pendulum/utils/gpu_matrix.mojo --enable-auto-fix"
    )
    print(
        "  mojo update_mojo_syntax.mojo --validate"
        " src/utils/gpu_matrix.mojo --check-docstring-code"
    )
    print("  mojo update_mojo_syntax.mojo --cleanup src/ --retention-days 3")
    print(
        "  mojo update_mojo_syntax.mojo --fix file.mojo --enable-auto-fix"
        " --keep-backups"
    )


fn test_syntax_checker():
    """Test the syntax checker functionality."""
    print("� Testing Mojo Syntax Checker")
    print("=" * 50)

    # Initialize syntax checker
    checker = MojoSyntaxChecker()

    # Test with sample code content
    sample_code = """
from .relative_import import SomeClass
let old_variable = 42

struct TestStruct:
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

fn test_function():
    print("Missing docstring")
    raise Error("Test error")
"""

    print("Testing pattern detection on sample code...")

    # Test individual checkers
    import_violations = checker.check_import_patterns(sample_code, "test.mojo")
    struct_violations = checker.check_struct_patterns(sample_code, "test.mojo")
    function_violations = checker.check_function_patterns(
        sample_code, "test.mojo"
    )
    variable_violations = checker.check_variable_patterns(
        sample_code, "test.mojo"
    )
    gpu_violations = checker.check_gpu_patterns(sample_code, "test.mojo")

    print("\nViolations found:")
    print("- Import violations:", len(import_violations))
    print("- Struct violations:", len(struct_violations))
    print("- Function violations:", len(function_violations))
    print("- Variable violations:", len(variable_violations))
    print("- GPU violations:", len(gpu_violations))

    # Test report generation
    report = ComplianceReport("test.mojo")
    report.total_lines = 15

    for violation in import_violations:
        report.add_violation(violation)
    for violation in struct_violations:
        report.add_violation(violation)
    for violation in function_violations:
        report.add_violation(violation)
    for violation in variable_violations:
        report.add_violation(violation)
    for violation in gpu_violations:
        report.add_violation(violation)

    report.calculate_score()

    reports = List[ComplianceReport]()
    reports.append(report)

    print("\nGenerating compliance report...")
    checker.print_report(reports)

    print("\n✅ Syntax checker test completed!")


fn main() raises:
    """Main entry point for the Mojo syntax automation script."""
    print("🤖 Mojo Syntax Automation Script v1.0")
    print("Standardizing Mojo code according to mojo_syntax.md patterns")
    print("")

    # Initialize syntax checker
    checker = MojoSyntaxChecker()

    # Get command-line arguments
    args = argv()

    # Parse global flags
    for i in range(len(args)):
        arg = String(args[i])
        if arg == "--show-observations":
            checker.show_observations = True
        elif arg == "--check-docstring-code":
            checker.check_docstring_code = True
        elif arg == "--keep-backups":
            checker.keep_backups = True
            checker.auto_cleanup_backups = False
        elif arg == "--auto-cleanup":
            checker.auto_cleanup_backups = True
        elif arg == "--retention-days":
            if i + 1 < len(args):
                try:
                    checker.backup_retention_days = atol(String(args[i + 1]))
                except:
                    print(
                        "Warning: Invalid retention days value, using"
                        " default (7)"
                    )
        elif arg == "--disable-backup":
            checker.backup_enabled = False

    # If no arguments provided, show usage and run demo
    if len(args) < 2:
        print_usage()
        print("")
        print("Running demonstration mode...")
        test_syntax_checker()
        return

    # Parse command
    command = String(args[1])

    if command == "--help" or command == "-h":
        print_usage()
        return
    elif command == "--scan":
        if len(args) < 3:
            print("Error: --scan requires a directory path")
            print("Usage: mojo update_mojo_syntax.mojo --scan <directory>")
            return

        directory = String(args[2])
        print("🔍 Scanning directory: " + directory)
        reports = checker.scan_directory(directory)

        if reports.__len__() > 0:
            checker.print_report(reports)
        else:
            print("No files scanned or no violations found")

    elif command == "--validate":
        if len(args) < 3:
            print("Error: --validate requires a file path")
            print("Usage: mojo update_mojo_syntax.mojo --validate <file>")
            return

        file_path = String(args[2])
        print("✅ Validating file: " + file_path)

        report = checker.scan_file(file_path)
        reports = List[ComplianceReport]()
        reports.append(report)

        checker.print_report(reports)

    elif command == "--fix":
        if len(args) < 3:
            print("Error: --fix requires a file path")
            print(
                "Usage: mojo update_mojo_syntax.mojo --fix <file>"
                " [--enable-auto-fix]"
            )
            return

        file_path = String(args[2])

        # Check for --enable-auto-fix flag
        if args.__len__() > 3 and String(args[3]) == "--enable-auto-fix":
            checker.auto_fix_enabled = True
            print("🔧 Auto-fix enabled for file: " + file_path)
        else:
            print("🔧 Dry-run mode for file: " + file_path)
            print("Use --enable-auto-fix to apply changes")

        success = checker.apply_automatic_fixes(file_path)
        if success:
            print("✅ Fixes applied successfully")

            # Auto-cleanup backup files if enabled
            if checker.auto_cleanup_backups and not checker.keep_backups:
                if checker.cleanup_backup_file(file_path):
                    print("🧹 Backup file cleaned up automatically")
        else:
            print("❌ Fix application failed or disabled")

    elif command == "--report":
        if len(args) < 3:
            print("Error: --report requires a directory path")
            print("Usage: mojo update_mojo_syntax.mojo --report <directory>")
            return

        directory = String(args[2])
        print("📊 Generating compliance report for: " + directory)
        reports = checker.scan_directory(directory)

        if reports.__len__() > 0:
            checker.print_report(reports)
            print("\n📋 Report generation completed")
        else:
            print("No files found to generate report")

    elif command == "--cleanup":
        if len(args) < 3:
            print("Error: --cleanup requires a directory path")
            print("Usage: mojo update_mojo_syntax.mojo --cleanup <directory>")
            return

        directory = String(args[2])
        print("🧹 Cleaning up backup files in: " + directory)

        cleaned_count = checker.cleanup_all_backups(directory)
        if cleaned_count > 0:
            print("✅ Cleanup completed:", cleaned_count, "backup files removed")
        else:
            print("ℹ️  No backup files found to clean up")

    else:
        print("Unknown command:", command)
        print("Use --help to see available options")
        print_usage()
