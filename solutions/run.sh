#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 [PUZZLE_NAME]"
    echo "  PUZZLE_NAME: Optional puzzle name (e.g., p23, p14, etc.)"
    echo "  If no puzzle specified, runs all puzzles"
    echo ""
    echo "Examples:"
    echo "  $0        # Run all puzzles"
    echo "  $0 p23    # Run only p23 tests"
}

run_mojo_files() {
  local path_prefix="$1"

  for f in *.mojo; do
    if [ -f "$f" ] && [ "$f" != "__init__.mojo" ]; then
      # Extract flags for Mojo files (skip demo flags)
      flags=$(grep -o 'argv()\[1\] == "--[^"]*"\|test_type == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')

      if [ -z "$flags" ]; then
        echo "=== Running ${path_prefix}$f ==="
        mojo "$f" || echo "Failed: ${path_prefix}$f"
      else
        for flag in $flags; do
          echo "=== Running ${path_prefix}$f with flag: $flag ==="
          mojo "$f" "$flag" || echo "Failed: ${path_prefix}$f with $flag"
        done
      fi
    fi
  done
}

run_python_files() {
  local path_prefix="$1"

  for f in *.py; do
    if [ -f "$f" ]; then
      # Extract flags for Python files (sys.argv[1] pattern, skip demo flags)
      flags=$(grep -o 'sys\.argv\[1\] == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')

      if [ -z "$flags" ]; then
        echo "=== Running ${path_prefix}$f ==="
        python "$f" || echo "Failed: ${path_prefix}$f"
      else
        for flag in $flags; do
          echo "=== Running ${path_prefix}$f with flag: $flag ==="
          python "$f" "$flag" || echo "Failed: ${path_prefix}$f with $flag"
        done
      fi
    fi
  done
}

process_directory() {
  local path_prefix="$1"

  run_mojo_files "$path_prefix"
  run_python_files "$path_prefix"
}

# Parse command line arguments
SPECIFIC_PUZZLE=""
if [ $# -eq 1 ]; then
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        usage
        exit 0
    fi
    SPECIFIC_PUZZLE="$1"
elif [ $# -gt 1 ]; then
    echo "Error: Too many arguments"
    usage
    exit 1
fi

cd solutions || exit 1

# Function to test a specific directory
test_puzzle_directory() {
    local dir="$1"
    echo "=== Testing solutions in ${dir} ==="
    cd "$dir" || return 1

    process_directory "${dir}"

    # Check for test directory and run mojo test
    if [ -d "test" ] || [ -d "tests" ]; then
        echo "=== Running tests in ${dir} ==="
        mojo test . || echo "Failed: mojo test in ${dir}"
    fi

    cd ..
}

if [ -n "$SPECIFIC_PUZZLE" ]; then
    # Run specific puzzle
    if [ -d "${SPECIFIC_PUZZLE}/" ]; then
        test_puzzle_directory "${SPECIFIC_PUZZLE}/"
    else
        echo "Error: Puzzle directory '${SPECIFIC_PUZZLE}' not found"
        echo "Available puzzles:"
        ls -d p*/ 2>/dev/null | tr -d '/' | sort
        exit 1
    fi
else
    # Run all puzzles (original behavior)
    for dir in p*/; do
        if [ -d "$dir" ]; then
            test_puzzle_directory "$dir"
        fi
    done
fi

cd ..
