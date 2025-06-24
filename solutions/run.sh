#!/bin/bash

run_mojo_files() {
  local path_prefix="$1"

  for f in *.mojo; do
    if [ -f "$f" ] && [ "$f" != "__init__.mojo" ]; then
      # Extract flags for Mojo files (skip demo flags)
      flags=$(grep -o 'argv()\[1\] == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')

      if [ -z "$flags" ]; then
        echo "=== Running ${path_prefix}$f ==="
        mojo "$f" --sanitize address --sanitize thread || echo "Failed: ${path_prefix}$f"
      else
        for flag in $flags; do
          echo "=== Running ${path_prefix}$f with flag: $flag ==="
          mojo "$f" "$flag" --sanitize address --sanitize thread || echo "Failed: ${path_prefix}$f with $flag"
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

cd solutions || exit 1

for dir in p*/; do
  if [ -d "$dir" ]; then
    echo "=== Testing solutions in ${dir} ==="
    cd "$dir" || continue

    process_directory "${dir}"

    # Check for test directory and run mojo test
    if [ -d "test" ] || [ -d "tests" ]; then
      echo "=== Running tests in ${dir} ==="
      mojo test . || echo "Failed: mojo test in ${dir}"
    fi

    cd ..
  fi
done

cd ..
