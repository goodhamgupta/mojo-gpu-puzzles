#!/bin/bash

run_mojo_files() {
  local path_prefix="$1"
  local current_dir=$(pwd)

  for f in *.mojo; do
    if [ -f "$f" ] && [ "$f" != "__init__.mojo" ]; then
      echo "=== Running ${path_prefix}$f ==="
      # Extract flags for Mojo files
      flags=$(grep -o 'argv()\[1\] == "--[^"]*"' "$f" | cut -d'"' -f2)

      # Get the directory of the file and change to it
      file_dir=$(dirname "$f")
      file_name=$(basename "$f")
      cd "$file_dir" || continue

      if [ -z "$flags" ]; then
        mojo "$file_name" || echo "Failed: ${path_prefix}$f"
      else
        for flag in $flags; do
          echo "Running ${path_prefix}$f with $flag"
          mojo "$file_name" "$flag" || echo "Failed: ${path_prefix}$f with $flag"
        done
      fi

      cd "$current_dir" || exit 1
    fi
  done
}

run_python_files() {
  local path_prefix="$1"
  local current_dir=$(pwd)

  for f in *.py; do
    if [ -f "$f" ]; then
      echo "=== Running ${path_prefix}$f ==="
      # Extract flags for Python files (sys.argv[1] pattern)
      flags=$(grep -o 'sys\.argv\[1\] == "--[^"]*"' "$f" | cut -d'"' -f2)

      # Get the directory of the file and change to it
      file_dir=$(dirname "$f")
      file_name=$(basename "$f")
      cd "$file_dir" || continue

      if [ -z "$flags" ]; then
        python "$file_name" || echo "Failed: ${path_prefix}$f"
      else
        for flag in $flags; do
          echo "Running ${path_prefix}$f with $flag"
          python "$file_name" "$flag" || echo "Failed: ${path_prefix}$f with $flag"
        done
      fi

      cd "$current_dir" || exit 1
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

    process_directory ""

    # Check for test directory and run mojo test
    if [ -d "test" ] || [ -d "tests" ]; then
      echo "=== Running tests in ${dir} ==="
      mojo test . || echo "Failed: mojo test in ${dir}"
    fi

    cd ..
  fi
done

cd ..
