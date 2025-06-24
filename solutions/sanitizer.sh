#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -eq 0 ]; then
    echo "Usage: $0 <tool>"
    echo "Available tools: memcheck, racecheck, synccheck, initcheck"
    exit 1
fi

TOOL="$1"

case "$TOOL" in
    memcheck|racecheck|synccheck|initcheck)
        ;;
    *)
        echo "Error: Invalid tool '$TOOL'"
        echo "Available tools: memcheck, racecheck, synccheck, initcheck"
        exit 1
        ;;
esac

case "$TOOL" in
    memcheck)
        GREP_PATTERN="(========= COMPUTE-SANITIZER|========= ERROR SUMMARY|========= Memory Error|========= Invalid|========= Out of bounds|out:|expected:)"
        ;;
    racecheck)
        GREP_PATTERN="(========= COMPUTE-SANITIZER|========= ERROR SUMMARY|========= Race|========= Hazard|out:|expected:)"
        ;;
    synccheck)
        GREP_PATTERN="(========= COMPUTE-SANITIZER|========= ERROR SUMMARY|========= Sync|========= Deadlock|out:|expected:)"
        ;;
    initcheck)
        GREP_PATTERN="(========= COMPUTE-SANITIZER|========= ERROR SUMMARY|========= Uninitialized|========= Unused|out:|expected:)"
        ;;
esac

TOTAL_ERRORS=0

run_mojo_files_with_sanitizer() {
  local path_prefix="$1"
  local tool="$2"
  local grep_pattern="$3"

  for f in *.mojo; do
    if [ -f "$f" ] && [ "$f" != "__init__.mojo" ]; then
      flags=$(grep -o 'argv()\[1\] == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')

      if [ -z "$flags" ]; then
        echo "=== Running compute-sanitizer --tool $tool on ${path_prefix}$f ==="
        output=$(compute-sanitizer --tool "$tool" mojo "$f" 2>&1)
        filtered_output=$(echo "$output" | grep -E "$grep_pattern")

        error_count=$(echo "$output" | grep -E "========= ERROR SUMMARY: [0-9]+ error" | sed -n 's/.*ERROR SUMMARY: \([0-9]\+\) error.*/\1/p')

        if [ -n "$error_count" ] && [ "$error_count" -gt 0 ]; then
          echo -e "${RED}🚨 FOUND $error_count ERRORS! 🚨${NC}"
          TOTAL_ERRORS=$((TOTAL_ERRORS + error_count))
        fi

        if [ -n "$filtered_output" ]; then
          echo "$filtered_output"
        else
          echo "Failed: compute-sanitizer $tool ${path_prefix}$f"
        fi
      else
        for flag in $flags; do
          echo "=== Running compute-sanitizer --tool $tool on ${path_prefix}$f with flag: $flag ==="
          output=$(compute-sanitizer --tool "$tool" mojo "$f" "$flag" 2>&1)
          filtered_output=$(echo "$output" | grep -E "$grep_pattern")

          error_count=$(echo "$output" | grep -E "========= ERROR SUMMARY: [0-9]+ error" | sed -n 's/.*ERROR SUMMARY: \([0-9]\+\) error.*/\1/p')

          if [ -n "$error_count" ] && [ "$error_count" -gt 0 ]; then
            echo -e "${RED}🚨 FOUND $error_count ERRORS! 🚨${NC}"
            TOTAL_ERRORS=$((TOTAL_ERRORS + error_count))
          fi

          if [ -n "$filtered_output" ]; then
            echo "$filtered_output"
          else
            echo "Failed: compute-sanitizer $tool ${path_prefix}$f with $flag"
          fi
        done
      fi
    fi
  done
}

cd solutions || exit 1

for dir in p*/; do
  if [ -d "$dir" ]; then
    echo "=== Running compute-sanitizer $TOOL on solutions in ${dir} ==="
    cd "$dir" || continue

    run_mojo_files_with_sanitizer "$dir" "$TOOL" "$GREP_PATTERN"

    cd ..
  fi
done

cd ..

echo ""
echo "========================================"
if [ "$TOTAL_ERRORS" -gt 0 ]; then
  echo -e "${RED}🚨 TOTAL ERRORS FOUND: $TOTAL_ERRORS 🚨${NC}"
  echo -e "${YELLOW}Please review the errors above and fix them.${NC}"
  exit 1
else
  echo -e "${GREEN}✅ NO ERRORS FOUND! All tests passed clean.${NC}"
  exit 0
fi
