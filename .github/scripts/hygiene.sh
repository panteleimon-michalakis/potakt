#!/usr/bin/env bash
# Repository hygiene (CLAUDE.md, Definition of done 8), run by the CI security job and locally
# from the repository root: bash .github/scripts/hygiene.sh [BASE]
# BASE is a commit: the commits in BASE..HEAD must carry no AI attribution line. Without BASE, or
# with the all-zero SHA of a new branch (scheduled and manual runs), that check is skipped; a BASE
# that is not in the clone is an error.
set -euo pipefail
status=0
indent() { while IFS= read -r line; do echo "  $line"; done; }

tracked="$(git ls-files -- .tasks .claude)"
if [ -n "$tracked" ]; then
  echo "::error::files under .tasks/ or .claude/ are tracked; they must stay local:"
  indent <<<"$tracked"
  status=1
fi

crlf="$(git ls-files --eol | grep 'i/crlf' | grep -v -E '\.(ps1|cmd|bat)$' || true)"
if [ -n "$crlf" ]; then
  echo "::error::files are stored with CRLF line endings (only *.ps1, *.cmd and *.bat may be):"
  indent <<<"$crlf"
  status=1
fi

base="${1:-}"
if [ -z "$base" ] || [ "$base" = "0000000000000000000000000000000000000000" ]; then
  echo "Commit messages: not checked (no base commit for this run)."
elif ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
  echo "::error::the base commit ${base} is not in this clone; check out the full history."
  status=1
else
  pattern='^(co-authored-by|claude-session):|noreply@anthropic\.com|generated with \[?claude'
  count=0
  for commit in $(git rev-list "${base}..HEAD"); do
    count=$((count + 1))
    message="$(git log -1 --format=%B "$commit")"
    if grep -q -i -E "$pattern" <<<"$message"; then
      echo "::error::commit ${commit:0:7} carries an AI attribution line (CLAUDE.md, rule 4):"
      git log -1 --format='  %s' "$commit"
      status=1
    fi
  done
  echo "Commit messages: ${count} checked."
fi

if [ "$status" -eq 0 ]; then
  echo "Hygiene: no tracked .tasks/.claude files, no CRLF files, no AI attribution lines."
fi
exit "$status"
