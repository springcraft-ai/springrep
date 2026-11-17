#!/bin/bash
# SPDX-License-Identifier: MIT
# Fails when a tracked file below the current folder is a binary, or is of a type that is not
# listed here, unless it is in a folder named `fixtures`.
#
# The list says what is allowed, so that it need not name what is refused: a CAD file, a compiled
# library and a vendor's SDK file are all refused because they are not on it.
set -euo pipefail

ALLOWED_EXTENSIONS=" css html js json lock md mjs py rs sh svg toml ts tsx txt yaml yml "
ALLOWED_NAMES=" .gitignore .prettierignore CODEOWNERS LICENSE NOTICE "

refused=()
while IFS= read -r -d '' entry; do
    info="${entry%%$'\t'*}"
    path="${entry#*$'\t'}"
    case "/$path" in
        */fixtures/*) continue ;;
    esac
    name="${path##*/}"
    if [[ "$info" == i/-text* ]]; then
        refused+=("$path: a binary file")
    elif [[ "$ALLOWED_NAMES" == *" $name "* ]]; then
        continue
    elif [[ "$name" != *.* || "$ALLOWED_EXTENSIONS" != *" ${name##*.} "* ]]; then
        refused+=("$path: a file type that is not on the list")
    fi
done < <(git ls-files --eol -z)

if [ "${#refused[@]}" -gt 0 ]; then
    echo "These files are not allowed outside a folder named fixtures:"
    printf '  %s\n' "${refused[@]}"
    echo "A fixture goes in a fixtures folder, with a notice of its terms beside it."
    echo "A new kind of source file is added to the list in .github/allowed-files.sh."
    echo ".github/CONTRIBUTING.md states the rule."
    exit 1
fi
echo "Every tracked file is of an allowed type."
