if test "$( basename $(pwd) )" != ".aiassistant" ; then
  echo "## not in .aiassistant, exiting"
  exit 1
fi

## COMMONS

cat <<EOF | tee rules/0_MD_METHOD.md 1>/dev/null
---
apply: by file patterns
patterns: *.md, *.markdown
---

EOF
cat ../guidelines/documentation/MD_METHOD.md | tee -a rules/0_MD_METHOD.md 1>/dev/null

cat ../app/README.md | tee rules/1_README.md 1>/dev/null

cat <<EOF | tee rules/2_HEADER.md 1>/dev/null
---
apply: by file patterns
patterns: *.py
---

EOF
cat ../guidelines/project/HEADER.md | tee -a rules/2_HEADER.md 1>/dev/null

cat <<EOF | tee rules/3_RULES.md 1>/dev/null
---
apply: by file patterns
patterns: *.py
---

EOF
cat ../guidelines/project/RULES.md | tee -a rules/3_RULES.md 1>/dev/null

cat <<EOF | tee rules/9_METHOD.md 1>/dev/null
---
apply: by file patterns
patterns: *.py
---

EOF
cat ../guidelines/project/METHOD.md | tee -a rules/9_METHOD.md 1>/dev/null



