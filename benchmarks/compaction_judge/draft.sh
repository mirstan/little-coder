#!/bin/zsh
# draft.sh <ID> <claude|gpt> : write drafts/<ID>.<family>.json
cd ${0:a:h}
id=$1; fam=$2; mkdir -p drafts logs
in=logs/$id.$fam.prompt.txt
{ cat draft-prompt.md; cat points/$id/pre.txt; echo "\n=== TRANSCRIPT END ==="; } > $in
if [[ $fam == claude ]]; then
  claude -p --model claude-opus-5-5 --output-format text < $in > drafts/$id.claude.raw 2> logs/$id.claude.err
else
  codex exec --skip-git-repo-check -s read-only --color never -o drafts/$id.gpt.raw - < $in > logs/$id.gpt.out 2> logs/$id.gpt.err
fi
echo "$id $fam exit=$?"
