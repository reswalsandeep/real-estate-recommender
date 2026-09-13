#!/usr/bin/env bash
set -euo pipefail

# Run this from the root of your local clone of real-estate-recommender, on main.
# It rewrites AUTHOR and COMMITTER dates for the 13 commits, keeping content/order
# identical, evenly spacing timestamps from the root commit to HEAD.

git filter-branch -f --env-filter '
case "$GIT_COMMIT" in
  bd613da*) export GIT_AUTHOR_DATE="2026-09-10T20:45:00" ; export GIT_COMMITTER_DATE="2026-09-10T20:45:00" ;;
  ed1eb30*) export GIT_AUTHOR_DATE="2026-09-11T02:32:00" ; export GIT_COMMITTER_DATE="2026-09-11T02:32:00" ;;
  83d7d89*) export GIT_AUTHOR_DATE="2026-09-11T08:19:00" ; export GIT_COMMITTER_DATE="2026-09-11T08:19:00" ;;
  5deb097*) export GIT_AUTHOR_DATE="2026-09-11T14:06:00" ; export GIT_COMMITTER_DATE="2026-09-11T14:06:00" ;;
  e6cd711*) export GIT_AUTHOR_DATE="2026-09-11T19:53:00" ; export GIT_COMMITTER_DATE="2026-09-11T19:53:00" ;;
  bbdca2d*) export GIT_AUTHOR_DATE="2026-09-12T01:40:00" ; export GIT_COMMITTER_DATE="2026-09-12T01:40:00" ;;
  04f6443*) export GIT_AUTHOR_DATE="2026-09-12T07:27:00" ; export GIT_COMMITTER_DATE="2026-09-12T07:27:00" ;;
  d2b6966*) export GIT_AUTHOR_DATE="2026-09-12T13:14:00" ; export GIT_COMMITTER_DATE="2026-09-12T13:14:00" ;;
  8682192*) export GIT_AUTHOR_DATE="2026-09-12T19:01:00" ; export GIT_COMMITTER_DATE="2026-09-12T19:01:00" ;;
  cc25dae*) export GIT_AUTHOR_DATE="2026-09-13T00:48:00" ; export GIT_COMMITTER_DATE="2026-09-13T00:48:00" ;;
  6bbb2ed*) export GIT_AUTHOR_DATE="2026-09-13T06:35:00" ; export GIT_COMMITTER_DATE="2026-09-13T06:35:00" ;;
  1d39623*) export GIT_AUTHOR_DATE="2026-09-13T12:22:00" ; export GIT_COMMITTER_DATE="2026-09-13T12:22:00" ;;
  beb7ba1*) export GIT_AUTHOR_DATE="2026-09-13T18:09:00" ; export GIT_COMMITTER_DATE="2026-09-13T18:09:00" ;;
esac
' -- --all

echo "Done rewriting dates. Verify with: git log --pretty=format:'%h %ad %s' --date=iso"
echo "If it looks right, force-push with: git push --force origin main"
