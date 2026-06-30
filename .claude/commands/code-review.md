---
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), mcp__github_inline_comment__create_inline_comment
description: Code review a pull request, posting inline comments on the relevant lines
---

Provide a code review for the given pull request.

To do this, follow these steps precisely:

1. Launch a haiku agent to check if any of the following are true:
   - The pull request is closed
   - The pull request is a draft
   - The pull request does not need code review (e.g. automated PR, trivial change that is obviously correct)
   - You have already submitted a code review on this pull request

   If any condition is true, stop and do not proceed.

Note: Still review Claude generated PR's.

2. Launch a haiku agent to return a list of file paths (not their contents) for all relevant CLAUDE.md files including:
   - The root CLAUDE.md file, if it exists
   - Any CLAUDE.md files in directories containing files modified by the pull request

3. Launch a sonnet agent to view the pull request and return a summary of the changes

4. Launch 4 agents in parallel to independently review the changes. Each agent should return the list of issues, where each issue includes a description and the reason it was flagged (e.g. "CLAUDE.md adherence", "bug"). The agents should do the following:

   Agents 1 + 2: CLAUDE.md compliance sonnet agents
   Audit changes for CLAUDE.md compliance in parallel. Note: When evaluating CLAUDE.md compliance for a file, you should only consider CLAUDE.md files that share a file path with the file or parents.

   Agent 3: Opus bug agent (parallel subagent with agent 4)
   Scan for obvious bugs. Focus only on the diff itself without reading extra context. Flag only significant bugs; ignore nitpicks and likely false positives. Do not flag issues that you cannot validate without looking at context outside of the git diff.

   Agent 4: Opus bug agent (parallel subagent with agent 3)
   Look for problems that exist in the introduced code. This could be security issues, incorrect logic, etc. Only look for issues that fall within the changed code.

   **CRITICAL: We only want HIGH SIGNAL issues.** This means:
   - Objective bugs that will cause incorrect behavior at runtime
   - Clear, unambiguous CLAUDE.md violations where you can quote the exact rule being broken

   We do NOT want:
   - Subjective concerns or "suggestions"
   - Style preferences not explicitly required by CLAUDE.md
   - Potential issues that "might" be problems
   - Anything requiring interpretation or judgment calls

   If you are not certain an issue is real, do not flag it. False positives erode trust and waste reviewer time.

   In addition to the above, each subagent should be told the PR title and description. This will help provide context regarding the author's intent.

5. For each issue found in the previous step by agents 3 and 4, launch parallel subagents to validate the issue. These subagents should get the PR title and description along with a description of the issue. The agent's job is to review the issue to validate that the stated issue is truly an issue with high confidence. For example, if an issue such as "variable is not defined" was flagged, the subagent's job would be to validate that is actually true in the code. Another example would be CLAUDE.md issues. The agent should validate that the CLAUDE.md rule that was violated is scoped for this file and is actually violated. Use Opus subagents for bugs and logic issues, and sonnet agents for CLAUDE.md violations.

6. Filter out any issues that were not validated in step 5. This step will give us our list of high signal issues for our review.

7. Finally, post the review on the pull request as inline comments, one per validated issue.
   For each issue, use the `mcp__github_inline_comment__create_inline_comment` tool to attach a comment to the exact file and line(s) where the issue occurs. Pass `confirmed: true`.
   When writing each inline comment, follow these guidelines:
   a. Keep your output brief
   b. Avoid emojis
   c. Anchor the comment to the precise line range of the offending code
   d. When citing CLAUDE.md violations, you MUST quote the exact text from CLAUDE.md that is being violated (e.g., CLAUDE.md says: "Use snake_case for variable names")
   e. Only post GitHub comments — do not submit the review text as a chat/message response

Use this list when evaluating issues in Steps 4 and 5 (these are false positives, do NOT flag):

- Pre-existing issues
- Something that appears to be a bug but is actually correct
- Pedantic nitpicks that a senior engineer would not flag
- Issues that a linter will catch (do not run the linter to verify)
- General code quality concerns (e.g., lack of test coverage, general security issues) unless explicitly required in CLAUDE.md
- Issues mentioned in CLAUDE.md but explicitly silenced in the code (e.g., via a lint ignore comment)

Notes:

- Use the `mcp__github_inline_comment__create_inline_comment` tool to post inline comments. Use the gh CLI for everything else (e.g., fetching pull requests, posting the summary comment). Do not use web fetch.
- Create a todo list before starting.
- Each inline comment must be anchored to the specific file and line range of the issue it describes.
- For each issue, the body of the inline comment should follow this format precisely (assuming for this example you found a bug and a CLAUDE.md violation):

---

<brief description of bug> (bug)

---

<brief description of violation> (CLAUDE.md says: "<exact quote from CLAUDE.md>")

---

- After posting the inline comments, post one brief summary comment via `gh pr comment` using the following format precisely (assuming for this example that you found 3 issues):

---

## Code review

Found 3 issues — see the inline comments.

---

- Or, if you found no issues, post a single summary comment via `gh pr comment` and do not post any inline comments:

---

## Auto code review

No issues found. Checked for bugs and CLAUDE.md compliance.

---
