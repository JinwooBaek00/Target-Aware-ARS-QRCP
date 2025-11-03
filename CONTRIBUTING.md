# Contributing Guide

This guide outlines how to set up, code, test, review, and release contributions to ensure they meet the project’s Definition of Done (DoD). All contributors are expected to follow these guidelines to maintain quality, consistency, and reproducibility across the research project.

## Code of Conduct

**Expected Behavior**
- Communicate respectfully and provide constructive feedback.  
- Support collaboration and inclusivity.  

**Unacceptable Behavior**
- Harassment, discrimination, or disrespectful communication of any form.  

**Reporting Process**
- Report any issues to **Team Lead (Jinwoo Baek)** or the **TA**, who will escalate as necessary through university or project partner channels.


## Getting Started

List prerequisites, setup steps, environment variables/secrets handling, and how to run the app locally.

Prerequisites
- Python 3.10+  
- Git and GitHub account  
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pytest`  
- Access to the private GitHub repository  

Setup:
N/A  

Environment Variables & Secrets:
N/A  

Running Locally:
N/A  

## Branching & Workflow
Describe the workflow (e.g., trunk-based or GitFlow), default branch, branch naming, and when to rebase vs. merge.  

Workflow Type: GitFlow
- Default Branch: main (stable)
- Development Branch: dev (active work)

Branch Naming Conventions:
- feature/short-description — for new features or experiments
- bugfix/short-description — for fixing issues
- doc/short-description — for documentation updates
Rebasing vs. Merging:
- Use rebase to update your feature branch with the latest dev changes.
- Use merge only when merging approved PRs into main or dev.
- 
## Issues & Planning
Explain how to file issues, required templates/labels, estimation, and triage/assignment practices.  

Creating an Issue:
- Use the “New Issue” template in GitHub.
- Clearly describe the problem, proposed solution, and context.
- Label appropriately (e.g., bug, enhancement, research, documentation).

Estimation & Assignment:
- Each issue must include an estimated completion time.
- Issues are assigned during team meetings based on workload distribution.

## Commit Messages
State the convention (e.g., Conventional Commits), include examples, and how to reference issues.

Convention: Conventional Commits
Format: <type>(scope): <short summary>  
Examples:
- feat(experiment): add random forest baseline selector
- fix(core): resolve NaN handling in correlation selector
- docs: update contributing guide

## Code Style, Linting & Formatting
Name the formatter/linter, config file locations, and the exact commands to check/fix locally.

Tools:
- Linter: flake8
- Formatter: black
- Import sorter: isort

Commands:
- Run linter: flake8 src/
- Auto-format code: black src/
- Sort imports: isort src/
- Configuration files are stored at the project root (.flake8, pyproject.toml).

## Testing
Define required test types, how to run tests, expected coverage thresholds, and when new/updated tests are mandatory.

Required Test Types:
- Integration tests for baseline and TAAQ algorithms
- Reproducibility tests for random sampling functions

Commands:
- N/A

Coverage Requirement:
- N/A

New Code:
Every new feature or fix must include at least one corresponding test.

## Pull Requests & Reviews
Outline PR requirements (template, checklist, size limits), reviewer expectations, approval rules, and required status checks.

Pull Request (PR)
- Include a brief description, screenshots (if applicable), and related issue references.
- PRs should not exceed 400 lines of change.

Review Process:
- Minimum of one peer review and one approval from the Team Lead.
- CI must pass before approval.
- Requested changes must be resolved before merge.

## CI/CD
Link to pipeline definitions, list mandatory jobs, how to view logs/re-run jobs, and what must pass before merge/release.

**Pipeline Definition:**  
The CI pipeline is defined in [`.github/workflows/ci.yml`](https://github.com/JinwooBaek00/Target-Aware-ARS-QRCP/blob/main/.github/workflows/ci.yml). It automatically runs on every push or pull request to the `main` branch.

**Mandatory Jobs:**
- **Lint Check:** `flake8 src/` — ensure code style compliance  
- **Format Check:** `black --check src/` — verify code formatting  
- **Import Order Check:** `isort --check-only src/` — verify import sorting  
- **Unit Tests:** `pytest` — run all tests and report coverage  

**View Logs:**  
Go to your PR → “Checks” tab → select a job name to view logs and artifacts.

**Re-run Jobs:**  
If a job fails, authorized users (Team Lead or assigned maintainer) can re-run it via the “Re-run jobs” button in GitHub Actions.

**Merge Requirements:**  
All mandatory jobs must pass before merging into `main`.  
PRs with failing checks or unresolved review comments cannot be merged.


## Security & Secrets
State how to report vulnerabilities, prohibited patterns (hard-coded secrets), dependency update policy, and scanning tools.

Vulnerability Reporting:
- Report security concerns privately to the Team Lead.
- Do not post security issues publicly in GitHub Issues.

Prohibited Practices:
Never commit hard-coded credentials or API keys.
Never expose our algorithm to public spaces.

## Documentation Expectations
Specify what must be updated (README, docs/, API refs, CHANGELOG) and docstring/comment standards.

Required Updates:
- Update README.md after major code or workflow changes.
- Add/update relevant files in docs/ directory.
- Maintain clear inline documentation and docstrings using the NumPy style.

## Release Process
Describe versioning scheme, tagging, changelog generation, packaging/publishing steps, and rollback process.

N/A

## Support & Contact
Provide maintainer contact channel, expected response windows, and where to ask questions.

Maintainers Contact:
- Jinwoo Baek (Team Lead) - baekji@oregonstate.edu  

Response Expectations:
- Issues/PR comments: within 24–48 hours
- Security or technical concerns: within 12–24 hours  

Questions:
- Jinwoo Baek (Team Lead) - baekji@oregonstate.edu
- Joy Lim - limjoy@oregonstate.edu
- Kevin Tran - trank8@oregonstate.edu
- Kevin Nguyen - nguykev2@oregonstate.edu
