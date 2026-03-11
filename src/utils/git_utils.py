"""
git_utils.py
~~~~~~~~~~~~
Uses GitPython to extract change velocity and history for files.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from git import Repo
except ImportError:
    Repo = None

log = logging.getLogger(__name__)

class GitMetrics:
    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)
        self.repo = None
        if Repo:
            try:
                self.repo = Repo(self.repo_path, search_parent_directories=True)
            except Exception as e:
                log.warning(f"Could not initialize git repo at {repo_path}: {e}")

    def get_velocity(self, file_path: str | Path) -> float:
        """
        Return the average number of commits per month for the given file
        over the last 6 months.
        """
        if not self.repo:
            return 0.0
        
        try:
            p = Path(file_path).relative_to(self.repo.working_dir)
            six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
            
            commits = list(self.repo.iter_commits(paths=str(p), since=six_months_ago))
            return round(len(commits) / 6.0, 2)
        except Exception:
            return 0.0

    def get_creation_date(self, file_path: str | Path) -> Optional[str]:
        if not self.repo:
            return None
        try:
            p = Path(file_path).relative_to(self.repo.working_dir)
            commits = list(self.repo.iter_commits(paths=str(p), reverse=True))
            if commits:
                return datetime.fromtimestamp(commits[0].committed_date, timezone.utc).isoformat()
        except Exception:
            pass
        return None
