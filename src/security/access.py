"""Explicit book ACLs; unlisted books belong to the shared demo corpus."""
from pathlib import Path
from storage.local import read_json


class BookAccess:
    def __init__(self, data_dir):
        self.root = Path(data_dir)

    def allows(self, user_id, work_id):
        policy = read_json(self.root / 'library/access_policy.json', {})
        rule = policy.get('books', {}).get(work_id)
        if rule is None:
            return policy.get('default', 'shared') == 'shared'
        if not isinstance(rule, dict):
            return False
        if rule.get('visibility') == 'shared':
            return True
        groups = set(policy.get('user_groups', {}).get(user_id, []))
        return (user_id == rule.get('owner') or user_id in rule.get('users', [])
                or bool(groups.intersection(rule.get('groups', []))))

    def require(self, user_id, work_id):
        if not self.allows(user_id, work_id):
            raise PermissionError('Accès à ce livre non autorisé.')
