import os
from mock import Mock, patch, MagicMock
from unittest import TestCase

from atm.database import Database, db_session


DB_PATH = '/tmp/atm.db'


class TestDatabase(TestCase):

    def setUp(self):
        os.remove(DB_PATH)
        self.db = Database(dialect='sqlite', database=DB_PATH)

    def test_get_datasets(self):
        self.db.session = MagicMock()
        self.db.session.query = MagicMock()
        self.db.session.query.return_value = self.db.session.query

        self.db._filter_by_like = MagicMock()
        self.db._filter_by_like.return_value = self.db.session.query

        self.db._filter_by_comparison = MagicMock()
        self.db._filter_by_comparison.return_value = self.db.session.query

        self.db.get_datasets()

        assert self.db.session.query.called
        assert self.db._filter_by_like.called
        assert self.db._filter_by_comparison.called
        assert self.db.session.query.all.called


class TestDatabaseFilters(TestCase):
    def setUp(self):
        os.remove(DB_PATH)
        self.db = Database(dialect='sqlite', database=DB_PATH)

    def test_filter_by_like_with_value_to_query(self):
        query = MagicMock(name='query')
        query.filter.return_value = MagicMock('filter')
        substring = 'foo'

        self.db._filter_by_like(
            query=query, class_to_filter=MagicMock(), substring=substring)

        assert query.filter.called

    def test_filter_by_like_with_no_value_to_query(self):
        query = MagicMock(name='query')
        query.filter.return_value = MagicMock('filter')
        substring = None

        self.db._filter_by_like(
            query=query, class_to_filter=MagicMock(), substring=substring)

        assert not query.filter.called
