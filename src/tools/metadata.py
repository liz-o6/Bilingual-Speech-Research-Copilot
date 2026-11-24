"""Read the paper manifest and execute restricted SQLite queries."""
import json
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MANIFEST = Path(__file__).resolve().parents[2] / 'data/corpus/manifest.jsonl'


class MetadataQuery(BaseModel):
    """Gemini selects parameters, not SQL or calculated answers."""
    model_config = ConfigDict(extra='forbid')
    operation: Literal['list', 'count', 'group_count', 'unsupported'] = 'list'
    group_by: Literal['year', 'journal'] = 'year'
    title_contains: str = ''
    author_contains: str = ''
    journal_contains: str = ''
    doc_ids: list[str] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    unsupported_reason: str = ''


def query_metadata(query, manifest_path=MANIFEST):
    """Tool computes all counts. No arbitrary SQL, file path or model arithmetic."""
    query = MetadataQuery.model_validate(query)
    if query.operation == 'unsupported':
        raise ValueError('Unsupported metadata request')
    if query.year_from is not None and query.year_to is not None:
        if query.year_from > query.year_to:
            raise ValueError('Invalid year range')

    papers = []
    for line in Path(manifest_path).read_text().splitlines():
        if line.strip():
            papers.append(json.loads(line))

    # Build a small in-memory database from the unchanged manifest.
    connection = sqlite3.connect(':memory:')
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('CREATE TABLE papers (doc_id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, journal TEXT, doi TEXT, source_url TEXT)')
        for paper in papers:
            authors = '; '.join(paper.get('authors', []))
            connection.execute('INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?)', (
                paper['doc_id'], paper['title'], authors, paper.get('year'),
                paper.get('journal'), paper.get('doi'), paper.get('source_url'),
            ))
        connection.execute('PRAGMA query_only = ON')

        conditions = []
        parameters = []
        # instr treats %, _, quotes and SQL-like text as literal search text.
        for column, value in [('title', query.title_contains),
                              ('authors', query.author_contains),
                              ('journal', query.journal_contains)]:
            if value:
                conditions.append('instr(lower(' + column + '), lower(?)) > 0')
                parameters.append(value)
        if query.doc_ids:
            placeholders = ','.join(['?'] * len(query.doc_ids))
            conditions.append('doc_id IN (' + placeholders + ')')
            parameters.extend(query.doc_ids)
        if query.year_from is not None:
            conditions.append('year >= ?')
            parameters.append(query.year_from)
        if query.year_to is not None:
            conditions.append('year <= ?')
            parameters.append(query.year_to)
        where = ''
        if conditions:
            where = ' WHERE ' + ' AND '.join(conditions)

        matched = connection.execute('SELECT * FROM papers' + where + ' ORDER BY doc_id', parameters).fetchall()
        sources = {}
        for paper in matched:
            source_id = 'M_' + paper['doc_id']  # Separate from Research's S1 labels.
            sources[source_id] = {
                'doc_id': paper['doc_id'], 'title': paper['title'],
                'source_url': paper['source_url'], 'source_type': 'metadata',
                'source_locator': 'data/corpus/manifest.jsonl:' + paper['doc_id'],
                'text': json.dumps(dict(paper), ensure_ascii=False),
            }

        if query.operation == 'list':
            rows = []
            for paper in matched:
                row = dict(paper)
                row['source_ids'] = ['M_' + paper['doc_id']]
                rows.append(row)
        elif query.operation == 'count':
            record = connection.execute('SELECT COUNT(*) AS count FROM papers' + where, parameters).fetchone()
            rows = [{'count': record['count'], 'unit': 'papers', 'source_ids': list(sources)}]
        else:
            # group_by is validated as year or journal, never arbitrary user SQL.
            column = query.group_by
            records = connection.execute(
                'SELECT ' + column + ', COUNT(*) AS count FROM papers' + where +
                ' GROUP BY ' + column + ' ORDER BY ' + column, parameters).fetchall()
            rows = []
            for record in records:
                source_ids = []
                for paper in matched:
                    if paper[column] == record[column]:
                        source_ids.append('M_' + paper['doc_id'])
                rows.append({column: record[column], 'count': record['count'],
                             'unit': 'papers', 'source_ids': source_ids})

        return {'rows': rows, 'sources': sources, 'filters': query.model_dump(),
                'scope': 'Local paper manifest only; one row per doc_id, not experiments or participants.',
                'corpus_size': len(papers), 'matched_count': len(matched)}
    finally:
        connection.close()
