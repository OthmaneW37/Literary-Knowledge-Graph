import json
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from ebooklib import epub

from catalog import CatalogBook, GutendexClient, LibraryImporter
from catalog.gutendex import CatalogError
from catalog.discovery import BookDiscovery
from catalog.context import catalog_context, candidate_people


def test_discovery_keeps_partial_results_and_provenance():
    def handler(request):
        if request.url.path == '/search.json':
            return httpx.Response(200, json={'numFound': 1, 'docs': [{'key': '/works/OL1W', 'title': 'Alice', 'author_name': ['Carroll'], 'cover_i': 12}]})
        return httpx.Response(200, json={'title': 'Alice', 'description': {'value': '<p>A synopsis</p>'}, 'subject_people': ['Alice (Fictional character)']})
    class Offline:
        def search(self, *args, **kwargs):
            raise CatalogError('offline')
    service = BookDiscovery(Offline(), httpx.Client(transport=httpx.MockTransport(handler)))
    result = service.search('Alice')
    assert len(result['books']) == 1 and len(result['warnings']) == 1
    detail = service.details('openlibrary', 'OL1W')
    assert detail['summary'] == 'A synopsis'
    assert detail['author'] == 'Carroll'
    assert not detail['can_download']
    assert detail['source_url'] == 'https://openlibrary.org/works/OL1W'
    with pytest.raises(ValueError):
        service.details('openlibrary', '../../secret')


def test_query_model_falls_back_and_rewrites():
    query = 'Find the story of Alice in Wonderland'
    provider = SimpleNamespace(chat=lambda **kwargs: {'message': {'content': '{"query":"Alice in Wonderland"}'}})
    assert BookDiscovery.plan_query(query, provider, 'model') == 'Alice in Wonderland'
    provider.chat = lambda **kwargs: {'message': {'content': 'broken'}}
    assert BookDiscovery.plan_query(query, provider, 'model') == query


@pytest.mark.parametrize('response', [httpx.Response(302, headers={'location':'http://127.0.0.1/secret'}), httpx.Response(200, content=b'<html>challenge</html>'), httpx.Response(200, headers={'content-length':str(26*1024*1024)})])
def test_download_rejects_unsafe_or_invalid_files(response):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return response
    client = GutendexClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    book = CatalogBook(provider_id=1, title='Alice', authors=('Carroll',), languages=('en',), formats={'application/epub+zip':'https://www.gutenberg.org/ebooks/1.epub'})
    with pytest.raises(CatalogError):
        client.download(book)
    assert len(calls) == 1


def test_epub_cover_metadata_and_external_context_are_separate(tmp_path):
    book = epub.EpubBook()
    book.set_identifier('test-book')
    book.set_title('A source book')
    book.set_language('en')
    book.add_author('An author')
    book.add_metadata('DC', 'description', 'External synopsis only')
    book.add_metadata('DC', 'subject', 'Fiction')
    book.set_cover('cover.png', b'\x89PNG\r\n\x1a\ncover-fixture')
    chapter = epub.EpubHtml(title='Chapter 1', file_name='chapter.xhtml', lang='en')
    chapter.content = '<h1>Chapter 1</h1><p>' + 'Alice meets the rabbit in the garden. ' * 100 + '</p>'
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ['nav', chapter]
    book.toc = (chapter,)
    content = BytesIO()
    epub.write_epub(content, book)
    data = tmp_path / 'data'
    (data / 'annotations').mkdir(parents=True)
    manifest = data / 'annotations/work_manifest.json'
    manifest.write_text('[]')
    importer = LibraryImporter(data)
    installed = importer.install_upload(content.getvalue(), 'book.epub')
    record = importer.records()[0]
    assert record['title'] == 'A source book'
    assert record['author'] == 'An author'
    assert record['cover_url'].startswith('/api/visuals/')
    assert record['summary'] == 'External synopsis only'
    chunks = (data / 'processed' / f'{installed.work_id}.chunks.json').read_text(encoding='utf-8')
    assert 'External synopsis only' not in chunks
    index = SimpleNamespace(manifest_path=manifest)
    assert 'External synopsis only' in catalog_context(index, [installed.work_id])
    assert catalog_context(index, [installed.work_id], max_chapter=1) == ''
    assert catalog_context(index, ['different-work']) == ''
    importer.attach_metadata(installed.work_id, {'source_url':'https://openlibrary.org/works/OL1W', 'source_name':'Open Library', 'summary':'New synopsis', 'subjects':['Adventure'], 'people':['Alice (Fictional character)'], 'cover_url':'https://example.org/cover.jpg'})
    assert importer.records()[0]['cover_url'] == record['cover_url']
    assert candidate_people(index, installed.work_id) == ['Alice']

def test_upload_reads_multipart_metadata_fields(monkeypatch, tmp_path):
    import importlib
    from fastapi.testclient import TestClient
    module = importlib.import_module('app.api')
    importer = LibraryImporter(tmp_path / 'data')
    monkeypatch.setattr(module, 'library', importer)
    assistant = lambda: SimpleNamespace(hybrid_retriever=SimpleNamespace(semantic=None))
    assistant.cache_clear = lambda: None
    monkeypatch.setattr(module, 'get_assistant', assistant)
    module.api.dependency_overrides[module.current_user_id] = lambda: 'test-user'
    try:
        with TestClient(module.api) as client:
            response = client.post('/api/books/upload', files={'file':('book.txt', b'Alice walks through the garden. '*100, 'text/plain')}, data={'title':'Chosen title', 'author':'Chosen author', 'language':'en'})
        assert response.status_code == 200, response.text
        record = importer.records()[0]
        assert record['title'] == 'Chosen title'
        assert record['author'] == 'Chosen author'
        assert record['language'] == 'en'
    finally:
        module.api.dependency_overrides.pop(module.current_user_id, None)
