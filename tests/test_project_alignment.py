import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app import api as api_module
from security.budget import RequestBudget, BudgetExceeded, current_budget, consume
from security.access import BookAccess
from training.splits import grouped_split
from training.evaluate_reranker import metrics
from training.build_dataset import validate_records


def test_ndcg_ignores_hits_after_rank_five_and_duplicates():
    assert metrics([(['a','b','c','d','e','hit'], ['hit'])])['ndcg_at_5'] == 0
    assert metrics([(['hit','hit','hit'], ['hit'])])['ndcg_at_5'] == 1


def test_training_separates_books_and_rejects_leakage():
    rows = [{'work_id':str(i), 'question':'question', 'positive':f'positive {i}', 'hard_negatives':[f'negative {i}']} for i in range(10)]
    partitions, groups = grouped_split(validate_records(rows))
    assert set(groups['train']).isdisjoint(groups['validation'])
    assert set(groups['test']).isdisjoint(groups['train'])
    assert sum(map(len, partitions.values())) == 10
    with pytest.raises(ValueError): grouped_split([{'question':'q','positive':'p','hard_negatives':['n']}])
    with pytest.raises(ValueError): validate_records([{'question':'q','positive':'p','hard_negatives':[' ']}])
    duplicate = [dict(row, positive='same passage') for row in rows]
    with pytest.raises(ValueError, match='partitions'): grouped_split(duplicate)


def test_model_mcp_and_time_budgets():
    budget = RequestBudget(max_model=1, max_mcp=1)
    token = current_budget.set(budget)
    try:
        consume('model'); consume('mcp')
        with pytest.raises(BudgetExceeded): consume('model')
        with pytest.raises(BudgetExceeded): consume('mcp')
        budget.started -= 100
        with pytest.raises(BudgetExceeded): budget.remaining()
    finally: current_budget.reset(token)


def test_document_acl_groups_and_closed_default(tmp_path):
    (tmp_path/'library').mkdir()
    (tmp_path/'library/access_policy.json').write_text(json.dumps({'default':'private','user_groups':{'alice':['readers']},'books':{'book':{'groups':['readers']}}}))
    policy = BookAccess(tmp_path)
    assert policy.allows('alice','book')
    assert not policy.allows('bob','book')
    assert not policy.allows('alice','unknown')


def test_api_denies_image_without_login_and_book_outside_group(tmp_path, monkeypatch):
    from rag.models import Passage
    monkeypatch.setenv('AUTH_ENABLED','true')
    monkeypatch.setattr(api_module, 'PROJECT_ROOT', tmp_path)
    client = TestClient(api_module.api)
    assert client.get('/api/visuals/'+'a'*20).status_code == 401
    monkeypatch.setenv('AUTH_ENABLED','false')
    (tmp_path/'data/library').mkdir(parents=True)
    (tmp_path/'data/library/access_policy.json').write_text(json.dumps({'books':{'private':{'users':['alice']}}}))
    assistant = SimpleNamespace(index=SimpleNamespace(works={'private':object()},passages=[Passage('private','Secret',1,'p','secret')]))
    monkeypatch.setattr(api_module,'get_assistant',lambda:assistant)
    assert client.get('/api/books/private/passages').status_code == 403
    assert client.post('/api/chat',json={'question':'secret?','work_ids':['private']}).status_code == 403
    assert client.post('/api/search',json={'query':'secret','work_ids':['private']}).status_code == 403


def test_mcp_real_transport_approval_and_private_notes(tmp_path):
    pytest.importorskip('mcp')
    from mcp_server.client import call_tool
    (tmp_path/'processed').mkdir()
    (tmp_path/'processed/book.chunks.json').write_text(json.dumps([{'chunk_id':'p','chapter':1,'text':'Alice reads.'}]))
    secret='integration-test-approval-secret'
    params={'work_id':'book','chapter':1,'text':'Alice reads.','evidence_ids':['p']}
    draft=call_tool('prepare_annotation',params,tmp_path,'alice',secret)
    assert call_tool('get_user_annotations',{'work_id':'book'},tmp_path,'alice',secret) == []
    result=call_tool('save_annotation',{**params,'approval_token':draft['approval_token']},tmp_path,'alice',secret)
    assert result['saved']
    duplicate=call_tool('save_annotation',{**params,'approval_token':draft['approval_token']},tmp_path,'alice',secret)
    assert duplicate['already_saved']
    assert call_tool('get_user_annotations',{'work_id':'book'},tmp_path,'bob',secret) == []
    with pytest.raises(ValueError): call_tool('send_all_documents',{},tmp_path,'alice',secret)

def test_mcp_rejects_modified_approval(tmp_path):
    pytest.importorskip('mcp')
    from mcp_server.client import call_tool
    (tmp_path/'processed').mkdir()
    (tmp_path/'processed/book.chunks.json').write_text(json.dumps([{'chunk_id':'p','chapter':1}]))
    args={'work_id':'book','chapter':1,'text':'Original note','evidence_ids':['p']}
    draft=call_tool('prepare_annotation',args,tmp_path,'alice','test-secret')
    with pytest.raises(ValueError, match='MCP'):
        call_tool('save_annotation',{**args,'text':'Changed note','approval_token':draft['approval_token']},tmp_path,'alice','test-secret')
