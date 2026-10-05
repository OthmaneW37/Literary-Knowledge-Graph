export type Book = {
  id: string; title: string; author: string; language: string; passage_count: number;
  chapters: number[]; cover_url: string; source: string; archived: boolean;
}
export type Passage = {
  work_id: string; work_title: string; author: string; chapter: number | string;
  chunk_id: string; text: string; score: number; section: string; page: number | null;
  pages: number[]; citation_label: string;
}
export type Edge = {
  source: string; target: string; label: string; evidence_chunk_id: string;
  evidence: string; work_id: string; source_kind: string; target_kind: string;
}
export type Graph = { nodes: {id: string; label: string; kind: string}[]; edges: Edge[] }
export type Message = {
  role: 'user' | 'assistant'; content: string; citations?: Passage[];
  visualization?: {title?: string; nodes?: {id: string; label: string; kind: string}[]; edges?: {source: string; target: string; label: string; evidence_chunk_id: string}[]};
  elapsed_ms?: number; used_model?: boolean;
}
export type CatalogBook = {
  provider_id: number; title: string; author: string; languages: string[];
  language_display: string; subjects: string[]; summary: string; cover_url: string; download_count: number;
}
const API = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(data.detail || `Erreur ${response.status}`)
  }
  return response.json() as Promise<T>
}
const json = (data: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
const params = (values: Record<string, string | number | null | undefined>) => {
  const search = new URLSearchParams()
  Object.entries(values).forEach(([key, value]) => { if (value !== null && value !== undefined && value !== '') search.set(key, String(value)) })
  return search.toString()
}

export const api = {
  library: () => request<{works: Book[]; model: string; embedding_model: string; stats: Record<string, number>; progress: Record<string, number>; max_upload_bytes: number}>('/library'),
  chat: (data: {question: string; work_ids: string[]; top_k: number; max_chapter: number | null; mode: string; history: {role: string; content: string}[]; conversation_id: string | null}) => request<Message & {conversation_id: string}>('/chat', json(data)),
  search: (data: {query: string; work_ids: string[]; method: string; top_k: number; max_chapter: number | null}) => request<{passages: Passage[]}>('/search', json(data)),
  passages: (id: string, chapter?: number) => request<{total: number; passages: Passage[]}>(`/books/${encodeURIComponent(id)}/passages?${params({chapter, limit: 100})}`),
  graph: (workIds: string[], chapter: number | null) => {
    const search = new URLSearchParams(); workIds.forEach(id => search.append('work_ids', id)); if (chapter) search.set('max_chapter', String(chapter))
    return request<Graph>(`/graph?${search}`)
  },
  buildGraph: (data: {work_ids: string[]; limit: number; max_chapter: number | null}) => request<Graph & {processed: number; remaining: number}>('/graph/build', json(data)),
  syncGraph: (data: {work_ids: string[]}) => request<{synced: number}>('/graph/sync', json(data)),
  catalogSearch: (query: string, language: string, page: number) => request<{count: number; page: number; has_next: boolean; has_previous: boolean; books: CatalogBook[]}>(`/catalog/search?${params({query, language, page})}`),
  installCatalog: (provider_id: number) => request<{work_id: string; title: string; warning?: string|null}>('/catalog/install', json({provider_id})),
  upload: (file: File, title: string, author: string, language: string) => {
    const data = new FormData(); data.set('file', file); data.set('title', title); data.set('author', author); data.set('language', language)
    return request<{work_id: string; title: string; warning?: string|null}>('/books/upload', {method: 'POST', body: data})
  },
  archiveBook: (id: string) => request(`/library/archive/${encodeURIComponent(id)}`, {method: 'POST'}),
  restoreBook: (id: string) => request(`/library/restore/${encodeURIComponent(id)}`, {method: 'POST'}),
  reindexBook: (id: string) => request(`/library/reindex/${encodeURIComponent(id)}`, {method: 'POST'}),
  progress: (work_ids: string[], chapter: number) => request('/progress', json({work_ids, chapter})),
  conversations: (work_ids: string[], max_chapter: number | null) => request<{conversations: {id: string; title: string; updated_at: string}[]}>('/conversations', json({work_ids, max_chapter})),
  loadConversation: (id: string, workIds: string[], chapter: number | null) => request<{id: string; messages: Message[]}>(`/conversations/${id}?${(() => { const p = new URLSearchParams(); workIds.forEach(w => p.append('work_ids',w)); if(chapter)p.set('max_chapter',String(chapter)); return p })()}`),
  archiveConversation: (id: string) => request(`/conversations/${id}/archive`, {method: 'POST'}),
  health: () => request<{ok: boolean; model?: string; ollama: boolean; neo4j: boolean}>('/health'),
  embeddings: () => request<{count: number; embedding_model: string}>('/insights/embeddings', {method: 'POST'}),
}
