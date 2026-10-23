export type Book = {
  id: string; title: string; author: string; language: string; passage_count: number;
  chapters: number[]; cover_url: string; source: string; archived: boolean;
  summary?:string; subjects?:string[]; first_publish_year?:number|null; cover_origin?:string;
  metadata_sources?:{source_name:string;source_url:string;retrieved_at?:string;summary?:string}[];
}
export type Passage = {
  work_id: string; work_title: string; author: string; chapter: number | string;
  chunk_id: string; text: string; score: number; section: string; page: number | null;
  pages: number[]; citation_label: string; highlight?: number[]|null;
}
export type Edge = {
  source: string; target: string; label: string; evidence_chunk_id: string;
  source_label?: string; target_label?: string;
  source_mention?: string; target_mention?: string; category?: string;
  evidence: string; work_id: string; source_kind: string; target_kind: string;
}
export type GraphNode = {id: string; label: string; kind: string; work_id?: string; mentions?: Passage[]; aliases?: string[]}
export type Graph = { nodes: GraphNode[]; edges: Edge[]; coverage?: {processed: number; total: number; remaining: number}; identities?: {remaining:number;unresolved_mentions:number} }
export type Message = {
  role: 'user' | 'assistant'; content: string; citations?: Passage[];
  image_url?: string;
  visual_citations?: {type:'visual';work_id:string;visual_id:string;chapter:number|null;page:number|null;caption:string;local_url:string}[];
  visual_observation?: {ocr_text:string;visual_description:string;visible_entities:string[];objects:string[];possible_scene:string;uncertainties:string[]}|null;
  visual_analysis_available?: boolean;
  visual_note?: string|null;
  status?: 'answered'|'needs_clarification'|'refused'|'action_pending';
  clarification_question?: string|null;
  model_version?: string; vlm_version?: string|null; prompt_version?: string;
  visual_prompt_version?: string|null; reranker_version?: string; index_version?: string;
  visualization?: {title?: string; nodes?: {id: string; label: string; kind: string}[]; edges?: {source: string; target: string; label: string; evidence_chunk_id: string}[]};
  elapsed_ms?: number; used_model?: boolean;
  answer_kind?: 'answered'|'quoted'|'fallback'|'premise_rejected'|'spoiler_guarded'|'abstained';
}
export type CatalogBook = {
  provider_id: number; title: string; author: string; languages: string[];
  language_display: string; subjects: string[]; summary: string; cover_url: string; download_count: number;
}
export type DiscoveredBook = {
  provider:'gutendex'|'openlibrary'|'wikisource';provider_id:string;title:string;author:string;languages:string[];work_id?:string;
  cover_url:string;summary:string;subjects:string[];people:string[];first_publish_year?:number|null;
  source_name:string;source_url:string;can_download:boolean;format:string;anna_url:string;
}
export type EvalCheck = {name:string;passed:boolean;detail:string}
export type EvalCitation = {work_id:string;work_title:string;chapter:number|string;chunk_id:string;text:string;page:number|null;citation_label:string}
export type EvalResult = {
  scenario_id:string;category:string;question:string;outcome:'answered'|'quoted'|'fallback'|'premise_rejected'|'spoiler_guarded'|'abstained';
  outcome_label:string;passed:boolean;checks:EvalCheck[];elapsed_ms:number;model:string;ran_at:string;
  answer:string;status:string;citations:EvalCitation[];used_model:boolean;error:string|null;
  visual_observation?:{ocr_text:string;visual_description:string}|null;
}
export type EvalScenario = {
  id:string;category:string;title:string;question:string;work_ids:string[];max_chapter:number|null;mode:string;
  expect:'answer'|'abstain'|'reject_premise'|'safe';expected:string;gold:{work_id:string;contains:string}[];
  must_not_include:string[];image:string|null;showcase:boolean;available:boolean;last_result:EvalResult|null;
}
export type EvalCategory = {label:string;icon:string;goal:string}
export type EvalSummary = {total:number;passed:number;errors:number;by_category:Record<string,{passed:number;total:number}>;median_ms:number|null}
export type EvalRun = {run_id:string;started_at:string;finished_at:string;state:string;model:string;scenario_count:number;summary:EvalSummary}
export type EvalProgress = {state:'idle'|'running'|'done'|'cancelled'|'failed';run_id?:string;done?:number;total?:number;passed?:number;current?:string|null;error?:string}
export type OpenJob = {job_id:string;provider_id:number|string;provider:'gutenberg'|'wikisource';title:string;state:'queued'|'downloading'|'indexing'|'ready'|'done'|'failed';stage:string;work_id:string|null;already_installed:boolean;embedded:number;to_embed:number;error:string|null;warning:string|null}
export type DiscoveryResult ={books:DiscoveredBook[];count:number;page:number;has_next:boolean;query:string;warnings:string[];anna_url:string}
const API = '/api'

export class ApiError extends Error { constructor(message:string, public status:number) { super(message) } }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = localStorage.getItem('narrativelens:token')
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  let response: Response
  try { response = await fetch(`${API}${path}`, {...init, headers}) }
  catch { throw new ApiError('Le serveur est injoignable. Vérifie que l’API est démarrée sur le port 8000.',0) }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    const detail = typeof data.detail === 'string' ? data.detail : Array.isArray(data.detail) ? data.detail.map((item:{msg:string})=>item.msg).join(' · ') : null
    throw new ApiError(detail || (response.status>=500?'Le serveur ne répond pas correctement. Vérifie le service API puis réessaie.':`Erreur ${response.status}`),response.status)
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
  discoverBooks:(data:{query:string;language:string;page:number;source:string;use_model:boolean})=>request<DiscoveryResult>('/catalog/discover',json(data)),
  bookDetails:(book:Pick<DiscoveredBook,'provider'|'provider_id'>)=>request<DiscoveredBook>(`/catalog/details?${params(book)}`),
  attachBookMetadata:(id:string,book:Pick<DiscoveredBook,'provider'|'provider_id'>)=>request<{saved:boolean}>(`/books/${encodeURIComponent(id)}/metadata`,json(book)),
  login: (username:string,password:string) => request<{access_token:string;token_type:string}>('/auth/login',json({username,password})),
  register: (username:string,password:string) => request<{created:boolean}>('/auth/register',json({username,password})),
  library: () => request<{works: Book[]; model: string; embedding_model: string; stats: Record<string, number>; progress: Record<string, number>; max_upload_bytes: number}>('/library'),
  chat: (data: {question: string; work_ids: string[]; top_k: number; max_chapter: number | null; mode: string; history: {role: string; content: string}[]; conversation_id: string | null}) => request<Message & {conversation_id: string}>('/chat', json(data)),
  multimodalChat: (data: {question:string;work_ids:string[];top_k:number;max_chapter:number|null;mode:string;file:File|null;history:{role:string;content:string}[];conversation_id:string|null}) => {
    const form = new FormData(); form.set('question',data.question); data.work_ids.forEach(id=>form.append('work_ids',id));
    form.set('top_k',String(data.top_k)); form.set('mode',data.mode); form.set('history_json',JSON.stringify(data.history)); if(data.conversation_id)form.set('conversation_id',data.conversation_id); if(data.max_chapter!==null)form.set('max_chapter',String(data.max_chapter)); if(data.file)form.set('file',data.file);
    return request<Message & {conversation_id:string}>('/chat/multimodal',{method:'POST',body:form})
  },
  prepareAnnotation: (data:{work_id:string;chapter:number;text:string;evidence_ids:string[];max_chapter:number|null}) => request<{action:string;work_id:string;chapter:number;text:string;evidence_ids:string[];approval_token:string}>('/annotations/prepare',json(data)),
  saveAnnotation: (data:{work_id:string;chapter:number;text:string;evidence_ids:string[];max_chapter:number|null;approval_token:string}) => request<{saved:boolean;id:string;already_saved:boolean}>('/annotations/save',json(data)),
  search: (data: {query: string; work_ids: string[]; method: string; top_k: number; max_chapter: number | null}) => request<{passages: Passage[]}>('/search', json(data)),
  passages: async (id: string, chapter?: number) => {
    const passages: Passage[] = []; let total=0
    do {
      const page=await request<{total:number;passages:Passage[]}>(`/books/${encodeURIComponent(id)}/passages?${params({chapter,limit:100,offset:passages.length})}`)
      total=page.total; passages.push(...page.passages)
      if(!page.passages.length)break
    } while(passages.length<total)
    return {total,passages}
  },
  graph: (workIds: string[], chapter: number | null) => {
    const search = new URLSearchParams(); workIds.forEach(id => search.append('work_ids', id)); if (chapter) search.set('max_chapter', String(chapter))
    return request<Graph>(`/graph?${search}`)
  },
  buildGraph: (data: {work_ids: string[]; limit: number; max_chapter: number | null}) => request<Graph & {processed: number; remaining: number}>('/graph/build', json(data)),
  syncGraph: (data: {work_ids: string[]}) => request<{synced: number}>('/graph/sync', json(data)),
  catalogSearch: (query: string, language: string, page: number) => request<{count: number; page: number; has_next: boolean; has_previous: boolean; books: CatalogBook[]}>(`/catalog/search?${params({query, language, page})}`),
  installCatalog: (provider_id: number) => request<{work_id: string; title: string; warning?: string|null}>('/catalog/install', json({provider_id})),
  upload: (file: File, title: string, author: string, language: string, metadata?:Pick<DiscoveredBook,'provider'|'provider_id'>) => {
    const data = new FormData(); data.set('file', file); data.set('title', title); data.set('author', author); data.set('language', language)
    if(metadata){data.set('metadata_provider',metadata.provider);data.set('metadata_id',metadata.provider_id)}
    return request<{work_id: string; title: string; warning?: string|null}>('/books/upload', {method: 'POST', body: data})
  },
  archiveBook: (id: string) => request(`/library/archive/${encodeURIComponent(id)}`, {method: 'POST'}),
  restoreBook: (id: string) => request(`/library/restore/${encodeURIComponent(id)}`, {method: 'POST'}),
  reindexBook: (id: string) => request(`/library/reindex/${encodeURIComponent(id)}`, {method: 'POST'}),
  progress: (work_ids: string[], chapter: number) => request('/progress', json({work_ids, chapter})),
  conversations: (work_ids: string[], max_chapter: number | null) => request<{conversations: {id: string; title: string; updated_at: string}[]}>('/conversations', json({work_ids, max_chapter})),
  loadConversation: (id: string, workIds: string[], chapter: number | null) => request<{id: string; messages: Message[]}>(`/conversations/${id}?${(() => { const p = new URLSearchParams(); workIds.forEach(w => p.append('work_ids',w)); if(chapter)p.set('max_chapter',String(chapter)); return p })()}`),
  archiveConversation: (id: string) => request(`/conversations/${id}/archive`, {method: 'POST'}),
  health: () => request<{ok: boolean; model?: string; ollama: boolean; neo4j: boolean; auth_enabled?:boolean}>('/health'),
  embeddings: () => request<{count: number; embedding_model: string}>('/insights/embeddings', {method: 'POST'}),
  openBook: (book:Pick<DiscoveredBook,'provider'|'provider_id'>) => request<OpenJob>('/catalog/open',json({provider:book.provider==='wikisource'?'wikisource':'gutenberg',provider_id:book.provider==='wikisource'?book.provider_id:Number(book.provider_id)})),
  translate: (text:string) => request<{translation:string;target:string;machine_translation:boolean}>('/translate',json({text,target:'fr'})),
  openJob: (job_id:string) => request<OpenJob>(`/catalog/jobs/${encodeURIComponent(job_id)}`),
  evaluation: () => request<{categories:Record<string,EvalCategory>;scenarios:EvalScenario[];progress:EvalProgress;runs:EvalRun[]}>('/evaluation/scenarios'),
  runScenario: (id:string) => request<EvalResult>(`/evaluation/scenarios/${encodeURIComponent(id)}/run`,{method:'POST'}),
  startEvaluation: (scenario_ids:string[]|null) => request<EvalProgress>('/evaluation/runs',json({scenario_ids})),
  evaluationProgress: () => request<EvalProgress>('/evaluation/runs/current'),
  cancelEvaluation: () => request<EvalProgress>('/evaluation/runs/cancel',{method:'POST'}),
}
