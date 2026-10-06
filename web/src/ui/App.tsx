import {ProtectedImage, ProtectedVisual} from './ProtectedMedia'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError, type Book, type Graph, type Message, type Passage } from '../api'
import { GraphCanvas } from './GraphCanvas'
import { BookImport } from './BookImport'

type Section = 'Discussion' | 'Bibliothèque' | 'Chapitres' | 'Recherche' | 'Personnages' | 'Graphe' | 'Aperçu'
const sections: {id: Section; icon: string}[] = [
  {id:'Discussion',icon:'◉'}, {id:'Bibliothèque',icon:'▤'}, {id:'Chapitres',icon:'☷'},
  {id:'Recherche',icon:'⌕'}, {id:'Personnages',icon:'♙'}, {id:'Graphe',icon:'⌘'}, {id:'Aperçu',icon:'◷'},
]
const modes: Record<string,string> = {Question:'Ask',Expliquer:'Explain',Analyser:'Analyze',Résumer:'Summarize',Personnages:'Characters',"Citations et recherche":'Quotes / Search',Comparer:'Compare'}

function App() {
  const [library, setLibrary] = useState<Book[]>([])
  const [libraryReady,setLibraryReady] = useState(false)
  const [model, setModel] = useState('')
  const [embedding, setEmbedding] = useState('')
  const [progress, setProgress] = useState<Record<string,number>>({})
  const [selected, setSelected] = useState<string[]>([])
  const [section, setSection] = useState<Section>('Discussion')
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string|null>(null)
  const [chapterMode, setChapterMode] = useState<'none'|'chapter'|'finished'>('none')
  const [chapter, setChapter] = useState<number|null>(null)
  const [topK, setTopK] = useState(4)
  const [mode, setMode] = useState('Question')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [mobileNav, setMobileNav] = useState(false)
  const [catalogOpen, setCatalogOpen] = useState(false)
  const [metadataTarget,setMetadataTarget]=useState<Book|null>(null)
  const [composer, setComposer] = useState('')
  const [questionImage, setQuestionImage] = useState<File|null>(null)
  const [questionImagePreview, setQuestionImagePreview] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMethod, setSearchMethod] = useState('Hybrid')
  const [searchResults, setSearchResults] = useState<Passage[]>([])
  const [bookChapter, setBookChapter] = useState<number|null>(null)
  const [readingBookId,setReadingBookId] = useState('')
  const [chapterPassages, setChapterPassages] = useState<Passage[]>([])
  const [graph, setGraph] = useState<Graph>({nodes:[],edges:[]})
  const [graphSearch, setGraphSearch] = useState('')
  const [graphDepth, setGraphDepth] = useState(1)
  const [graphBuilding,setGraphBuilding] = useState(false)
  const stopGraph = useRef(false)
  const graphRun = useRef(false)
  const scopeRef = useRef('')
  const conversationRun = useRef(0)
  const [health, setHealth] = useState<{ollama:boolean;neo4j:boolean}|null>(null)
  const [conversations, setConversations] = useState<{id:string;title:string;updated_at:string}[]>([])
  const [conversationChoice, setConversationChoice] = useState('')
  const [authEnabled, setAuthEnabled] = useState(false)
  const [authReady, setAuthReady] = useState(false)
  const [authMode, setAuthMode] = useState<'login'|'register'>('login')
  const [authUser, setAuthUser] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authenticated, setAuthenticated] = useState(!!localStorage.getItem('narrativelens:token'))
  const [annotationDraft, setAnnotationDraft] = useState<{work_id:string;chapter:number;text:string;evidence_ids:string[]}|null>(null)
  const [preparedAnnotation, setPreparedAnnotation] = useState<{approval_token:string}|null>(null)

  const refreshLibrary = useCallback(async (preferred?: string) => {
    const data = await api.library()
    setLibrary(data.works)
    setModel(data.model)
    setEmbedding(data.embedding_model)
    setProgress(data.progress)
    setSelected(current => {
      const available = data.works.filter(book => !book.archived).map(book => book.id)
      let saved: string[] = current
      if(!current.length)try { const parsed=JSON.parse(localStorage.getItem('literary:selected')||'[]'); saved=Array.isArray(parsed)?parsed:[] } catch { saved=[] }
      const kept = saved.filter((id: string) => available.includes(id))
      const next = preferred && available.includes(preferred) ? [preferred] : kept.length ? kept : available.slice(0,1)
      return next
    })
    setLibraryReady(true)
  }, [])
  useEffect(() => {
    api.health().then(result=>{setAuthEnabled(!!result.auth_enabled);setHealth(result);setAuthReady(true)}).catch(()=>setAuthReady(true))
    refreshLibrary().catch(e=>{if(e instanceof ApiError&&e.status===401){localStorage.removeItem('narrativelens:token');setAuthenticated(false);setAuthEnabled(true);setAuthReady(true)}else setError(e.message)})
  }, [refreshLibrary])
  useEffect(() => { if(libraryReady)localStorage.setItem('literary:selected', JSON.stringify(selected)) }, [selected,libraryReady])

  const selectedBooks = useMemo(() => library.filter(book => selected.includes(book.id)), [library,selected])
  const activeChapters = useMemo(() => [...new Set(selectedBooks.flatMap(book => book.chapters))].sort((a,b)=>a-b), [selectedBooks])
  const maxChapter = chapterMode === 'none' ? null : chapterMode === 'finished' ? (activeChapters.at(-1) ?? null) : (chapter ?? activeChapters[0] ?? 1)
  const scope = `${selected.join('|')}:${maxChapter}`
  scopeRef.current = scope
  const activeBook = selectedBooks.find(book=>book.id===readingBookId)||selectedBooks[0]
  const archivedBooks = library.filter(book => book.archived)

  useEffect(() => {
    conversationRun.current++;setMessages([]); setConversationId(null); setBookChapter(null); setChapterPassages([]); setSearchResults([]); setGraph({nodes:[],edges:[]}); setGraphSearch(''); stopGraph.current=true
  }, [selected.join('|'), maxChapter])
  useEffect(() => {
    if (section !== 'Graphe' && section !== 'Personnages') return
    if (!selected.length) { setGraph({nodes:[],edges:[]}); return }
    let active=true
    api.graph(selected,maxChapter).then(data=>{if(active)setGraph(data)}).catch(e=>{if(active)setError(e.message)})
    return ()=>{active=false}
  }, [section,selected.join('|'),maxChapter])
  useEffect(() => {
    if (section !== 'Chapitres' || !activeBook || !bookChapter) return
    let active=true;setChapterPassages([])
    api.passages(activeBook.id,bookChapter).then(data=>{if(active)setChapterPassages(data.passages)}).catch(e=>{if(active)setError(e.message)})
    return ()=>{active=false}
  }, [section,activeBook?.id,bookChapter])
  useEffect(() => {
    if (!selected.length) { setConversations([]); return }
    let active=true
    api.conversations(selected,maxChapter).then(data=>{if(active)setConversations(data.conversations)}).catch(()=>{if(active)setConversations([])})
    return ()=>{active=false}
  }, [selected.join('|'),maxChapter,conversationId])

  const report = (action: () => Promise<unknown>, done?: (value:any)=>void, fallback='Action terminée.') => {
    setLoading(true); setError(''); setNotice('')
    action().then(async value=>{ setNotice(fallback); await done?.(value) }).catch(e=>setError(e.message)).finally(()=>setLoading(false))
  }
  const chooseSection = (next: Section) => { setSection(next); setMobileNav(false); setError('') }
  const submitQuestion = async (question = composer) => {
    const prompt=question.trim()
    if (!prompt || !selected.length || loading) return
    setComposer(''); setError(''); setLoading(true)
    const requestScope=scope
    const requestRun=++conversationRun.current
    const history=messages.slice(-12).map(({role,content})=>({role,content}))
    setMessages(current=>[...current,{role:'user',content:prompt}])
    try {
      const imageUrl=questionImagePreview||undefined
      setMessages(current=>[...current.slice(0,-1),{role:'user',content:prompt,image_url:imageUrl}])
      const answer=questionImage
        ? await api.multimodalChat({question:prompt,work_ids:selected,top_k:topK,max_chapter:maxChapter,mode:modes[mode]||'Ask',file:questionImage,history,conversation_id:conversationId})
        : await api.chat({question:prompt,work_ids:selected,top_k:topK,max_chapter:maxChapter,mode:modes[mode]||'Ask',history,conversation_id:conversationId})
      if(scopeRef.current!==requestScope||conversationRun.current!==requestRun)return
      setMessages(current=>[...current,answer]); setConversationId(answer.conversation_id)
      setQuestionImage(null)
      setQuestionImagePreview('')
      const saved=await api.conversations(selected,maxChapter); if(scopeRef.current===requestScope&&conversationRun.current===requestRun)setConversations(saved.conversations)
    } catch(e) { if(scopeRef.current===requestScope&&conversationRun.current===requestRun){setError((e as Error).message);setComposer(prompt)} }
    finally { setLoading(false) }
  }
  const newConversation = () => { conversationRun.current++;setMessages([]);setConversationChoice(''); setConversationId(null); setNotice('Nouvelle discussion prête.') }
  const loadConversation = async () => {
    if (!conversationChoice) return
    const requestScope=scope, requestRun=++conversationRun.current
    try { const data=await api.loadConversation(conversationChoice,selected,maxChapter); if(scopeRef.current!==requestScope||conversationRun.current!==requestRun)return;setMessages(data.messages); setConversationId(data.id); setNotice('Discussion ouverte.') }
    catch(e) { setError((e as Error).message) }
  }
  const exportConversation = () => {
    const text=['# NarrativeLens',...messages.map(m=>`## ${m.role==='user'?'Question':'Réponse'}\n\n${m.content}\n${(m.citations||[]).map(c=>`\n### ${c.citation_label}\n\n> ${c.text.replaceAll('\n','\n> ')}\n`).join('')}`)].join('\n')
    const url=URL.createObjectURL(new Blob([text],{type:'text/markdown'})); const link=document.createElement('a'); link.href=url; link.download='literary-chat.md'; link.click(); URL.revokeObjectURL(url)
  }
  const saveProgress = () => {
    if(chapter===null||!selected.length)return
    report(()=>api.progress(selected,chapter),result=>setProgress(result.progress),'Progression enregistrée.')
  }
  const doSearch = () => {
    if(loading||searchQuery.trim().length<2)return
    const requestScope=scope
    report(()=>api.search({query:searchQuery.trim(),work_ids:selected,method:searchMethod,top_k:8,max_chapter:maxChapter}),result=>{if(scopeRef.current===requestScope)setSearchResults(result.passages)},'Recherche terminée.')
  }
  const buildGraph = async () => {
    if(graphRun.current)return
    graphRun.current=true;stopGraph.current=false;setGraphBuilding(true);setError('');setNotice('')
    const requestScope=scope
    try {
      while(!stopGraph.current&&scopeRef.current===requestScope){
        const result=await api.buildGraph({work_ids:selected,limit:1,max_chapter:maxChapter})
        if(scopeRef.current!==requestScope)break
        setGraph(result)
        if(result.remaining===0 && !result.identities?.remaining){setNotice('Passages analysés et personnages regroupés.');break}
      }
    } catch(e){if(scopeRef.current===requestScope)setError((e as Error).message)}
    finally {graphRun.current=false;setGraphBuilding(false)}
  }
  const checkServices = () => report(()=>api.health(),result=>setHealth(result),'État des services actualisé.')
  const authenticate = async () => {
    setError('')
    try {
      if(authMode==='register')await api.register(authUser,authPassword)
      const result=await api.login(authUser,authPassword);localStorage.setItem('narrativelens:token',result.access_token);setAuthenticated(true);await refreshLibrary()
    } catch(e) { setError((e as Error).message) }
  }
  const logout = () => { localStorage.removeItem('narrativelens:token');setAuthenticated(false);setLibrary([]);setSelected([]);setMessages([]) }
  const prepareAnnotation = async () => {
    if(!annotationDraft)return
    try { const result=await api.prepareAnnotation({...annotationDraft,max_chapter:maxChapter});setPreparedAnnotation({approval_token:result.approval_token}) }
    catch(e){setError((e as Error).message)}
  }
  const confirmAnnotation = async () => {
    if(!annotationDraft||!preparedAnnotation)return
    try { await api.saveAnnotation({...annotationDraft,max_chapter:maxChapter,approval_token:preparedAnnotation.approval_token});setNotice('Annotation enregistrée.');setAnnotationDraft(null);setPreparedAnnotation(null) }
    catch(e){setError((e as Error).message);setPreparedAnnotation(null)}
  }
  const archiveBook = (book:Book) => report(()=>api.archiveBook(book.id),async()=>{await refreshLibrary();setSelected(current=>current.filter(id=>id!==book.id))},'Livre archivé. Ses fichiers sont conservés.')
  const restoreBook = (book:Book) => report(()=>api.restoreBook(book.id),async()=>refreshLibrary(book.id),`« ${book.title} » a été restauré.`)
  const reindexBook = (book:Book) => report(()=>api.reindexBook(book.id),async()=>refreshLibrary(book.id),`« ${book.title} » a été réindexé.`)

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav?'sidebar--open':''}`}>
      <div className="brand"><div className="brand-mark" aria-hidden="true"><span/><span/><span/></div><div><div className="brand-name">NarrativeLens</div><div className="brand-caption">analyse multimodale de livres</div></div></div>
      <div className="shelf-head"><span>Ma bibliothèque</span><button className="icon-button" title="Ajouter un livre" onClick={()=>setCatalogOpen(true)}>＋</button></div>
      <div className="book-list">
        {library.filter(book=>!book.archived).map(book=><button key={book.id} className={`book-row ${selected.includes(book.id)?'is-selected':''}`} onClick={()=>setSelected(current=>current.includes(book.id)?current.filter(id=>id!==book.id):[...current,book.id])}>
          {book.cover_url?<ProtectedImage src={book.cover_url} alt=""/>:<div className="cover-placeholder"><span>✧</span></div>}
          <span className="book-row-copy"><strong>{book.title}</strong><small>{book.author}</small></span><span className="book-check" aria-hidden="true">{selected.includes(book.id)?'✓':''}</span>
        </button>)}
        {!library.some(book=>!book.archived)&&<button className="empty-library" onClick={()=>setCatalogOpen(true)}>Ajouter ton premier livre</button>}
      </div>
      <div className="side-rule"/>
      <div className="sidebar-label">Ta lecture</div>
      <label className="field-label" htmlFor="spoiler-mode">Limite anti-spoiler</label>
      <select id="spoiler-mode" value={chapterMode} onChange={e=>{const next=e.target.value as typeof chapterMode;setChapterMode(next);setChapter(next==='chapter'?(progress[selected[0]]??activeChapters[0]??1):null)}}>
        <option value="none">Aucune restriction</option><option value="chapter">Jusqu’à un chapitre</option><option value="finished">Livre terminé</option>
      </select>
      {chapterMode==='chapter'&&<><label className="field-label" htmlFor="chapter-limit">Je suis au chapitre…</label><div className="inline-field"><select id="chapter-limit" value={chapter??progress[selected[0]]??activeChapters[0]??1} onChange={e=>setChapter(Number(e.target.value))}>{activeChapters.map(n=><option key={n} value={n}>{n}</option>)}</select><button className="subtle-button" onClick={saveProgress} disabled={!activeChapters.length}>Enregistrer</button></div></>}
      <label className="field-label" htmlFor="answer-mode">Mode de réponse</label>
      <select id="answer-mode" value={mode} onChange={e=>setMode(e.target.value)}>{Object.keys(modes).map(item=><option key={item}>{item}</option>)}</select>
      <div className="model-note"><span className="status-dot"/>Réponses locales avec <strong>{model||'Ollama'}</strong></div>
      <div className="side-rule"/>
      <nav className="side-nav" aria-label="Navigation principale">{sections.map(item=><button key={item.id} className={section===item.id?'nav-item nav-item--active':'nav-item'} onClick={()=>chooseSection(item.id)}><span className="nav-icon">{item.icon}</span>{item.id}</button>)}</nav>
      <div className="sidebar-bottom"><span className="privacy-mark">◈</span><span>Les livres restent sur cette machine.</span>{authEnabled&&authenticated&&<button className="text-button" onClick={logout}>Se déconnecter</button>}</div>
    </aside>

    <main className="main-area">
      <header className="topbar"><button className="mobile-menu icon-button" aria-label="Ouvrir la navigation" onClick={()=>setMobileNav(!mobileNav)}>☰</button><div className="breadcrumb"><span>{section}</span>{activeBook&&<><span className="crumb-divider">/</span><strong>{selectedBooks.length>1?`${selectedBooks.length} livres`:activeBook.title}</strong></>}</div><div className="topbar-state"><span className="status-dot"/>Traitement sur cet appareil</div></header>
      <div className="page-wrap">
        {error&&<div className="notice notice--error" role="alert"><span>!</span><p>{error}</p><button onClick={()=>setError('')} aria-label="Fermer">×</button></div>}
        {notice&&<div className="notice notice--success" role="status"><span>✓</span><p>{notice}</p><button onClick={()=>setNotice('')} aria-label="Fermer">×</button></div>}
        {!selectedBooks.length&&<div className="empty-state"><div className="empty-ornament">✧</div><h1>Choisis un livre pour commencer.</h1><p>Ajoute un roman à ta bibliothèque, puis pose-lui tes questions.</p><button className="primary-button" onClick={()=>setCatalogOpen(true)}>Ajouter un livre</button></div>}
        {selectedBooks.length>0&&section==='Discussion'&&<section className="chat-page">
          <div className="chat-heading"><div><div className="page-kicker">ESPACE DE LECTURE</div><h1>{activeBook?.title}</h1><p>{activeBook?.author}{selectedBooks.length>1?` et ${selectedBooks.length-1} autre(s) livre(s)`:''} · {activeBook?.passage_count.toLocaleString('fr-FR')} passages indexés</p></div><div className="chat-tools"><button className="secondary-button" onClick={newConversation}>＋ Nouvelle discussion</button><button className="icon-button bordered" title="Exporter en Markdown" disabled={!messages.length} onClick={exportConversation}>⇩</button></div></div>
          <div className="chat-options"><label>Analyser <select value={topK} onChange={e=>setTopK(Number(e.target.value))}>{[3,4,5,6,7,8].map(k=><option key={k} value={k}>{k} passages</option>)}</select></label><div className="conversation-picker"><select value={conversationChoice} onChange={e=>setConversationChoice(e.target.value)} aria-label="Discussions enregistrées"><option value="">Discussions enregistrées</option>{conversations.map(c=><option key={c.id} value={c.id}>{c.title}</option>)}</select>{conversationChoice&&<><button className="text-button" onClick={loadConversation}>Ouvrir</button><button className="text-button text-danger" onClick={()=>report(()=>api.archiveConversation(conversationChoice),()=>{setConversationChoice('');setConversations(current=>current.filter(c=>c.id!==conversationChoice))},'Discussion archivée.')}>Archiver</button></>}</div></div>
          <div className="conversation" aria-live="polite">
            {!messages.length&&!loading&&<div className="welcome-note"><div className="bookmark">✦</div><div><h2>À quoi penses-tu en lisant ?</h2><p>Demande une explication, retrouve une scène ou explore les liens entre les personnages. Les réponses renvoient aux passages du livre.</p></div></div>}
            {!messages.length&&!loading&&<div className="prompt-grid">{['Pourquoi Gregor se sent-il coupable ?','Quels passages parlent de la famille ?','Montre-moi les liens entre les personnages.'].map(q=><button key={q} className="prompt-card" onClick={()=>submitQuestion(q)}><span>↳</span>{q}</button>)}</div>}
            {messages.map((message,index)=><MessageView key={`${index}-${message.content.slice(0,10)}`} message={message} goGraph={()=>chooseSection('Graphe')} onAnnotate={()=>{const citation=message.citations?.[0];if(citation){setAnnotationDraft({work_id:citation.work_id,chapter:Number(citation.chapter)||1,text:message.content,evidence_ids:[citation.chunk_id]});setPreparedAnnotation(null)}}} />)}
            {loading&&<div className="thinking"><span className="loader"/><div><strong>Je relis les passages pertinents…</strong><small>La réponse se construit à partir du texte sélectionné.</small></div></div>}
          </div>
          <form className="composer" onSubmit={e=>{e.preventDefault();submitQuestion()}}>
            {questionImage&&<div className="image-preview"><img src={questionImagePreview} alt="Aperçu de l’image jointe"/><span>{questionImage.name}</span><button type="button" className="text-button" onClick={()=>{setQuestionImage(null);setQuestionImagePreview('')}}>Retirer</button></div>}
            <textarea value={composer} onChange={e=>setComposer(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submitQuestion()}}} placeholder="Pose une question sur le livre…" aria-label="Question sur le livre" disabled={loading}/><div className="composer-footer"><label className="image-pick">＋ Image<input type="file" accept="image/jpeg,image/png,image/webp" onChange={e=>{const file=e.target.files?.[0]||null;if(file&&file.size>10*1024*1024){setError('L’image doit faire 10 Mo maximum.');return}setQuestionImage(file);if(file){const reader=new FileReader();reader.onload=()=>setQuestionImagePreview(String(reader.result||''));reader.readAsDataURL(file)}else setQuestionImagePreview('')}}/></label><span>Entrée pour envoyer · Maj + Entrée pour une nouvelle ligne</span><button className="send-button" type="submit" aria-label="Envoyer la question" disabled={!composer.trim()||loading}>↑</button></div></form>
          <div className="chat-footnote">Les réponses peuvent se tromper : ouvre les passages cités pour vérifier le contexte.</div>
        </section>}
        {section==='Bibliothèque'&&<section className="library-page"><PageTitle title="Les livres de ta table" subtitle="Parcours les textes indexés, consulte leurs détails ou gère ta collection." action={<button className="primary-button" onClick={()=>setCatalogOpen(true)}>＋ Ajouter un livre</button>}/><div className="book-grid">{selectedBooks.map(book=><article className="book-detail" key={book.id}>{book.cover_url?<ProtectedImage className="detail-cover" src={book.cover_url} alt={`Couverture de ${book.title}`}/>:<div className="detail-cover cover-placeholder"><span>✧</span></div>}<div className="book-detail-copy"><span className="book-lang">{book.language||'LANGUE NON PRÉCISÉE'}</span><h2>{book.title}</h2><p className="book-author">{book.author}</p><p>{book.passage_count.toLocaleString('fr-FR')} passages · {book.chapters.length} chapitres détectés</p><p className="source-line">Source : {book.source}</p>{book.cover_origin&&<p className="source-line">Couverture : {book.cover_origin}</p>}{book.summary&&<details className="book-summary"><summary>Résumé de la fiche · peut révéler l’intrigue</summary><p>{book.summary}</p>{book.metadata_sources?.filter(m=>m.summary).map((m,i)=><small key={i}>{m.source_url?<a href={m.source_url} target="_blank" rel="noreferrer">{m.source_name} ↗</a>:m.source_name}</small>)}</details>}{!!book.subjects?.length&&<p className="book-subjects">{book.subjects.slice(0,6).join(' · ')}</p>}<div className="button-row"><button className="secondary-button" onClick={()=>{setMetadataTarget(book);setCatalogOpen(true)}}>Compléter la fiche</button><button className="secondary-button" onClick={()=>reindexBook(book)} disabled={loading}>Réindexer</button><button className="text-button text-danger" onClick={()=>archiveBook(book)} disabled={loading}>Archiver</button></div></div></article>)}</div>{archivedBooks.length>0&&<div className="archived-list"><h2>Livres archivés</h2>{archivedBooks.map(book=><div key={book.id}><span>{book.title} · {book.author}</span><button className="text-button" onClick={()=>restoreBook(book)}>Restaurer</button></div>)}</div>}</section>}
        {selectedBooks.length>0&&section==='Chapitres'&&<section className="reading-page"><PageTitle title="Lire par chapitre" subtitle="Retrouve les passages indexés, en respectant ta limite anti-spoiler."/><div className="chapter-layout"><aside className="chapter-list"><label className="field-label" htmlFor="chapter-book">Livre</label><select id="chapter-book" value={activeBook?.id||''} onChange={e=>{setReadingBookId(e.target.value);setBookChapter(null);setChapterPassages([])}}>{selectedBooks.map(book=><option key={book.id} value={book.id}>{book.title}</option>)}</select>{activeBook?.chapters.filter(n=>maxChapter===null||n<=maxChapter).map(n=><button key={n} className={bookChapter===n?'chapter-link chapter-link--active':'chapter-link'} onClick={()=>setBookChapter(n)}>Chapitre {n}<span>›</span></button>)}</aside><div className="chapter-content">{bookChapter?<><div className="page-kicker">{activeBook?.title}</div><h2>Chapitre {bookChapter}</h2>{chapterPassages.map(p=><PassageCard key={p.chunk_id} passage={p}/>)}</>:<div className="empty-reading"><span>☷</span><h2>Choisis un chapitre</h2><p>Les passages disponibles apparaîtront ici dans l’ordre du livre.</p></div>}</div></div></section>}
        {selectedBooks.length>0&&section==='Recherche'&&<section className="search-page"><PageTitle title="Chercher dans les livres" subtitle="Un mot, un thème, une citation : retrouve le passage qui te revient."/><div className="search-panel"><div className="search-input-wrap"><span>⌕</span><input value={searchQuery} onChange={e=>setSearchQuery(e.target.value)} onKeyDown={e=>e.key==='Enter'&&doSearch()} placeholder="Que souhaites-tu retrouver ?"/></div><select value={searchMethod} onChange={e=>setSearchMethod(e.target.value)}><option value="Hybrid">Combinée</option><option value="BM25">Mots-clés (BM25)</option><option value="Sémantique">Sémantique</option></select><button className="primary-button" onClick={doSearch} disabled={loading||searchQuery.trim().length<2}>Rechercher</button></div>{searchResults.length>0?<div className="results-list"><p className="result-count">{searchResults.length} passages pertinents</p>{searchResults.map(p=><PassageCard key={p.chunk_id} passage={p}/>)}</div>:<div className="quiet-empty"><span>⌕</span><p>La recherche explore seulement les livres sélectionnés et les chapitres accessibles.</p></div>}</section>}
        {selectedBooks.length>0&&(section==='Graphe'||section==='Personnages')&&<section className="graph-page"><PageTitle title={section==='Graphe'?'Les liens dans le récit':'Personnages'} subtitle="Chaque relation est appuyée par une citation du texte."/><div className="graph-toolbar"><div className="search-input-wrap compact"><span>⌕</span><input value={graphSearch} onChange={e=>setGraphSearch(e.target.value)} placeholder="Chercher un personnage…"/></div><label>Voisinage <select value={graphDepth} onChange={e=>setGraphDepth(Number(e.target.value))}><option value={1}>1 relation</option><option value={2}>2 relations</option></select></label><button className="secondary-button" onClick={buildGraph} disabled={loading||graphBuilding||(graph.coverage?.remaining===0&&!graph.identities?.remaining)}>{graphBuilding?(graph.coverage?.remaining===0?'Regroupement des personnages…':'Analyse en cours…'):graph.coverage?.remaining===0?(graph.identities?.remaining?'Regrouper les personnages':'Analyse terminée'):'Analyser tous les passages'}</button>{graphBuilding&&<button className="text-button" onClick={()=>{stopGraph.current=true;setNotice('Pause demandée : le passage en cours sera conservé.')}}>Mettre en pause</button>}<button className="icon-button bordered" title="Synchroniser Neo4j" onClick={()=>report(()=>api.syncGraph({work_ids:selected}),result=>setNotice(`${result.synced} relations synchronisées dans Neo4j.`))}>⟳</button></div>{graph.coverage&&<div className="graph-progress" role="status"><progress max={graph.coverage.total||1} value={graph.coverage.processed}/><span>{graph.coverage.processed} / {graph.coverage.total} passages analysés{graph.identities?.remaining ? ' · Identités à résoudre : '+graph.identities.remaining+' lots' : ' · Identités regroupées'}{graph.coverage.remaining>0?' · Graphe partiel':' · Tous les passages accessibles ont été traités'}</span><small>L’extraction peut manquer des personnages ou des relations. Chaque élément affiché conserve sa source.</small></div>}{section==='Graphe'?<GraphCanvas key={scope} graph={graph} query={graphSearch} depth={graphDepth}/>:<CharacterList key={scope} graph={graph} query={graphSearch}/>}</section>}
        {selectedBooks.length>0&&section==='Aperçu'&&<section className="insights-page"><PageTitle title="Vue d’ensemble" subtitle="L’état de l’index et des services locaux."/><div className="insight-stats">{selectedBooks.map(book=><div className="stat-paper" key={book.id}><span>{book.title}</span><strong>{book.passage_count.toLocaleString('fr-FR')}</strong><small>passages indexés</small><div className="stat-rule"/><strong>{book.chapters.length}</strong><small>chapitres détectés</small></div>)}</div><div className="service-panel"><div><h2>Services de lecture</h2><p>La recherche sémantique utilise {embedding||'les réglages locaux'}. Les textes restent sur cet appareil.</p></div><button className="secondary-button" onClick={checkServices} disabled={loading}>Vérifier les services</button>{health&&<div className="service-badges"><span className={health.ollama?'service-ok':'service-off'}><i/>Ollama {health.ollama?'disponible':'indisponible'}</span><span className={health.neo4j?'service-ok':'service-off'}><i/>Neo4j {health.neo4j?'disponible':'indisponible'}</span></div>}<button className="text-button" onClick={()=>report(()=>api.embeddings(),result=>setNotice(`${result.count} passages vectorisés avec ${result.embedding_model}.`),'Index sémantique actualisé.')}>Préparer ou réparer les embeddings</button></div><div className="about-local"><span>◈</span><div><strong>Un compagnon privé</strong><p>Les livres, les discussions et les citations restent stockés localement. La recherche de catalogue utilise Gutendex, uniquement pour les œuvres du domaine public.</p></div></div></section>}
      </div>
    </main>
    {mobileNav&&<button className="scrim" aria-label="Fermer la navigation" onClick={()=>setMobileNav(false)}/>}
    {annotationDraft&&<div className="modal-backdrop" role="presentation" onClick={e=>e.target===e.currentTarget&&(setAnnotationDraft(null),setPreparedAnnotation(null))}><section className="annotation-modal" role="dialog" aria-modal="true" aria-labelledby="annotation-title"><div className="modal-head"><div><div className="page-kicker">ANNOTATION PERSONNELLE</div><h2 id="annotation-title">Ajouter à mes notes</h2></div><button className="icon-button" onClick={()=>{setAnnotationDraft(null);setPreparedAnnotation(null)}} aria-label="Fermer">×</button></div><label className="field-label" htmlFor="annotation-text">Texte qui sera enregistré</label><textarea id="annotation-text" value={annotationDraft.text} onChange={e=>{setAnnotationDraft({...annotationDraft,text:e.target.value});setPreparedAnnotation(null)}} disabled={!!preparedAnnotation}/><p>Chapitre {annotationDraft.chapter} · Preuve : {annotationDraft.evidence_ids.join(', ')}</p>{preparedAnnotation?<><div className="annotation-exact"><strong>Prêt à enregistrer après confirmation</strong><p>{annotationDraft.text}</p></div><button className="primary-button" onClick={confirmAnnotation}>Confirmer et enregistrer</button></>:<button className="primary-button" onClick={prepareAnnotation} disabled={!annotationDraft.text.trim()}>Préparer l’annotation</button>}</section></div>}
    {authReady&&authEnabled&&!authenticated&&<div className="auth-overlay"><form className="auth-card" onSubmit={e=>{e.preventDefault();authenticate()}}><div className="page-kicker">NARRATIVELENS</div><h1>{authMode==='login'?'Se connecter':'Créer un compte local'}</h1><p>Les conversations, la progression et les annotations restent séparées par compte.</p>{error&&<div className="notice notice--error" role="alert"><p>{error}</p></div>}<label className="field-label" htmlFor="auth-user">Nom d’utilisateur</label><input id="auth-user" autoComplete="username" value={authUser} onChange={e=>setAuthUser(e.target.value)}/><label className="field-label" htmlFor="auth-password">Mot de passe</label><input id="auth-password" type="password" autoComplete={authMode==='login'?'current-password':'new-password'} value={authPassword} onChange={e=>setAuthPassword(e.target.value)}/><button className="primary-button" type="submit">{authMode==='login'?'Se connecter':'Créer le compte'}</button><button className="text-button" type="button" onClick={()=>setAuthMode(authMode==='login'?'register':'login')}>{authMode==='login'?'Créer un compte':'J’ai déjà un compte'}</button></form></div>}
    {catalogOpen&&<BookImport target={metadataTarget} onClose={()=>{setCatalogOpen(false);setMetadataTarget(null)}} onDone={async(id,message)=>{await refreshLibrary(id);setCatalogOpen(false);setMetadataTarget(null);setNotice(message);setSection('Bibliothèque')}}/>}
  </div>
}

function PageTitle({title,subtitle,action}:{title:string;subtitle:string;action?:React.ReactNode}) { return <div className="page-title"><div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div> }

function MessageView({message,goGraph,onAnnotate}:{message:Message;goGraph:()=>void;onAnnotate:()=>void}) {
  const [open,setOpen]=useState<number|null>(null)
  const visual=message.visualization
  const messageGraph:Graph={nodes:visual?.nodes||[],edges:(visual?.edges||[]).map(edge=>({source:edge.source,target:edge.target,source_label:visual?.nodes?.find(n=>n.id===edge.source)?.label,target_label:visual?.nodes?.find(n=>n.id===edge.target)?.label,label:edge.label,evidence_chunk_id:edge.evidence_chunk_id,evidence:message.citations?.find(c=>c.chunk_id===edge.evidence_chunk_id)?.text||'',work_id:'',source_kind:'character',target_kind:'character'}))}
  return <article className={`message message--${message.role}`}><div className="message-avatar">{message.role==='assistant'?<span className="mini-book">✧</span>:<span>◉</span>}</div><div className="message-body"><div className="message-author">{message.role==='assistant'?'NarrativeLens':'Toi'}{message.elapsed_ms&&<span>{(message.elapsed_ms/1000).toFixed(1)} s</span>}</div>{message.image_url&&<img className="message-image" src={message.image_url} alt="Image envoyée avec la question"/>}<div className="message-text">{message.content}</div>{message.visual_observation&&<section className="visual-observation"><strong>Observation visuelle · à confirmer dans le livre</strong><p>{message.visual_observation.visual_description}</p>{message.visual_observation.ocr_text&&<small>Texte lu dans l’image : {message.visual_observation.ocr_text}</small>}{message.visual_observation.uncertainties.length>0&&<small>Incertitudes : {message.visual_observation.uncertainties.join(' · ')}</small>}</section>}{message.visual_note&&<p className="visual-note">{message.visual_note}</p>}{message.visual_citations&&message.visual_citations.length>0&&<div className="visual-citations"><div className="citation-heading">IMAGES RETROUVÉES DANS LE LIVRE</div>{message.visual_citations.map(c=><ProtectedVisual key={c.visual_id} src={c.local_url} caption={c.caption||`Chapitre ${c.chapter??'inconnu'}${c.page?`, page ${c.page}`:''}`}/>)}</div>}{message.citations&&message.citations.length>0&&<div className="citation-list"><div className="citation-heading">PASSAGES À L’APPUI</div>{message.citations.map((citation,index)=><div className="citation" key={citation.chunk_id}><button onClick={()=>setOpen(open===index?null:index)}><span className="citation-glyph">⌞</span><span><strong>{citation.citation_label}</strong><small>{citation.text.slice(0,130).replace(/\s+/g,' ')}{citation.text.length>130?'…':''}</small></span><span className="citation-chevron">{open===index?'−':'+'}</span></button>{open===index&&<blockquote>{citation.text}</blockquote>}</div>)}</div>}{message.role==='assistant'&&message.citations&&message.citations.length>0&&<button className="text-button" onClick={onAnnotate}>＋ Ajouter une annotation</button>}{messageGraph.edges.length>0&&<div className="answer-graph"><div className="answer-graph-head"><div><span className="page-kicker">RELATIONS DANS LE TEXTE</span><h3>{visual?.title||'Les liens entre personnages'}</h3></div><button className="text-button" onClick={goGraph}>Explorer le graphe</button></div><div className="answer-edges">{messageGraph.edges.slice(0,6).map((edge,index)=><div key={`${edge.source}-${edge.target}-${index}`}><span>{edge.source_label||edge.source}</span><small>{edge.label}</small><span>{edge.target_label||edge.target}</span></div>)}</div></div>}</div></article>
}

function PassageCard({passage}:{passage:Passage}) { return <article className="passage-card"><div className="passage-cite"><span>⌞</span>{passage.citation_label}</div><blockquote>{passage.text}</blockquote></article> }

function CharacterList({graph,query}:{graph:Graph;query:string}) {
  const [selectedId,setSelectedId]=useState<string|null>(null)
  const characters=graph.nodes.filter(n=>(n.kind==='character'||n.kind==='group')&&[n.label,...(n.aliases||[])].some(label=>label.toLocaleLowerCase().includes(query.toLocaleLowerCase())))
  const person=characters.find(n=>n.id===selectedId)
  const relations=graph.edges.filter(e=>e.source===person?.id||e.target===person?.id)
  return <div className="character-layout"><div className="character-list">{characters.map(person=>{const count=graph.edges.filter(e=>e.source===person.id||e.target===person.id).length;return <button key={person.id} className="character-row" onClick={()=>setSelectedId(person.id)}><span className="character-seal">{person.label.slice(0,1)}</span><span><strong>{person.label}</strong><small>{count} relations · {person.mentions?.length||0} passages</small></span><i>›</i></button>})}{characters.length===0&&<div className="quiet-empty"><p>Aucun personnage ne correspond à cette recherche. Vérifie la couverture de l’analyse.</p></div>}</div><div className="evidence-panel"><div className="page-kicker">PREUVES DANS LE TEXTE</div>{person?<><h3>{person.label}</h3>{relations.length===0&&<p>Personnage repéré ; aucune relation validée pour l’instant.</p>}{relations.map((edge,i)=><details key={i}><summary>{edge.source_label||edge.source} · {edge.label} · {edge.target_label||edge.target}</summary><blockquote>{edge.evidence}</blockquote><small>{edge.evidence_chunk_id}</small></details>)}<h4>Mentions dans le livre</h4>{person.mentions?.map(m=><details key={m.chunk_id}><summary>{m.citation_label}</summary><blockquote>{m.text}</blockquote></details>)}</>:<p>Choisis un personnage pour consulter toutes ses relations et les passages où il apparaît.</p>}</div></div>
}

export { App }
