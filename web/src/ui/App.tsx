import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type Book, type CatalogBook, type Edge, type Graph, type Message, type Passage } from '../api'
import { GraphCanvas } from './GraphCanvas'

type Section = 'Discussion' | 'Bibliothèque' | 'Chapitres' | 'Recherche' | 'Personnages' | 'Graphe' | 'Aperçu'
const sections: {id: Section; icon: string}[] = [
  {id:'Discussion',icon:'◉'}, {id:'Bibliothèque',icon:'▤'}, {id:'Chapitres',icon:'☷'},
  {id:'Recherche',icon:'⌕'}, {id:'Personnages',icon:'♙'}, {id:'Graphe',icon:'⌘'}, {id:'Aperçu',icon:'◷'},
]
const modes: Record<string,string> = {Question:'Ask',Expliquer:'Explain',Analyser:'Analyze',Résumer:'Summarize',Personnages:'Characters',"Citations et recherche":'Quotes / Search',Comparer:'Compare'}
const languages = [['Toutes les langues',''],['Français','fr'],['Anglais','en'],['Espagnol','es'],['Allemand','de'],['Italien','it'],['Portugais','pt']]

function App() {
  const [library, setLibrary] = useState<Book[]>([])
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
  const [catalogQuery, setCatalogQuery] = useState('')
  const [catalogLanguage, setCatalogLanguage] = useState('')
  const [catalogBooks, setCatalogBooks] = useState<CatalogBook[]>([])
  const [catalogCount, setCatalogCount] = useState(0)
  const [catalogPage, setCatalogPage] = useState(1)
  const [catalogHasNext, setCatalogHasNext] = useState(false)
  const [upload, setUpload] = useState<File|null>(null)
  const [uploadTitle, setUploadTitle] = useState('')
  const [uploadAuthor, setUploadAuthor] = useState('')
  const [uploadLanguage, setUploadLanguage] = useState('')
  const [composer, setComposer] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMethod, setSearchMethod] = useState('Hybrid')
  const [searchResults, setSearchResults] = useState<Passage[]>([])
  const [bookChapter, setBookChapter] = useState<number|null>(null)
  const [chapterPassages, setChapterPassages] = useState<Passage[]>([])
  const [graph, setGraph] = useState<Graph>({nodes:[],edges:[]})
  const [graphSearch, setGraphSearch] = useState('')
  const [graphDepth, setGraphDepth] = useState(1)
  const [health, setHealth] = useState<{ollama:boolean;neo4j:boolean}|null>(null)
  const [conversations, setConversations] = useState<{id:string;title:string;updated_at:string}[]>([])
  const [conversationChoice, setConversationChoice] = useState('')

  const refreshLibrary = useCallback(async (preferred?: string) => {
    const data = await api.library()
    setLibrary(data.works)
    setModel(data.model)
    setEmbedding(data.embedding_model)
    setProgress(data.progress)
    setSelected(current => {
      const available = data.works.filter(book => !book.archived).map(book => book.id)
      const saved = current.length ? current : JSON.parse(localStorage.getItem('literary:selected') || '[]') as string[]
      const kept = saved.filter((id: string) => available.includes(id))
      const next = preferred && available.includes(preferred) ? [preferred] : kept.length ? kept : available.slice(0,1)
      return next
    })
  }, [])
  useEffect(() => { refreshLibrary().catch(e => setError(e.message)) }, [refreshLibrary])
  useEffect(() => { localStorage.setItem('literary:selected', JSON.stringify(selected)) }, [selected])

  const selectedBooks = useMemo(() => library.filter(book => selected.includes(book.id)), [library,selected])
  const activeChapters = useMemo(() => [...new Set(selectedBooks.flatMap(book => book.chapters))].sort((a,b)=>a-b), [selectedBooks])
  const maxChapter = chapterMode === 'none' ? null : chapterMode === 'finished' ? (activeChapters.at(-1) ?? null) : chapter
  const activeBook = selectedBooks[0]
  const archivedBooks = library.filter(book => book.archived)

  useEffect(() => {
    setMessages([]); setConversationId(null); setChapter(null); setBookChapter(null)
  }, [selected.join('|'), maxChapter])
  useEffect(() => {
    if (section !== 'Graphe' && section !== 'Personnages') return
    if (!selected.length) { setGraph({nodes:[],edges:[]}); return }
    api.graph(selected,maxChapter).then(setGraph).catch(e=>setError(e.message))
  }, [section,selected.join('|'),maxChapter])
  useEffect(() => {
    if (section !== 'Chapitres' || !activeBook || !bookChapter) return
    api.passages(activeBook.id,bookChapter).then(data=>setChapterPassages(data.passages)).catch(e=>setError(e.message))
  }, [section,activeBook?.id,bookChapter])
  useEffect(() => {
    if (!selected.length) { setConversations([]); return }
    api.conversations(selected,maxChapter).then(data=>setConversations(data.conversations)).catch(()=>setConversations([]))
  }, [selected.join('|'),maxChapter,conversationId])

  const report = (action: () => Promise<unknown>, done?: (value:any)=>void, fallback='Action terminée.') => {
    setLoading(true); setError(''); setNotice('')
    action().then(value=>{ done?.(value); setNotice(fallback) }).catch(e=>setError(e.message)).finally(()=>setLoading(false))
  }
  const chooseSection = (next: Section) => { setSection(next); setMobileNav(false); setError('') }
  const submitQuestion = async (question = composer) => {
    const prompt=question.trim()
    if (!prompt || !selected.length || loading) return
    setComposer(''); setError(''); setLoading(true)
    const history=messages.slice(-12).map(({role,content})=>({role,content}))
    setMessages(current=>[...current,{role:'user',content:prompt}])
    try {
      const answer=await api.chat({question:prompt,work_ids:selected,top_k:topK,max_chapter:maxChapter,mode:modes[mode]||'Ask',history,conversation_id:conversationId})
      setMessages(current=>[...current,answer]); setConversationId(answer.conversation_id)
      const saved=await api.conversations(selected,maxChapter); setConversations(saved.conversations)
    } catch(e) { setError((e as Error).message); setMessages(current=>[...current,{role:'assistant',content:`Je n’ai pas pu terminer la recherche. ${(e as Error).message}`}]) }
    finally { setLoading(false) }
  }
  const newConversation = () => { setMessages([]); setConversationId(null); setNotice('Nouvelle discussion prête.') }
  const loadConversation = async () => {
    if (!conversationChoice) return
    try { const data=await api.loadConversation(conversationChoice,selected,maxChapter); setMessages(data.messages); setConversationId(data.id); setNotice('Discussion ouverte.') }
    catch(e) { setError((e as Error).message) }
  }
  const exportConversation = () => {
    const text=['# Literary Chat',...messages.map(m=>`## ${m.role==='user'?'Question':'Réponse'}\n\n${m.content}\n${(m.citations||[]).map(c=>`\n### ${c.citation_label}\n\n> ${c.text.replaceAll('\n','\n> ')}\n`).join('')}`)].join('\n')
    const url=URL.createObjectURL(new Blob([text],{type:'text/markdown'})); const link=document.createElement('a'); link.href=url; link.download='literary-chat.md'; link.click(); URL.revokeObjectURL(url)
  }
  const searchCatalog = async (page=1) => {
    if(catalogQuery.trim().length<2){setError('Saisis au moins deux caractères pour lancer la recherche.');return}
    setLoading(true);setError('')
    try{const data=await api.catalogSearch(catalogQuery,catalogLanguage,page);setCatalogBooks(data.books);setCatalogCount(data.count);setCatalogPage(page);setCatalogHasNext(data.has_next)}catch(e){setError((e as Error).message)}finally{setLoading(false)}
  }
  const installCatalog = (book: CatalogBook) => report(()=>api.installCatalog(book.provider_id),async result=>{await refreshLibrary(result.work_id);setCatalogOpen(false);if(result.warning)setNotice(result.warning)},`« ${book.title} » est prêt dans la bibliothèque.`)
  const installUpload = () => {
    if(!upload)return
    report(()=>api.upload(upload,uploadTitle,uploadAuthor,uploadLanguage),async result=>{await refreshLibrary(result.work_id);setUpload(null);setCatalogOpen(false);if(result.warning)setNotice(result.warning)},'Le livre a été indexé et ajouté à la bibliothèque.')
  }
  const saveProgress = () => {
    if(chapter===null||!selected.length)return
    report(()=>api.progress(selected,chapter),result=>setProgress(result.progress),'Progression enregistrée.')
  }
  const doSearch = () => report(()=>api.search({query:searchQuery,work_ids:selected,method:searchMethod,top_k:8,max_chapter:maxChapter}),result=>setSearchResults(result.passages),'Recherche terminée.')
  const buildGraph = () => report(()=>api.buildGraph({work_ids:selected,limit:5,max_chapter:maxChapter}),result=>setGraph(result),'Le graphe a été mis à jour.')
  const checkServices = () => report(()=>api.health(),result=>setHealth(result),'État des services actualisé.')
  const archiveBook = (book:Book) => report(()=>api.archiveBook(book.id),async()=>{await refreshLibrary();setSelected(current=>current.filter(id=>id!==book.id))},'Livre archivé. Ses fichiers sont conservés.')
  const restoreBook = (book:Book) => report(()=>api.restoreBook(book.id),async()=>refreshLibrary(book.id),`« ${book.title} » a été restauré.`)
  const reindexBook = (book:Book) => report(()=>api.reindexBook(book.id),async()=>refreshLibrary(book.id),`« ${book.title} » a été réindexé.`)

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav?'sidebar--open':''}`}>
      <div className="brand"><div className="brand-mark" aria-hidden="true"><span/><span/><span/></div><div><div className="brand-name">Literary Chat</div><div className="brand-caption">compagnon de lecture</div></div></div>
      <div className="shelf-head"><span>Ma bibliothèque</span><button className="icon-button" title="Ajouter un livre" onClick={()=>setCatalogOpen(true)}>＋</button></div>
      <div className="book-list">
        {library.filter(book=>!book.archived).map(book=><button key={book.id} className={`book-row ${selected.includes(book.id)?'is-selected':''}`} onClick={()=>setSelected(current=>current.includes(book.id)?current.filter(id=>id!==book.id):[...current,book.id])}>
          {book.cover_url?<img src={book.cover_url} alt=""/>:<div className="cover-placeholder"><span>✧</span></div>}
          <span className="book-row-copy"><strong>{book.title}</strong><small>{book.author}</small></span><span className="book-check" aria-hidden="true">{selected.includes(book.id)?'✓':''}</span>
        </button>)}
        {!library.some(book=>!book.archived)&&<button className="empty-library" onClick={()=>setCatalogOpen(true)}>Ajouter ton premier livre</button>}
      </div>
      <div className="side-rule"/>
      <div className="sidebar-label">Ta lecture</div>
      <label className="field-label" htmlFor="spoiler-mode">Limite anti-spoiler</label>
      <select id="spoiler-mode" value={chapterMode} onChange={e=>{const next=e.target.value as typeof chapterMode;setChapterMode(next);setChapter(next==='chapter'?(progress[selected[0]]??activeChapters.at(-1)??1):null)}}>
        <option value="none">Aucune restriction</option><option value="chapter">Jusqu’à un chapitre</option><option value="finished">Livre terminé</option>
      </select>
      {chapterMode==='chapter'&&<><label className="field-label" htmlFor="chapter-limit">Je suis au chapitre…</label><div className="inline-field"><select id="chapter-limit" value={chapter??progress[selected[0]]??activeChapters[0]??1} onChange={e=>setChapter(Number(e.target.value))}>{activeChapters.map(n=><option key={n} value={n}>{n}</option>)}</select><button className="subtle-button" onClick={saveProgress} disabled={!activeChapters.length}>Enregistrer</button></div></>}
      <label className="field-label" htmlFor="answer-mode">Mode de réponse</label>
      <select id="answer-mode" value={mode} onChange={e=>setMode(e.target.value)}>{Object.keys(modes).map(item=><option key={item}>{item}</option>)}</select>
      <div className="model-note"><span className="status-dot"/>Réponses locales avec <strong>{model||'Ollama'}</strong></div>
      <div className="side-rule"/>
      <nav className="side-nav" aria-label="Navigation principale">{sections.map(item=><button key={item.id} className={section===item.id?'nav-item nav-item--active':'nav-item'} onClick={()=>chooseSection(item.id)}><span className="nav-icon">{item.icon}</span>{item.id}</button>)}</nav>
      <div className="sidebar-bottom"><span className="privacy-mark">◈</span><span>Les livres restent sur cette machine.</span></div>
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
            {messages.map((message,index)=><MessageView key={`${index}-${message.content.slice(0,10)}`} message={message} goGraph={()=>chooseSection('Graphe')} />)}
            {loading&&<div className="thinking"><span className="loader"/><div><strong>Je relis les passages pertinents…</strong><small>La réponse se construit à partir du texte sélectionné.</small></div></div>}
          </div>
          <form className="composer" onSubmit={e=>{e.preventDefault();submitQuestion()}}><textarea value={composer} onChange={e=>setComposer(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submitQuestion()}}} placeholder="Pose une question sur le roman…" aria-label="Question sur le roman" disabled={loading}/><div className="composer-footer"><span>Entrée pour envoyer · Maj + Entrée pour une nouvelle ligne</span><button className="send-button" type="submit" aria-label="Envoyer la question" disabled={!composer.trim()||loading}>↑</button></div></form>
          <div className="chat-footnote">Les réponses peuvent se tromper : ouvre les passages cités pour vérifier le contexte.</div>
        </section>}
        {selectedBooks.length>0&&section==='Bibliothèque'&&<section className="library-page"><PageTitle title="Les livres de ta table" subtitle="Parcours les textes indexés, consulte leurs détails ou gère ta collection." action={<button className="primary-button" onClick={()=>setCatalogOpen(true)}>＋ Ajouter un livre</button>}/><div className="book-grid">{selectedBooks.map(book=><article className="book-detail" key={book.id}>{book.cover_url?<img className="detail-cover" src={book.cover_url} alt={`Couverture de ${book.title}`}/>:<div className="detail-cover cover-placeholder"><span>✧</span></div>}<div className="book-detail-copy"><span className="book-lang">{book.language||'LANGUE NON PRÉCISÉE'}</span><h2>{book.title}</h2><p className="book-author">{book.author}</p><p>{book.passage_count.toLocaleString('fr-FR')} passages · {book.chapters.length} chapitres détectés</p><p className="source-line">Source : {book.source}</p><div className="button-row"><button className="secondary-button" onClick={()=>reindexBook(book)} disabled={loading}>Réindexer</button><button className="text-button text-danger" onClick={()=>archiveBook(book)} disabled={loading}>Archiver</button></div></div></article>)}</div>{archivedBooks.length>0&&<div className="archived-list"><h2>Livres archivés</h2>{archivedBooks.map(book=><div key={book.id}><span>{book.title} · {book.author}</span><button className="text-button" onClick={()=>restoreBook(book)}>Restaurer</button></div>)}</div>}</section>}
        {selectedBooks.length>0&&section==='Chapitres'&&<section className="reading-page"><PageTitle title="Lire par chapitre" subtitle="Retrouve les passages indexés, en respectant ta limite anti-spoiler."/><div className="chapter-layout"><aside className="chapter-list"><label className="field-label" htmlFor="chapter-book">Livre</label><select id="chapter-book" value={activeBook?.id||''} onChange={e=>setSelected([e.target.value])}>{selectedBooks.map(book=><option key={book.id} value={book.id}>{book.title}</option>)}</select>{activeBook?.chapters.filter(n=>maxChapter===null||n<=maxChapter).map(n=><button key={n} className={bookChapter===n?'chapter-link chapter-link--active':'chapter-link'} onClick={()=>setBookChapter(n)}>Chapitre {n}<span>›</span></button>)}</aside><div className="chapter-content">{bookChapter?<><div className="page-kicker">{activeBook?.title}</div><h2>Chapitre {bookChapter}</h2>{chapterPassages.map(p=><PassageCard key={p.chunk_id} passage={p}/>)}</>:<div className="empty-reading"><span>☷</span><h2>Choisis un chapitre</h2><p>Les passages disponibles apparaîtront ici dans l’ordre du livre.</p></div>}</div></div></section>}
        {selectedBooks.length>0&&section==='Recherche'&&<section className="search-page"><PageTitle title="Chercher dans les livres" subtitle="Un mot, un thème, une citation : retrouve le passage qui te revient."/><div className="search-panel"><div className="search-input-wrap"><span>⌕</span><input value={searchQuery} onChange={e=>setSearchQuery(e.target.value)} onKeyDown={e=>e.key==='Enter'&&doSearch()} placeholder="Que souhaites-tu retrouver ?"/></div><select value={searchMethod} onChange={e=>setSearchMethod(e.target.value)}><option value="Hybrid">Combinée</option><option value="BM25">Mots-clés (BM25)</option><option value="Sémantique">Sémantique</option></select><button className="primary-button" onClick={doSearch} disabled={loading||searchQuery.trim().length<2}>Rechercher</button></div>{searchResults.length>0?<div className="results-list"><p className="result-count">{searchResults.length} passages pertinents</p>{searchResults.map(p=><PassageCard key={p.chunk_id} passage={p}/>)}</div>:<div className="quiet-empty"><span>⌕</span><p>La recherche explore seulement les livres sélectionnés et les chapitres accessibles.</p></div>}</section>}
        {selectedBooks.length>0&&(section==='Graphe'||section==='Personnages')&&<section className="graph-page"><PageTitle title={section==='Graphe'?'Les liens dans le récit':'Personnages'} subtitle="Chaque relation est appuyée par une citation du texte."/><div className="graph-toolbar"><div className="search-input-wrap compact"><span>⌕</span><input value={graphSearch} onChange={e=>setGraphSearch(e.target.value)} placeholder="Chercher un personnage…"/></div><label>Voisinage <select value={graphDepth} onChange={e=>setGraphDepth(Number(e.target.value))}><option value={1}>1 relation</option><option value={2}>2 relations</option></select></label><button className="secondary-button" onClick={buildGraph} disabled={loading}>Construire le graphe</button><button className="icon-button bordered" title="Synchroniser Neo4j" onClick={()=>report(()=>api.syncGraph({work_ids:selected}),result=>setNotice(`${result.synced} relations synchronisées dans Neo4j.`))}>⟳</button></div>{section==='Graphe'?<GraphCanvas graph={graph} query={graphSearch} depth={graphDepth}/>:<CharacterList graph={graph} query={graphSearch}/>}</section>}
        {selectedBooks.length>0&&section==='Aperçu'&&<section className="insights-page"><PageTitle title="Vue d’ensemble" subtitle="L’état de l’index et des services locaux."/><div className="insight-stats">{selectedBooks.map(book=><div className="stat-paper" key={book.id}><span>{book.title}</span><strong>{book.passage_count.toLocaleString('fr-FR')}</strong><small>passages indexés</small><div className="stat-rule"/><strong>{book.chapters.length}</strong><small>chapitres détectés</small></div>)}</div><div className="service-panel"><div><h2>Services de lecture</h2><p>La recherche sémantique utilise {embedding||'les réglages locaux'}. Les textes restent sur cet appareil.</p></div><button className="secondary-button" onClick={checkServices} disabled={loading}>Vérifier les services</button>{health&&<div className="service-badges"><span className={health.ollama?'service-ok':'service-off'}><i/>Ollama {health.ollama?'disponible':'indisponible'}</span><span className={health.neo4j?'service-ok':'service-off'}><i/>Neo4j {health.neo4j?'disponible':'indisponible'}</span></div>}<button className="text-button" onClick={()=>report(()=>api.embeddings(),result=>setNotice(`${result.count} passages vectorisés avec ${result.embedding_model}.`),'Index sémantique actualisé.')}>Préparer ou réparer les embeddings</button></div><div className="about-local"><span>◈</span><div><strong>Un compagnon privé</strong><p>Les livres, les discussions et les citations restent stockés localement. La recherche de catalogue utilise Gutendex, uniquement pour les œuvres du domaine public.</p></div></div></section>}
      </div>
    </main>
    {mobileNav&&<button className="scrim" aria-label="Fermer la navigation" onClick={()=>setMobileNav(false)}/>}
    {catalogOpen&&<div className="modal-backdrop" role="presentation" onClick={e=>e.target===e.currentTarget&&setCatalogOpen(false)}><section className="catalog-modal" role="dialog" aria-modal="true" aria-labelledby="catalog-title"><div className="modal-head"><div><div className="page-kicker">ÉLARGIR LA BIBLIOTHÈQUE</div><h2 id="catalog-title">Ajouter un livre</h2></div><button className="icon-button" onClick={()=>setCatalogOpen(false)} aria-label="Fermer">×</button></div><div className="catalog-tabs"><span className="catalog-tab catalog-tab--active">Catalogue public</span><span className="catalog-tab-rule"/><span className="catalog-tab-note">ou importe ton fichier</span></div><form className="catalog-search" onSubmit={e=>{e.preventDefault();searchCatalog(1)}}><input value={catalogQuery} onChange={e=>setCatalogQuery(e.target.value)} placeholder="Titre ou auteur, par ex. Jane Austen" aria-label="Chercher par titre ou auteur"/><select value={catalogLanguage} onChange={e=>setCatalogLanguage(e.target.value)} aria-label="Langue">{languages.map(([label,value])=><option key={value} value={value}>{label}</option>)}</select><button className="primary-button" disabled={loading}>Chercher</button></form>{catalogBooks.length>0&&<><div className="catalog-results-head">{catalogCount.toLocaleString('fr-FR')} résultats dans Project Gutenberg</div><div className="catalog-results">{catalogBooks.map(book=><article className="catalog-book" key={book.provider_id}>{book.cover_url?<img src={book.cover_url} alt=""/>:<div className="cover-placeholder"><span>✧</span></div>}<div><h3>{book.title}</h3><p>{book.author} · {book.language_display}</p>{book.subjects.length>0&&<small>{book.subjects.slice(0,2).join(' · ')}</small>}</div><button className="text-button" onClick={()=>installCatalog(book)} disabled={loading}>{loading?'Indexation…':'Ajouter'}</button></article>)}</div><div className="catalog-pager"><button className="text-button" disabled={catalogPage<=1||loading} onClick={()=>searchCatalog(catalogPage-1)}>Précédent</button><span>Page {catalogPage}</span><button className="text-button" disabled={loading||!catalogHasNext} onClick={()=>searchCatalog(catalogPage+1)}>Suivant</button></div></>}{catalogBooks.length===0&&<div className="catalog-empty"><span>⌕</span><p>Explore le catalogue public des œuvres du domaine public.</p></div>}<div className="upload-divider"><span>FICHIER PERSONNEL</span></div><div className="upload-zone"><label className="file-pick">{upload?'Changer le fichier':'Choisir un EPUB, TXT, MD, PDF ou DOCX'}<input type="file" accept=".epub,.txt,.md,.pdf,.docx" onChange={e=>{const file=e.target.files?.[0]||null;setUpload(file);if(file&&!uploadTitle)setUploadTitle(file.name.replace(/\.[^.]+$/,''))}}/></label>{upload&&<><p className="file-selected">{upload.name} · {(upload.size/1024/1024).toFixed(1)} Mo</p><div className="upload-fields"><input value={uploadTitle} onChange={e=>setUploadTitle(e.target.value)} placeholder="Titre du livre"/><input value={uploadAuthor} onChange={e=>setUploadAuthor(e.target.value)} placeholder="Auteur"/><select value={uploadLanguage} onChange={e=>setUploadLanguage(e.target.value)}><option value="">Langue à détecter</option>{languages.slice(1).map(([l,v])=><option key={v} value={v}>{l}</option>)}</select><button className="primary-button" onClick={installUpload} disabled={loading}>Importer et indexer</button></div></>}</div><p className="modal-footnote">Les livres locaux sont analysés sur cet appareil. N’importe que des fichiers que tu possèdes ou peux utiliser.</p></section></div>}
  </div>
}

function PageTitle({title,subtitle,action}:{title:string;subtitle:string;action?:React.ReactNode}) { return <div className="page-title"><div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div> }

function MessageView({message,goGraph}:{message:Message;goGraph:()=>void}) {
  const [open,setOpen]=useState<number|null>(null)
  const visual=message.visualization
  const messageGraph:Graph={nodes:visual?.nodes||[],edges:(visual?.edges||[]).map(edge=>({source:edge.source,target:edge.target,label:edge.label,evidence_chunk_id:edge.evidence_chunk_id,evidence:message.citations?.find(c=>c.chunk_id===edge.evidence_chunk_id)?.text||'',work_id:'',source_kind:'character',target_kind:'character'}))}
  return <article className={`message message--${message.role}`}><div className="message-avatar">{message.role==='assistant'?<span className="mini-book">✧</span>:<span>◉</span>}</div><div className="message-body"><div className="message-author">{message.role==='assistant'?'Literary Chat':'Toi'}{message.elapsed_ms&&<span>{(message.elapsed_ms/1000).toFixed(1)} s</span>}</div><div className="message-text">{message.content}</div>{message.citations&&message.citations.length>0&&<div className="citation-list"><div className="citation-heading">PASSAGES À L’APPUI</div>{message.citations.map((citation,index)=><div className="citation" key={citation.chunk_id}><button onClick={()=>setOpen(open===index?null:index)}><span className="citation-glyph">⌞</span><span><strong>{citation.citation_label}</strong><small>{citation.text.slice(0,130).replace(/\s+/g,' ')}{citation.text.length>130?'…':''}</small></span><span className="citation-chevron">{open===index?'−':'+'}</span></button>{open===index&&<blockquote>{citation.text}</blockquote>}</div>)}</div>}{messageGraph.edges.length>0&&<div className="answer-graph"><div className="answer-graph-head"><div><span className="page-kicker">RELATIONS DANS LE TEXTE</span><h3>{visual?.title||'Les liens entre personnages'}</h3></div><button className="text-button" onClick={goGraph}>Explorer le graphe</button></div><div className="answer-edges">{messageGraph.edges.slice(0,6).map((edge,index)=><div key={`${edge.source}-${edge.target}-${index}`}><span>{edge.source}</span><small>{edge.label}</small><span>{edge.target}</span></div>)}</div></div>}</div></article>
}

function PassageCard({passage}:{passage:Passage}) { return <article className="passage-card"><div className="passage-cite"><span>⌞</span>{passage.citation_label}</div><blockquote>{passage.text}</blockquote></article> }

function CharacterList({graph,query}:{graph:Graph;query:string}) {
  const [selectedEdge,setSelectedEdge]=useState<Edge|null>(null)
  const characters=graph.nodes.filter(n=>n.kind==='character'&&n.label.toLocaleLowerCase().includes(query.toLocaleLowerCase()))
  return <div className="character-layout"><div className="character-list">{characters.map(person=>{const rel=graph.edges.filter(e=>e.source===person.label||e.target===person.label);return <button key={person.id} className="character-row" onClick={()=>setSelectedEdge(rel[0]||null)}><span className="character-seal">{person.label.slice(0,1)}</span><span><strong>{person.label}</strong><small>{rel.length} relation{rel.length>1?'s':''} attestée{rel.length>1?'s':''}</small></span><i>›</i></button>})}{characters.length===0&&<div className="quiet-empty"><p>Aucun personnage trouvé pour l’instant. Construis quelques relations pour commencer.</p></div>}</div><div className="evidence-panel"><div className="page-kicker">PREUVE DANS LE TEXTE</div>{selectedEdge?<><h3>{selectedEdge.source} <span>{selectedEdge.label.toLocaleLowerCase()}</span> {selectedEdge.target}</h3><blockquote>{selectedEdge.evidence||'Le passage source est consultable dans la vue graphe.'}</blockquote><small>Passage : {selectedEdge.evidence_chunk_id}</small></>:<><span className="evidence-mark">⌞</span><p>Choisis un personnage pour consulter ses relations et le passage qui les appuie.</p></>}</div></div>
}

export { App }
