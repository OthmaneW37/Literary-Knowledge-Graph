import { useEffect, useRef, useState } from 'react'
import { api, type Book, type DiscoveredBook, type DiscoveryResult, type OpenJob } from '../api'
import { jobKey, localMatches, openProgress } from './exploreSearch'

const suggestions=['Les Frères Karamazov','Les Misérables','Orgueil et préjugés','Le Comte de Monte-Cristo','Frankenstein','Alice au pays des merveilles']
const languages=[['Toutes les langues',''],['Français','fr'],['Anglais','en'],['Espagnol','es'],['Allemand','de'],['Italien','it']]

export function ExplorePage({library,jobs,onOpen,onRead,onImportFile}:{library:Book[];jobs:Record<string,OpenJob>;onOpen:(book:DiscoveredBook)=>void;onRead:(workId:string)=>void;onImportFile:()=>void}) {
  const [query,setQuery]=useState('')
  const [language,setLanguage]=useState('')
  const [result,setResult]=useState<DiscoveryResult|null>(null)
  const [searching,setSearching]=useState(false)
  const [error,setError]=useState('')
  const run=useRef(0)
  const input=useRef<HTMLInputElement>(null)
  useEffect(()=>{input.current?.focus()},[])

  const search=async(text=query,page=1)=>{
    const value=text.trim()
    if(value.length<2)return
    const current=++run.current
    setQuery(value);setSearching(true);setError('')
    try {
      const data=await api.discoverBooks({query:value,language,page,source:'all',use_model:true})
      if(current===run.current)setResult(data)
    } catch(e){if(current===run.current)setError((e as Error).message)}
    finally{if(current===run.current)setSearching(false)}
  }
  const owned=localMatches(library,query)
  const readable=(result?.books||[]).filter(book=>book.can_download)
  const recordsOnly=(result?.books||[]).filter(book=>!book.can_download)

  return <section className="explore-page">
    <div className="page-title"><div><div className="page-kicker">EXPLORER</div><h1>Tous les livres du domaine public</h1><p>Plus de 75 000 œuvres de Project Gutenberg, en français, en anglais et dans d’autres langues. Choisis un livre : il s’ouvre directement, prêt pour tes questions.</p></div><button className="secondary-button" onClick={onImportFile}>Importer un fichier personnel</button></div>
    <form className="explore-search" onSubmit={e=>{e.preventDefault();search()}}>
      <div className="search-input-wrap"><span>⌕</span><input ref={input} value={query} onChange={e=>setQuery(e.target.value)} placeholder="Titre, auteur… ex. Les Frères Karamazov" aria-label="Chercher un livre" maxLength={200}/></div>
      <select value={language} onChange={e=>setLanguage(e.target.value)} aria-label="Langue">{languages.map(([label,value])=><option key={value} value={value}>{label}</option>)}</select>
      <button className="primary-button" disabled={searching||query.trim().length<2}>{searching?'Recherche…':'Chercher'}</button>
    </form>
    {!result&&!searching&&<div className="explore-suggestions"><span>Idées :</span>{suggestions.map(item=><button key={item} className="explore-chip" onClick={()=>search(item)}>{item}</button>)}</div>}
    {error&&<div className="notice notice--error" role="alert"><span>!</span><p>{error}</p><button onClick={()=>setError('')} aria-label="Fermer">×</button></div>}
    {searching&&<div className="thinking"><span className="loader"/><div><strong>Recherche dans les catalogues…</strong><small>Project Gutenberg et Open Library</small></div></div>}

    {owned.length>0&&<><h2 className="explore-heading">Déjà dans ta bibliothèque</h2><div className="explore-grid">{owned.map(book=><article key={book.id} className="explore-card"><Cover url={book.cover_url} title={book.title}/><div><strong>{book.title}</strong><small>{book.author}</small><button className="primary-button" onClick={()=>onRead(book.id)}>Discuter</button></div></article>)}</div></>}

    {result&&!searching&&<>
      <h2 className="explore-heading">Lire maintenant <small>{readable.length?`${result.count.toLocaleString('fr-FR')} résultats · « ${result.query} »`:''}</small></h2>
      {result.warnings.map(w=><p className="book-import-warning" key={w}>{w}</p>)}
      {!readable.length&&<p className="explore-empty">Aucun texte libre de droits trouvé. Essaie le titre original ou le nom de l’auteur (ex. « Dostoyevsky », « Karamazov »).</p>}
      {readable.some(book=>book.provider==='wikisource')&&<p className="explore-hint">Les textes Wikisource sont en français ; ceux de Project Gutenberg sont souvent en anglais (les réponses restent en français et chaque passage peut être traduit).</p>}
      <div className="explore-grid">{readable.map(book=>{
        const installed=library.find(item=>item.id===(book.work_id||`gutenberg_${book.provider_id}`)&&!item.archived)
        const job=jobs[jobKey(book)]
        const progress=openProgress(job)
        return <article key={`${book.provider}:${book.provider_id}`} className="explore-card">
          <Cover url={book.cover_url} title={book.title}/>
          <div>
            <strong>{book.title}</strong><small>{book.author}</small>
            <span className="explore-meta"><b className={`explore-lang explore-lang--${book.languages[0]||'x'}`}>{book.languages.map(l=>l.toUpperCase()).join(' · ')||'—'}</b> {book.source_name} · {book.format}</span>
            {installed?<button className="primary-button" onClick={()=>onRead(installed.id)}>Discuter</button>
              :job&&job.state!=='failed'?<div className="explore-progress" role="status"><div className="eval-bar"><span style={{width:`${progress}%`}}/></div><small>{job.stage}</small></div>
              :<button className="primary-button" onClick={()=>onOpen(book)}>Ouvrir le livre</button>}
            {job?.state==='failed'&&<small className="eval-warning">{job.error}</small>}
          </div>
        </article>})}</div>
      {recordsOnly.length>0&&<details className="explore-records"><summary>{recordsOnly.length} fiches bibliographiques sans texte libre (Open Library)</summary><p>Ces éditions sont encore protégées par le droit d’auteur : leur texte n’est pas disponible librement. Si tu possèdes le fichier, importe-le.</p><ul>{recordsOnly.map(book=><li key={book.provider_id}><a href={book.source_url} target="_blank" rel="noreferrer">{book.title}</a> · {book.author}{book.first_publish_year?` · ${book.first_publish_year}`:''}</li>)}</ul><button className="secondary-button" onClick={onImportFile}>Importer un fichier</button></details>}
      <div className="catalog-pager"><button className="text-button" disabled={searching||result.page<=1} onClick={()=>search(result.query,result.page-1)}>Précédent</button><button className="text-button" disabled={searching||!result.has_next} onClick={()=>search(result.query,result.page+1)}>Suivant</button></div>
    </>}
  </section>
}

function Cover({url,title}:{url:string;title:string}) {
  const [failed,setFailed]=useState(!url)
  return failed?<div className="explore-cover cover-placeholder"><span>{title.slice(0,1)}</span></div>
    :<img className="explore-cover" src={url} alt="" loading="lazy" onError={()=>setFailed(true)}/>
}
