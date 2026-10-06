import {useEffect,useRef,useState} from 'react'
import {api,type Book,type DiscoveredBook,type DiscoveryResult} from '../api'

const languages=[['Toutes les langues',''],['Français','fr'],['Anglais','en'],['Espagnol','es'],['Allemand','de'],['Italien','it'],['Portugais','pt']]
const languageCodes:Record<string,string>={fre:'fr',eng:'en',spa:'es',ger:'de',ita:'it',por:'pt'}

export function BookImport({target,onClose,onDone}:{target:Book|null;onClose:()=>void;onDone:(id:string,notice:string)=>Promise<void>}){
  const [query,setQuery]=useState(target?`${target.title} ${target.author}`:'')
  const [language,setLanguage]=useState('')
  const [source,setSource]=useState('all')
  const [assisted,setAssisted]=useState(true)
  const [result,setResult]=useState<DiscoveryResult|null>(null)
  const [selection,setSelection]=useState<DiscoveredBook|null>(null)
  const [stage,setStage]=useState('')
  const [error,setError]=useState('')
  const [file,setFile]=useState<File|null>(null)
  const [title,setTitle]=useState('')
  const [author,setAuthor]=useState('')
  const [fileLanguage,setFileLanguage]=useState('')
  const epoch=useRef(0)
  const lastSearch=useRef({query:'',language:'',source:'all'})
  const dialog=useRef<HTMLElement>(null)
  useEffect(()=>{const previous=document.activeElement as HTMLElement|null;dialog.current?.focus();return()=>{epoch.current++;previous?.focus()}},[])
  const close=()=>{if(!stage)onClose()}
  const search=async(page=1)=>{
    const request=page===1?{query:query.trim(),language,source}:lastSearch.current
    if(request.query.length<2||stage)return
    const run=++epoch.current;setStage('Recherche dans les catalogues en ligne…');setError('');setSelection(null);setFile(null);setTitle('');setAuthor('');setFileLanguage('')
    try{
      const data=await api.discoverBooks({...request,page,use_model:assisted&&page===1})
      if(run!==epoch.current)return
      setResult(data);lastSearch.current={...request,query:data.query}
    }catch(e){if(run===epoch.current)setError((e as Error).message)}finally{if(run===epoch.current)setStage('')}
  }
  const choose=async(book:DiscoveredBook)=>{
    const run=++epoch.current;setStage('Récupération de la fiche du livre…');setError('');setFile(null)
    try{
      const detail=await api.bookDetails(book)
      if(run!==epoch.current)return
      setSelection(detail);setTitle(detail.title);setAuthor(detail.author)
      setFileLanguage(detail.languages.length===1?(languageCodes[detail.languages[0]]||detail.languages[0]):'')
    }catch(e){if(run===epoch.current)setError((e as Error).message)}finally{if(run===epoch.current)setStage('')}
  }
  const acquire=async()=>{
    if(stage||!selection)return
    setError('');setStage(target?'Enregistrement de la fiche…':'Téléchargement, lecture du livre et indexation…')
    try{
      if(target){await api.attachBookMetadata(target.id,selection);await onDone(target.id,'Fiche, couverture et contexte bibliographique enregistrés.');return}
      const installed=await api.installCatalog(Number(selection.provider_id))
      await onDone(installed.work_id,installed.warning||'Le livre, sa couverture et sa fiche sont prêts.')
    }catch(e){setError((e as Error).message)}finally{setStage('')}
  }
  const upload=async()=>{
    if(!file||stage)return
    if(file.size>50*1024*1024){setError('Le fichier dépasse la limite de 50 Mo.');return}
    setError('');setStage('Lecture du fichier, récupération de la couverture et indexation…')
    try{
      const installed=await api.upload(file,title,author,fileLanguage,selection||undefined)
      await onDone(installed.work_id,installed.warning||'Le livre et ses informations sont prêts.')
    }catch(e){setError((e as Error).message)}finally{setStage('')}
  }
  const annaUrl=`https://annas-archive.gd/search?${new URLSearchParams({q:query,ext:'epub'})}`
  return <div className="modal-backdrop" onClick={e=>e.target===e.currentTarget&&close()}>
    <section className="book-import" role="dialog" aria-modal="true" aria-labelledby="book-import-title" tabIndex={-1} ref={dialog} onKeyDown={e=>{
      if(e.key==='Escape')close()
      if(e.key==='Tab'){
        const items=Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),summary')||[]).filter(el=>el.getClientRects().length)
        const first=items[0],last=items.at(-1)
        if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus()}
      }
    }}>
      <header className="modal-head"><div><div className="page-kicker">TA PROCHAINE LECTURE</div><h2 id="book-import-title">{target?'Compléter la fiche du livre':'Trouver et ajouter un livre'}</h2><p>{target?`${target.title} · ${target.author}`:'Cherche un titre, un auteur, un ISBN ou décris le livre.'}</p></div><button className="icon-button" disabled={!!stage} onClick={close} aria-label="Fermer l’ajout de livre">×</button></header>
      <form className="book-web-search" onSubmit={e=>{e.preventDefault();search()}}><label className="field-label" htmlFor="book-query">Quel livre cherches-tu ?</label><div><input id="book-query" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Ex. le roman de Kafka où un homme devient un insecte" maxLength={200}/><button className="primary-button" disabled={!!stage||query.trim().length<2}>Chercher sur Internet</button></div><div className="book-search-options"><select value={source} onChange={e=>setSource(e.target.value)} aria-label="Catalogue de recherche" disabled={!!stage}><option value="all">Tous les catalogues</option><option value="gutendex">EPUB disponibles · Gutenberg</option><option value="openlibrary">Livres et éditions · Open Library</option></select><select aria-label="Langue recherchée" value={language} onChange={e=>setLanguage(e.target.value)} disabled={!!stage}>{languages.map(([name,value])=><option key={value} value={value}>{name}</option>)}</select><label><input type="checkbox" checked={assisted} onChange={e=>setAssisted(e.target.checked)} disabled={!!stage}/>Aider la recherche avec le modèle</label></div></form>
      <div className="book-import-scroll">
        {!!stage&&<p className="book-import-status" role="status">{stage}</p>}{!!error&&<p className="book-import-error" role="alert">{error}</p>}
        {result?.warnings.map(w=><p className="book-import-warning" key={w}>{w}</p>)}
        {result&&<div className="book-results-heading"><span>{result.count.toLocaleString('fr-FR')} résultats des catalogues · recherche « {result.query} »</span><span>Page {result.page}</span></div>}
        <div className="book-discovery-layout">
          <div className="book-web-results">{result?.books.map(book=><button key={`${book.provider}:${book.provider_id}`} disabled={!!stage} className={`book-web-result${selection?.provider===book.provider&&selection.provider_id===book.provider_id?' book-web-result--selected':''}`} onClick={()=>choose(book)}>
            {book.cover_url?<img src={book.cover_url} alt="" loading="lazy" onError={e=>{e.currentTarget.style.visibility='hidden'}}/>:<span className="book-result-cover">✧</span>}
            <span><strong>{book.title}</strong><small>{book.author}{book.first_publish_year?` · ${book.first_publish_year}`:''}</small><span className="book-result-source">{book.source_name} · {book.can_download?`${book.format} disponible`:'Fiche · fichier à importer'}</span></span><span aria-hidden="true">›</span>
          </button>)}{result&&!result.books.length&&<p>Aucun résultat accessible. Essaie le titre original, l’auteur ou la recherche Anna’s Archive ci-dessous.</p>}</div>
          {selection&&<aside className="book-preview"><div className="page-kicker">FICHE DU LIVRE</div>{selection.cover_url&&<img className="book-preview-cover" src={selection.cover_url} alt={`Couverture de ${selection.title}`} onError={e=>{e.currentTarget.style.display='none'}}/>}<h3>{selection.title}</h3><p>{selection.author}</p><a href={selection.source_url} target="_blank" rel="noreferrer">Source : {selection.source_name} ↗</a>{!!selection.summary&&<details className="book-summary"><summary>Résumé externe · peut révéler l’intrigue</summary><p>{selection.summary}</p></details>}{!!selection.subjects.length&&<p className="book-subjects">{selection.subjects.slice(0,6).join(' · ')}</p>}{target||selection.can_download?<button className="primary-button" disabled={!!stage} onClick={acquire}>{target?'Associer cette fiche':`Télécharger le ${selection.format} et ajouter`}</button>:<p className="book-import-warning">Cette fiche décrit l’œuvre. Ajoute ton EPUB ci-dessous pour lire et analyser son texte.</p>}<a className="text-button" href={selection.anna_url} target="_blank" rel="noreferrer">Rechercher ce livre sur Anna’s Archive ↗</a></aside>}
        </div>
        {result&&<div className="catalog-pager"><button className="text-button" disabled={!!stage||result.page<=1} onClick={()=>search(result.page-1)}>Précédent</button><button className="text-button" disabled={!!stage||!result.has_next} onClick={()=>search(result.page+1)}>Suivant</button></div>}
        <div className="book-anna"><div><strong>Anna’s Archive</strong><p>Ouvre la recherche dans ton navigateur, puis importe le fichier EPUB obtenu. Les vérifications du site s’effectuent dans le navigateur.</p></div><a className="secondary-button" href={annaUrl} target="_blank" rel="noreferrer">Ouvrir la recherche ↗</a></div>
        {!target&&<section className="book-local-upload"><h3>{selection?'Importer le fichier de ce livre':'Ou ajouter un fichier personnel'}</h3><label className="file-pick">{file?'Changer le fichier':'Choisir un EPUB, PDF, DOCX ou texte'}<input type="file" accept=".epub,.pdf,.docx,.txt,.md" disabled={!!stage} onChange={e=>{setFile(e.target.files?.[0]||null);setError('')}}/></label><p>La couverture, le titre et l’auteur sont récupérés dans l’EPUB lorsqu’ils y figurent.</p>{file&&<><p>{file.name} · {(file.size/1024/1024).toFixed(1)} Mo{selection?` · Fiche associée : ${selection.source_name}`:''}</p><div className="upload-fields"><input aria-label="Titre du fichier" placeholder="Titre · détecté dans l’EPUB" value={title} onChange={e=>setTitle(e.target.value)}/><input aria-label="Auteur du fichier" placeholder="Auteur · détecté dans l’EPUB" value={author} onChange={e=>setAuthor(e.target.value)}/><select aria-label="Langue du fichier" value={fileLanguage} onChange={e=>setFileLanguage(e.target.value)}><option value="">Détection automatique</option>{languages.slice(1).map(([name,value])=><option key={value} value={value}>{name}</option>)}</select></div><button className="primary-button" disabled={!!stage} onClick={upload}>Importer le livre et sa fiche</button></>}</section>}
      </div>
      <footer className="book-import-footer">Le texte du livre fournit les citations. Les résumés de catalogue apportent un contexte identifié séparément.</footer>
    </section>
  </div>
}
