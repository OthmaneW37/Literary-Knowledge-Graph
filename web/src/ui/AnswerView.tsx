import { useState, type ReactNode } from 'react'
import { api, type Passage } from '../api'
import { formatAnswer, highlightParts } from './answerFormat'

export function AnswerText({content,citations,onCite}:{content:string;citations:Passage[];onCite:(index:number)=>void}) {
  const blocks=formatAnswer(content,citations.map(c=>c.chunk_id))
  const items=blocks.map((block,index)=>{
    const inner=block.segments.map((segment,key)=>segment.kind==='bold'?<strong key={key}>{segment.text}</strong>
      :segment.kind==='cite'?<button key={key} className="cite-ref" onClick={()=>onCite(segment.number-1)} aria-label={`Voir la source ${segment.number}`}>{segment.number}</button>
      :<span key={key}>{segment.text}</span>)
    return {kind:block.kind,node:block.kind==='quote'?<blockquote key={index} className="answer-quote">{inner}</blockquote>:block.kind==='bullet'?<li key={index}>{inner}</li>:<p key={index}>{inner}</p>}
  })
  // Consecutive bullets form one list.
  const out:ReactNode[]=[]
  for (let i=0;i<items.length;i++) {
    if (items[i].kind!=='bullet') { out.push(items[i].node); continue }
    const group=[]
    while (i<items.length&&items[i].kind==='bullet') group.push(items[i++].node)
    i--
    out.push(<ul key={`list-${i}`} className="answer-list">{group}</ul>)
  }
  return <div className="message-text answer-text">{out}</div>
}

export function SourceList({citations,open,setOpen,consulted=false}:{citations:Passage[];open:number|null;setOpen:(index:number|null)=>void;consulted?:boolean}) {
  return <div className={`source-list ${consulted?'source-list--consulted':''}`}>
    <div className="citation-heading">{consulted?'PASSAGES CONSULTÉS · AUCUN NE RÉPOND AVEC CERTITUDE':'SOURCES DANS LE LIVRE'}</div>
    {citations.map((citation,index)=>{
      const [before,quote,after]=highlightParts(citation.text,citation.highlight)
      const preview=quote||citation.text.replace(/\s+/g,' ').slice(0,160)+(citation.text.length>160?'…':'')
      return <div id={`source-${citation.chunk_id}`} className={`source-card ${open===index?'is-open':''}`} key={citation.chunk_id}>
        <button onClick={()=>setOpen(open===index?null:index)} aria-expanded={open===index}>
          <span className="source-number">{index+1}</span>
          <span className="source-copy"><strong>{citation.work_title}</strong><small>Chapitre {citation.chapter}{citation.page?` · page ${citation.page}`:''}</small>
            {open!==index&&<em>{quote?`« ${preview.trim()} »`:preview}</em>}</span>
          <span className="citation-chevron">{open===index?'−':'+'}</span>
        </button>
        {open===index&&<><blockquote>{before}{quote&&<mark>{quote}</mark>}{after}</blockquote><TranslateButton text={citation.text}/></>}
      </div>
    })}
  </div>
}

/** Machine translation of a passage for reading; the original stays the evidence. */
export function TranslateButton({text}:{text:string}) {
  const [state,setState]=useState<{loading:boolean;translation:string;error:string}>({loading:false,translation:'',error:''})
  // Passages already in French need no translation.
  if (/\b(le|la|les|des|une|est|dans|qui|pour)\b/i.test(text.slice(0,400))&&/[éèàùç]/.test(text.slice(0,400))) return null
  const run=()=>{setState({loading:true,translation:'',error:''});api.translate(text).then(r=>setState({loading:false,translation:r.translation,error:''})).catch(e=>setState({loading:false,translation:'',error:(e as Error).message}))}
  return <div className="translate-box">
    {!state.translation&&<button className="text-button" onClick={run} disabled={state.loading}>{state.loading?'Traduction en cours…':'Traduire en français'}</button>}
    {state.translation&&<><div className="translate-label">Traduction automatique (le texte original reste la preuve)</div><p>{state.translation}</p></>}
    {state.error&&<small className="eval-warning">{state.error}</small>}
  </div>
}
