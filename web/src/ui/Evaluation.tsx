import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type Book, type EvalCategory, type EvalProgress, type EvalResult, type EvalRun, type EvalScenario } from '../api'

import { ProtectedImage } from './ProtectedMedia'
import { ago, runDelta, verdictMark } from './evaluationStats'

const expectations: Record<EvalScenario['expect'],string> = {
  answer:'Réponse prouvée attendue', abstain:'Refus ou abstention attendu', reject_premise:'Correction de la prémisse attendue', safe:'Aucune fuite ni obéissance attendue',
}
const outcomeTone: Record<EvalResult['outcome'],string> = {answered:'answer',quoted:'answer',fallback:'warn',premise_rejected:'guard',spoiler_guarded:'guard',abstained:'guard'}

export function EvaluationPage({books}:{books:Book[]}) {
  const [categories,setCategories]=useState<Record<string,EvalCategory>>({})
  const [scenarios,setScenarios]=useState<EvalScenario[]>([])
  const [runs,setRuns]=useState<EvalRun[]>([])
  const [progress,setProgress]=useState<EvalProgress>({state:'idle'})
  const [category,setCategory]=useState('all')
  const [showcase,setShowcase]=useState(true)
  const [selectedId,setSelectedId]=useState('')
  const [running,setRunning]=useState('')
  const [error,setError]=useState('')
  const [ready,setReady]=useState(false)

  const load = useCallback(async()=>{
    const data=await api.evaluation()
    setCategories(data.categories);setScenarios(data.scenarios);setRuns(data.runs);setProgress(data.progress);setReady(true)
  },[])
  useEffect(()=>{load().catch(e=>{setError((e as Error).message);setReady(true)})},[load])
  useEffect(()=>{
    if(progress.state!=='running')return
    const timer=window.setInterval(()=>{
      api.evaluationProgress().then(next=>{setProgress(next);if(next.state!=='running')load().catch(e=>setError(e.message))}).catch(()=>{})
    },2500)
    return()=>window.clearInterval(timer)
  },[progress.state,load])

  const visible=useMemo(()=>scenarios.filter(s=>(category==='all'||s.category===category)&&(!showcase||s.showcase)),[scenarios,category,showcase])
  const selected=visible.find(s=>s.id===selectedId)||visible[0]
  const titles=useMemo(()=>Object.fromEntries(books.map(book=>[book.id,book.title])),[books])
  // Only runs of the whole set count as complete evaluations (category runs are partial).
  const fullRuns=runs.filter(run=>run.scenario_count===scenarios.length)
  const latestRun=fullRuns.find(run=>run.state==='done')
  const delta=runDelta(fullRuns)

  const play=async(scenario:EvalScenario)=>{
    if(running||progress.state==='running')return
    setRunning(scenario.id);setError('')
    try {
      const result=await api.runScenario(scenario.id)
      setScenarios(current=>current.map(s=>s.id===scenario.id?{...s,last_result:result}:s))
    } catch(e){setError((e as Error).message)}
    finally{setRunning('')}
  }
  const next=()=>{ if(!selected)return; const index=visible.findIndex(s=>s.id===selected.id); setSelectedId(visible[(index+1)%visible.length].id) }
  const startAll=()=>api.startEvaluation(null).then(setProgress).catch(e=>setError(e.message))
  const cancelAll=()=>api.cancelEvaluation().then(setProgress).catch(e=>setError(e.message))

  if(!ready)return <section className="eval-page"><div className="thinking"><span className="loader"/><div><strong>Chargement du banc d’essai…</strong></div></div></section>
  return <section className="eval-page">
    <div className="page-title"><div><div className="page-kicker">BANC D’ESSAI</div><h1>Le moteur à l’épreuve</h1><p>{scenarios.length} scénarios annotés, chacun vérifié mot pour mot dans le texte. Lance-les en direct : la réponse, ses preuves et chaque contrôle s’affichent.</p></div></div>
    {error&&<div className="notice notice--error" role="alert"><span>!</span><p>{error}</p><button onClick={()=>setError('')} aria-label="Fermer">×</button></div>}

    <div className="eval-summary">
      <div className="eval-score">
        {latestRun?<><strong>{latestRun.summary.passed}<small> / {latestRun.summary.total}</small></strong><span>réussis lors de la dernière évaluation complète · {ago(latestRun.finished_at)}{delta!==null&&delta!==0&&<em className={delta>0?'eval-up':'eval-down'}> {delta>0?'▲ +':'▼ '}{delta} depuis la précédente</em>}</span></>
          :<><strong>—</strong><span>Aucune évaluation complète pour l’instant.</span></>}
      </div>
      {progress.state==='running'
        ?<div className="eval-progress"><div className="eval-bar"><span style={{width:`${100*(progress.done||0)/Math.max(1,progress.total||1)}%`}}/></div><small>{progress.done}/{progress.total} · {progress.passed} réussis · en cours : {progress.current}</small><button className="text-button" onClick={cancelAll}>Arrêter</button></div>
        :<button className="secondary-button" onClick={startAll} disabled={!!running}>▶ Tout jouer</button>}
      {latestRun&&<div className="eval-categories">{Object.entries(latestRun.summary.by_category).map(([key,value])=><div key={key} title={categories[key]?.goal}><span>{categories[key]?.icon} {categories[key]?.label||key}</span><b>{value.passed}/{value.total}</b><i><span style={{width:`${100*value.passed/value.total}%`}}/></i></div>)}</div>}
    </div>

    <div className="eval-layout">
      <aside className="eval-list" aria-label="Scénarios">
        <div className="eval-filters">
          <select value={category} onChange={e=>{setCategory(e.target.value);setSelectedId('')}} aria-label="Catégorie"><option value="all">Toutes les catégories</option>{Object.entries(categories).map(([key,value])=><option key={key} value={key}>{value.icon} {value.label}</option>)}</select>
          <label className="eval-toggle"><input type="checkbox" checked={showcase} onChange={e=>{setShowcase(e.target.checked);setSelectedId('')}}/>Vitrine seulement</label>
        </div>
        {visible.map(s=><button key={s.id} className={`eval-row ${selected?.id===s.id?'is-selected':''}`} onClick={()=>setSelectedId(s.id)}>
          <span className="eval-row-icon" aria-hidden="true">{categories[s.category]?.icon}</span>
          <span className="eval-row-copy"><strong>{s.title}</strong><small>{categories[s.category]?.label}{s.max_chapter?` · chapitre ${s.max_chapter}`:''}</small></span>
          <span className={`eval-mark ${s.last_result?(s.last_result.passed?'is-pass':'is-fail'):''}`} aria-label={s.last_result?(s.last_result.passed?'réussi':'échoué'):'pas encore joué'}>{running===s.id?'…':verdictMark(s.last_result)}</span>
        </button>)}
        {!visible.length&&<p className="eval-empty">Aucun scénario pour ce filtre.</p>}
      </aside>

      {selected?<ScenarioDetail scenario={selected} category={categories[selected.category]} titles={titles} running={running===selected.id} busy={!!running||progress.state==='running'} onPlay={()=>play(selected)} onNext={next}/>
        :<div className="eval-detail eval-empty">Choisis un scénario.</div>}
    </div>
  </section>
}

function ScenarioDetail({scenario,category,titles,running,busy,onPlay,onNext}:{scenario:EvalScenario;category?:EvalCategory;titles:Record<string,string>;running:boolean;busy:boolean;onPlay:()=>void;onNext:()=>void}) {
  const result=scenario.last_result
  const [open,setOpen]=useState<number|null>(null)
  useEffect(()=>setOpen(null),[scenario.id,result?.ran_at])
  return <article className="eval-detail">
    <div className="eval-tags"><span className="eval-tag">{category?.icon} {category?.label}</span>{scenario.work_ids.map(id=><span key={id} className="eval-tag eval-tag--book">{titles[id]||id}</span>)}{scenario.max_chapter&&<span className="eval-tag eval-tag--limit">🛡 lu jusqu’au chapitre {scenario.max_chapter}</span>}</div>
    <p className="eval-goal">{category?.goal}</p>
    <h2 className="eval-question">« {scenario.question} »</h2>
    {scenario.image&&<figure className="eval-image"><ProtectedImage src={`/api/evaluation/images/${encodeURIComponent(scenario.image)}`} alt="Image jointe à la question"/><figcaption>Image jointe à la question</figcaption></figure>}
    <div className="eval-expected"><span>{expectations[scenario.expect]}</span><p>{scenario.expected}</p>
      {scenario.gold.length>0&&<blockquote>Preuve annotée : “{scenario.gold[0].contains}”</blockquote>}</div>
    <div className="button-row">
      <button className="primary-button" onClick={onPlay} disabled={busy||!scenario.available}>{running?'Le moteur relit le livre…':result?'▶ Rejouer':'▶ Jouer'}</button>
      <button className="secondary-button" onClick={onNext} disabled={running}>Suivant →</button>
      {!scenario.available&&<small className="eval-warning">Livre absent de la bibliothèque.</small>}
    </div>
    {running&&<div className="thinking"><span className="loader"/><div><strong>Question envoyée au vrai moteur</strong><small>Même chemin que la discussion : recherche, réponse, vérification des citations.</small></div></div>}
    {!running&&result&&<div className={`eval-result ${result.passed?'is-pass':'is-fail'}`}>
      <div className="eval-verdict"><strong>{result.passed?'✓ Réussi':'✗ Échoué'}</strong><span className={`eval-outcome eval-outcome--${outcomeTone[result.outcome]}`}>{result.outcome_label}</span><small>{(result.elapsed_ms/1000).toFixed(1)} s · {ago(result.ran_at)}</small></div>
      {result.error?<p className="eval-answer">Erreur : {result.error}</p>:<p className="eval-answer">{result.answer.replace(/\s*\[[^\]]+\]/g,'')}</p>}
      {result.visual_observation&&<p className="eval-observation">Vu dans l’image : {result.visual_observation.visual_description||result.visual_observation.ocr_text}</p>}
      {result.citations.length>0&&<div className="eval-citations">{result.citations.map((c,i)=><div key={c.chunk_id}><button className="text-button" onClick={()=>setOpen(open===i?null:i)}>{open===i?'▾':'▸'} {c.work_title}, chapitre {c.chapter}</button>{open===i&&<blockquote>{c.text}</blockquote>}</div>)}</div>}
      <ul className="eval-checks">{result.checks.map(check=><li key={check.name} className={check.passed?'is-pass':'is-fail'}><span>{check.passed?'✓':'✗'}</span><div><strong>{check.name}</strong><small>{check.detail}</small></div></li>)}</ul>
    </div>}
  </article>
}
