import { useEffect, useMemo, useRef, useState } from 'react'
import type { Graph, Edge } from '../api'
import { graphNeighborhood } from './graphFilters'
import { graphLayout, groupRelations, pairKey, wrapName, edgeLabelPositions, type Point } from './graphLayout'

export function GraphCanvas({graph,query,depth}:{graph:Graph;query:string;depth:number}) {
  const [selection,setSelection]=useState<Edge|null>(null)
  const [focus,setFocus]=useState<string|null>(null)
  const [zoom,setZoom]=useState(1)
  const [showOther,setShowOther]=useState(false)
  const [manual,setManual]=useState<Record<string,Point>>({})
  const dragging=useRef(false)
  const releaseDrag=useRef<()=>void>(()=>{})
  useEffect(()=>()=>releaseDrag.current(),[])
  const visible=graphNeighborhood(graph,query,focus,depth,showOther)
  const groups=groupRelations(visible.edges)
  const connected=new Set(visible.edges.flatMap(e=>[e.source,e.target]))
  const drawn=visible.nodes.filter(n=>connected.has(n.id))
  const isolated=visible.nodes.filter(n=>!connected.has(n.id))
  const automatic=useMemo(()=>graphLayout(drawn,visible.edges),[graph,query,focus,depth,showOther])
  const positions={...automatic,...manual}
  const labelPositions=edgeLabelPositions(groups,positions)
  const focusedNode=visible.nodes.find(n=>n.id===focus)
  const selected=selection&&visible.edges.find(e=>e.source===selection.source&&e.target===selection.target&&e.label===selection.label&&e.evidence_chunk_id===selection.evidence_chunk_id)
  const selectedGroup=selected?groups.find(g=>pairKey(g[0])===pairKey(selected)):null
  const chooseNode=(id:string)=>{setFocus(current=>current===id?null:id);setSelection(null)}
  const reset=()=>{setZoom(1);setFocus(null);setSelection(null);setManual({})}
  const startDrag=(event:React.PointerEvent<SVGCircleElement>,id:string)=>{
    releaseDrag.current();dragging.current=false
    const svg=event.currentTarget.ownerSVGElement;if(!svg)return
    const start={x:event.clientX,y:event.clientY}
    event.currentTarget.setPointerCapture(event.pointerId)
    const move=(e:PointerEvent)=>{
      if(Math.hypot(e.clientX-start.x,e.clientY-start.y)<4&&!dragging.current)return
      dragging.current=true
      const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY
      const xy=p.matrixTransform(svg.getScreenCTM()?.inverse())
      setManual(current=>({...current,[id]:{x:Math.max(75,Math.min(845,(xy.x-460)/zoom+460)),y:Math.max(50,Math.min(480,(xy.y-285)/zoom+285))}}))
    }
    const end=()=>{window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',end);window.removeEventListener('pointercancel',end)}
    releaseDrag.current=end
    window.addEventListener('pointermove',move);window.addEventListener('pointerup',end);window.addEventListener('pointercancel',end)
  }
  if(!graph.nodes.length)return <div className="graph-empty"><h2>Le récit attend ses premières relations.</h2><p>Analyse les passages accessibles pour découvrir les personnages et leurs liens.</p></div>
  return <div className="graph-workspace">
    <div className="graph-canvas-wrap">
      <div className="graph-canvas-meta"><span><i className="legend-dot"/>Personnage</span><span><i className="legend-dot legend-group"/>Groupe</span><span className="graph-count">{visible.nodes.filter(n=>n.kind==='character').length} personnages{visible.nodes.some(n=>n.kind==='group')?` · ${visible.nodes.filter(n=>n.kind==='group').length} groupe(s)`:''} · {groups.length} liens</span></div>
      <div className="graph-view-options"><label><input type="checkbox" checked={showOther} onChange={e=>{setShowOther(e.target.checked);setFocus(null);setSelection(null)}}/> Lieux, thèmes et événements</label>{focus&&<button className="text-button" onClick={reset}>Effacer la sélection</button>}</div>
      {!!drawn.length&&<div className="graph-scroll" tabIndex={0} aria-label="Zone du graphe, défilement horizontal possible"><svg className="relation-graph" viewBox="0 0 920 560" role="group" aria-label="Graphe interactif des relations entre personnages">
        <defs><marker id="graph-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#718679" strokeWidth="1.5"/></marker></defs>
        <g transform={`translate(${460-460*zoom} ${285-285*zoom}) scale(${zoom})`}>
          {groups.map(items=>{
            const edge=items[0],a=positions[edge.source],b=positions[edge.target];if(!a||!b)return null
            const active=!!selected&&pairKey(selected)===pairKey(edge)
            const distance=Math.max(Math.hypot(b.x-a.x,b.y-a.y),1),dx=(b.x-a.x)/distance,dy=(b.y-a.y)/distance
            const center=labelPositions[pairKey(edge)]
            const label=edge.label.replaceAll('_',' '),lines=wrapName(label,24),width=Math.max(...lines.map(l=>l.length))*6.9+22
            return <g key={pairKey(edge)} className={`graph-edge${active?' graph-edge--selected':''}${groups.length<=16?' graph-edge--readable':''}`} role="button" tabIndex={0} aria-label={`${edge.source_label} ${label} ${edge.target_label}`} onClick={()=>setSelection(edge)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setSelection(edge)}}}>
              <title>{edge.source_label} {label} {edge.target_label} · {items.length} relation(s)</title>
              <line x1={a.x+dx*25} y1={a.y+dy*25} x2={b.x-dx*28} y2={b.y-dy*28} className={`edge-line${active?' edge-line--active':''}`} markerEnd="url(#graph-arrow)"/>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="edge-hit"/>
              <rect x={center.x-width/2} y={center.y-12} width={width} height={lines.length*17+7} rx="4" className="edge-label-bg"/>
              <text x={center.x} y={center.y+4} className="edge-label">{lines.map((line,i)=><tspan key={i} x={center.x} dy={i?17:0}>{line}</tspan>)}</text>
            </g>
          })}
          {drawn.map(node=>{
            const p=positions[node.id],active=focus===node.id
            return <g key={node.id} className="graph-node" role="button" tabIndex={0} aria-label={`${node.kind==='group'?'Groupe':'Personnage'} : ${node.label}`} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();chooseNode(node.id)}}} onClick={()=>{if(!dragging.current)chooseNode(node.id);dragging.current=false}}>
              <title>{node.label}</title><circle cx={p.x} cy={p.y} r={active?26:23} fill={node.kind==='group'?'#927445':node.kind==='character'?'#416b5a':'#788f61'} className={`node-orb${active?' node-orb--active':''}`} onPointerDown={e=>startDrag(e,node.id)}/>
              <text x={p.x} y={p.y+42} textAnchor="middle" className="node-name">{wrapName(node.label).map((line,i)=><tspan key={i} x={p.x} dy={i?18:0}>{line}</tspan>)}</text>
            </g>
          })}
        </g>
      </svg></div>}
      {!!drawn.length&&<p className="graph-scroll-hint">↔ Fais défiler le graphe horizontalement pour voir tous les personnages.</p>}
      {!visible.nodes.length&&<p className="graph-no-results">Aucun personnage ne correspond à cette recherche.</p>}
      {!!isolated.length&&<div className="graph-isolated"><span>Repérés dans le texte · aucun lien établi</span><div>{isolated.map(n=><button key={n.id} onClick={()=>chooseNode(n.id)} aria-pressed={focus===n.id}>{n.label}</button>)}</div></div>}
      <div className="graph-bottom"><p className="graph-hint">Clique sur un lien pour lire sa source, sur un personnage pour explorer ses relations.</p><div className="graph-zoom"><button onClick={()=>setZoom(z=>Math.min(1.6,z+.12))} aria-label="Zoom avant">＋</button><button onClick={()=>setZoom(z=>Math.max(.7,z-.12))} aria-label="Zoom arrière">−</button><button onClick={reset}>Recentrer</button></div></div>
    </div>
    <aside className="graph-evidence"><div className="page-kicker">PERSONNAGES & SOURCES</div>
      {selected?<><div className="relation-title"><span>{selected.source_label}</span><small>{selected.label}</small><span>{selected.target_label}</span></div>
        {(selectedGroup?.length||0)>1&&<div className="relation-options"><p>Autres liens entre ces personnages</p>{selectedGroup?.map((edge,i)=><button key={i} className="text-button" aria-pressed={edge===selected} onClick={()=>setSelection(edge)}>{edge.label}</button>)}</div>}
        <blockquote>{selected.evidence}</blockquote><div className="evidence-citation">{selected.evidence_chunk_id}</div>
      </>:focusedNode?<><h3>{focusedNode.label}</h3>{!!focusedNode.aliases?.length&&<div className="character-aliases"><span>Appellations dans le texte</span><p>{focusedNode.aliases.join(' · ')}</p></div>}
        <div className="character-connections">{groups.filter(items=>items[0].source===focusedNode.id||items[0].target===focusedNode.id).map(items=>{const edge=items[0];return <button key={pairKey(edge)} onClick={()=>setSelection(edge)}>{edge.source_label} <em>{edge.label}</em> {edge.target_label}</button>})}</div>
        <h4>Passages où ce personnage apparaît</h4>{focusedNode.mentions?.map(m=><details key={m.chunk_id}><summary>{m.citation_label}</summary><blockquote>{m.text}</blockquote></details>)}
      </>:<div className="graph-reading-key"><h3>Lire les relations</h3><p>La flèche se lit du premier personnage vers le second. Chaque libellé décrit le lien dans ce sens.</p><p>Un personnage peut porter plusieurs noms dans le texte. Ses appellations sont regroupées dans sa fiche.</p><h4>Explorer un personnage</h4><div className="graph-cast-index">{visible.nodes.filter(n=>connected.has(n.id)).map(n=><button key={n.id} onClick={()=>chooseNode(n.id)}>{n.label}<span>→</span></button>)}</div>
        {!!graph.identities?.unresolved_mentions&&<small>{graph.identities.unresolved_mentions} mentions sans identité suffisamment précise ne créent pas de personnage supplémentaire.</small>}
      </div>}
    </aside>
  </div>
}
