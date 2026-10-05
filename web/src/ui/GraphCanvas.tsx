import { useMemo, useState } from 'react'
import type { Graph, Edge } from '../api'

type Point = {x:number;y:number}
const colors:Record<string,string>={character:'#416b5a',person:'#416b5a',place:'#788f61',event:'#bc974f',organization:'#796687',theme:'#a86f55'}

export function GraphCanvas({graph,query,depth}:{graph:Graph;query:string;depth:number}) {
  const [selected,setSelected]=useState<Edge|null>(null)
  const [focus,setFocus]=useState<string|null>(null)
  const [zoom,setZoom]=useState(1)
  const [points,setPoints]=useState<Record<string,Point>>({})
  const nodes=useMemo(()=>graph.nodes.filter(n=>n.label.toLocaleLowerCase().includes(query.toLocaleLowerCase())),[graph.nodes,query])
  const positions=useMemo(()=>{
    const output:Record<string,Point>={}
    graph.nodes.forEach((node,index)=>{
      const angle=(index/Math.max(graph.nodes.length,1))*Math.PI*2-Math.PI/2
      output[node.id]=points[node.id]||{x:460+Math.cos(angle)*Math.min(235,55+graph.nodes.length*17),y:280+Math.sin(angle)*Math.min(190,42+graph.nodes.length*14)}
    })
    return output
  },[graph.nodes,points])
  const connected=useMemo(()=>{
    if(!focus)return null
    let near=new Set([focus])
    for(let i=0;i<depth;i++)near=new Set(graph.edges.filter(e=>near.has(e.source)||near.has(e.target)).flatMap(e=>[e.source,e.target]))
    return near
  },[focus,depth,graph.edges])
  const shownEdges=graph.edges.filter(edge=>
    nodes.some(n=>n.id===edge.source)&&nodes.some(n=>n.id===edge.target)&&(!connected||connected.has(edge.source)&&connected.has(edge.target)),
  )
  const visibleNodes=nodes.filter(n=>!connected||connected.has(n.id))
  const focusNode=(event:React.PointerEvent<SVGCircleElement>,id:string)=>{
    event.currentTarget.setPointerCapture(event.pointerId)
    const svg=event.currentTarget.ownerSVGElement
    if(!svg)return
    const move=(e:PointerEvent)=>{
      const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY
      const xy=p.matrixTransform(svg.getScreenCTM()?.inverse())
      setPoints(current=>({...current,[id]:{x:xy.x,y:xy.y}}))
    }
    const end=()=>{window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',end)}
    window.addEventListener('pointermove',move);window.addEventListener('pointerup',end)
  }
  if(graph.edges.length===0)return <div className="graph-empty"><div className="empty-ornament">⌘</div><h2>Le récit attend ses premières relations.</h2><p>Construis un premier lot de passages. Le modèle repère les personnages et leurs liens, puis associe chaque relation à une citation vérifiable.</p></div>
  return <div className="graph-workspace"><div className="graph-canvas-wrap"><div className="graph-canvas-meta"><span><i className="legend-dot"/>Personnage</span><span><i className="legend-square"/>Lieu, thème ou événement</span><span className="graph-count">{visibleNodes.length} personnages · {shownEdges.length} liens</span></div><svg className="relation-graph" viewBox="0 0 920 560" role="img" aria-label="Graphe interactif des relations entre personnages"><defs><marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#89948e" strokeWidth="1.4"/></marker><marker id="arrowhead-active" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#416b5a" strokeWidth="1.8"/></marker></defs><g transform={`translate(${460-460*zoom} ${280-280*zoom}) scale(${zoom})`}>
    {shownEdges.map((edge,index)=>{const a=positions[edge.source],b=positions[edge.target];if(!a||!b)return null;const active=selected===edge;const center={x:(a.x+b.x)/2,y:(a.y+b.y)/2};return <g key={`${edge.evidence_chunk_id}-${index}`} className="graph-edge" onClick={()=>setSelected(edge)} role="button" tabIndex={0} aria-label={`${edge.source} ${edge.label} ${edge.target}`} onKeyDown={e=>e.key==='Enter'&&setSelected(edge)}><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={active?'edge-line edge-line--active':'edge-line'} markerEnd={active?'url(#arrowhead-active)':'url(#arrowhead)'}/><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="edge-hit"/><rect x={center.x-55} y={center.y-13} width="110" height="22" rx="6" className="edge-label-bg"/><text x={center.x} y={center.y+3} className="edge-label">{edge.label.length>21?`${edge.label.slice(0,20)}…`:edge.label}</text></g>})}
    {visibleNodes.map(node=>{const p=positions[node.id];const isFocused=focus===node.id;return <g key={node.id} className="graph-node" onClick={()=>setFocus(current=>current===node.id?null:node.id)}><circle cx={p.x} cy={p.y} r={isFocused?30:26} fill={colors[node.kind]||'#8b9690'} className={isFocused?'node-orb node-orb--active':'node-orb'} onPointerDown={e=>focusNode(e,node.id)}/><text x={p.x} y={p.y+47} textAnchor="middle" className="node-name">{node.label.length>22?`${node.label.slice(0,21)}…`:node.label}</text></g>})}
    </g></svg><div className="graph-zoom"><button onClick={()=>setZoom(z=>Math.min(1.6,z+.12))} aria-label="Zoom avant">＋</button><button onClick={()=>setZoom(z=>Math.max(.7,z-.12))} aria-label="Zoom arrière">−</button><button onClick={()=>{setZoom(1);setFocus(null);setSelected(null);setPoints({})}}>Recentrer</button></div><p className="graph-hint">Choisis un personnage pour filtrer son voisinage, ou une relation pour lire sa preuve. Fais glisser les nœuds pour les déplacer.</p></div><aside className="graph-evidence"><div className="page-kicker">PASSAGE SOURCE</div>{selected?<><div className="relation-title"><span>{selected.source}</span><small>{selected.label}</small><span>{selected.target}</span></div><blockquote>{selected.evidence||'Le passage est bien rattaché au graphe. Ouvre le chapitre correspondant pour lire son contexte.'}</blockquote><div className="evidence-citation"><span>⌞</span>Référence <strong>{selected.evidence_chunk_id}</strong></div>{focus&&<button className="text-button" onClick={()=>setFocus(null)}>Effacer le filtre de personnage</button>}</>:<div className="evidence-placeholder"><span>⌞</span><p>Les liens du graphe sont toujours accompagnés de leur source. Sélectionne une ligne pour consulter l’extrait.</p></div>}</aside></div>
}
