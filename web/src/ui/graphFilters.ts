import type { Graph } from '../api'

export function graphNeighborhood(graph:Graph, query:string, focus:string|null, depth:number, showOther:boolean) {
  const nodes=graph.nodes.filter(n=>showOther||n.kind==='character'||n.kind==='group')
  const allowed=new Set(nodes.map(n=>n.id))
  const edges=graph.edges.filter(e=>allowed.has(e.source)&&allowed.has(e.target))
  const term=query.trim().toLocaleLowerCase()
  if(!term&&!focus)return {nodes,edges}
  let visible=new Set(term?nodes.filter(n=>[n.label,...(n.aliases||[])].some(label=>label.toLocaleLowerCase().includes(term))).map(n=>n.id):focus?[focus]:[])
  for(let i=0;i<depth;i++){
    const expanded=new Set(visible)
    edges.forEach(edge=>{if(visible.has(edge.source)||visible.has(edge.target)){expanded.add(edge.source);expanded.add(edge.target)}})
    visible=expanded
  }
  return {nodes:nodes.filter(n=>visible.has(n.id)),edges:edges.filter(e=>visible.has(e.source)&&visible.has(e.target))}
}
