import type { Edge, GraphNode } from '../api'
export type Point = {x:number;y:number}
export const pairKey = (edge:Edge) => [edge.source,edge.target].sort().join('\0')

export function groupRelations(edges:Edge[]) {
  const groups=new Map<string,Edge[]>()
  edges.forEach(edge=>{const key=pairKey(edge);groups.set(key,[...(groups.get(key)||[]),edge])})
  groups.forEach(items=>items.sort((a,b)=>Number(b.category==='family')-Number(a.category==='family')||a.label.length-b.label.length))
  return [...groups.values()]
}

// Stable, bounded force layout. One spring per pair, so repeated citations do
// not pull a pair of characters closer than their other relationships.
export function graphLayout(nodes:GraphNode[], edges:Edge[]):Record<string,Point> {
  const ordered=[...nodes].sort((a,b)=>a.id.localeCompare(b.id))
  const points:Record<string,Point>={}
  ordered.forEach((n,i)=>{const t=2*Math.PI*i/Math.max(ordered.length,1);points[n.id]={x:460+280*Math.cos(t),y:290+190*Math.sin(t)}})
  const links=groupRelations(edges).map(g=>g[0])
  for(let step=0;step<240;step++){
    const forces=Object.fromEntries(ordered.map(n=>[n.id,{x:(460-points[n.id].x)*.025,y:(285-points[n.id].y)*.025}]))
    for(let i=0;i<ordered.length;i++)for(let j=i+1;j<ordered.length;j++){
      const a=ordered[i].id,b=ordered[j].id,dx=points[a].x-points[b].x,dy=points[a].y-points[b].y
      const distance=Math.max(Math.hypot(dx,dy),1),force=18000/(distance*distance)
      forces[a].x+=dx/distance*force;forces[a].y+=dy/distance*force
      forces[b].x-=dx/distance*force;forces[b].y-=dy/distance*force
    }
    links.forEach(e=>{
      const a=points[e.source],b=points[e.target];if(!a||!b)return
      const dx=b.x-a.x,dy=b.y-a.y,d=Math.max(Math.hypot(dx,dy),1),f=(d-235)*.025
      forces[e.source].x+=dx/d*f;forces[e.source].y+=dy/d*f
      forces[e.target].x-=dx/d*f;forces[e.target].y-=dy/d*f
    })
    const speed=1-step/300
    ordered.forEach(n=>{const p=points[n.id],f=forces[n.id];p.x=Math.max(100,Math.min(820,p.x+Math.max(-8,Math.min(8,f.x))*speed));p.y=Math.max(60,Math.min(470,p.y+Math.max(-8,Math.min(8,f.y))*speed))})
  }
  if(ordered.length>1){
    const xs=ordered.map(n=>points[n.id].x),ys=ordered.map(n=>points[n.id].y)
    const left=Math.min(...xs),top=Math.min(...ys),width=Math.max(...xs)-left,height=Math.max(...ys)-top
    ordered.forEach(n=>{const p=points[n.id];p.x=width>1?90+(p.x-left)/width*740:460;p.y=height>1?65+(p.y-top)/height*390:285})
  }
  return points
}

export function edgeLabelPositions(groups:Edge[][], positions:Record<string,Point>) {
  const labels:Record<string,Point>={}
  const placed:{x:number;y:number;width:number;height:number}[]=[]
  const overlap=(a:typeof placed[number],b:typeof placed[number])=>Math.max(0,(a.width+b.width)/2-Math.abs(a.x-b.x))*Math.max(0,(a.height+b.height)/2-Math.abs(a.y-b.y))
  groups.forEach(items=>{
    const edge=items[0],a=positions[edge.source],b=positions[edge.target];if(!a||!b)return
    const lines=wrapName(edge.label,24),width=Math.max(...lines.map(l=>l.length))*6.9+28,height=lines.length*17+12
    let best={x:(a.x+b.x)/2,y:(a.y+b.y)/2},cost=Infinity
    for(const t of [.5,.35,.65,.22,.78]){
      const candidate={x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t,width,height}
      let score=Math.abs(t-.5)*50
      Object.values(positions).forEach(p=>{score+=overlap(candidate,{x:p.x,y:p.y+15,width:155,height:105})})
      placed.forEach(p=>{score+=overlap(candidate,p)*3})
      if(score<cost){best=candidate;cost=score}
    }
    labels[pairKey(edge)]=best;placed.push({...best,width,height})
  })
  return labels
}

export function wrapName(label:string, max=22):string[] {
  const lines=['']
  label.split(/\s+/).forEach(word=>{const i=lines.length-1;if(lines[i]&&lines[i].length+word.length+1>max)lines.push(word);else lines[i]+=(lines[i]?' ':'')+word})
  return lines
}
