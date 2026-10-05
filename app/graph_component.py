from __future__ import annotations

import json

import streamlit.components.v1 as components

from rag.models import Visualization


def render_relationship_graph(
    visualization: Visualization,
    evidence: dict[str, str],
    height: int = 480,
) -> None:
    """Render a self-contained, local interactive graph with evidence on edge selection."""
    data = {
        "nodes": [
            {"id": node.id, "label": node.label, "kind": node.kind}
            for node in visualization.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
                "evidence": edge.evidence_chunk_id,
                "text": evidence.get(edge.evidence_chunk_id, "Source non disponible."),
            }
            for edge in visualization.edges
        ],
    }
    serialized = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    html = r"""<!doctype html>
<html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#fcfcf8;color:#26343a;font:14px "Aptos","Segoe UI",sans-serif}
.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px;background:#e4e9e4;border:1px solid #d0d9d3;border-radius:12px 12px 0 0}
input,select,button{color:#26343a;background:#fcfcf8;border:1px solid #cbd6cf;border-radius:8px;padding:7px 9px}
input{min-width:150px;flex:1}button{cursor:pointer}button:hover{background:#edf2ed}
.legend{color:#65716f;font-size:12px;margin-left:auto}.stage{display:grid;grid-template-columns:minmax(0,1fr) 250px;background:#fcfcf8;border:1px solid #dce3de;border-top:0;border-radius:0 0 12px 12px;overflow:hidden}
canvas{width:100%;height:380px;display:block;touch-action:none}.detail{border-left:1px solid #dce3de;padding:14px;overflow:auto;max-height:380px;color:#46544f}
.detail h3{margin:0 0 8px;color:#26343a;font-size:15px}.detail blockquote{margin:10px 0;padding:9px;border-left:3px solid #a8b58a;background:#f1f4ef;white-space:pre-wrap;font-family:Georgia,"Times New Roman",serif;line-height:1.7}
@media(max-width:680px){.stage{grid-template-columns:1fr}.detail{border-left:0;border-top:1px solid #263246;max-height:170px}.legend{width:100%;margin:0}}
</style></head><body>
<div class="toolbar"><input id="search" placeholder="Rechercher un personnage…"><select id="filter"><option value="">Toutes les relations</option></select><button id="zoomIn">＋</button><button id="zoomOut">−</button><button id="reset">Recentrer</button><span class="legend">● Personnage　■ Autre entité　Cliquer une relation pour sa source</span></div>
<div class="stage"><canvas id="graph"></canvas><aside class="detail" id="detail"><h3>Graphe des relations</h3><p>Déplace les nœuds. Fais glisser le fond pour déplacer le graphe, utilise la molette pour zoomer. Clique sur une arête pour lire son passage source.</p></aside></div>
<script>
const DATA=__DATA__, canvas=document.getElementById('graph'),ctx=canvas.getContext('2d'),detail=document.getElementById('detail');
const colors={character:'#416b5a',person:'#416b5a',place:'#788f61',event:'#c59a4b',organization:'#7a6385',theme:'#a86f55'};
const W=()=>canvas.clientWidth,H=()=>canvas.clientHeight;
let scale=1,pan={x:0,y:0},drag=null,selected=null,query='',relationFilter='',nodes=DATA.nodes.map((n,i)=>({...n,x:W()/2+Math.cos(i*2*Math.PI/DATA.nodes.length)*Math.min(W(),H())*.28,y:H()/2+Math.sin(i*2*Math.PI/DATA.nodes.length)*Math.min(W(),H())*.28,r:22}));
const edges=DATA.edges.map(e=>({...e}));
function resize(){let dpr=window.devicePixelRatio||1;canvas.width=W()*dpr;canvas.height=H()*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
function screen(p){return{x:p.x*scale+pan.x,y:p.y*scale+pan.y}}
function visibleNode(n){return !query||n.label.toLowerCase().includes(query)}
function visibleEdge(e){return(!relationFilter||e.label===relationFilter)&&nodes.some(n=>n.id===e.source&&visibleNode(n))&&nodes.some(n=>n.id===e.target&&visibleNode(n))}
function draw(){ctx.clearRect(0,0,W(),H());ctx.save();ctx.translate(pan.x,pan.y);ctx.scale(scale,scale);
for(const e of edges){if(!visibleEdge(e))continue;let a=nodes.find(n=>n.id===e.source),b=nodes.find(n=>n.id===e.target);if(!a||!b)continue;let active=selected===e;ctx.strokeStyle=active?'#788f61':'#84918c';ctx.lineWidth=(active?2.5:1.4)/scale;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();let angle=Math.atan2(b.y-a.y,b.x-a.x);let x=b.x-Math.cos(angle)*b.r,y=b.y-Math.sin(angle)*b.r;ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-10/scale*Math.cos(angle-.45),y-10/scale*Math.sin(angle-.45));ctx.lineTo(x-10/scale*Math.cos(angle+.45),y-10/scale*Math.sin(angle+.45));ctx.closePath();ctx.fill();ctx.fillStyle='#53625d';ctx.font=`${12/scale}px system-ui`;ctx.textAlign='center';ctx.fillText(e.label,(a.x+b.x)/2,(a.y+b.y)/2-7/scale)}
for(const n of nodes){if(!visibleNode(n))continue;ctx.beginPath();ctx.fillStyle=colors[n.kind]||'#8b9690';ctx.strokeStyle='#fcfcf8';ctx.lineWidth=3/scale;ctx.arc(n.x,n.y,n.r,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#26343a';ctx.font=`${12/scale}px system-ui`;ctx.textAlign='center';ctx.fillText(n.label,n.x,n.y+n.r+16/scale)}ctx.restore()}
function hitNode(x,y){return nodes.find(n=>visibleNode(n)&&Math.hypot(n.x-x,n.y-y)<n.r+8)}
function hitEdge(x,y){let best=null,dist=12/scale;for(let e of edges){if(!visibleEdge(e))continue;let a=nodes.find(n=>n.id===e.source),b=nodes.find(n=>n.id===e.target);if(!a||!b)continue;let dx=b.x-a.x,dy=b.y-a.y,t=Math.max(0,Math.min(1,((x-a.x)*dx+(y-a.y)*dy)/(dx*dx+dy*dy||1))),d=Math.hypot(x-a.x-t*dx,y-a.y-t*dy);if(d<dist){best=e;dist=d}}return best}
function showEdge(e){selected=e;detail.innerHTML=`<h3>${esc(e.source)} → ${esc(e.target)}</h3><p><b>${esc(e.label)}</b></p><small>Source : ${esc(e.evidence)}</small><blockquote>${esc(e.text)}</blockquote>`;draw()}
function esc(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function point(ev){let r=canvas.getBoundingClientRect(),sx=ev.clientX-r.left,sy=ev.clientY-r.top;return{x:(sx-pan.x)/scale,y:(sy-pan.y)/scale,sx,sy}}
canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);let p=point(e),n=hitNode(p.x,p.y);drag=n?{node:n,ox:p.x-n.x,oy:p.y-n.y,lastX:p.sx,lastY:p.sy,moved:false}:{pan:true,lastX:p.sx,lastY:p.sy,moved:false}});
canvas.addEventListener('pointermove',e=>{if(!drag)return;let p=point(e);if(Math.hypot(p.sx-drag.lastX,p.sy-drag.lastY)>2)drag.moved=true;if(drag.node){drag.node.x=p.x-drag.ox;drag.node.y=p.y-drag.oy}else{pan.x+=p.sx-drag.lastX;pan.y+=p.sy-drag.lastY;drag.lastX=p.sx;drag.lastY=p.sy}draw()});
canvas.addEventListener('pointerup',e=>{let p=point(e);if(!drag?.node&&!drag?.moved){let edge=hitEdge(p.x,p.y);if(edge)showEdge(edge)}drag=null});
canvas.addEventListener('wheel',e=>{e.preventDefault();let p=point(e),before={x:(p.sx-pan.x)/scale,y:(p.sy-pan.y)/scale};scale=Math.max(.45,Math.min(2.5,scale*(e.deltaY<0?1.12:.89)));pan.x=p.sx-before.x*scale;pan.y=p.sy-before.y*scale;draw()},{passive:false});
document.getElementById('search').addEventListener('input',e=>{query=e.target.value.toLowerCase();draw()});
let select=document.getElementById('filter');[...new Set(edges.map(e=>e.label))].sort().forEach(x=>{let o=document.createElement('option');o.value=x;o.textContent=x;select.appendChild(o)});select.addEventListener('change',e=>{relationFilter=e.target.value;draw()});
document.getElementById('zoomIn').onclick=()=>{scale=Math.min(2.5,scale*1.15);draw()};document.getElementById('zoomOut').onclick=()=>{scale=Math.max(.45,scale/1.15);draw()};document.getElementById('reset').onclick=()=>{scale=1;pan={x:0,y:0};selected=null;detail.innerHTML='<h3>Graphe des relations</h3><p>Déplace les nœuds. Fais glisser le fond pour déplacer le graphe, utilise la molette pour zoomer. Clique sur une arête pour lire son passage source.</p>';draw()};
window.addEventListener('resize',resize);resize();
</script></body></html>""".replace("__DATA__", serialized)
    components.html(html, height=height, scrolling=False)
