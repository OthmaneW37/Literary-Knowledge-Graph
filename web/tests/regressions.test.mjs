import test from 'node:test'
import assert from 'node:assert/strict'
import {readFile} from 'node:fs/promises'
import ts from 'typescript'

async function loadSource(path) {
  const source=await readFile(new URL(path,import.meta.url),'utf8')
  const result=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}})
  return import(`data:text/javascript;base64,${Buffer.from(result.outputText).toString('base64')}`)
}
const {graphNeighborhood}=await loadSource('../src/ui/graphFilters.ts')
const {api,ApiError}=await loadSource('../src/api.ts')
const graph={nodes:[
  {id:'book:a',label:'Alice',kind:'character'},
  {id:'book:b',label:'Bernard',kind:'character'},
  {id:'book:c',label:'Clara',kind:'character'},
  {id:'book:d',label:'David',kind:'character'},
  {id:'book:alone',label:'Isolated',kind:'character'},
  {id:'other:a',label:'Alice',kind:'character'},
  {id:'book:place',label:'Paris',kind:'place'},
],edges:[{source:'book:a',target:'book:b'},{source:'book:b',target:'book:c'},
  {source:'book:c',target:'book:d'},{source:'book:a',target:'book:place'}]}

test('search retains matching people and their immediate neighbours',()=>{
  const result=graphNeighborhood(graph,' ALICE ',null,1,false)
  assert.deepEqual(result.nodes.map(n=>n.id),['book:a','book:b','other:a'])
  assert.equal(result.edges.length,1)
})
test('two-hop exploration stops before the third hop',()=>{
  const result=graphNeighborhood(graph,'','book:a',2,false)
  assert.deepEqual(result.nodes.map(n=>n.id),['book:a','book:b','book:c'])
})
test('an isolated character does not disappear when selected',()=>{
  const result=graphNeighborhood(graph,'','book:alone',2,false)
  assert.deepEqual(result.nodes.map(n=>n.id),['book:alone'])
  assert.deepEqual(result.edges,[])
})
test('new searches override a previous node focus',()=>{
  assert.deepEqual(graphNeighborhood(graph,'Isolated','book:a',1,false).nodes.map(n=>n.id),['book:alone'])
})
test('non-character nodes remain opt-in and absent queries return no nodes',()=>{
  assert.equal(graphNeighborhood(graph,'',null,1,false).nodes.length,6)
  assert.equal(graphNeighborhood(graph,'',null,1,true).nodes.length,7)
  assert.equal(graphNeighborhood(graph,'unknown',null,1,true).nodes.length,0)
})

function mockNetwork(t,fetch) {
  const originalFetch=globalThis.fetch
  const originalStorage=Object.getOwnPropertyDescriptor(globalThis,'localStorage')
  globalThis.fetch=fetch
  Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:()=>null}})
  t.after(()=>{globalThis.fetch=originalFetch;if(originalStorage)Object.defineProperty(globalThis,'localStorage',originalStorage);else delete globalThis.localStorage})
}

test('reading loads all pages of a chapter longer than 100 passages',async t=>{
  const offsets=[]
  mockNetwork(t,async path=>{
    const offset=Number(new URL(path,'http://localhost').searchParams.get('offset'));offsets.push(offset)
    const count=Math.min(100,205-offset)
    return Response.json({total:205,passages:Array.from({length:count},(_,i)=>({chunk_id:String(offset+i)}))})
  })
  const result=await api.passages('book',1)
  assert.equal(result.passages.length,205)
  assert.deepEqual(offsets,[0,100,200])
  assert.equal(result.passages[204].chunk_id,'204')
})
test('network failures explain how to restore the API',async t=>{
  mockNetwork(t,async()=>{throw new TypeError('Failed to fetch')})
  await assert.rejects(api.library(),e=>e instanceof ApiError&&e.message.includes('8000'))
})
test('validation errors are readable strings instead of object Object',async t=>{
  mockNetwork(t,async()=>Response.json({detail:[{msg:'Question trop courte'}]},{status:422}))
  await assert.rejects(api.library(),e=>e.message==='Question trop courte')
})

const {graphLayout,groupRelations,wrapName,edgeLabelPositions}=await loadSource('../src/ui/graphLayout.ts')
test('alias searches find one canonical character, including groups',()=>{
  const data={nodes:[{id:'mother',label:'Mère de Gregor',kind:'character',aliases:['his mother',"Gregor's mother"]},{id:'lodgers',label:'Locataires',kind:'group'}],edges:[]}
  assert.deepEqual(graphNeighborhood(data,'his mother',null,1,false).nodes.map(n=>n.id),['mother'])
  assert.equal(graphNeighborhood(data,'',null,1,false).nodes.length,2)
})
test('family relation is the readable representative of a pair',()=>{
  const data=[{source:'a',target:'b',label:'menace'},{source:'b',target:'a',label:'père de',category:'family'}]
  const groups=groupRelations(data)
  assert.equal(groups.length,1);assert.equal(groups[0][0].label,'père de');assert.equal(data[0].label,'menace')
})
test('layout is deterministic, bounded, and spreads nodes across the canvas',()=>{
  const nodes=Array.from({length:10},(_,i)=>({id:String(i),label:'Person '+i,kind:'character'}))
  const edges=nodes.slice(1).map(n=>({source:'0',target:n.id,label:'parle à'}))
  const first=graphLayout(nodes,edges),second=graphLayout([...nodes].reverse(),edges)
  assert.deepEqual(first,second)
  const values=Object.values(first)
  assert.ok(values.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)&&p.x>=75&&p.x<=845&&p.y>=50&&p.y<=480))
  assert.ok(Math.max(...values.map(p=>p.x))-Math.min(...values.map(p=>p.x))>650)
  assert.equal(Object.keys(edgeLabelPositions(groupRelations(edges),first)).length,9)
})
test('long names wrap instead of losing the distinguishing last word',()=>{
  const label='Le père du jeune voyageur'
  assert.equal(wrapName(label,15).join(' '),label)
  assert.ok(wrapName(label,15).length>1)
})

const {runDelta,verdictMark,ago}=await loadSource('../src/ui/evaluationStats.ts')
const run=(passed,state='done',count=76)=>({state,scenario_count:count,summary:{passed}})

test('evaluation delta compares the two latest complete runs of the same set',()=>{
  assert.equal(runDelta([run(60),run(40,'cancelled'),run(34)]),26)
  assert.equal(runDelta([run(60)]),null)
  assert.equal(runDelta([run(60,'done',76),run(30,'done',70)]),null)
})
test('evaluation verdict marks and relative times read naturally',()=>{
  assert.equal(verdictMark(null),'–')
  assert.equal(verdictMark({passed:true}),'✓')
  assert.equal(verdictMark({passed:false}),'✗')
  const now=Date.parse('2026-10-22T12:00:00Z')
  assert.equal(ago('2026-10-22T11:58:00Z',now),'il y a 2 min')
  assert.equal(ago('2026-10-22T09:00:00Z',now),'il y a 3 h')
})
test('a demo scenario run posts to its own route',async t=>{
  const calls=[]
  mockNetwork(t,async(path,init)=>{calls.push([path,init?.method]);return Response.json({passed:true})})
  await api.runScenario('spoil-01')
  await api.startEvaluation(null)
  assert.deepEqual(calls,[['/api/evaluation/scenarios/spoil-01/run','POST'],['/api/evaluation/runs','POST']])
})

const {localMatches,openProgress}=await loadSource('../src/ui/exploreSearch.ts')
test('explorer finds owned books by title or author, ignoring accents and case',()=>{
  const library=[{id:'a',title:'Les Misérables',author:'Victor Hugo',archived:false},{id:'b',title:'The Trial',author:'Franz Kafka',archived:false},{id:'c',title:'Old',author:'Hugo',archived:true}]
  assert.deepEqual(localMatches(library,'miserables').map(b=>b.id),['a'])
  assert.deepEqual(localMatches(library,'HUGO').map(b=>b.id),['a'])
  assert.deepEqual(localMatches(library,'de'),[])
})
test('opening progress moves from download to embeddings',()=>{
  assert.equal(openProgress(undefined),0)
  assert.equal(openProgress({state:'downloading'}),15)
  assert.equal(openProgress({state:'ready',embedded:30,to_embed:60}),70)
  assert.equal(openProgress({state:'done'}),100)
})
test('opening a public-domain book posts its Gutenberg number',async t=>{
  const calls=[]
  mockNetwork(t,async(path,init)=>{calls.push([path,init?.body]);return Response.json({job_id:'j',state:'queued'})})
  await api.openBook({provider:'gutendex',provider_id:'28054'})
  await api.openJob('j')
  assert.deepEqual(calls,[['/api/catalog/open','{"provider":"gutenberg","provider_id":28054}'],['/api/catalog/jobs/j',undefined]])
})

const {formatAnswer,highlightParts}=await loadSource('../src/ui/answerFormat.ts')
test('answers replace source ids with numbered references and keep lists, bold and quotes',()=>{
  const blocks=formatAnswer('Voici les liens :\n\n- **Polina** → belle-fille → **General** [b_ch08_p016]\n> « Exact words » [b_ch17_p015]\nFin [unknown_id].',['b_ch08_p016','b_ch17_p015'])
  assert.deepEqual(blocks.map(b=>b.kind),['paragraph','bullet','quote','paragraph'])
  assert.deepEqual(blocks[1].segments.filter(s=>s.kind!=='text'),[{kind:'bold',text:'Polina'},{kind:'bold',text:'General'},{kind:'cite',number:1}])
  assert.deepEqual(blocks[2].segments.at(-1),{kind:'cite',number:2})
  assert.equal(JSON.stringify(blocks[3]).includes('unknown_id'),false)
})
test('the verified quote is highlighted inside its passage',()=>{
  assert.deepEqual(highlightParts('He said yes today.',[3,11]),['He ','said yes',' today.'])
  assert.deepEqual(highlightParts('abc',null),['abc','',''])
})
