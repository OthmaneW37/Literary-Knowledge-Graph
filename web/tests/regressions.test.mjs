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
