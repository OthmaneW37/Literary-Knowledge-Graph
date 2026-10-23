export type Segment = {kind:'text'|'bold';text:string} | {kind:'cite';number:number}
export type Block = {kind:'paragraph'|'bullet'|'quote';segments:Segment[]}

/**
 * Turn an answer's plain text into blocks: "- " bullets, "> " quotations,
 * **bold** names and [chunk_id] markers replaced by the 1-based number of the
 * matching citation. Markers that match no citation are dropped.
 */
export function formatAnswer(content:string, citationIds:string[]):Block[] {
  const blocks:Block[]=[]
  for (const raw of content.split('\n')) {
    const line=raw.trim()
    if (!line) continue
    const kind=line.startsWith('- ')||line.startsWith('• ')?'bullet':line.startsWith('> ')?'quote':'paragraph'
    const body=kind==='paragraph'?line:line.slice(2)
    const segments:Segment[]=[]
    for (const part of body.split(/(\*\*[^*]+\*\*|\[[^\]]+\])/)) {
      if (!part) continue
      if (part.startsWith('**')&&part.endsWith('**')) segments.push({kind:'bold',text:part.slice(2,-2)})
      else if (part.startsWith('[')&&part.endsWith(']')) {
        for (const id of part.slice(1,-1).split(',').map(item=>item.trim().replace(/^SOURCE\s+/,''))) {
          const index=citationIds.indexOf(id)
          if (index>=0&&!segments.some(s=>s.kind==='cite'&&s.number===index+1)) segments.push({kind:'cite',number:index+1})
        }
      } else segments.push({kind:'text',text:part})
    }
    // Drop the space left before a removed marker at the end of a sentence.
    for (let i=0;i<segments.length;i++) {
      const segment=segments[i], next=segments[i+1]
      if (segment.kind==='text'&&(!next||next.kind==='cite')) segment.text=segment.text.replace(/\s+$/,'')
    }
    blocks.push({kind,segments})
  }
  return blocks
}

/** Split a passage around its highlighted quote span. */
export function highlightParts(text:string, span?:number[]|null):[string,string,string] {
  if (!span||span.length!==2||span[0]<0||span[1]>text.length||span[0]>=span[1]) return [text,'','']
  return [text.slice(0,span[0]),text.slice(span[0],span[1]),text.slice(span[1])]
}
